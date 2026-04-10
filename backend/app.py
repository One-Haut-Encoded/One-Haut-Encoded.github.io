# AI-assisted (Claude Code, claude.ai) -- https://claude.ai
"""FastAPI backend for One Haut Encoded — serves live model inference.

Downloads trained models from HF Hub (DTanzillo/one-haut-encoded) on startup.
Exposes /recommend and /purchase_history endpoints for the GitHub Pages frontend.
"""

import os
import pickle
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import snapshot_download
from sklearn.preprocessing import LabelEncoder, normalize

# Add bundled scripts to path for model class definitions (needed for pickle)
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from train_baseline import PopularityRecommender
from train_knn import KNNRecommender
from train_ncf import NCFModel


# ── Global state ──────────────────────────────────────────────
_models: dict = {}
_articles: pd.DataFrame | None = None
_interactions: pd.DataFrame | None = None
_ncf_mappings: dict | None = None
_ncf_meta_tensor: torch.Tensor | None = None


def _register_pickle_classes():
    """Pickles saved from __main__ need the classes registered there."""
    import __main__
    __main__.PopularityRecommender = PopularityRecommender
    __main__.KNNRecommender = KNNRecommender
    __main__.NCFModel = NCFModel


def _build_meta_tensor(articles: pd.DataFrame) -> torch.Tensor:
    """Build the same one-hot metadata tensor used during NCF training."""
    feature_cols = [
        "product_type_name", "product_group_name", "colour_group_name",
        "department_name", "index_group_name",
    ]
    parts = []
    for col in feature_cols:
        if col in articles.columns:
            le = LabelEncoder()
            vals = articles[col].fillna("unknown").values
            encoded = le.fit_transform(vals)
            one_hot = np.zeros((len(vals), len(le.classes_)), dtype=np.float32)
            one_hot[np.arange(len(vals)), encoded] = 1
            parts.append(one_hot)
    meta_np = np.hstack(parts)
    return torch.tensor(meta_np, dtype=torch.float32)


def _load_all():
    """Download from HF Hub and load all models + data."""
    global _models, _articles, _interactions, _ncf_mappings, _ncf_meta_tensor

    repo_id = "DTanzillo/one-haut-encoded"
    local_dir = os.environ.get("ONEHAUT_DATA_DIR", "/app/hf_data")

    print(f"Downloading from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )

    data_dir = Path(local_dir) / "data" / "processed"
    model_dir = Path(local_dir) / "models"

    # Load metadata CSVs
    _articles = pd.read_csv(data_dir / "articles_subset.csv", dtype={"article_id": str})
    _interactions = pd.read_csv(
        data_dir / "interactions.csv",
        dtype={"customer_id": str, "article_id": str},
    )
    print(f"Loaded {len(_articles)} articles, {len(_interactions)} interactions")

    _register_pickle_classes()

    # Popularity baseline
    pop_path = model_dir / "baseline" / "popularity.pkl"
    if pop_path.exists():
        with open(pop_path, "rb") as f:
            _models["popularity"] = pickle.load(f)
        print("Loaded: popularity")

    # KNN content-based
    knn_path = model_dir / "knn" / "knn.pkl"
    if knn_path.exists():
        with open(knn_path, "rb") as f:
            _models["knn"] = pickle.load(f)
        print("Loaded: knn")

    # NCF + metadata
    ncf_pt = model_dir / "ncf" / "ncf-meta.pt"
    ncf_map = model_dir / "ncf" / "ncf-meta_mappings.pkl"
    if ncf_pt.exists() and ncf_map.exists():
        with open(ncf_map, "rb") as f:
            _ncf_mappings = pickle.load(f)

        user_to_idx = _ncf_mappings["user_to_idx"]
        item_to_idx = _ncf_mappings["item_to_idx"]

        sd = torch.load(ncf_pt, map_location="cpu", weights_only=True)
        embed_dim = sd["user_embed.weight"].shape[1]
        n_meta = sd["mlp.0.weight"].shape[1] - embed_dim * 2

        model = NCFModel(
            n_users=len(user_to_idx),
            n_items=len(item_to_idx),
            embed_dim=embed_dim,
            n_meta_features=n_meta,
        )
        model.load_state_dict(sd)
        model.eval()
        _models["ncf"] = model

        _ncf_meta_tensor = _build_meta_tensor(_articles)
        print(f"Loaded: ncf-meta (embed_dim={embed_dim}, meta_features={n_meta})")

    print(f"Ready — {len(_models)} models loaded")


# ── App lifecycle ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_all()
    yield

app = FastAPI(title="One Haut Encoded API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://one-haut-encoded.github.io",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────
def _enrich(article_ids: list[str]) -> list[dict]:
    """Convert article IDs to card-ready dicts with metadata."""
    if _articles is None or not article_ids:
        return [{"article_id": aid} for aid in article_ids]

    by_id = _articles.set_index("article_id")
    out = []
    for aid in article_ids:
        if aid in by_id.index:
            row = by_id.loc[aid]
            out.append({
                "article_id": aid,
                "product_name": str(row.get("prod_name", "")),
                "product_type": str(row.get("product_type_name", "")),
                "colour": str(row.get("colour_group_name", "")),
                "department": str(row.get("department_name", "")),
            })
        else:
            out.append({"article_id": aid})
    return out


def _seen_articles(customer_id: str) -> set[str]:
    """Articles this customer has already purchased."""
    if _interactions is None:
        return set()
    sub = _interactions[_interactions["customer_id"] == customer_id]
    return set(sub["article_id"].astype(str))


def _recommend_popularity(customer_id: str, k: int) -> list[str]:
    model = _models.get("popularity")
    if not model:
        return []
    return model.recommend(customer_id, k=k, mode="global")


def _recommend_knn(customer_id: str, k: int) -> list[str]:
    model = _models.get("knn")
    if not model:
        return []
    recs = model.recommend(customer_id, k=k)
    if recs:
        return recs
    # Cold start fallback to popularity
    return _recommend_popularity(customer_id, k)


def _recommend_ncf(customer_id: str, k: int) -> list[str]:
    model = _models.get("ncf")
    if not model or not _ncf_mappings:
        return []

    user_to_idx = _ncf_mappings["user_to_idx"]
    item_to_idx = _ncf_mappings["item_to_idx"]

    uid = user_to_idx.get(customer_id)
    if uid is None:
        return _recommend_popularity(customer_id, k)

    idx_to_item = {v: k for k, v in item_to_idx.items()}
    n_items = len(item_to_idx)
    seen = _seen_articles(customer_id)

    scores = []
    batch_size = 512
    with torch.no_grad():
        for start in range(0, n_items, batch_size):
            end = min(start + batch_size, n_items)
            bsz = end - start
            u_batch = torch.full((bsz,), uid, dtype=torch.long)
            i_batch = torch.arange(start, end, dtype=torch.long)
            meta_batch = _ncf_meta_tensor[start:end] if _ncf_meta_tensor is not None else None
            preds = model(u_batch, i_batch, meta_batch)
            for j in range(bsz):
                scores.append((float(preds[j].item()), start + j))

    scores.sort(key=lambda x: x[0], reverse=True)

    picked = []
    for score, idx in scores:
        aid = idx_to_item.get(idx)
        if aid and aid not in seen:
            picked.append(aid)
        if len(picked) >= k:
            break
    return picked


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/status")
def status():
    return {
        "models_loaded": list(_models.keys()),
        "articles_count": len(_articles) if _articles is not None else 0,
        "interactions_count": len(_interactions) if _interactions is not None else 0,
    }


@app.get("/recommend")
def recommend(
    customer_id: str = Query(..., description="Customer ID hash"),
    model: str = Query("knn", description="popularity | knn | ncf"),
    k: int = Query(12, ge=1, le=50),
):
    dispatch = {
        "popularity": _recommend_popularity,
        "knn": _recommend_knn,
        "ncf": _recommend_ncf,
    }
    fn = dispatch.get(model, _recommend_knn)
    article_ids = fn(customer_id, k)
    items = _enrich(article_ids)
    return {
        "customer_id": customer_id,
        "model": model,
        "count": len(items),
        "items": items,
    }


@app.get("/purchase_history")
def purchase_history(
    customer_id: str = Query(..., description="Customer ID hash"),
    max_items: int = Query(24, ge=1, le=200),
):
    if _interactions is None or _articles is None:
        return {"customer_id": customer_id, "count": 0, "items": []}

    sub = _interactions[_interactions["customer_id"] == customer_id]
    if sub.empty:
        return {"customer_id": customer_id, "count": 0, "items": []}

    if "purchase_count" in sub.columns:
        sub = sub.sort_values("purchase_count", ascending=False)

    seen = set()
    aids = []
    for a in sub["article_id"].astype(str):
        if a not in seen:
            seen.add(a)
            aids.append(a)

    items = _enrich(aids[:max_items])
    return {"customer_id": customer_id, "count": len(items), "items": items}


@app.get("/recommend_from_selection")
def recommend_from_selection(
    article_ids: str = Query(..., description="Comma-separated article IDs"),
    k: int = Query(12, ge=1, le=50),
):
    """Build a KNN profile from user-selected articles and return recommendations."""
    knn = _models.get("knn")
    if not knn or knn.article_features is None:
        return {"count": 0, "items": [], "error": "KNN model not loaded"}

    selected = [aid.strip() for aid in article_ids.split(",") if aid.strip()]
    if not selected:
        return {"count": 0, "items": [], "error": "No articles provided"}

    # Build ad-hoc user profile from selected articles
    article_id_to_idx = {aid: i for i, aid in enumerate(knn.article_ids)}
    indices = [article_id_to_idx[aid] for aid in selected if aid in article_id_to_idx]

    if not indices:
        return {"count": 0, "items": [], "error": "None of the selected articles found"}

    profile = knn.article_features[indices].mean(axis=0, keepdims=True)
    profile = normalize(profile)

    n_neighbors = min(k * 3, len(knn.article_ids))
    _, neighbor_indices = knn.knn.kneighbors(profile, n_neighbors=n_neighbors)

    exclude = set(selected)
    picked = []
    for idx in neighbor_indices[0]:
        aid = str(knn.article_ids[idx])
        if aid not in exclude:
            picked.append(aid)
        if len(picked) >= k:
            break

    items = _enrich(picked)
    return {"count": len(items), "items": items}
