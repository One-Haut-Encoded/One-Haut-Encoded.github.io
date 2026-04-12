# AI-assisted (Cursor) -- https://cursor.com
"""Load trained recommenders from disk; live inference for Popularity, KNN (dynamic profile), NCF."""

from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, normalize

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"
_APP_SCRIPTS = Path(__file__).resolve().parent / "scripts"


def _ensure_scripts_path() -> None:
    """Add scripts directories to sys.path so train_*.py class definitions are importable."""
    for scripts_dir in (_SCRIPTS, _APP_SCRIPTS):
        s = str(scripts_dir)
        if scripts_dir.is_dir() and s not in sys.path:
            sys.path.insert(0, s)


def _register_main_classes_for_pickles(
    popularity_cls: type,
    knn_cls: type,
    ncf_model_cls: type | None = None,
) -> None:
    """Pickles from `python train_*.py` store __main__.ClassName."""
    main = sys.modules.get("__main__")
    if main is None:
        return
    setattr(main, "PopularityRecommender", popularity_cls)
    setattr(main, "KNNRecommender", knn_cls)
    if ncf_model_cls is not None:
        setattr(main, "NCFModel", ncf_model_cls)


from .paths import find_metadata_file, find_model_root  # noqa: E402


def _articles_path() -> Path | None:
    p = find_metadata_file("articles_subset.csv")
    return Path(p) if p else None


def _data_dir_for_embeddings() -> Path | None:
    ap = _articles_path()
    return ap.parent if ap else None


def _ncf_embedding_roots() -> list[Path]:
    """Directories that may contain image_embeddings.npy + article_embedding_map.csv."""
    roots: list[Path] = []
    extra = os.environ.get("ONEHAUT_EMBEDDING_DIR", "").strip()
    if extra:
        roots.append(Path(extra))
    ap = _articles_path()
    if ap:
        roots.append(ap.parent)
    roots.extend(
        [
            _REPO_ROOT / "data/processed",
            Path(__file__).resolve().parent / "mnt/metadata",
        ]
    )
    seen: set[Path] = set()
    out: list[Path] = []
    for r in roots:
        try:
            rr = r.resolve()
        except OSError:
            continue
        if rr not in seen:
            seen.add(rr)
            out.append(rr)
    return out


def _find_image_embedding_paths() -> tuple[Path | None, Path | None]:
    for root in _ncf_embedding_roots():
        emb = root / "image_embeddings.npy"
        mp = root / "article_embedding_map.csv"
        if emb.is_file() and mp.is_file():
            return emb, mp
    return None, None


def _load_articles() -> pd.DataFrame | None:
    ap = _articles_path()
    if not ap:
        return None
    return pd.read_csv(ap, dtype={"article_id": str})


def _load_interactions() -> pd.DataFrame | None:
    p = find_metadata_file("interactions.csv")
    if not p:
        return None
    return pd.read_csv(p, dtype={"customer_id": str, "article_id": str})


@dataclass
class RecommendResult:
    items: list[dict]
    warning: str | None = None


def _article_to_card_dict(row: pd.Series) -> dict:
    return {
        "article_id": str(row["article_id"]),
        "product_name": str(row.get("prod_name") or ""),
        "product_type": str(row.get("product_type_name") or ""),
        "colour": str(row.get("colour_group_name") or ""),
        "department": str(row.get("department_name") or ""),
    }


def enrich_article_ids(articles_df: pd.DataFrame | None, article_ids: list[str]) -> list[dict]:
    if not article_ids:
        return []
    if articles_df is None:
        return [
            {
                "article_id": aid,
                "product_name": "",
                "product_type": "",
                "colour": "",
                "department": "",
            }
            for aid in article_ids
        ]
    want = set(article_ids)
    sub = articles_df[articles_df["article_id"].isin(want)]
    by_id = {str(r["article_id"]): r for _, r in sub.iterrows()}
    out: list[dict] = []
    for aid in article_ids:
        row = by_id.get(aid)
        if row is not None:
            out.append(_article_to_card_dict(row))
        else:
            out.append(
                {
                    "article_id": aid,
                    "product_name": "",
                    "product_type": "",
                    "colour": "",
                    "department": "",
                }
            )
    return out


class InferenceService:
    """Loads pickles and NCF checkpoint from ONEHAUT_MODEL_ROOT / repo models/."""

    def __init__(self) -> None:
        self._articles: pd.DataFrame | None = None
        self._interactions: pd.DataFrame | None = None

        self._popularity: object | None = None
        self._knn: object | None = None

        self._ncf_model: torch.nn.Module | None = None
        self._ncf_device = torch.device("cpu")
        self._ncf_user_to_idx: dict[str, int] | None = None
        self._ncf_item_to_idx: dict[str, int] | None = None
        self._ncf_idx_to_item: list[str] | None = None
        self._ncf_meta_tensor: torch.Tensor | None = None
        self._ncf_image_tensor: torch.Tensor | None = None
        self._ncf_variant: str | None = None

        self._articles_path: str | None = None
        self._interactions_path: str | None = None
        self._model_root: str | None = None
        self._load_error: str | None = None

    def load(self) -> None:
        try:
            ap = _articles_path()
            self._articles_path = str(ap) if ap else None
            self._articles = _load_articles()

            ip = find_metadata_file("interactions.csv")
            self._interactions_path = str(ip) if ip else None
            self._interactions = _load_interactions()
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"Data load failed: {exc}"
            return

        try:
            _ensure_scripts_path()
            from train_baseline import PopularityRecommender  # noqa: PLC0415
            from train_knn import KNNRecommender  # noqa: PLC0415
            from train_ncf import NCFModel  # noqa: PLC0415

            _register_main_classes_for_pickles(
                PopularityRecommender, KNNRecommender, NCFModel
            )

            root = find_model_root()
            if not root:
                self._load_error = (
                    "No model directory found. "
                    "Set ONEHAUT_MODEL_ROOT or mount models/ at /mnt/models."
                )
                return
            self._model_root = str(root)

            pop_path = root / "baseline" / "popularity.pkl"
            if pop_path.is_file():
                self._popularity = PopularityRecommender.load(str(pop_path))

            knn_vis = root / "knn" / "knn_visual.pkl"
            knn_std = root / "knn" / "knn.pkl"
            if knn_vis.is_file():
                self._knn = KNNRecommender.load(str(knn_vis))
            elif knn_std.is_file():
                self._knn = KNNRecommender.load(str(knn_std))

            self._load_ncf(root)

        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)

    def _load_ncf(self, root: Path) -> None:
        from train_ncf import NCFModel  # noqa: PLC0415

        ncf_dir = root / "ncf"
        if not ncf_dir.is_dir():
            return

        env_var = os.environ.get("ONEHAUT_NCF_VARIANT", "").strip().replace(".pt", "")
        if env_var:
            variants = [env_var]
        else:
            variants = ["ncf-full", "ncf-meta", "ncf-visual", "ncf"]

        for variant in variants:
            if self._try_load_ncf_variant(ncf_dir, variant, NCFModel):
                return

    def _try_load_ncf_variant(
        self,
        ncf_dir: Path,
        variant: str,
        NCFModel: type,
    ) -> bool:
        pt_path = ncf_dir / f"{variant}.pt"
        map_path = ncf_dir / f"{variant}_mappings.pkl"
        if not pt_path.is_file() or not map_path.is_file():
            return False

        try:
            with open(map_path, "rb") as f:
                mappings = pickle.load(f)
        except (OSError, pickle.UnpicklingError, KeyError, TypeError):
            return False

        user_to_idx = mappings["user_to_idx"]
        item_to_idx: dict[str, int] = mappings["item_to_idx"]

        articles = self._articles
        if articles is None:
            return False
        data_dir = _data_dir_for_embeddings()
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent / "mnt/metadata"

        use_meta = "meta" in variant or variant == "ncf-full"
        use_images = "visual" in variant or variant == "ncf-full"

        meta_t, img_t, n_meta, n_img = _build_ncf_tensors(
            articles, data_dir, use_meta, use_images, self._ncf_device
        )

        try:
            sd = torch.load(pt_path, map_location=self._ncf_device, weights_only=True)
        except TypeError:
            sd = torch.load(pt_path, map_location=self._ncf_device)
        except OSError:
            return False

        embed_dim = int(sd["user_embed.weight"].shape[1])
        n_users = int(sd["user_embed.weight"].shape[0])
        n_items_sd = int(sd["item_embed.weight"].shape[0])

        first_w = sd.get("mlp.0.weight")
        if first_w is None:
            first_linear_key = next(
                (k for k in sd if k.endswith(".weight") and "embed" not in k),
                None,
            )
        else:
            first_linear_key = "mlp.0.weight"

        if first_linear_key:
            in_dim = int(sd[first_linear_key].shape[1])
            inferred_side = in_dim - embed_dim * 2
            if inferred_side != n_meta + n_img:
                if inferred_side == 0:
                    meta_t, img_t, n_meta, n_img = None, None, 0, 0
                else:
                    return False

        try:
            model = NCFModel(
                n_users=n_users,
                n_items=n_items_sd,
                embed_dim=embed_dim,
                n_meta_features=n_meta,
                n_image_features=n_img,
            )
            model.load_state_dict(sd)
        except Exception:
            return False

        max_ix = max((int(ix) for ix in item_to_idx.values()), default=-1)
        idx_to_item: list[str | None] = [None] * (max_ix + 1)
        for aid, ix in item_to_idx.items():
            idx_to_item[int(ix)] = str(aid)

        model.to(self._ncf_device)
        model.eval()

        self._ncf_user_to_idx = user_to_idx
        self._ncf_item_to_idx = item_to_idx
        self._ncf_idx_to_item = idx_to_item
        self._ncf_model = model
        self._ncf_meta_tensor = meta_t
        self._ncf_image_tensor = img_t
        self._ncf_variant = variant
        return True

    def status_summary(self) -> dict:
        return {
            "articles_loaded": self._articles is not None,
            "articles_count": len(self._articles) if self._articles is not None else 0,
            "articles_path": self._articles_path,
            "interactions_loaded": self._interactions is not None,
            "interactions_count": len(self._interactions) if self._interactions is not None else 0,
            "interactions_path": self._interactions_path,
            "model_root": self._model_root,
            "popularity_loaded": self._popularity is not None,
            "knn_loaded": self._knn is not None,
            "ncf_loaded": self._ncf_model is not None,
            "ncf_variant": self._ncf_variant,
            "load_error": self._load_error,
        }

    def available(self) -> bool:
        return bool(self._popularity or self._knn or self._ncf_model)

    def load_message(self) -> str | None:
        return self._load_error

    def purchase_items(self, customer_id: str, max_items: int = 48) -> list[dict]:
        """Return articles this customer purchased, sorted by purchase_count descending."""
        if self._articles is None or self._interactions is None:
            return []
        df = self._interactions[self._interactions["customer_id"] == customer_id]
        if df.empty:
            return []
        if "purchase_count" in df.columns:
            df = df.sort_values("purchase_count", ascending=False)
        seen: set[str] = set()
        aids: list[str] = []
        for a in df["article_id"].astype(str):
            if a not in seen:
                seen.add(a)
                aids.append(a)
        return enrich_article_ids(self._articles, aids[:max_items])

    def _seen_articles(self, customer_id: str) -> set[str]:
        if self._interactions is None:
            return set()
        sub = self._interactions[self._interactions["customer_id"] == customer_id]
        return set(sub["article_id"].astype(str))

    def recommend(
        self,
        customer_id: str,
        model_key: str,
        k: int = 12,
    ) -> RecommendResult:
        mk = (model_key or "knn").lower()
        if mk == "popularity":
            return self._recommend_popularity(customer_id, k)
        if mk == "knn":
            return self._recommend_knn(customer_id, k)
        if mk in ("ncf", "ncf_meta"):
            return self._recommend_ncf(customer_id, k)
        return RecommendResult([], f"Unknown model key: {model_key!r}. Use popularity|knn|ncf.")

    def recommend_from_selection(self, article_ids: list[str], k: int = 12) -> RecommendResult:
        """KNN neighbors from mean pooled features over the selected article IDs."""
        knn = self._knn
        if knn is None:
            return RecommendResult([], "KNN model not loaded.")
        cleaned = [a.strip() for a in article_ids if a and str(a).strip()]
        if not cleaned:
            return RecommendResult([], "No article IDs provided.")
        if knn.article_ids is None or knn.article_features is None:
            return RecommendResult([], "KNN model incomplete.")
        article_id_to_idx = {str(aid): i for i, aid in enumerate(knn.article_ids)}
        indices = [article_id_to_idx[a] for a in cleaned if a in article_id_to_idx]
        if not indices:
            return RecommendResult([], "No valid article IDs in KNN index.")
        profile = knn.article_features[indices].mean(axis=0, keepdims=True)
        profile = normalize(profile)
        cap = min(knn.n_neighbors or 50, len(knn.article_ids))
        n_nbr = max(min(k * 3, cap), 1)
        _, nbr_ix = knn.knn.kneighbors(profile, n_neighbors=n_nbr)
        cand = [str(knn.article_ids[i]) for i in nbr_ix[0]]
        exclude = set(cleaned)
        out: list[str] = []
        for aid in cand:
            if aid not in exclude:
                out.append(aid)
            if len(out) >= k:
                break
        if len(out) < k:
            for aid in cand:
                if aid not in exclude and aid not in out:
                    out.append(aid)
                if len(out) >= k:
                    break
        return RecommendResult(enrich_article_ids(self._articles, out[:k]), None)

    def _recommend_popularity(self, customer_id: str, k: int) -> RecommendResult:
        if not self._popularity:
            return RecommendResult([], "Popularity model not loaded.")
        ids: list[str] = self._popularity.recommend(customer_id, k=k, mode="global")
        return RecommendResult(enrich_article_ids(self._articles, ids))

    def _knn_dynamic_profile(self, customer_id: str) -> np.ndarray | None:
        knn = self._knn
        if knn is None or self._interactions is None:
            return None
        article_ids = knn.article_ids
        if article_ids is None or knn.article_features is None:
            return None
        article_id_to_idx = {aid: i for i, aid in enumerate(article_ids)}
        sub = self._interactions[self._interactions["customer_id"] == customer_id]
        purchased = sub["article_id"].astype(str).unique()
        indices = [article_id_to_idx[a] for a in purchased if a in article_id_to_idx]
        if not indices:
            return None
        profile = knn.article_features[indices].mean(axis=0, keepdims=True)
        return normalize(profile)

    def _recommend_knn(self, customer_id: str, k: int) -> RecommendResult:
        knn = self._knn
        if knn is None:
            return RecommendResult([], "KNN model not loaded.")
        seen = self._seen_articles(customer_id)

        if knn.user_profiles and customer_id in knn.user_profiles:
            profile = knn.user_profiles[customer_id]
            profile_source = "stored"
        else:
            dyn = self._knn_dynamic_profile(customer_id)
            if dyn is None:
                r = self._recommend_popularity(customer_id, k)
                return RecommendResult(
                    r.items,
                    "KNN: no purchase history in article index; returning Popularity fallback.",
                )
            profile = dyn
            profile_source = "dynamic"

        cap = min(knn.n_neighbors or 50, len(knn.article_ids))
        n_nbr = max(min(k * 3, cap), 1)
        _, indices = knn.knn.kneighbors(profile, n_neighbors=n_nbr)
        cand = [str(knn.article_ids[i]) for i in indices[0]]

        out: list[str] = []
        for aid in cand:
            if aid not in seen:
                out.append(aid)
            if len(out) >= k:
                break

        if len(out) < k:
            for aid in cand:
                if aid not in out:
                    out.append(aid)
                if len(out) >= k:
                    break

        warning = f"KNN profile: {profile_source}." if profile_source == "dynamic" else None
        return RecommendResult(enrich_article_ids(self._articles, out[:k]), warning)

    def _recommend_ncf(self, customer_id: str, k: int) -> RecommendResult:
        if not self._ncf_model or not self._ncf_user_to_idx or not self._ncf_idx_to_item:
            return RecommendResult([], "NCF model not loaded.")
        uid = self._ncf_user_to_idx.get(customer_id)
        if uid is None:
            return RecommendResult([], "NCF: customer not in training set (OOV cold start).")

        model = self._ncf_model
        seen = self._seen_articles(customer_id)
        n_items = len(self._ncf_idx_to_item)
        meta_t = self._ncf_meta_tensor
        img_t = self._ncf_image_tensor
        batch = 512

        scores: list[tuple[float, int]] = []
        u_batch = torch.tensor([uid] * batch, dtype=torch.long, device=self._ncf_device)

        with torch.no_grad():
            for start in range(0, n_items, batch):
                end = min(start + batch, n_items)
                bsz = end - start
                if bsz < batch:
                    u_batch = torch.tensor([uid] * bsz, dtype=torch.long, device=self._ncf_device)
                item_idx = torch.arange(start, end, dtype=torch.long, device=self._ncf_device)
                mb = meta_t[item_idx] if meta_t is not None else None
                ib = img_t[item_idx] if img_t is not None else None
                logits = model(u_batch[:bsz], item_idx, mb, ib)
                for j in range(bsz):
                    scores.append((float(logits[j].item()), start + j))

        scores.sort(key=lambda x: x[0], reverse=True)

        picked: list[str] = []
        for _, ix in scores:
            aid = self._ncf_idx_to_item[ix]
            if aid is None or aid in seen:
                continue
            picked.append(aid)
            if len(picked) >= k:
                break

        if len(picked) < k:
            for _, ix in scores:
                aid = self._ncf_idx_to_item[ix]
                if aid is None or aid in picked:
                    continue
                picked.append(aid)
                if len(picked) >= k:
                    break

        return RecommendResult(enrich_article_ids(self._articles, picked[:k]))


def _build_ncf_tensors(
    articles: pd.DataFrame,
    data_dir: Path,
    use_meta: bool,
    use_images: bool,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int, int]:
    n_meta = 0
    n_img = 0
    meta_tensor: torch.Tensor | None = None
    image_tensor: torch.Tensor | None = None

    if use_meta:
        feature_cols = [
            "product_type_name",
            "product_group_name",
            "colour_group_name",
            "department_name",
            "index_group_name",
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
        if parts:
            meta_np = np.hstack(parts)
            n_meta = meta_np.shape[1]
            meta_tensor = torch.tensor(meta_np, dtype=torch.float32, device=device)

    if use_images:
        emb_path, map_path = _find_image_embedding_paths()
        if emb_path is None or map_path is None:
            emb_path = data_dir / "image_embeddings.npy"
            map_path = data_dir / "article_embedding_map.csv"
        item_ids = articles["article_id"].values
        if emb_path.is_file() and map_path.is_file():
            all_embeddings = np.load(emb_path)
            emb_map = pd.read_csv(map_path, dtype={"article_id": str})
            emb_id_to_idx = dict(zip(emb_map["article_id"], emb_map["embedding_idx"]))
            subset_indices = [emb_id_to_idx[aid] for aid in item_ids if aid in emb_id_to_idx]
            if len(subset_indices) == len(item_ids):
                img_np = all_embeddings[subset_indices]
                n_img = img_np.shape[1]
                image_tensor = torch.tensor(img_np, dtype=torch.float32, device=device)

    return meta_tensor, image_tensor, n_meta, n_img


_SERVICE: InferenceService | None = None


def get_inference_service() -> InferenceService:
    global _SERVICE  # noqa: PLW0603
    if _SERVICE is None:
        _SERVICE = InferenceService()
        _SERVICE.load()
    return _SERVICE
