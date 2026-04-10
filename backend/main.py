"""FastAPI app: public JSON API for GitHub Pages (CORS) and HF Spaces."""

from __future__ import annotations

import os
import re
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from .inference import get_inference_service
from .paths import safe_is_file


def _image_cdn_base() -> str:
    return os.environ.get(
        "ONEHAUT_IMAGE_CDN_BASE",
        "https://huggingface.co/datasets/alexoh2020/onehautapp-storage/resolve/main/images",
    ).strip().rstrip("/")


def _mnt_data_roots() -> list[Path]:
    extra = os.environ.get("ONEHAUT_MNT_DATA", "").strip()
    roots: list[Path] = []
    if extra:
        roots.append(Path(extra))
    roots.append(Path("/mnt/data"))
    _repo = Path(__file__).resolve().parent.parent
    roots.append(_repo / "onehautapp/mnt/data")
    roots.append(Path(__file__).resolve().parent / "mnt/data")
    return roots


def _local_image_file(article_id: str) -> Path | None:
    rel = Path("images") / article_id[:3] / f"{article_id}.jpg"
    for root in _mnt_data_roots():
        p = (root / rel).resolve()
        if safe_is_file(p):
            return p
    return None


def _normalize_model_param(model: str) -> str:
    m = (model or "knn").strip().lower()
    if m in ("ncf", "ncf_meta"):
        return "ncf_meta"
    return m


def create_app() -> FastAPI:
    app = FastAPI(title="One Haut API", version="1.0.0")

    origins_raw = os.environ.get(
        "ONEHAUT_CORS_ORIGINS",
        "https://one-haut-encoded.github.io,http://localhost:8080,http://127.0.0.1:8080,http://localhost:5500,http://127.0.0.1:5500",
    )
    origins = [o.strip() for o in origins_raw.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=r"https://.*\.hf\.space",
        allow_credentials=True,
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/status")
    def status():
        svc = get_inference_service()
        return svc.status_summary()

    @app.get("/recommend")
    def recommend(
        customer_id: str = Query(..., description="Customer ID (hash string)"),
        model: str = Query("knn", description="popularity | knn | ncf_meta"),
        k: int = Query(12, ge=1, le=50),
    ):
        svc = get_inference_service()
        mk = _normalize_model_param(model)
        r = svc.recommend(customer_id, mk, k=k)
        return {"items": r.items}

    @app.get("/recommend_from_selection")
    def recommend_from_selection(
        article_ids: str = Query("", description="Comma-separated article IDs"),
        k: int = Query(12, ge=1, le=50),
    ):
        raw = [x.strip() for x in re.split(r"[\s,]+", article_ids) if x.strip()]
        if not raw:
            return {"items": []}
        svc = get_inference_service()
        r = svc.recommend_from_selection(raw, k=k)
        return {"items": r.items}

    @app.get("/purchase_history")
    def purchase_history(
        customer_id: str = Query(..., description="Customer ID (hash string)"),
        max_items: int = Query(48, ge=1, le=200),
    ):
        svc = get_inference_service()
        items = svc.purchase_items(customer_id, max_items=max_items)
        return {
            "customer_id": customer_id,
            "count": len(items),
            "items": items,
            "interactions_loaded": svc.status_summary()["interactions_loaded"],
        }

    @app.get("/images/{prefix}/{filename}")
    def serve_image(prefix: str, filename: str):
        if ".." in prefix or ".." in filename or "/" in prefix or "/" in filename:
            return RedirectResponse(
                url=f"{_image_cdn_base()}/{prefix}/{filename}", status_code=302
            )
        article_id = filename.removesuffix(".jpg")
        local = _local_image_file(article_id)
        if local is not None:
            return FileResponse(str(local), media_type="image/jpeg")
        cdn = f"{_image_cdn_base()}/{prefix}/{filename}"
        return RedirectResponse(url=cdn, status_code=302)

    return app


app = create_app()
