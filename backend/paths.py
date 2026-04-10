"""Resolve metadata CSVs, model artifacts, and live user IDs (test ∩ customers_subset)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent

LIVE_USER_LIMIT = 150


def safe_is_file(path: Path) -> bool:
    """True if path is a regular file; False on missing path or OSError (e.g. flaky /mnt on HF)."""
    try:
        return path.is_file()
    except OSError:
        return False


def safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _metadata_dir_candidates() -> list[Path]:
    extra = os.environ.get("ONEHAUT_METADATA_DIR", "").strip()
    out: list[Path] = []
    if extra:
        out.append(Path(extra))
    out.extend(
        [
            Path("/mnt/metadata"),
            _APP_DIR / "mnt/metadata",
            _REPO_ROOT / "onehautapp/mnt/metadata",
            _REPO_ROOT / "data/processed",
            _REPO_ROOT / "one-haut-encoded/data/processed",
        ]
    )
    return out


def find_metadata_file(name: str) -> Path | None:
    for d in _metadata_dir_candidates():
        try:
            p = (d / name).resolve()
        except OSError:
            continue
        if safe_is_file(p):
            return p
    return None


def model_root_candidates() -> list[Path]:
    extra = os.environ.get("ONEHAUT_MODEL_ROOT", "").strip()
    out: list[Path] = []
    if extra:
        out.append(Path(extra))
    out.extend(
        [
            Path("/mnt/models"),
            _APP_DIR / "mnt/models",
            _REPO_ROOT / "onehautapp/mnt/models",
            _REPO_ROOT / "models",
            _REPO_ROOT / "onehautapp/models",
        ]
    )
    return out


def find_model_root() -> Path | None:
    for d in model_root_candidates():
        if safe_is_dir(d):
            try:
                return d.resolve()
            except OSError:
                continue
    return None


def compute_live_customer_ids(limit: int = LIVE_USER_LIMIT) -> tuple[list[str], Path | None]:
    """Unique customer_ids from test.csv that appear in customers_subset.csv, sorted, first `limit`."""
    test_path = find_metadata_file("test.csv")
    cust_path = find_metadata_file("customers_subset.csv")
    if not test_path or not cust_path:
        return [], test_path or cust_path

    test = pd.read_csv(test_path, dtype={"customer_id": str, "article_id": str})
    customers = pd.read_csv(cust_path, dtype={"customer_id": str})
    test_ids = set(test["customer_id"].unique())
    subset_ids = set(customers["customer_id"].unique())
    common = sorted(test_ids & subset_ids)
    return common[:limit], test_path


def live_user_dropdown_choices(limit: int = LIVE_USER_LIMIT) -> list[tuple[str, str]]:
    ids, _ = compute_live_customer_ids(limit)
    if not ids:
        return []
    cust_path = find_metadata_file("customers_subset.csv")
    if not cust_path:
        return [(f"{cid[:8]}…", cid) for cid in ids]
    cust = pd.read_csv(cust_path, dtype={"customer_id": str})
    d = cust.set_index("customer_id")
    out: list[tuple[str, str]] = []
    for cid in ids:
        short = f"{cid[:8]}…"
        label = short
        if cid in d.index:
            age = d.loc[cid, "age"] if "age" in d.columns else None
            if pd.notna(age):
                label = f"{short} · age {int(float(age))}"
        out.append((label, cid))
    return out
