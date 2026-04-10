#!/usr/bin/env python3
"""
Checklist: demo images for onehautapp precomputed data.

1. Derive unique article_ids from precomputed_recommendations.json
2. Extract only matching files from Kaggle images.zip OR copy from --source-dir
3. Verify total size (warn if >= 1 GiB)
4. Upload folder to Hugging Face dataset repo (e.g. alexoh2020/onehautapp-storage)

Examples:
  # List paths and counts (no files read)
  uv run python scripts/sync_demo_images_to_hf.py --dry-run

  # Extract from local Kaggle archive (download images.zip via Kaggle UI/API first)
  uv run python scripts/sync_demo_images_to_hf.py \\
    --zip ~/Downloads/images.zip --out data/demo_images_subset

  # Copy from already-unpacked data/raw/images
  uv run python scripts/sync_demo_images_to_hf.py \\
    --source-dir data/raw --out data/demo_images_subset

  # Upload resized images from onehautapp/mnt/data (HF_TOKEN or huggingface-cli login)
  uv run python scripts/sync_demo_images_to_hf.py \\
    --source-dir onehautapp/mnt/data --upload --upload-only --out onehautapp/mnt/data \\
    --repo-id alexoh2020/onehautapp-storage
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    HfApi = None  # type: ignore
    create_repo = None  # type: ignore

ONE_GIB = 1024**3


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def collect_article_ids(json_path: Path) -> list[str]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    ids: set[str] = set()
    for u in data:
        for item in u.get("purchase_history") or []:
            ids.add(str(item.get("article_id")))
        recs = u.get("recommendations") or {}
        for key in ("popularity", "knn", "ncf_meta"):
            for item in recs.get(key) or []:
                ids.add(str(item.get("article_id")))
    return sorted(ids)


def candidate_zip_names(article_id: str) -> list[str]:
    """Paths as commonly used in H&M Kaggle zips and our JSON image_path."""
    p = article_id[:3]
    base = f"{p}/{article_id}.jpg"
    return [
        f"images/{base}",
        f"images/{p}/{article_id}.JPG",
        f"data/images/{base}",
    ]


def build_zip_index(z: zipfile.ZipFile) -> dict[str, str]:
    """Map normalized path -> raw name in zip."""
    index: dict[str, str] = {}
    for raw in z.namelist():
        norm = raw.replace("\\", "/").lstrip("/")
        index[norm] = raw
    return index


def find_in_zip(index: dict[str, str], article_id: str) -> str | None:
    for cand in candidate_zip_names(article_id):
        if cand in index:
            return index[cand]
    tail = f"{article_id}.jpg"
    for norm, raw in index.items():
        if norm.endswith("/" + tail):
            return raw
    return None


def dest_relpath(article_id: str) -> str:
    return f"images/{article_id[:3]}/{article_id}.jpg"


def extract_from_zip(zip_path: Path, article_ids: list[str], out_dir: Path) -> tuple[int, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    found = 0
    with zipfile.ZipFile(zip_path, "r") as z:
        index = build_zip_index(z)
        for aid in article_ids:
            member = find_in_zip(index, aid)
            rel = dest_relpath(aid)
            target = out_dir / rel
            if member is None:
                missing.append(aid)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            found += 1
    return found, missing


def copy_from_source_tree(source_dir: Path, article_ids: list[str], out_dir: Path) -> tuple[int, list[str]]:
    """
    source_dir may be .../images or .../data/raw containing images/XXX/XXX.jpg
    """
    source_dir = source_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    found = 0
    for aid in article_ids:
        rel = dest_relpath(aid)
        candidates = [
            source_dir / rel,
            source_dir / "images" / aid[:3] / f"{aid}.jpg",
            source_dir / rel.lstrip("images/"),
        ]
        src = next((p for p in candidates if p.is_file()), None)
        target = out_dir / rel
        if src is None:
            missing.append(aid)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        found += 1
    return found, missing


def total_bytes(root: Path) -> int:
    total = 0
    for p in root.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def format_size(n: int) -> str:
    if n >= ONE_GIB:
        return f"{n / ONE_GIB:.2f} GiB"
    return f"{n / (1024**2):.1f} MiB"


def upload_folder(folder: Path, repo_id: str, repo_type: str, private: bool) -> None:
    if HfApi is None:
        raise SystemExit("Install huggingface_hub: pip install huggingface_hub")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    try:
        create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
            token=token,
        )
    except Exception as e:
        print(f"create_repo (ok if exists): {e}", file=sys.stderr)
    api.upload_folder(
        folder_path=str(folder),
        path_in_repo="",
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
    )
    print(f"Uploaded {folder} -> {repo_id} ({repo_type})")


def main() -> None:
    root = repo_root()
    default_json = root / "onehautapp/mnt/metadata/precomputed_recommendations.json"
    ap = argparse.ArgumentParser(description="Subset demo images + optional HF upload")
    ap.add_argument("--json", type=Path, default=default_json, help="precomputed_recommendations.json")
    ap.add_argument("--zip", type=Path, default=None, help="Path to Kaggle images.zip")
    ap.add_argument("--source-dir", type=Path, default=None, help="Extracted images tree (e.g. data/raw)")
    ap.add_argument("--out", type=Path, default=root / "data/demo_images_subset", help="Output folder")
    ap.add_argument("--dry-run", action="store_true", help="Only print counts and paths")
    ap.add_argument("--upload", action="store_true", help="Upload --out folder to Hugging Face")
    ap.add_argument("--repo-id", default="alexoh2020/onehautapp-storage", help="HF dataset repo id")
    ap.add_argument("--repo-type", default="dataset", choices=("dataset", "space"), help="Hub repo type")
    ap.add_argument("--private", action="store_true", help="Create repo private if missing")
    ap.add_argument(
        "--upload-only",
        action="store_true",
        help="Upload existing --out folder without --zip/--source-dir (after a prior extract)",
    )
    args = ap.parse_args()

    json_path = args.json.resolve()
    if not json_path.is_file():
        raise SystemExit(f"Missing JSON: {json_path}")

    article_ids = collect_article_ids(json_path)
    print(f"Unique article_ids from {json_path.name}: {len(article_ids)}")

    if args.upload_only:
        if not args.upload:
            raise SystemExit("--upload-only requires --upload")
        out_dir = args.out.resolve()
        if not out_dir.is_dir() or not any(out_dir.rglob("*.jpg")):
            raise SystemExit(f"--upload-only: need existing images under {out_dir}")
        size = total_bytes(out_dir)
        print(f"Total size under {out_dir}: {format_size(size)} ({size} bytes)")
        if size >= ONE_GIB:
            print(
                "WARNING: size is >= 1 GiB; trim or reduce scope before relying on a 1 GiB bucket.",
                file=sys.stderr,
            )
        upload_folder(out_dir, args.repo_id, args.repo_type, args.private)
        return

    if args.dry_run:
        for aid in article_ids[:5]:
            print(f"  example: {dest_relpath(aid)}")
        if len(article_ids) > 5:
            print(f"  ... +{len(article_ids) - 5} more")
        print("Dry run only; no extraction.")
        return

    out_dir = args.out.resolve()
    if args.zip and args.source_dir:
        raise SystemExit("Use only one of --zip or --source-dir")

    if args.zip:
        zp = args.zip.resolve()
        if not zp.is_file():
            raise SystemExit(f"Zip not found: {zp}")
        print(f"Extracting from {zp} -> {out_dir} ...")
        found, missing = extract_from_zip(zp, article_ids, out_dir)
        print(f"Extracted: {found} / {len(article_ids)}")
        if missing:
            print(f"Missing in zip ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")
    elif args.source_dir:
        sd = args.source_dir.resolve()
        if not sd.is_dir():
            raise SystemExit(f"Source dir not found: {sd}")
        print(f"Copying from {sd} -> {out_dir} ...")
        found, missing = copy_from_source_tree(sd, article_ids, out_dir)
        print(f"Copied: {found} / {len(article_ids)}")
        if missing:
            print(f"Missing on disk ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")
    else:
        raise SystemExit("Provide --zip or --source-dir (or use --dry-run)")

    if not out_dir.is_dir():
        raise SystemExit("No output produced.")

    size = total_bytes(out_dir)
    print(f"Total size under {out_dir}: {format_size(size)} ({size} bytes)")
    if size >= ONE_GIB:
        print("WARNING: size is >= 1 GiB; trim or reduce scope before relying on a 1 GiB bucket.", file=sys.stderr)

    if args.upload:
        upload_folder(out_dir, args.repo_id, args.repo_type, args.private)
    else:
        print("Skipping upload (pass --upload). Set HF_TOKEN if using --upload.")


if __name__ == "__main__":
    main()
