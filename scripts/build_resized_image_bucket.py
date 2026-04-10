#!/usr/bin/env python3
# AI-assisted (Cursor) -- https://cursor.com
"""
Build ~3k resized JPEGs (240×320, quality 80) for the HF dataset bucket.

ID set:
  1) Union of article_ids from precomputed_recommendations.json (purchase + 3 models).
  2) Then fill from unique article_id in transactions_subset.csv (sorted), up to --max-images.
  Precomputed IDs are processed first (deterministic sort), then the rest.

Output layout: {out_dir}/images/{article_id[:3]}/{article_id}.jpg
Default out_dir: onehautapp/mnt/data (Space /mnt/data convention).

Resize: cover + center crop to 240×320 (optional --fit for letterbox).

After running, upload with:
  uv run python scripts/sync_demo_images_to_hf.py \\
    --source-dir onehautapp/mnt/data --upload --upload-only --out onehautapp/mnt/data \\
    --repo-id alexoh2020/onehautapp-storage
(Adjust --source-dir/--out to match; upload_folder uses the folder containing images/.)

Examples:
  uv run python scripts/build_resized_image_bucket.py --input-dir data/raw --max-images 5
  uv run python scripts/build_resized_image_bucket.py --input-dir data/raw

"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRECOMPUTED = REPO_ROOT / "onehautapp/mnt/metadata/precomputed_recommendations.json"
DEFAULT_TX = REPO_ROOT / "onehautapp/mnt/metadata/transactions_subset.csv"
DEFAULT_OUT = REPO_ROOT / "onehautapp/mnt/data"

TARGET_W, TARGET_H = 240, 320
JPEG_QUALITY = 80
ONE_GIB = 1024**3


def collect_precomputed_ids(json_path: Path) -> list[str]:
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


def collect_transaction_ids(csv_path: Path) -> list[str]:
    seen: set[str] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "article_id" not in reader.fieldnames:
            raise SystemExit("transactions CSV must have article_id column")
        for row in reader:
            seen.add(str(row["article_id"]).strip())
    return sorted(seen)


def ordered_article_ids(
    precomputed_path: Path,
    tx_path: Path,
    max_images: int,
) -> list[str]:
    pc = collect_precomputed_ids(precomputed_path)
    pc_set = set(pc)
    tx = collect_transaction_ids(tx_path)
    ordered: list[str] = []
    for aid in pc:
        if len(ordered) >= max_images:
            break
        ordered.append(aid)
    for aid in tx:
        if len(ordered) >= max_images:
            break
        if aid not in pc_set:
            ordered.append(aid)
    return ordered


def find_source_image(input_root: Path, article_id: str) -> Path | None:
    p = article_id[:3]
    rel = Path("images") / p / f"{article_id}.jpg"
    candidates = [
        input_root / rel,
        input_root / "images" / p / f"{article_id}.jpg",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def resize_image(img: Image.Image, fit: bool) -> Image.Image:
    img = img.convert("RGB")
    tw, th = TARGET_W, TARGET_H
    if fit:
        img.thumbnail((tw, th), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (tw, th), (255, 255, 255))
        w, h = img.size
        left = (tw - w) // 2
        top = (th - h) // 2
        canvas.paste(img, (left, top))
        return canvas
    # cover + center crop
    w, h = img.size
    scale = max(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return img.crop((left, top, left + tw, top + th))


def dest_path(out_dir: Path, article_id: str) -> Path:
    return out_dir / "images" / article_id[:3] / f"{article_id}.jpg"


def total_bytes(root: Path) -> int:
    n = 0
    for p in root.rglob("*"):
        if p.is_file():
            n += p.stat().st_size
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Resize H&M product images for HF bucket")
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "data/raw",
        help="Root containing images/<prefix>/<id>.jpg (default: data/raw)",
    )
    ap.add_argument("--precomputed-json", type=Path, default=DEFAULT_PRECOMPUTED)
    ap.add_argument("--transactions-csv", type=Path, default=DEFAULT_TX)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-images", type=int, default=3000)
    ap.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip missing sources instead of failing",
    )
    ap.add_argument(
        "--fit",
        action="store_true",
        help="Letterbox fit inside 240×320 instead of cover crop",
    )
    args = ap.parse_args()

    input_root = args.input_dir.resolve()
    out_dir = args.out_dir.resolve()
    pre_path = args.precomputed_json.resolve()
    tx_path = args.transactions_csv.resolve()

    if not pre_path.is_file():
        raise SystemExit(f"Missing precomputed JSON: {pre_path}")
    if not tx_path.is_file():
        raise SystemExit(f"Missing transactions CSV: {tx_path}")

    ids = ordered_article_ids(pre_path, tx_path, args.max_images)
    print(f"Article IDs to process (max {args.max_images}): {len(ids)}")

    ok = 0
    missing: list[str] = []
    errors: list[str] = []

    for aid in ids:
        src = find_source_image(input_root, aid)
        if src is None:
            missing.append(aid)
            if not args.skip_missing:
                print(f"Missing source for {aid}", file=sys.stderr)
            continue
        dst = dest_path(out_dir, aid)
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(src) as im:
                out = resize_image(im, args.fit)
            out.save(
                dst,
                "JPEG",
                quality=JPEG_QUALITY,
                optimize=True,
                subsampling=2,
            )
            ok += 1
        except OSError as e:
            errors.append(f"{aid}: {e}")

    print(f"Written: {ok} JPEGs -> {out_dir / 'images'}")
    if missing:
        print(f"Missing sources ({len(missing)}): {missing[:15]}{'...' if len(missing) > 15 else ''}")
    if errors:
        print(f"Errors ({len(errors)}): {errors[:5]}")

    if not (out_dir / "images").exists():
        raise SystemExit("No output produced.")

    size = total_bytes(out_dir)
    print(f"Total size under {out_dir}: {size / (1024**2):.1f} MiB ({size} bytes)")
    if size >= ONE_GIB:
        print("WARNING: output >= 1 GiB", file=sys.stderr)


if __name__ == "__main__":
    main()
