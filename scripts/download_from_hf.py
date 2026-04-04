# AI-assisted (Claude Code, claude.ai) -- https://claude.ai
"""Download all model artifacts and processed data from Hugging Face Hub.

Run this once to get everything you need without retraining:
    python scripts/download_from_hf.py

For product images, download from Kaggle separately:
    kaggle competitions download -c h-and-m-personalized-fashion-recommendations
    unzip h-and-m-personalized-fashion-recommendations.zip -d data/raw/
"""

from huggingface_hub import snapshot_download
from pathlib import Path


def download():
    repo_id = "DTanzillo/one-haut-encoded"
    local_dir = Path(".")

    print(f"Downloading from {repo_id}...")
    print("This includes models and processed data (~300 MB)")

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )

    print("\nDone! You now have:")
    print("  models/          — all 7 trained model artifacts (baseline, KNN, NCF variants)")
    print("  data/processed/  — subset CSVs, parquets, demo users, precomputed recommendations")
    print()
    print("For product images, download from Kaggle:")
    print("  kaggle competitions download -c h-and-m-personalized-fashion-recommendations")
    print("  unzip h-and-m-personalized-fashion-recommendations.zip -d data/raw/")
    print()
    print("Image path pattern: data/raw/images/{article_id[:3]}/{article_id}.jpg")


if __name__ == "__main__":
    download()
