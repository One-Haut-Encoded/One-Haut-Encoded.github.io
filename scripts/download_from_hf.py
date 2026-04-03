# AI-assisted (Claude Code, claude.ai) -- https://claude.ai
"""Download all model artifacts and processed data from Hugging Face Hub.

Run this once to get everything you need without retraining:
    python scripts/download_from_hf.py
"""

from huggingface_hub import snapshot_download
from pathlib import Path


def download():
    repo_id = "DTanzillo/one-haut-encoded"
    local_dir = Path(".")

    print(f"Downloading from {repo_id}...")
    print("This includes models, processed data, and subset images (~2.7 GB)")

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )

    print("\nDone! You now have:")
    print("  models/          — all trained model artifacts")
    print("  data/processed/  — subset CSVs, parquets, embeddings, demo users")
    print("  data/images/     — product images for the 3K article subset")
    print("\nYou can now run the app without any training.")


if __name__ == "__main__":
    download()
