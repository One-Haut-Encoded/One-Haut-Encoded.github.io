# AI-assisted (Claude Code, claude.ai) -- https://claude.ai
"""Extract ResNet50 image embeddings for articles in the subset."""

import torch
import numpy as np
import pandas as pd
from torchvision import models, transforms
from PIL import Image
from pathlib import Path


def extract_features(
    data_dir: str = "data",
    batch_size: int = 32,
    full_catalog: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.5, 0)
        print(f"GPU memory capped at 50%")
    print(f"Using device: {device}")

    # Load pretrained ResNet50, remove classification head
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # -> 2048-dim
    model = model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    data_dir = Path(data_dir)
    if full_catalog:
        articles = pd.read_csv(data_dir / "raw" / "articles.csv", dtype={"article_id": str})
        print(f"Extracting for FULL catalog: {len(articles):,} articles")
    else:
        articles = pd.read_csv(data_dir / "processed" / "articles_subset.csv", dtype={"article_id": str})
        print(f"Extracting for subset: {len(articles):,} articles")
    image_dir = data_dir / "raw" / "images"

    article_ids = articles["article_id"].values
    embeddings = np.zeros((len(article_ids), 2048), dtype=np.float32)
    missing = []

    # Process in batches
    for start in range(0, len(article_ids), batch_size):
        end = min(start + batch_size, len(article_ids))
        batch_ids = article_ids[start:end]
        batch_tensors = []
        batch_indices = []

        for i, aid in enumerate(batch_ids):
            img_path = image_dir / aid[:3] / f"{aid}.jpg"
            if not img_path.exists():
                missing.append(aid)
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                batch_tensors.append(preprocess(img))
                batch_indices.append(start + i)
            except Exception as e:
                print(f"  Error loading {aid}: {e}")
                missing.append(aid)

        if batch_tensors:
            batch = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                features = model(batch).squeeze(-1).squeeze(-1).cpu().numpy()
            for j, idx in enumerate(batch_indices):
                embeddings[idx] = features[j]

        if (start // batch_size) % 10 == 0:
            print(f"  Processed {end}/{len(article_ids)} articles...")

    # Save
    out_dir = data_dir / "processed"
    np.save(out_dir / "image_embeddings.npy", embeddings)

    # Save article_id -> index mapping
    id_map = pd.DataFrame({"article_id": article_ids, "embedding_idx": range(len(article_ids))})
    id_map.to_csv(out_dir / "article_embedding_map.csv", index=False)

    print(f"\nDone! Saved {len(article_ids)} embeddings to {out_dir / 'image_embeddings.npy'}")
    print(f"Shape: {embeddings.shape}")
    if missing:
        print(f"Missing images: {len(missing)} articles")

    return embeddings, article_ids


if __name__ == "__main__":
    import sys
    full = "--subset-only" not in sys.argv
    extract_features(full_catalog=full)
