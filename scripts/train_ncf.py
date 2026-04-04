# AI-assisted (Claude Code, claude.ai) -- https://claude.ai
"""Neural Collaborative Filtering with optional metadata and image features."""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class InteractionDataset(Dataset):
    """Binary implicit feedback dataset with negative sampling."""

    def __init__(self, interactions, n_items, neg_ratio: int = 4):
        self.user_ids = interactions["user_idx"].values
        self.item_ids = interactions["item_idx"].values
        self.n_items = n_items
        self.neg_ratio = neg_ratio

        # Build positive set per user for fast negative sampling
        self.user_positives = {}
        for uid, iid in zip(self.user_ids, self.item_ids):
            self.user_positives.setdefault(uid, set()).add(iid)

    def __len__(self):
        return len(self.user_ids) * (1 + self.neg_ratio)

    def __getitem__(self, idx):
        n_pos = len(self.user_ids)
        if idx < n_pos:
            return self.user_ids[idx], self.item_ids[idx], 1.0
        else:
            # Negative sample
            pos_idx = idx % n_pos
            uid = self.user_ids[pos_idx]
            neg_iid = np.random.randint(self.n_items)
            while neg_iid in self.user_positives.get(uid, set()):
                neg_iid = np.random.randint(self.n_items)
            return uid, neg_iid, 0.0


class NCFModel(nn.Module):
    """Neural Collaborative Filtering with optional side features."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        mlp_dims: list = None,
        n_meta_features: int = 0,
        n_image_features: int = 0,
    ):
        super().__init__()
        if mlp_dims is None:
            mlp_dims = [128, 64, 32]

        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.item_embed = nn.Embedding(n_items, embed_dim)

        input_dim = embed_dim * 2 + n_meta_features + n_image_features

        layers = []
        for dim in mlp_dims:
            layers.extend([nn.Linear(input_dim, dim), nn.ReLU(), nn.Dropout(0.2)])
            input_dim = dim
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)

    def forward(self, user_ids, item_ids, meta_features=None, image_features=None):
        u = self.user_embed(user_ids)
        i = self.item_embed(item_ids)
        x = torch.cat([u, i], dim=-1)

        if meta_features is not None:
            x = torch.cat([x, meta_features], dim=-1)
        if image_features is not None:
            x = torch.cat([x, image_features], dim=-1)

        return self.mlp(x).squeeze(-1)


def train_ncf(
    data_dir: str = "data/processed",
    embed_dim: int = 64,
    epochs: int = 10,
    batch_size: int = 1024,
    lr: float = 0.001,
    use_meta: bool = False,
    use_images: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.5, 0)
        print(f"GPU memory capped at 50%")
    print(f"Using device: {device}")

    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv", dtype={"article_id": str})
    articles = pd.read_csv(data_dir / "articles_subset.csv", dtype={"article_id": str})

    # Build ID mappings
    user_ids = train["customer_id"].unique()
    item_ids = articles["article_id"].values

    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    item_to_idx = {iid: i for i, iid in enumerate(item_ids)}

    n_users = len(user_ids)
    n_items = len(item_ids)
    print(f"Users: {n_users:,}, Items: {n_items:,}")

    # Map to indices
    train_mapped = train.copy()
    train_mapped["user_idx"] = train_mapped["customer_id"].map(user_to_idx)
    train_mapped["item_idx"] = train_mapped["article_id"].map(item_to_idx)
    train_mapped = train_mapped.dropna(subset=["user_idx", "item_idx"])
    train_mapped["user_idx"] = train_mapped["user_idx"].astype(int)
    train_mapped["item_idx"] = train_mapped["item_idx"].astype(int)

    # Deduplicate
    interactions = train_mapped.groupby(["user_idx", "item_idx"]).size().reset_index(name="count")

    # Optional features
    n_meta_features = 0
    n_image_features = 0
    meta_tensor = None
    image_tensor = None

    if use_meta:
        from sklearn.preprocessing import LabelEncoder
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
        n_meta_features = meta_np.shape[1]
        meta_tensor = torch.tensor(meta_np, dtype=torch.float32).to(device)
        print(f"Metadata features: {n_meta_features}")

    if use_images:
        emb_path = data_dir / "image_embeddings.npy"
        map_path = data_dir / "article_embedding_map.csv"
        if emb_path.exists() and map_path.exists():
            all_embeddings = np.load(emb_path)
            emb_map = pd.read_csv(map_path, dtype={"article_id": str})
            # Align full catalog embeddings to subset articles
            emb_id_to_idx = dict(zip(emb_map["article_id"], emb_map["embedding_idx"]))
            subset_indices = [emb_id_to_idx[aid] for aid in item_ids if aid in emb_id_to_idx]
            img_np = all_embeddings[subset_indices]
            n_image_features = img_np.shape[1]
            image_tensor = torch.tensor(img_np, dtype=torch.float32).to(device)
            print(f"Image features: {n_image_features} (aligned to subset)")
        else:
            print("Warning: image embeddings not found, skipping")
            use_images = False

    # Model
    model = NCFModel(
        n_users=n_users,
        n_items=n_items,
        embed_dim=embed_dim,
        n_meta_features=n_meta_features,
        n_image_features=n_image_features,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    dataset = InteractionDataset(interactions, n_items)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for user_idx, item_idx, labels in loader:
            user_idx = user_idx.long().to(device)
            item_idx = item_idx.long().to(device)
            labels = labels.float().to(device)

            # Gather side features if needed
            meta_batch = meta_tensor[item_idx] if meta_tensor is not None else None
            img_batch = image_tensor[item_idx] if image_tensor is not None else None

            preds = model(user_idx, item_idx, meta_batch, img_batch)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}")

    # Save model and mappings
    variant = "ncf"
    if use_meta:
        variant += "-meta"
    if use_images:
        variant += "-visual"
    if use_meta and use_images:
        variant = "ncf-full"

    out_dir = Path("models/ncf")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / f"{variant}.pt")

    mappings = {"user_to_idx": user_to_idx, "item_to_idx": item_to_idx}
    with open(out_dir / f"{variant}_mappings.pkl", "wb") as f:
        pickle.dump(mappings, f)

    print(f"Saved {variant} to {out_dir}")
    return model, mappings


if __name__ == "__main__":
    import sys
    use_meta = "--meta" in sys.argv
    use_images = "--images" in sys.argv
    train_ncf(use_meta=use_meta, use_images=use_images)
