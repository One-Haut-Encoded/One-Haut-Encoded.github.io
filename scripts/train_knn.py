# AI-assisted (Claude Code, claude.ai) -- https://claude.ai
"""KNN content-based recommender using article metadata + optional image features."""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, normalize


class KNNRecommender:
    """Content-based filtering via cosine similarity on article feature vectors."""

    def __init__(self, n_neighbors: int = 50, use_images: bool = False):
        self.n_neighbors = n_neighbors
        self.use_images = use_images
        self.knn = None
        self.article_ids = None
        self.article_features = None
        self.user_profiles = None

    def _build_article_features(self, articles: pd.DataFrame, image_embeddings=None):
        """One-hot encode metadata, optionally concat image embeddings."""
        feature_cols = [
            "product_type_name", "product_group_name", "colour_group_name",
            "department_name", "index_group_name", "section_name",
            "garment_group_name",
        ]

        # Label encode then one-hot
        encoded_parts = []
        for col in feature_cols:
            if col in articles.columns:
                le = LabelEncoder()
                vals = articles[col].fillna("unknown").values
                encoded = le.fit_transform(vals)
                one_hot = np.zeros((len(vals), len(le.classes_)))
                one_hot[np.arange(len(vals)), encoded] = 1
                encoded_parts.append(one_hot)

        features = np.hstack(encoded_parts).astype(np.float32)

        if self.use_images and image_embeddings is not None:
            # Normalize image embeddings before concat
            img_norm = normalize(image_embeddings, axis=1)
            # Weight image features (metadata has fewer dims, so scale up)
            features = np.hstack([features, img_norm * 0.5])

        return normalize(features, axis=1)

    def fit(self, train: pd.DataFrame, articles: pd.DataFrame, image_embeddings=None):
        self.article_ids = articles["article_id"].values

        self.article_features = self._build_article_features(articles, image_embeddings)

        self.knn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric="cosine",
            algorithm="brute",
        )
        self.knn.fit(self.article_features)

        # Build user profiles: average of purchased article feature vectors
        article_id_to_idx = {aid: i for i, aid in enumerate(self.article_ids)}
        user_profiles = {}

        for cid, group in train.groupby("customer_id"):
            purchased_ids = group["article_id"].unique()
            indices = [article_id_to_idx[aid] for aid in purchased_ids if aid in article_id_to_idx]
            if indices:
                profile = self.article_features[indices].mean(axis=0, keepdims=True)
                user_profiles[cid] = normalize(profile)

        self.user_profiles = user_profiles
        print(f"Fitted KNN on {len(self.article_ids)} articles, {len(user_profiles)} user profiles")

    def recommend(self, customer_id: str, k: int = 12):
        if customer_id not in self.user_profiles:
            return []

        profile = self.user_profiles[customer_id]
        distances, indices = self.knn.kneighbors(profile, n_neighbors=k)
        return self.article_ids[indices[0]].tolist()

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


def train_knn(data_dir: str = "data/processed", use_images: bool = False):
    data_dir = Path(data_dir)

    train = pd.read_csv(data_dir / "train.csv", dtype={"article_id": str})
    articles = pd.read_csv(data_dir / "articles_subset.csv", dtype={"article_id": str})

    image_embeddings = None
    if use_images:
        emb_path = data_dir / "image_embeddings.npy"
        map_path = data_dir / "article_embedding_map.csv"
        if emb_path.exists() and map_path.exists():
            all_embeddings = np.load(emb_path)
            emb_map = pd.read_csv(map_path, dtype={"article_id": str})
            # Align full catalog embeddings to subset articles
            subset_ids = articles["article_id"].values
            emb_id_to_idx = dict(zip(emb_map["article_id"], emb_map["embedding_idx"]))
            subset_indices = [emb_id_to_idx[aid] for aid in subset_ids if aid in emb_id_to_idx]
            image_embeddings = all_embeddings[subset_indices]
            print(f"Loaded image embeddings: {image_embeddings.shape} (aligned to subset)")
        else:
            print("Warning: image embeddings not found, running without images")
            use_images = False

    model = KNNRecommender(use_images=use_images)
    model.fit(train, articles, image_embeddings)

    suffix = "_visual" if use_images else ""
    out_path = Path(f"models/knn/knn{suffix}.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"Saved to {out_path}")

    return model


if __name__ == "__main__":
    import sys
    use_images = "--images" in sys.argv
    train_knn(use_images=use_images)
