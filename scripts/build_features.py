# AI-assisted (Claude Code, claude.ai) -- https://claude.ai
"""Build interaction matrix and train/test split from subsampled data."""

import pandas as pd
import numpy as np
from pathlib import Path


def build_interaction_matrix(data_dir: str = "data/processed"):
    data_dir = Path(data_dir)

    transactions = pd.read_csv(data_dir / "transactions_subset.csv", dtype={"article_id": str})
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])

    # Binary interaction matrix
    interactions = (
        transactions.groupby(["customer_id", "article_id"])
        .size()
        .reset_index(name="purchase_count")
    )
    interactions["purchased"] = 1

    # Temporal train/test split — last 14 days as test
    cutoff = transactions["t_dat"].max() - pd.Timedelta(days=14)
    train = transactions[transactions["t_dat"] < cutoff]
    test = transactions[transactions["t_dat"] >= cutoff]

    # Save splits
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    interactions.to_csv(data_dir / "interactions.csv", index=False)

    print(f"Train: {len(train):,} transactions ({train['t_dat'].min()} to {train['t_dat'].max()})")
    print(f"Test:  {len(test):,} transactions ({test['t_dat'].min()} to {test['t_dat'].max()})")
    print(f"Interactions: {len(interactions):,} unique user-item pairs")

    n_users = interactions["customer_id"].nunique()
    n_items = interactions["article_id"].nunique()
    sparsity = 1 - len(interactions) / (n_users * n_items)
    print(f"Matrix sparsity: {sparsity:.4%}")

    return train, test, interactions


if __name__ == "__main__":
    build_interaction_matrix()
