# AI-assisted (Claude Code, claude.ai) -- https://claude.ai
"""Evaluation metrics for recommendation models: HR@K, NDCG@K, Coverage, Novelty."""

import numpy as np
import pandas as pd
from pathlib import Path


def hit_rate_at_k(recommendations: dict, test: pd.DataFrame, k: int = 12):
    """Fraction of users where at least one test item appears in top-K."""
    test_items = test.groupby("customer_id")["article_id"].apply(set).to_dict()
    hits = 0
    total = 0

    for uid, rec_list in recommendations.items():
        if uid not in test_items:
            continue
        total += 1
        if set(rec_list[:k]) & test_items[uid]:
            hits += 1

    return hits / total if total > 0 else 0.0


def ndcg_at_k(recommendations: dict, test: pd.DataFrame, k: int = 12):
    """Normalized Discounted Cumulative Gain at K."""
    test_items = test.groupby("customer_id")["article_id"].apply(set).to_dict()
    ndcgs = []

    for uid, rec_list in recommendations.items():
        if uid not in test_items:
            continue
        relevant = test_items[uid]
        dcg = sum(
            1.0 / np.log2(i + 2)
            for i, item in enumerate(rec_list[:k])
            if item in relevant
        )
        # Ideal DCG: all relevant items ranked first
        ideal_dcg = sum(
            1.0 / np.log2(i + 2)
            for i in range(min(len(relevant), k))
        )
        ndcgs.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)

    return np.mean(ndcgs) if ndcgs else 0.0


def coverage(recommendations: dict, all_article_ids: list, k: int = 12):
    """Fraction of catalog that appears in any user's top-K."""
    recommended = set()
    for rec_list in recommendations.values():
        recommended.update(rec_list[:k])
    return len(recommended) / len(all_article_ids)


def novelty(recommendations: dict, train: pd.DataFrame, k: int = 12):
    """Average self-information of recommended items (higher = less popular = more novel)."""
    item_popularity = train.groupby("article_id").size()
    total = len(train)
    scores = []

    for rec_list in recommendations.values():
        for item in rec_list[:k]:
            pop = item_popularity.get(item, 1) / total
            scores.append(-np.log2(pop))

    return np.mean(scores) if scores else 0.0


def evaluate_model(model, test: pd.DataFrame, train: pd.DataFrame, all_article_ids: list, k: int = 12, model_name: str = "model"):
    """Run all metrics for a model that has a .recommend(customer_id, k) method."""
    test_users = test["customer_id"].unique()

    print(f"Generating recommendations for {len(test_users):,} test users...")
    recommendations = {}
    for uid in test_users:
        recs = model.recommend(uid, k=k)
        if recs:
            recommendations[uid] = recs

    hr = hit_rate_at_k(recommendations, test, k)
    ndcg = ndcg_at_k(recommendations, test, k)
    cov = coverage(recommendations, all_article_ids, k)
    nov = novelty(recommendations, train, k)

    print(f"\n{'=' * 40}")
    print(f"  {model_name} @ K={k}")
    print(f"{'=' * 40}")
    print(f"  Hit Rate:  {hr:.4f}")
    print(f"  NDCG:      {ndcg:.4f}")
    print(f"  Coverage:  {cov:.4f}")
    print(f"  Novelty:   {nov:.2f} bits")
    print(f"  Users evaluated: {len(recommendations):,}")
    print(f"{'=' * 40}")

    return {"model": model_name, "HR@K": hr, "NDCG@K": ndcg, "coverage": cov, "novelty": nov}


if __name__ == "__main__":
    # Quick test with baseline
    from train_baseline import PopularityRecommender

    data_dir = Path("data/processed")
    train = pd.read_csv(data_dir / "train.csv", dtype={"article_id": str})
    test = pd.read_csv(data_dir / "test.csv", dtype={"article_id": str})
    articles = pd.read_csv(data_dir / "articles_subset.csv", dtype={"article_id": str})

    model = PopularityRecommender.load("models/baseline/popularity.pkl")
    evaluate_model(model, test, train, articles["article_id"].tolist(), model_name="Popularity (global)")
