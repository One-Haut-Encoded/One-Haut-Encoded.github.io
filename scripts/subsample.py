# AI-assisted (Claude Code, claude.ai) -- https://claude.ai
"""Subsample the H&M dataset to a manageable size for development."""

import pandas as pd
from pathlib import Path


def subsample(
    raw_dir: str = "data/raw",
    out_dir: str = "data/processed",
    min_customer_purchases: int = 15,
    min_article_purchases: int = 50,
    start_date: str = "2020-08-01",
):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading transactions...")
    transactions = pd.read_csv(
        raw_dir / "transactions_train.csv", dtype={"article_id": str}
    )
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])

    # Keep only recent transactions
    recent = transactions[transactions["t_dat"] >= start_date].copy()
    print(f"After date filter ({start_date}): {len(recent):,} transactions")

    # Keep active customers
    customer_counts = recent.groupby("customer_id").size()
    active_customers = customer_counts[customer_counts >= min_customer_purchases].index
    recent = recent[recent["customer_id"].isin(active_customers)]
    print(f"After customer filter (>={min_customer_purchases}): {len(recent):,} transactions")

    # Keep popular articles
    article_counts = recent.groupby("article_id").size()
    popular_articles = article_counts[article_counts >= min_article_purchases].index
    recent = recent[recent["article_id"].isin(popular_articles)]
    print(f"After article filter (>={min_article_purchases}): {len(recent):,} transactions")

    # Save subset
    recent.to_csv(out_dir / "transactions_subset.csv", index=False)

    # Also subset articles and customers to match
    articles = pd.read_csv(raw_dir / "articles.csv", dtype={"article_id": str})
    customers = pd.read_csv(raw_dir / "customers.csv")

    articles_sub = articles[articles["article_id"].isin(recent["article_id"].unique())]
    customers_sub = customers[customers["customer_id"].isin(recent["customer_id"].unique())]

    articles_sub.to_csv(out_dir / "articles_subset.csv", index=False)
    customers_sub.to_csv(out_dir / "customers_subset.csv", index=False)

    print(
        f"\nSubset saved to {out_dir}/:\n"
        f"  {len(recent):,} transactions\n"
        f"  {recent['customer_id'].nunique():,} customers\n"
        f"  {recent['article_id'].nunique():,} articles"
    )


if __name__ == "__main__":
    subsample()
