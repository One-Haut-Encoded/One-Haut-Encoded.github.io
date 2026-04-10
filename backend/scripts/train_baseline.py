# AI-assisted (Claude Code, claude.ai) -- https://claude.ai
"""Popularity baseline recommender — the floor every other model must beat."""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path


class PopularityRecommender:
    """Recommend top-N most purchased items globally or per department."""

    def __init__(self):
        self.global_ranking = None
        self.dept_ranking = None
        self.user_top_dept = None

    def fit(self, train: pd.DataFrame, articles: pd.DataFrame):
        # Global popularity
        self.global_ranking = (
            train.groupby("article_id").size()
            .sort_values(ascending=False)
            .index.tolist()
        )

        # Per-department popularity
        train_with_dept = train.merge(
            articles[["article_id", "department_name"]],
            on="article_id",
            how="left",
        )
        self.dept_ranking = (
            train_with_dept.groupby(["department_name", "article_id"])
            .size()
            .reset_index(name="count")
            .sort_values(["department_name", "count"], ascending=[True, False])
            .groupby("department_name")["article_id"]
            .apply(list)
            .to_dict()
        )

        # Each user's most-purchased department
        user_dept = (
            train_with_dept.groupby(["customer_id", "department_name"])
            .size()
            .reset_index(name="count")
        )
        self.user_top_dept = (
            user_dept.loc[user_dept.groupby("customer_id")["count"].idxmax()]
            .set_index("customer_id")["department_name"]
            .to_dict()
        )

        print(f"Fitted on {len(train):,} transactions")
        print(f"Global top 5: {self.global_ranking[:5]}")

    def recommend(self, customer_id: str, k: int = 12, mode: str = "global"):
        if mode == "department" and customer_id in self.user_top_dept:
            dept = self.user_top_dept[customer_id]
            candidates = self.dept_ranking.get(dept, self.global_ranking)
        else:
            candidates = self.global_ranking
        return candidates[:k]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


def train_baseline(data_dir: str = "data/processed"):
    data_dir = Path(data_dir)

    train = pd.read_csv(data_dir / "train.csv", dtype={"article_id": str})
    articles = pd.read_csv(data_dir / "articles_subset.csv", dtype={"article_id": str})

    model = PopularityRecommender()
    model.fit(train, articles)

    out_path = Path("models/baseline/popularity.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"Saved to {out_path}")

    return model


if __name__ == "__main__":
    train_baseline()
