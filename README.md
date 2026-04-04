# One Haut Encoded

Fashion recommendation system combining visual features (CNN) with collaborative filtering.

**AIPI 540 Module Project** — Dominic Tanzillo & Alex Oh

## Dataset

[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) — 105K articles, 1.37M customers, 31M transactions.

## Setup

```bash
pip install -r requirements.txt
```

### Data

```bash
kaggle competitions download -c h-and-m-personalized-fashion-recommendations
unzip h-and-m-personalized-fashion-recommendations.zip -d data/raw/
python scripts/subsample.py
python scripts/build_features.py
```

## Project Structure

```
OneHautEncoded/
├── scripts/           <- Data pipeline & training scripts
├── notebooks/         <- EDA & experiment notebooks
├── models/            <- Saved model artifacts (gitignored)
├── data/              <- Raw & processed data (gitignored)
└── .github/           <- CI workflows, PR template, CODEOWNERS
```

## Models

1. **Popularity Baseline** — top-N most purchased items
2. **KNN Content-Based** — cosine similarity on article metadata + image features
3. **Neural Collaborative Filtering** — user/item embeddings + MLP (PyTorch)

## Evaluation

- Hit Rate @ 12
- NDCG @ 12
- Coverage & Novelty

## Git Workflow

- Feature branches → PR → review → merge to main
- All PRs require approval from a code owner
- Gitmoji commits
