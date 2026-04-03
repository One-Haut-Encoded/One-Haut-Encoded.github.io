# Next Steps — Alex

## What's Done

Dominic set up the full ML pipeline. Everything below is trained, evaluated, and saved locally (models + data are gitignored).

### Data
- **Raw data** downloaded from Kaggle H&M competition into `data/raw/` (articles.csv, customers.csv, transactions_train.csv, images/)
- **Subsampled** to a development set: ~373K transactions, 25K customers, 3K articles (`data/processed/`)
- **Temporal train/test split**: last 14 days held out as test set
- **Full catalog parquets** saved in `data/processed/` for fast loading (articles_full.parquet, customers_full.parquet, transactions_full.parquet)
- **ResNet50 image embeddings** extracted for all 105,542 articles (2048-dim vectors, `data/processed/image_embeddings.npy`)
- **Article-to-embedding index** mapping at `data/processed/article_embedding_map.csv`
- **10 demo users** pre-generated at `data/processed/demo_users.json` with purchase histories and image paths

### Models (all saved in `models/`, gitignored)
| Model | File | Description |
|-------|------|-------------|
| Popularity Baseline | `models/baseline/popularity.pkl` | Top-N most purchased globally or per department |
| KNN (metadata) | `models/knn/knn.pkl` | Cosine similarity on one-hot article metadata |
| KNN + images | `models/knn/knn_visual.pkl` | Same + ResNet50 embeddings |
| NCF base | `models/ncf/ncf.pt` | User + item embeddings only |
| NCF + metadata | `models/ncf/ncf-meta.pt` | + one-hot article metadata |
| NCF + visual | `models/ncf/ncf-visual.pt` | + ResNet50 image embeddings |
| NCF full | `models/ncf/ncf-full.pt` | + metadata + image embeddings |

Each NCF model has a `_mappings.pkl` file with `user_to_idx` and `item_to_idx` dicts.

### Evaluation Results (K=12, on held-out test set)

| Model | HR@12 | NDCG@12 | Coverage | Novelty |
|-------|-------|---------|----------|---------|
| Popularity (global) | 0.0998 | 0.0145 | 0.0040 | 8.91 |
| Popularity (dept) | 0.0898 | 0.0159 | 0.2365 | 10.02 |
| KNN (metadata) | 0.0921 | 0.0187 | 0.9920 | 11.85 |
| KNN (+ images) | 0.0920 | 0.0187 | 0.9923 | 11.85 |
| NCF (base) | 0.0724 | 0.0136 | 0.7215 | 10.33 |
| NCF (+ meta) | **0.0887** | **0.0178** | 0.8769 | 10.68 |
| NCF (+ visual) | 0.0718 | 0.0134 | 0.6891 | 10.18 |
| NCF (full) | 0.0873 | 0.0171 | 0.7942 | 10.45 |

**Key findings:**
- Popularity global has the highest raw hit rate (0.0998) — it just recommends what everyone buys
- KNN has the best NDCG (0.0187) and near-perfect coverage (99%) — most diverse recommendations
- NCF + meta is the best deep learning variant — metadata features help a lot
- Visual features alone don't improve over ID-only baselines — the hypothesis that images help for cold-start still needs testing
- KNN wins on coverage + novelty (recommends across the whole catalog, not just popular items)

### Scripts
| Script | What it does |
|--------|-------------|
| `scripts/subsample.py` | Filters raw H&M data to dev subset |
| `scripts/build_features.py` | Builds interaction matrix + temporal train/test split |
| `scripts/extract_image_features.py` | ResNet50 feature extraction (full catalog, GPU with 50% VRAM cap) |
| `scripts/train_baseline.py` | Trains popularity recommender |
| `scripts/train_knn.py` | Trains KNN content-based (use `--images` for visual variant) |
| `scripts/train_ncf.py` | Trains NCF (use `--meta`, `--images`, or both for variants) |
| `scripts/evaluate.py` | Computes HR@12, NDCG@12, Coverage, Novelty |

---

## What You Need to Do

### 1. Build the Live Website

Build an interactive web app where a user can test the recommendation system. This is the main deliverable.

**Core features:**
- User can select from demo users or enter a customer ID
- Display that customer's purchase history as a grid of product images
- Show recommendations from the best model (NCF + meta or KNN) as product image cards
- Let user toggle between models to compare recommendations visually
- Each product card shows: image, product name, product type, colour, department

**Pre-built assets for you:**
- `data/processed/demo_users.json` — 10 pre-selected demo users with varied purchase histories (heavy/medium/light buyers across different departments)
- Product images at `data/raw/images/{article_id[:3]}/{article_id}.jpg` (e.g., article `0751471001` is at `images/075/0751471001.jpg`)
- Full catalog metadata in `data/processed/articles_full.parquet`
- All models have a `.recommend(customer_id, k=12)` method (baseline + KNN) or can be wrapped (NCF)

**Loading models in your app:**
```python
# Baseline / KNN — pickle load, call .recommend()
import pickle
with open("models/knn/knn.pkl", "rb") as f:
    knn_model = pickle.load(f)
recs = knn_model.recommend(customer_id, k=12)  # returns list of article_ids

# NCF — load PyTorch model, score all items for a user
import torch
from scripts.train_ncf import NCFModel
model = NCFModel(n_users=23373, n_items=2998, embed_dim=64, n_meta_features=206)
model.load_state_dict(torch.load("models/ncf/ncf-meta.pt", map_location="cpu"))
model.eval()
# See scripts/evaluate.py for the full recommend wrapper pattern
```

**Tech choice:**
- Streamlit is fastest but needs polish — use custom CSS, sidebar navigation, image grids
- FastAPI + React if you want a more production feel
- Product images are local files — serve them from the app or copy a subset to a static folder

### 2. Git Workflow

**Important:** Main branch is protected. You must:
1. Accept the collaborator invite on GitHub (check your email)
2. Create a feature branch: `git checkout -b feature/frontend`
3. Push to your branch: `git push -u origin feature/frontend`
4. Open a PR on GitHub — the PR template auto-populates
5. Fill in Summary, Changes, Testing, and Code Quality sections
6. Dominic reviews and approves
7. PR checks must pass: PR Summary, Secret Scan, Large File Scan, AI Attribution

**AI attribution:** Every new `.py` file must have this at the top:
```python
# AI-assisted (Claude Code, claude.ai) -- https://claude.ai
```

**Never commit `data/` or `models/`** — they're gitignored. Never commit `.env` or credentials.

### 3. Getting the Data on Your Machine

You need to download the H&M dataset and run the pipeline locally:
```bash
pip install -r requirements.txt
pip install kaggle

# For CUDA PyTorch (if you have a GPU):
pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu126

# Download data (accept competition rules on kaggle.com first!)
# Place your kaggle.json in ~/.kaggle/
kaggle competitions download -c h-and-m-personalized-fashion-recommendations
mkdir -p data/raw
unzip h-and-m-personalized-fashion-recommendations.zip -d data/raw/

# Run full pipeline
python scripts/subsample.py
python scripts/build_features.py
python scripts/extract_image_features.py
python scripts/train_baseline.py
python scripts/train_knn.py
python scripts/train_knn.py --images
python scripts/train_ncf.py
python scripts/train_ncf.py --meta
python scripts/train_ncf.py --images
python scripts/train_ncf.py --meta --images
```

### 4. Demo Users Reference

10 pre-selected users in `data/processed/demo_users.json`:

| Customer ID (truncated) | Purchases | Unique Articles | Top Department |
|--------------------------|-----------|-----------------|----------------|
| 4308983955... | 137 | 85 | Casual Lingerie |
| 7f0ac43942... | 68 | 53 | Denim Trousers |
| a6c1c70b77... | 70 | 15 | Heavy Basic Jersey |
| bed7af8414... | 12 | 12 | Dress |
| 71d4eb174a... | 17 | 16 | Ladies Sport Bras |
| dfa8d2952c... | 13 | 13 | Swimwear |
| caf967aa1f... | 15 | 10 | Ladies Sport Bras |
| 38cce1f033... | 5 | 5 | Light Basic Jersey |
| 9420694f62... | 8 | 8 | Swimwear |
| 4330e0e73e... | 5 | 3 | Trouser |

These span heavy buyers (137 purchases) to light buyers (5 purchases) across diverse departments — good for demonstrating how recommendations differ by user profile.
