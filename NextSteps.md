# Next Steps — Alex

## Current State

The frontend is built and ready to merge. The backend (HF Spaces API) is yours to integrate with your existing asset bucket. Here's everything you need.

### Architecture

```
GitHub Pages (static frontend)  →  HF Space (live model API)
  one-haut-encoded.github.io        your-hf-space.hf.space
        ↓                                   ↓
  precomputed fallback             loads models from DTanzillo/one-haut-encoded
  images from your HF bucket       serves /recommend, /purchase_history, etc.
```

### What's Done

| Component | Status | Owner |
|-----------|--------|-------|
| Data pipeline + training scripts | Complete | Dominic |
| 7 trained models on HF Hub | Complete | `DTanzillo/one-haut-encoded` |
| Frontend (GitHub Pages) | Complete, PR #3 | Dominic |
| Precomputed fallback JSON | Complete | In repo root |
| Product image bucket | Complete | `alexoh2020/onehautapp-storage` |
| HF Spaces backend API | **Yours to build** | Alex |

---

## What You Need to Do

### 1. Review and Approve PR #3 (Frontend)

PR #3 adds the static frontend. Please review it and leave substantive comments — the rubric grades on PR review quality. When reviewing, consider:

- Does the frontend accurately represent the evaluation results?
- Is the UX clear for someone unfamiliar with the project?
- Any edge cases in the JS (error handling, empty states)?
- Does the precomputed fallback data look correct?

### 2. Build the HF Spaces API Backend

The frontend expects a FastAPI backend with these endpoints. You can use your existing `alexoh2020/onehautapp` Space or create a new one — just make sure it serves these routes:

#### Required Endpoints

**`GET /recommend`**
```
Params: customer_id (str), model (str: "popularity"|"knn"|"ncf"), k (int, default 12)
Response: { "customer_id": str, "model": str, "count": int, "items": [...] }

Each item: { "article_id": str, "product_name": str, "product_type": str, "colour": str, "department": str }
```

**`GET /purchase_history`**
```
Params: customer_id (str), max_items (int, default 24)
Response: { "customer_id": str, "count": int, "items": [...] }
```

**`GET /recommend_from_selection`**
This powers the "Curate Your Look" feature where users select items and get personalized recommendations.
```
Params: article_ids (str, comma-separated), k (int, default 12)
Response: { "count": int, "items": [...] }

Implementation: build a KNN user profile from the selected article feature vectors,
find nearest neighbors, exclude the selected items, return top-k.
```

**`GET /status`**
```
Response: { "models_loaded": [...], "articles_count": int, "interactions_count": int }
```

#### CORS

The API must allow requests from `https://one-haut-encoded.github.io`. Add this middleware:
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://one-haut-encoded.github.io", "http://localhost:8080"],
    allow_methods=["GET"],
    allow_headers=["*"],
)
```

#### Loading Models

All models are on HF Hub at `DTanzillo/one-haut-encoded`. Download on startup:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="DTanzillo/one-haut-encoded", repo_type="model", local_dir="/app/data")
```

Then load:
```python
# Popularity / KNN — pickle, call .recommend()
import pickle
with open("models/baseline/popularity.pkl", "rb") as f:
    pop_model = pickle.load(f)
with open("models/knn/knn.pkl", "rb") as f:
    knn_model = pickle.load(f)

# NCF — PyTorch checkpoint
import torch
from train_ncf import NCFModel
model = NCFModel(n_users=23373, n_items=2998, embed_dim=64, n_meta_features=206)
model.load_state_dict(torch.load("models/ncf/ncf-meta.pt", map_location="cpu"))
model.eval()
```

**Important:** The pickle files reference class definitions from `scripts/train_baseline.py`, `scripts/train_knn.py`, and `scripts/train_ncf.py`. You need these files accessible to unpickle. Either copy them into your Space or register the classes on `__main__` before loading.

#### Image Serving

Your bucket at `alexoh2020/onehautapp-storage` already has the images. The frontend loads them directly from the HF CDN:
```
https://huggingface.co/datasets/alexoh2020/onehautapp-storage/resolve/main/images/{article_id[:3]}/{article_id}.jpg
```

No need to serve images from the API.

### 3. Connect the Frontend to Your API

Once your Space is live, open a PR that updates one line in `app.js`:

```javascript
// Change this:
const API_BASE = "";

// To your Space URL:
const API_BASE = "https://your-space-url.hf.space";
```

That's it. The frontend already handles the API calls, error states, and precomputed fallback.

### 4. Fix PR #2

Your current PR needs to follow the repo's PR template before it can merge. Please update it with:

- **Descriptive title** — what does this PR add and why?
- **Summary** — 2-3 sentences on the motivation
- **Changes** — what files were added/modified and what they do
- **Testing** — how you verified it works
- **Code Quality** — AI attribution, no loose code, etc.

The PR template auto-populates when you create a PR — please fill in all sections. Good PR documentation is part of the rubric and reflects well on both of us.

### 5. Enable GitHub Pages

After PR #3 merges:

1. Repo **Settings > Pages**
2. Source: **Deploy from a branch**
3. Branch: `main`, folder: `/ (root)`
4. Save — site goes live at `https://one-haut-encoded.github.io`

### 6. Written Report

We need the final report. Required sections per the rubric:

- Problem Statement
- Data Sources
- Related Work (review prior approaches to fashion recommendation)
- Evaluation Strategy & Metrics (justify HR@K, NDCG@K, Coverage, Novelty)
- Modeling Approach (Popularity baseline, KNN, NCF — all 3 required)
- Data Processing Pipeline (subsample → features → temporal split)
- Hyperparameter Tuning Strategy
- Results (8-model comparison table + visualizations)
- Error Analysis (5 specific mispredictions, root causes, mitigations)
- Experiment Write-Up (ablation study: metadata vs. visual features)
- Conclusions & Future Work
- Commercial Viability Statement
- Ethics Statement

---

## Reference

### Demo Users (in `precomputed_recommendations.json`)

| Style Profile | Purchases | Top Department |
|---------------|-----------|----------------|
| The Everyday Essential | 137 | Casual Lingerie |
| Denim & Done | 68 | Denim Trousers |
| The Basics Builder | 70 | Heavy Basic Jersey |
| Dress to Impress | 12 | Dress |
| Athleisure | 17 | Ladies Sport Bras |
| Beach & Beyond | 13 | Swimwear |
| Active Chic | 15 | Ladies Sport Bras |
| Capsule Wardrobe | 5 | Light Basic Jersey |
| Resort Ready | 8 | Swimwear |
| The Tailored Edit | 5 | Trouser |

### Evaluation Results (K=12)

| Model | HR@12 | NDCG@12 | Coverage | Novelty |
|-------|-------|---------|----------|---------|
| Popularity (global) | **0.0998** | 0.0145 | 0.0040 | 8.91 |
| KNN (metadata) | 0.0921 | **0.0187** | **0.9920** | **11.85** |
| NCF (+ metadata) | 0.0887 | 0.0178 | 0.8769 | 10.68 |

### Git Workflow

**Main branch is protected.** Every change goes through a PR with the template filled out.

1. Feature branch: `git checkout -b feature/your-feature-name`
2. Push: `git push -u origin feature/your-feature-name`
3. Open PR — fill in **all** template sections
4. Both team members approve
5. CI checks pass: PR Summary, Secret Scan, Large File Scan, AI Attribution

**AI attribution** on every new `.py` file:
```python
# AI-assisted (Claude Code, claude.ai) -- https://claude.ai
```
