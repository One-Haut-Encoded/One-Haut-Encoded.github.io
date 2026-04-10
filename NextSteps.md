# Next Steps — Alex

## What's Changed Since Your Last Update

The frontend and deployment architecture have been built out. Here's the current state and what still needs your attention.

### Current Architecture

```
GitHub Pages (static frontend)  →  HF Space (live model API)
  one-haut-encoded.github.io        dtanzillo-one-haut-encoded.hf.space
```

- **Frontend** (`index.html`, `style.css`, `app.js`) — static site with named style profiles, interactive "Curate Your Look" feature, methodology writeup, and full evaluation results
- **Backend** (`backend/app.py`) — FastAPI on HF Spaces, downloads models from `DTanzillo/one-haut-encoded` on startup, serves live inference
- **Models** — all 7 variants on HF Hub at `DTanzillo/one-haut-encoded`
- **Images** — 330 resized thumbnails committed to `images/` for the demo users and curated browse items

### What's Done

| Component | Status | Location |
|-----------|--------|----------|
| Data pipeline (subsample, features, split) | Complete | `scripts/` |
| Model training (7 variants) | Complete | `scripts/`, models on HF Hub |
| Evaluation (HR@12, NDCG@12, Coverage, Novelty) | Complete | Results in `index.html` |
| Frontend (GitHub Pages) | Complete | `index.html`, `style.css`, `app.js` |
| Backend API (HF Space) | Deployed | `backend/`, live at HF Space |
| Precomputed fallback | Complete | `precomputed_recommendations.json` |
| Product thumbnails | Complete | `images/` (330 resized JPGs, 2 MB) |

---

## What You Need to Do

### 1. Review and Approve PR #3

The frontend deployment PR is open and needs your approval before it can merge to main.

**Please review it thoroughly and leave substantive comments.** The rubric requires good PR reviews from each team member. A one-line approval won't demonstrate that.

When reviewing, look at:
- Does the frontend accurately represent the model evaluation results?
- Is the UX clear for someone unfamiliar with the project?
- Are there any broken links or missing images?
- Does the API integration look correct?

### 2. Fix Your Open PR (#2)

Your current PR ("syncing images to app") needs work before it can be merged. Specifically:

- **Title** should be descriptive: what does this PR do and why? "syncing images to app" doesn't tell a reviewer what to expect
- **Description** needs to use the PR template (Summary, Changes, Testing, Code Quality sections). The current body is two sentence fragments
- **Branch name** should follow the convention: `feature/your-feature-name`
- **Scope** should be clear: does this PR add image processing scripts? If so, explain why they're needed and how they fit into the existing pipeline

The rubric specifically grades on PR quality and review quality. Take the time to write a proper summary — it reflects well on both of us.

### 3. Enable GitHub Pages

Once PR #3 is merged to main:

1. Go to **repo Settings > Pages**
2. Source: **Deploy from a branch**
3. Branch: `main`, folder: `/ (root)`
4. Save

Site goes live at `https://one-haut-encoded.github.io`.

### 4. Verify the Live Site

After Pages is enabled, check:
- [ ] Style profiles load and show purchase history
- [ ] Model toggle (Popularity / KNN / NCF) switches recommendations
- [ ] "Curate Your Look" lets you select items and get recommendations (requires API to be running)
- [ ] Methodology, pipeline, and results sections render correctly
- [ ] Footer links to GitHub and HF Hub work

### 5. Written Report

We still need the final report. The rubric requires these sections:

- Problem Statement
- Data Sources
- Related Work
- Evaluation Strategy & Metrics
- Modeling Approach (all 3: Popularity baseline, KNN, NCF)
- Data Processing Pipeline
- Hyperparameter Tuning Strategy
- Results (quantitative comparison + visualizations)
- Error Analysis (5 specific mispredictions with root causes)
- Experiment Write-Up (our ablation study: metadata vs. visual features)
- Conclusions & Future Work
- Commercial Viability Statement
- Ethics Statement

---

## Git Workflow Reminder

**Main branch is protected.** Every change must go through a PR.

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Push to your branch: `git push -u origin feature/your-feature-name`
3. Open a PR — fill in **all** sections of the PR template
4. Both team members must approve before merge
5. PR checks must pass: PR Summary, Secret Scan, Large File Scan, AI Attribution

**AI attribution:** Every new `.py` file must have this at the top:
```python
# AI-assisted (Claude Code, claude.ai) -- https://claude.ai
```

**Never commit `data/` or `models/`** — they're gitignored. Never commit `.env` or credentials.

---

## Reference

### API Endpoints (live at `https://dtanzillo-one-haut-encoded.hf.space`)

| Endpoint | Description |
|----------|-------------|
| `GET /status` | Model load status |
| `GET /recommend?customer_id=X&model=popularity\|knn\|ncf&k=12` | Recommendations for a customer |
| `GET /purchase_history?customer_id=X` | Customer's purchase history |
| `GET /recommend_from_selection?article_ids=X,Y,Z&k=12` | KNN recs from user-selected items |

### Demo Users

10 pre-selected users in `precomputed_recommendations.json`:

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
