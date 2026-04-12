---
title: One-Haut-Encoded
emoji: 🛍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.x.x
python_version: 3.10
app_file: backend/app.py
---

<!-- AI-assisted (Cursor) -- https://cursor.com -->

# One Haut Encoded

Fashion recommendation system combining visual features (CNN) with collaborative filtering.

**AIPI 540 Module Project** — Dominic Tanzillo & Alex Oh

## Dataset

[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) — 105K articles, 1.37M customers, 31M transactions.

## Setup

```bash
pip install -r requirements.txt
```

### Hugging Face Space (`alexoh2020/onehaut_backend`)

1. Connect the Space to **this** GitHub repository (no separate backend repo).
2. Set the Space **SDK** to **Docker** — the root [`Dockerfile`](Dockerfile) runs `uvicorn backend.main:app` (FastAPI under [`backend/`](backend/)).
3. On the Space, mount or upload models to `/mnt/models` and metadata CSVs to `/mnt/metadata`, or bake them into the image.
4. **GitHub Pages** calls the Space for live recommendations. Set `API_BASE` in [`app.js`](app.js) to your Space URL (for example `https://alexoh2020-onehaut-backend.hf.space`; confirm under Space **Settings → URL**).

### Local API (from repo root)

```bash
export PYTHONPATH=.
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 7860
```

### Local frontend + API (end-to-end)

[`app.js`](app.js) uses **`http://127.0.0.1:7860`** when the page is served from `localhost` or `127.0.0.1` (any port), and the Hugging Face Space URL otherwise.

```bash
# Terminal 1 — API
export PYTHONPATH=.
uvicorn backend.main:app --host 127.0.0.1 --port 7860

# Terminal 2 — static site (repo root)
python3 -m http.server 8080 --bind 127.0.0.1
```

Open **http://127.0.0.1:8080** and use the UI; the browser will call the local API with CORS allowed for `http://127.0.0.1:8080`.

Automated smoke test (starts both servers briefly, checks `/health`, CORS preflight, `/recommend`, `/recommend_from_selection`, and serves `index.html`):

```bash
bash scripts/e2e_local_test.sh
```

### Gradio demo (optional)

```bash
export PYTHONPATH=.
pip install -r onehautapp/requirements.txt
uvicorn onehautapp.app:app --reload --port 7860
```

Build the Gradio image from the repo root: `docker build -f onehautapp/Dockerfile -t onehaut-gradio .`

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
├── backend/           <- FastAPI inference API (HF Space Docker entrypoint)
├── onehautapp/        <- Optional Gradio UI (imports backend)
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
