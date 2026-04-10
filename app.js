// AI-assisted (Claude Code, claude.ai) -- https://claude.ai

// ── Configuration ────────────────────────────────────────────
// When deployed, set this to your HF Space URL for live inference.
// When empty, falls back to precomputed static data.
const API_BASE = "";

const MODEL_KEYS = { popularity: "popularity", knn: "knn", ncf: "ncf_meta" };
const MODEL_NAMES = { popularity: "Popularity", knn: "KNN", ncf: "NCF + Metadata" };

const STYLE_PROFILES = [
  { name: "The Everyday Essential", style: "Casual Lingerie", desc: "A loyalist who knows what she likes — comfort-first staples she reaches for every day." },
  { name: "Denim & Done", style: "Denim Trousers", desc: "Lived-in denim in every wash. Pairs everything with a great pair of jeans." },
  { name: "The Basics Builder", style: "Jersey & Knits", desc: "Invests in the perfect tee, the ideal layering piece. Quality over quantity." },
  { name: "Dress to Impress", style: "Dresses", desc: "One piece, done. Believes a great dress is the ultimate shortcut to looking put-together." },
  { name: "Athleisure", style: "Activewear", desc: "From studio to street. Performance fabrics that look as good at brunch as at barre." },
  { name: "Beach & Beyond", style: "Swimwear", desc: "Always planning the next getaway. Curates resort-ready looks year-round." },
  { name: "Active Chic", style: "Sport & Movement", desc: "Function meets form. Wants support, style, and versatility in equal measure." },
  { name: "Capsule Wardrobe", style: "Light Basics", desc: "Less is more. A few perfectly chosen pieces that work for everything." },
  { name: "Resort Ready", style: "Swim & Sun", desc: "Poolside polish. Mix-and-match swimwear with effortless cover-ups." },
  { name: "The Tailored Edit", style: "Trousers", desc: "Clean lines, sharp cuts. Every piece looks like it was made for her." },
];

// ── State ────────────────────────────────────────────────────
let userData = [];
let userById = {};
let curatedItems = [];
let selectedProfile = null;
let currentModel = "popularity";
let selectedArticles = new Set();

// ── Init ─────────────────────────────────────────────────────
async function init() {
  const [precomputed, curated] = await Promise.all([
    fetch("precomputed_recommendations.json").then((r) => r.json()).catch(() => []),
    fetch("curated_items.json").then((r) => r.json()).catch(() => []),
  ]);

  userData = precomputed;
  userById = {};
  userData.forEach((u) => { userById[u.customer_id] = u; });
  curatedItems = curated;

  buildProfileGrid();
  buildBrowseGrid();
  bindModelButtons();
  bindCurateButton();
}

// ── Style Profile Grid ───────────────────────────────────────
function buildProfileGrid() {
  const grid = document.getElementById("profile-grid");
  grid.innerHTML = "";

  userData.forEach((user, i) => {
    const profile = STYLE_PROFILES[i] || { name: `Profile ${i + 1}`, style: "", desc: "" };
    const card = document.createElement("div");
    card.className = "profile-card";
    card.innerHTML = `
      <div class="profile-name">${escapeHtml(profile.name)}</div>
      <div class="profile-style">${escapeHtml(profile.style)}</div>
      <div class="profile-purchases">${user.n_purchases} purchases</div>
    `;
    card.addEventListener("click", () => selectProfile(user.customer_id, i));
    grid.appendChild(card);
  });
}

function selectProfile(customerId, index) {
  selectedProfile = customerId;
  currentModel = "popularity";

  // Highlight active card
  document.querySelectorAll(".profile-card").forEach((c, i) => {
    c.classList.toggle("active", i === index);
  });

  // Reset model buttons
  document.querySelectorAll(".model-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.model === "popularity");
  });

  // Show detail section
  const detail = document.getElementById("profile-detail");
  detail.classList.remove("hidden");

  const profile = STYLE_PROFILES[index] || {};
  document.getElementById("profile-name").textContent = profile.name || "";
  document.getElementById("profile-desc").textContent = profile.desc || "";

  renderHistory();
  renderRecommendations();

  detail.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Render Profile Data ──────────────────────────────────────
function renderHistory() {
  const user = userById[selectedProfile];
  const grid = document.getElementById("history-grid");
  const meta = document.getElementById("history-meta");

  if (!user) { grid.innerHTML = ""; meta.textContent = ""; return; }

  const items = user.purchase_history || [];
  meta.textContent = `${items.length} items shown from ${user.n_purchases} total purchases`;
  renderCards(grid, items, false);
}

function renderRecommendations() {
  const user = userById[selectedProfile];
  const grid = document.getElementById("recs-grid");
  const meta = document.getElementById("recs-meta");

  if (!user) { grid.innerHTML = ""; meta.textContent = ""; return; }

  // Try live API first, fall back to precomputed
  if (API_BASE) {
    meta.textContent = "Loading live recommendations...";
    showSkeletons(grid, 12);
    fetchLiveRecs(selectedProfile, currentModel).then((items) => {
      if (items && items.length > 0) {
        meta.textContent = `${items.length} live recommendations from ${MODEL_NAMES[currentModel]}`;
        renderCards(grid, items, false);
      } else {
        renderStaticRecs(user, grid, meta);
      }
    }).catch(() => {
      renderStaticRecs(user, grid, meta);
    });
  } else {
    renderStaticRecs(user, grid, meta);
  }
}

function renderStaticRecs(user, grid, meta) {
  const key = MODEL_KEYS[currentModel] || "knn";
  const recs = (user.recommendations || {})[key] || [];
  meta.textContent = `${recs.length} recommendations from ${MODEL_NAMES[currentModel]}`;
  renderCards(grid, recs, false);
}

async function fetchLiveRecs(customerId, model) {
  const resp = await fetch(`${API_BASE}/recommend?customer_id=${customerId}&model=${model}&k=12`);
  if (!resp.ok) return null;
  const data = await resp.json();
  return data.items || [];
}

// ── Model Toggle ─────────────────────────────────────────────
function bindModelButtons() {
  document.querySelectorAll(".model-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".model-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentModel = btn.dataset.model;
      if (selectedProfile) renderRecommendations();
    });
  });
}

// ── Curate Your Look ─────────────────────────────────────────
function buildBrowseGrid() {
  const grid = document.getElementById("browse-grid");
  grid.innerHTML = "";

  if (curatedItems.length === 0) {
    grid.innerHTML = '<p class="placeholder">No items available.</p>';
    return;
  }

  curatedItems.forEach((item) => {
    const card = createCard(item, true);
    card.addEventListener("click", () => toggleSelection(item.article_id, card));
    grid.appendChild(card);
  });
}

function toggleSelection(articleId, card) {
  if (selectedArticles.has(articleId)) {
    selectedArticles.delete(articleId);
    card.classList.remove("selected");
  } else {
    selectedArticles.add(articleId);
    card.classList.add("selected");
  }
  updateCurateBar();
}

function updateCurateBar() {
  const count = selectedArticles.size;
  document.getElementById("curate-count").textContent = `${count} selected`;
  document.getElementById("curate-btn").disabled = count < 3;
}

function bindCurateButton() {
  document.getElementById("curate-btn").addEventListener("click", getCuratedRecs);
}

async function getCuratedRecs() {
  const results = document.getElementById("curate-results");
  const grid = document.getElementById("curate-recs-grid");
  const meta = document.getElementById("curate-meta");

  results.classList.remove("hidden");
  showSkeletons(grid, 12);
  meta.textContent = "Finding your perfect matches...";

  if (API_BASE) {
    try {
      const ids = Array.from(selectedArticles).join(",");
      const resp = await fetch(`${API_BASE}/recommend_from_selection?article_ids=${ids}&k=12`);
      if (resp.ok) {
        const data = await resp.json();
        const items = data.items || [];
        if (items.length > 0) {
          meta.textContent = `${items.length} items matched to your taste via KNN similarity`;
          renderCards(grid, items, false);
          results.scrollIntoView({ behavior: "smooth", block: "start" });
          return;
        }
      }
    } catch (e) { /* fall through */ }
  }

  // Offline fallback: find items from precomputed data that share departments
  const selectedDepts = new Set();
  curatedItems.forEach((item) => {
    if (selectedArticles.has(item.article_id)) {
      selectedDepts.add(item.department);
    }
  });

  // Gather recommendations from profiles with matching departments
  const fallback = [];
  const seen = new Set(selectedArticles);
  for (const user of userData) {
    const knnRecs = (user.recommendations || {}).knn || [];
    for (const rec of knnRecs) {
      if (!seen.has(rec.article_id) && selectedDepts.has(rec.department)) {
        fallback.push(rec);
        seen.add(rec.article_id);
      }
      if (fallback.length >= 12) break;
    }
    if (fallback.length >= 12) break;
  }

  if (fallback.length > 0) {
    meta.textContent = `${fallback.length} similar items (offline mode — deploy the API for live KNN inference)`;
    renderCards(grid, fallback, false);
  } else {
    meta.textContent = "Deploy the API backend for live recommendations from your selection.";
    grid.innerHTML = '<p class="placeholder">Live inference requires the API backend. See the deployment guide.</p>';
  }
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Card Rendering ───────────────────────────────────────────
function imageUrl(articleId) {
  return `images/${articleId.substring(0, 3)}/${articleId}.jpg`;
}

function createCard(item, selectable) {
  const card = document.createElement("div");
  card.className = "product-card" + (selectable ? " selectable" : "");

  const aid = item.article_id || "";
  const name = item.product_name || "";
  const type = item.product_type || "";
  const colour = item.colour || "";
  const dept = item.department || "";

  card.innerHTML = `
    <div class="card-img">
      <div class="img-placeholder">No image</div>
      <img src="${imageUrl(aid)}" alt="${escapeHtml(name)}" class="loading"
           onload="this.classList.remove('loading');this.classList.add('loaded')"
           onerror="this.style.display='none'" />
    </div>
    <div class="card-body">
      <div class="card-name">${escapeHtml(name)}</div>
      <div class="card-detail"><span class="label">Type</span>${escapeHtml(type)}</div>
      <div class="card-detail"><span class="label">Colour</span>${escapeHtml(colour)}</div>
      <div class="card-detail"><span class="label">Dept</span>${escapeHtml(dept)}</div>
    </div>
  `;
  return card;
}

function renderCards(container, items, selectable) {
  container.innerHTML = "";
  if (!items || items.length === 0) {
    container.innerHTML = '<p class="placeholder">No items found.</p>';
    return;
  }
  items.forEach((item) => container.appendChild(createCard(item, selectable)));
}

function showSkeletons(container, count) {
  container.innerHTML = Array.from({ length: count }, () => `
    <div class="skeleton-card">
      <div class="skeleton-img"></div>
      <div class="skeleton-body">
        <div class="skeleton-line"></div>
        <div class="skeleton-line short"></div>
      </div>
    </div>
  `).join("");
}

function showError(container, msg) {
  container.innerHTML = `<div class="error-msg">${escapeHtml(msg)}</div>`;
}

function escapeHtml(str) {
  const d = document.createElement("div");
  d.textContent = str;
  return d.innerHTML;
}

// ── Boot ─────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", init);
