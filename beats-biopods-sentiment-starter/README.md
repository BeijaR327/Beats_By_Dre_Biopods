# Beats BioPods Sentiment — NLP → Product Insights

Turn raw comments/surveys into **business-ready insights** for product, marketing, and CX decisions around Beats **BioPods™** (biometric-powered earbuds).

> 📌 **What this repo shows recruiters:** a clean, reproducible NLP pipeline, explainable metrics, and an executive-friendly summary with clear product recommendations.

---

## 1) Problem & Data

**Goal:** Convert unstructured text (surveys, app store, Reddit/Twitter, support tickets) into **sentiment, topics, and actionable insights** (e.g., *“Sustainability is a top driver; fit comfort is a recurring pain point.”*).

**Input data schema (`data/feedback_sample.csv`):**
- `text`: raw user comment
- `channel`: source (survey, reddit, twitter, app_store, support)
- `created_at`: ISO timestamp
- `rating` (optional): 1–5 if present
- `label` (optional): gold sentiment label if supervised training available (`pos/neg/neu`)

> You can replace `data/feedback_sample.csv` with your real dataset. Keep the same headers for plug‑and‑play.

---

## 2) Approach

- **Preprocessing:** lowercasing, punctuation-strip, emoji→text, stopwords, lemmatization
- **Sentiment:** VADER (rule-based) + optional Logistic Regression (TF‑IDF) if `label` exists
- **Topic signals:** class‑based TF‑IDF keywords + simple noun‑phrase keyphrases
- **Explainability:** top tokens by class; example comments by topic
- **Evaluation:** Accuracy/F1 (if labels), calibration vs. star ratings (if `rating` exists)

---

## 3) Quickstart

```bash
# (Recommended) create venv
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt

# (first time) download NLTK resources automatically when prompted
# Run end-to-end pipeline
python run.py --config configs/baseline.yaml

# Launch the Streamlit insights app
streamlit run app/app.py
```

**Artifacts produced:**
- `reports/figs/` plots (sentiment dist, top terms, confusion matrix)
- `reports/summary.md` one‑page executive summary
- `models/` (auto-created if supervised model saved)

---

## 4) Repo Structure
```
.
├─ app/                 # Streamlit UI
├─ configs/             # YAML configs for runs
├─ data/                # input CSVs (excluded from git LFS here)
├─ notebooks/           # EDA templates
├─ reports/figs/        # generated charts
├─ src/                 # reusable modules
├─ tests/               # unit tests
├─ .github/workflows/   # CI
├─ run.py               # pipeline entry point
└─ README.md
```

---

## 5) Results (sample)
- VADER baseline F1: ~0.7 on labeled sample
- TF‑IDF + Logistic Regression F1: ~0.78 (stratified 5‑fold)
- **Top drivers (example):** “battery life”, “comfort/fit”, “sustainability materials”, “ANC quality”
- **Business takeaways (example):**
  1. **Sustainability messaging resonates** in positive reviews; include materials + certifications in launch assets.
  2. **Comfort/fit** spikes in negatives; prioritize ear tip variety & fit assistant in onboarding.
  3. **Battery life** is a decisive delight/pain — make it a hero spec in marketing.

> Replace these with your real results after running on your dataset.

---

## 6) Config & Reproducibility
See `configs/baseline.yaml` for tokenization, model flags, and paths.

---

## 7) License
MIT
