# Content That Converts
### Groupon Global Operations — Case Study Submission

**Author:** Leonardo Bocchi | **Model:** llama-3.3-70b-versatile (Groq) | **Dataset:** 500 deals, 8 weeks of CVR data
**Scorer:** Weights derived from RF importances (Pearson r vs CVR: **0.70**) | **RF CV R²:** 0.665

---

## Deliverables

| # | Deliverable | Location |
|---|---|---|
| 1 | **Analysis** — what patterns drive conversion | [`notebooks/exploration_and_evaluation.ipynb`](notebooks/exploration_and_evaluation.ipynb) |
| 2 | **Working PoC** — scores, rewrites, and evaluates deals | [`src/`](src/), [`main.py`](main.py) |
| 3 | **Operations Blueprint** — path from 100 FTE to AI-first ops | [`docs/operations_blueprint.md`](docs/operations_blueprint.md) |

---

## Quickstart

### 1. Install dependencies
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Set your Groq API key
```bash
# Create a .env file in the project root:
GROQ_API_KEY=your_key_here
```
Get a free key at [console.groq.com](https://console.groq.com).

### 3. Run the full pipeline
```bash
python main.py
```
Processes the 10 lowest-converting deals: scores, rewrites, evaluates each one.
Results saved to `results/latest_results.json` and `results/latest_results.csv`.

> **Scale note:** The default `limit=10` is set for API cost control during demo. To process all 500 deals, edit `main.py` line ~25: `limit=500` (or `limit=None`). The pipeline is designed to handle 5,000+ deals/day with async batching — the current implementation runs sequentially for auditability.

### 4. Run validation tests
```bash
python test_pipeline.py
```
Validates the analyzer and scorer against the full 500-deal dataset.

### 5. Open the analysis notebook
```bash
jupyter notebook notebooks/exploration_and_evaluation.ipynb
```

---

## Deliverable 1 — Analysis

**File:** [`notebooks/exploration_and_evaluation.ipynb`](notebooks/exploration_and_evaluation.ipynb)

The notebook walks through 10 sections of analysis on the 500-deal dataset:

1. **Dataset overview** — CVR distribution, category & geo breakdowns
2. **Feature correlation heatmap** — 22 NLP features vs CVR (Pearson r + p-values)
3. **Random Forest feature importances** — 200 trees, 5-fold CV **R² = 0.665**
4. **Top vs bottom performer comparison** — mean feature values, top 20% vs bottom 20% CVR
5. **Category-stratified CVR analysis** — boxplots + word count scatter with trend line
6. **Weekly CVR decay analysis** — 8-week time series; top-quality content holds **3.5× CVR advantage every week**, gap never closes
7. **Before/after composite score comparison** — 10 rewritten deals, mean **+36.4 pts**
8. **Blinded A/B LLM judge breakdown** — 5 axes × 20 pts each
9. **Sample before/after deal showcase** — rendered HTML cards
10. **Full evaluation summary table** — all metrics, gradient-colored

### Key findings

| Feature | Pearson r with CVR | Takeaway |
|---|---|---|
| `structure_section_count` | **+0.563** | Structured sections (What/Why/Good-to-Know) drive biggest lift |
| `desc_word_count` | **+0.548** | Longer, denser descriptions convert better |
| `specificity_count` | **+0.500** | Concrete numbers, durations, credentials matter |
| `image_quality_score` | **+0.458** | Image quality is the strongest non-text signal |
| `title_length` | **+0.356** | Service + hook titles outperform brand-only titles |
| `desc_flesch_ease` | **−0.388** | Lower readability ease = more specific = higher CVR |
| `title_polarity` | **−0.357** | Hyperbolic adjectives (Amazing, Best Ever) hurt conversion |
| `desc_polarity` | **−0.231** | Factual tone beats promotional tone |

Pre-computed analysis saved to [`docs/analysis_findings.json`](docs/analysis_findings.json).
Generated charts saved to [`docs/`](docs/).

---

## Deliverable 2 — Working Proof of Concept

### Architecture

```
data/deals.csv
      │
      ▼
src/analyzer.py       22-feature NLP extractor + RF statistical model
      │
      ▼
src/scorer.py         Composite content quality score (0–100, 9 dimensions)
      │
      ▼  (triage: priority = 0.6×low-CVR + 0.4×improvement-opportunity)
      │
      ▼
src/optimizer.py      Category-aware LLM rewriter
                      • System prompt encodes 8 EDA-derived conversion rules
                      • Few-shot examples pulled from same-category top performers
                      • Chain-of-thought reasoning
                      • Guardrail validation (no price injection, fine print intact)
                      • Up to 3 retries on failure
      │
      ▼
src/evaluator.py      5-metric rigorous evaluation
                      • Composite score delta (before/after)
                      • Readability delta (Flesch ease + Flesch-Kincaid grade)
                      • Specificity delta (concrete entity count)
                      • ROUGE-L (anti-hallucination / content preservation)
                      • Blinded A/B LLM judge — randomises A/B assignment to
                        prevent positional bias, scores 5 axes × 20 pts
                      • Verdict: PASS / MARGINAL / FAIL
      │
      ▼
results/latest_results.json + .csv   (full audit trail per deal)
```

### Pipeline results (10 lowest-CVR deals)

| Metric | Result |
|---|---|
| Verdicts | 10/10 PASS |
| Mean composite score delta | **+36.4 pts** |
| Mean LLM judge score delta | **+37.9 pts** |
| All guardrails passed | ✓ |

Sample result — Deal #1 (Home Services, Birmingham, CVR: 0.35%):

| | Before | After |
|---|---|---|
| **Title** | Amazing Home Service at Spotless Care | Deep Home Cleaning for Up to 5 Rooms at Spotless Care |
| **Score** | 21/100 | 61/100 |
| **LLM Judge** | 30/100 | 85/100 |

### Source files

| File | Purpose |
|---|---|
| [`src/analyzer.py`](src/analyzer.py) | Feature engineering (22 NLP features), Pearson/Spearman correlations, Random Forest importance |
| [`src/scorer.py`](src/scorer.py) | Composite content quality scorer, 9 dimensions weighted by **RF importances** (auto-updating) |
| [`src/optimizer.py`](src/optimizer.py) | Category-aware few-shot LLM rewriter with guardrails |
| [`src/evaluator.py`](src/evaluator.py) | ROUGE-L + blinded A/B LLM judge evaluator |
| [`src/retrainer.py`](src/retrainer.py) | Feedback loop: ingests post-rewrite CVR, augments dataset, retrains RF, updates scorer weights |
| [`main.py`](main.py) | Production pipeline: triage → optimize → evaluate → log |
| [`test_pipeline.py`](test_pipeline.py) | Validation tests: analyzer, scorer, batch scoring |

---

## Deliverable 3 — Operations Blueprint

**File:** [`docs/operations_blueprint.md`](docs/operations_blueprint.md)

Covers:
- Three-tier system architecture diagram
- EDA findings encoded as conversion rules
- **3-phase FTE reduction: 100 → 60 → 25 → 12** over 18 months
- Quality control framework: automated guardrails + evaluation scorecard
- **"Permanently human" deal categories** — 5 deal types that always require human review (new merchants, regulated categories, brand-voice accounts, etc.)
- **Continuous learning (Section 5b)** — monthly retraining loop via `src/retrainer.py`
- Automation decision tree (when to auto-publish vs human review)
- Weekly KPIs (volume, quality, business)
- Financial model: **~$7M annual savings + ~$45.6M GMV upside**

---

## Repository Structure

```
├── data/
│   ├── deals.csv                   500 deals with 8 weeks of CVR data
│   └── data_dictionary.md          Field descriptions
│
├── src/
│   ├── analyzer.py                 NLP feature engineering + statistical analysis
│   ├── scorer.py                   Composite content quality scorer (0–100, RF-calibrated weights)
│   ├── optimizer.py                Category-aware LLM rewriter
│   ├── evaluator.py                Multi-metric evaluator + blinded LLM judge
│   └── retrainer.py                Feedback loop: post-CVR retraining of RF weights
│
├── notebooks/
│   └── exploration_and_evaluation.ipynb           Full analysis notebook (Deliverable 1)
│
├── docs/
│   ├── operations_blueprint.md     Ops blueprint (Deliverable 3)
│   ├── analysis_findings.json      Pre-computed EDA results + RF importances
│   └── fig_*.png                   Generated charts (including weekly CVR decay)
│
├── results/
│   ├── latest_results.json         Most recent pipeline run (full audit trail)
│   └── latest_results.csv          Same, CSV format
│
├── main.py                         Pipeline entry point (set limit= to scale up)
├── test_pipeline.py                Validation test harness
├── requirements.txt                Python dependencies
└── case_study/
    └── case_study.md               Original brief
```

---

## Dependencies

```
pandas, numpy, scikit-learn, scipy, statsmodels
textstat, textblob
rouge-score
seaborn, matplotlib
groq, python-dotenv
```

Full list in [`requirements.txt`](requirements.txt).

---

## Design Decisions

**Why Random Forest over linear regression?**
Non-linear interactions between features (e.g., word count only matters if structure is also present). CV R² = 0.665 validates it captures real signal.

**Why blinded A/B for LLM judging?**
Without randomising which version is "A" vs "B", LLMs systematically prefer the second option presented. Blinding eliminates positional bias from the evaluation.

**Why ROUGE-L?**
Ensures rewrites are grounded in the original content — not hallucinating new facts, prices, or terms. An anti-fabrication guardrail.

**Why category-aware few-shot prompting?**
Top-performing deals within the same category provide the most relevant signal. A spa deal and a home services deal have different conversion patterns; cross-category examples dilute the prompt.

**Does the system improve over time?**
Yes. The scorer weights are **derived directly from RF importances** at module import time (not hand-tuned constants). `src/retrainer.py` ingests post-rewrite CVR observations, augments the training dataset, re-fits the RF model, and updates `docs/analysis_findings.json` — from which `scorer.py` auto-loads new weights on next import. No code changes required. The retrainer includes drift detection (flags features that shift >5% importance), a minimum-observation guard (prevents retraining on noise), and a simulation mode for testing without live data.

```bash
# Simulate retraining with synthetic post-rewrite CVR data:
python -m src.retrainer --simulate --dry-run
# With real observed CVR:
python -m src.retrainer --observed-cvr data/observed_cvr.csv
```
