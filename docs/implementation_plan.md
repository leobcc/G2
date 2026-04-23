# Implementation Plan: Content That Converts

## Revised Architecture (v2 — Full Stack, Production Quality)

The v1 approach was a minimal proof-of-concept: a single Pearson correlation and a generic LLM prompt. This v2 replaces that with a statistically rigorous, industry-grade system.

---

## Phase 1: Deep Exploratory Data Analysis
**Goal:** Extract 20+ statistically validated content features and build an interpretable predictive model.

**Features Engineered:**
- Readability: Flesch-Kincaid Grade Level, Flesch Reading Ease (via `textstat`)
- Keyword signals: presence of social proof ("4-star", "over X customers"), urgency ("limited", "today"), specificity markers ("includes", "up to", list of services named)
- Structural signals: description section headers ("What We Offer", "Good to Know"), bullet-like formatting, fine print constraint count
- Sentiment: TextBlob polarity and subjectivity on title and description
- Content-to-fine-print ratio
- Option name quality: are options descriptive vs generic ("Option 1")?

**Modeling:**
- Bivariate Pearson + Spearman correlations for each feature vs CVR
- OLS regression with statsmodels to get p-values and R²
- Random Forest regressor for non-linear feature importance
- Category-stratified analysis (beauty vs automotive vs health convert differently)
- Temporal trend analysis on the 8 weeks of order data to detect momentum

---

## Phase 2: Composite Content Scoring System (`src/scorer.py`)
**Goal:** Assign every deal a heuristic Content Quality Score (0–100) before and after rewriting.

**Method:**
- Weighted combination of engineered features, weights derived from regression coefficients
- Dimensions: Clarity, Specificity, Persuasion, Structural Quality, Image Quality
- Thresholds: Score < 40 → rewrite required; 40–70 → suggest improvements; > 70 → pass

---

## Phase 3: Multi-Dimensional, Category-Aware Optimizer (`src/optimizer.py`)
**Goal:** Rewrite deals with surgical precision grounded in actual data findings.

**Architecture:**
- System prompt that encodes ALL EDA findings as explicit rules
- Few-shot examples: dynamically pull the 3 highest-CVR deals in the same category as positive examples (in-context learning)
- Chain-of-thought reasoning: model must first explain what is weak, then rewrite
- Strict JSON schema with all fields: title, description, fine_print, option_names, reasoning
- Guardrail validation: post-generation rule checks (price not mentioned, fine print not modified, no new restrictions added)

---

## Phase 4: Rigorous Multi-Metric Evaluation (`src/evaluator.py`)
**Goal:** Prove that rewrites are objectively better, not just longer.

**Metrics:**
- Composite score delta (before/after) using the scorer
- Readability delta (Flesch-Kincaid)
- Specificity score (named entities, number of concrete facts)
- LLM-as-a-judge with blinded A/B framing (randomise which is "A" and "B") and rubric scoring on 5 axes: Clarity, Persuasion, Specificity, Accuracy, Professionalism
- ROUGE-L score to ensure content fidelity (minimal hallucination)

---

## Phase 5: Production Pipeline (`main.py`)
**Goal:** Scalable, fault-tolerant, observable batch processor.

**Features:**
- Async batch processing with `asyncio`
- Exponential backoff retry logic for API errors
- Structured logging
- Triage prioritization: process worst-performing deals first
- Full audit trail: every run saved with timestamp, model version, metrics

---

## Phase 6: Comprehensive EDA Notebook (`notebooks/exploration.ipynb`)
**Goal:** The primary presentable deliverable — a visual, narrative, data-driven story of what converts and why.

**Sections:**
1. Dataset overview & distributions
2. Feature engineering walkthrough
3. Correlation analysis (heatmap, scatter plots, statistical significance)
4. Category-level deep dives
5. Top vs. bottom performer analysis
6. Regression model & feature importance
7. Before/after optimization showcase with composite score bar charts
8. Evaluation summary table

---

## Progress
- [x] Initial Planning & Git Setup
- [x] v1 baseline: basic correlation + naive LLM optimizer
- [x] Deep EDA with NLP feature engineering (readability, sentiment, specificity, structure)
- [x] Composite Content Score model (`src/scorer.py`)
- [x] Category-aware few-shot optimizer (`src/optimizer.py`)
- [x] Rigorous multi-metric evaluator with blinded A/B judge (`src/evaluator.py`)
- [x] Production pipeline with retry/logging (`main.py`)
- [x] Comprehensive EDA + Results Notebook (`notebooks/exploration.ipynb`)
