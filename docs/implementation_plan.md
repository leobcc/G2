# Implementation Plan: Content That Converts

## Phase 1: Exploratory Data Analysis & Feature Engineering
1. **Goal**: Identify patterns in content data (title, desc, fine print, image score, etc.) that correlate with conversion rate (CVR).
2. **Steps**:
   - Initialize a python environment and jupyter notebook for EDA.
   - Load and clean `deals.csv`.
   - Engineer text features: text length, readability scores, presence of urgent/action words, sentiment, formatting (bullet points, etc.).
   - Analyze correlation between content features and target variables (`cvr`, `weekly_orders_w1-w8`).
   - Deliverable: Analysis notebook/report summarizing the key drivers of conversion.

## Phase 2: Scoring Model & Generation Engine (POC)
1. **Goal**: Build an automated pipeline to score existing deals and generate optimized rewrites, utilizing LLMs.
2. **Steps**:
   - Design a scoring prompt/model that evaluates a deal against the key drivers identified in Phase 1.
   - Design a generation prompt/pipeline that rewrites underperforming deals to maximize the score without hallucinating constraints.
   - Implement an automated LLM-as-a-judge (or classical metric) evaluation mechanism to score the *new* content against the *original*.

## Phase 3: Project Scaffolding & Git Setup
1. **Goal**: Structure the repository for maintainability and scalability.
2. **Steps**:
   - Initialize git repo, create `.gitignore`, `requirements.txt`.
   - Organize into `src/` (data processing, llm calls, evaluation) and `notebooks/` (EDA, scratchpad).

## Phase 4: Operations Blueprint & Delivery
1. **Goal**: Outline how this POC scales to replace the 100 FTE manual operation.
2. **Steps**:
   - Write the blueprint in `docs/operations_blueprint.md`.
   - Finalize `docs/delivery.md` summarizing the presentation strategy.

## Progress
- [x] Initial Planning
- [ ] Requirements definition
- [ ] Git Setup
- [ ] EDA & Modeling
- [ ] Evaluation Framework
- [ ] Operations Blueprint
