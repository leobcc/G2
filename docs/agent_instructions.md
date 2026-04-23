# Agent Instructions: Groupon Deal Optimizer

## Context
You are tasked with building a system to optimize Groupon deal content (title, description, fine print). The baseline content currently converts suboptimally, and the goal is to fully automate the scoring and rewriting of deals using LLMs, effectively phasing out a 100 FTE manual review team.

## Objective
Implement a pipeline that ingests a deal, scores its content quality (based on predetermined conversion drivers), and rewrites it to maximize conversion likelihood.

## Methodology & Rules
1. **Data Driven**: Your features and optimizations MUST tie back to findings from exploring `deals.csv`. Prioritize attributes that actually corelate with `cvr`.
2. **Evaluation Rigor**: A standalone eval mechanism must be in place. Do not just output text; output a confidence score or LLM-judgment comparing the old version to the new version.
3. **No Hallucinations**: You may add marketing flair, but you MUST NOT invent new "fine print" or fundamentally alter the physical constraints or price of the deal.
4. **Automation First**: Structure the code so it expects batched data frames or async streaming, not a simple 1-by-1 CLI tool. Prepare it for scale.

## Deliverables Needed
- `src/analyzer.py` (Data analysis and basic scoring functions)
- `src/optimizer.py` (LLM-based rewriting module)
- `src/evaluator.py` (Rigorous A/B metrics and LLM-as-a-judge comparison)
- `notebooks/exploration.ipynb` (Exposed analysis data)