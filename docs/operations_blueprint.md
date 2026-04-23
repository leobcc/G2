# Operations Blueprint: AI-First Content Optimization
### Replacing 100 Manual FTEs with a Scalable, Auditable AI Pipeline

**Author:** Leo B | **Date:** April 2026 | **Context:** Groupon Chief of Staff — Global Operations

---

## Executive Summary

This blueprint replaces a 100-person manual content writing/QA operation with a three-tier AI system that delivers:

| Metric | Manual Baseline | AI System | Improvement |
|---|---|---|---|
| Deals optimized/day | ~200 (100 FTE × 2/hr) | 5,000+ | **25×** |
| Cost per deal | ~$4.50 | ~$0.03 | **99% reduction** |
| Quality consistency | High variance | Scored 0–100 | **Standardized** |
| Turnaround | 4–8 hours | <2 min | **240× faster** |
| Audit trail | Manual logs | Full JSON per deal | **100% traceable** |

**FTE reduction pathway:** 100 → 60 → 25 → 12 over 18 months (Phase 1–3 below)

---

## 1. System Architecture

### Three-Tier Pipeline

```
[Raw Deal Data]
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  TIER 1: Intelligence Layer                          │
│  • 22-feature NLP extractor (textstat + textblob)    │
│  • Random Forest scorer (CV R² = 0.665)              │
│  • Composite quality score 0–100 (9 dimensions)      │
│  • Priority triage: 0.6×(1−CVR%tile)+0.4×(opp%tile) │
└─────────────────────────────────────────────────────┘
      │  (ranked queue of deals needing optimization)
      ▼
┌─────────────────────────────────────────────────────┐
│  TIER 2: Generation Layer                            │
│  • Category-aware system prompt (8 conversion rules) │
│  • Few-shot examples from same-category top deals    │
│  • Chain-of-thought reasoning output                 │
│  • Model: llama-3.3-70b-versatile (Groq)             │
│  • Guardrail validation + up to 3 retries            │
└─────────────────────────────────────────────────────┘
      │  (improved title, description, option names)
      ▼
┌─────────────────────────────────────────────────────┐
│  TIER 3: Evaluation Layer                            │
│  • Composite score delta (before/after)              │
│  • Readability delta (Flesch ease + FK grade)        │
│  • Specificity delta (concrete entity count)         │
│  • ROUGE-L anti-hallucination check                  │
│  • Blinded A/B LLM judge (5 axes × 20 pts = 100)    │
│  • Verdict: PASS / MARGINAL / FAIL                   │
└─────────────────────────────────────────────────────┘
      │
      ▼
[Audit trail JSON + CSV → Human review queue]
```

### Key EDA Findings Encoded in Pipeline

From statistical analysis of 500 deals (RF CV R² = 0.665):

| Signal | Direction | Pearson r | Action Encoded |
|---|---|---|---|
| structure_section_count | ↑ | +0.563 | Require "What We Offer / Why Grab This / Good to Know" |
| desc_word_count | ↑ | +0.548 | Target 100–160 words |
| specificity_count | ↑ | +0.500 | Inject concrete numbers, durations, credentials |
| image_quality_score | ↑ | +0.458 | Flag low-image deals for photographer dispatch |
| title_length | ↑ | +0.356 | Target 50–80 chars with service + hook |
| social_proof_count | ↑ | +0.276 | Include review counts, star ratings, customer numbers |
| desc_flesch_ease | ↓ | −0.388 | Higher FK grade (8–12) = more specific, less generic fluff |
| title_polarity | ↓ | −0.357 | Avoid hyperbolic adjectives (Amazing, Incredible) |

---

## 2. Three-Phase FTE Transition Plan

### Phase 1: Augmentation (Months 1–6) — 100 → 60 FTE

**Goal:** Deploy AI as co-pilot. Humans review all AI outputs.

**New roles:**
- **AI Content Reviewers (40):** Review AI-generated content, approve/reject, provide feedback labels
- **Content Operations Managers (10):** Manage reviewer queues, handle escalations, monitor KPIs
- **ML Ops Engineer (1):** Monitor model performance, manage Groq API, run retraining cycles
- **Data Analyst (1):** Weekly quality reporting, CVR lift analysis

**Redeployed/exited:** ~40 FTE (content writers replaced by AI generation)

**Infrastructure:** Pipeline processes all 500+ deals/day in <30 min. Humans review MARGINAL/FAIL verdicts + 10% random PASS sample.

---

### Phase 2: Automation (Months 7–12) — 60 → 25 FTE

**Goal:** Auto-publish PASS deals (>80% of volume). Humans handle exceptions.

**Auto-publish criteria:**
- Verdict = PASS
- Composite score delta ≥ +20 pts
- ROUGE-L ≥ 0.08 (content preserved, no hallucination)
- LLM judge score ≥ 75/100
- All guardrails passed

**Redeployed/exited:** ~35 FTE

---

### Phase 3: Scale (Months 13–18) — 25 → 12 FTE

**Goal:** Fully autonomous pipeline with human oversight at category/geo level only.

**Capabilities added:**
- Automated A/B testing: AI generates 2 variants per deal, winner auto-selected after 72h
- Feedback loop: Live CVR data retrains scorer monthly
- Multi-market: All 15 geos with locale-specific prompt tuning
- Merchant self-serve: API endpoint for merchants to request rewrites

**Steady-state team (12 FTE):** Head of Content Ops (1), Category Leads (4), ML Scientists (2), Data Engineers (2), Ops Analysts (2), QA Specialist (1)

---

## 3. Quality Control Framework

### Automated Guardrails (every deal)
1. **Fine print integrity:** AI-generated fine print diff < 10% vs original
2. **Price injection check:** No new price figures not in original
3. **Length sanity:** Description 50–300 words, title 20–100 chars
4. **Required fields:** title, description, option_names all present and non-empty
5. **Anti-hallucination:** ROUGE-L ≥ 0.06 ensures factual continuity

### Evaluation Scorecard

| Metric | PASS threshold | MARGINAL | FAIL |
|---|---|---|---|
| Composite score delta | ≥ +15 pts | +5 to +15 | < +5 |
| LLM judge score (new) | ≥ 70/100 | 55–70 | < 55 |
| LLM judge delta | ≥ +20 | +10–20 | < +10 |
| ROUGE-L | ≥ 0.06 | 0.03–0.06 | < 0.03 |

---

## 4. Weekly KPIs & Reporting

### Volume KPIs
| KPI | Target | Owner |
|---|---|---|
| Deals processed/day | 500+ | ML Ops |
| Auto-publish rate | ≥85% | Content Ops |
| Human review SLA (24h) | ≥95% | Queue Managers |
| Pipeline uptime | 99.5% | ML Ops |

### Quality KPIs
| KPI | Target | Owner |
|---|---|---|
| Mean composite score delta | ≥ +30 pts | ML Scientists |
| Mean LLM judge delta | ≥ +30 pts | AI Trainers |
| PASS rate | ≥80% | Prompt Engineers |
| Merchant complaint rate | ≤1% | Category Leads |

### Business KPIs (monthly)
| KPI | Target | Owner |
|---|---|---|
| CVR lift on optimized deals | ≥5% relative | Data Analysts |
| Revenue uplift (GMV) | ≥$500K/month | Head of Content Ops |
| Cost per optimized deal | ≤$0.05 | ML Ops |

---

## 5. Automation Decision Tree

```
New/Updated Deal Submitted
          │
          ▼
   Composite Score < 40?
    ┌─────┴──────┐
   YES           NO
    │             │
    ▼             ▼
 Run AI      Archive as-is
 Optimizer   (already good)
    │
    ▼
 Verdict = PASS & all thresholds met?
    ┌─────┴──────┐
   YES           NO
    │             │
    ▼             ▼
Auto-publish   Verdict = MARGINAL?
               ┌─────┴──────┐
              YES            NO (= FAIL)
               │              │
               ▼              ▼
         Human review    Force human
         queue (24h SLA)  rewrite
```

---

## 6. Financial Model

### Year 1 Cost Comparison

| Item | Manual (100 FTE) | AI System | Year 1 Delta |
|---|---|---|---|
| Headcount (fully loaded $80K avg) | $8,000,000 | $960,000 (12 FTE) | −$7,040,000 |
| LLM API costs (500 deals/day × $0.03) | — | $5,475 | +$5,475 |
| Infrastructure (cloud VM + storage) | — | $12,000 | +$12,000 |
| **Total** | **$8,000,000** | **$977,475** | **−$7,022,525** |

**Payback period on implementation:** < 1 month (assuming $200K build cost)

**Revenue upside (Phase 2):** If AI-optimized deals improve CVR by 5% relative on 10% of GMV, and average deal GMV is $50: 500 deals/day × 0.1 × 0.05 × $50 × 365 = **~$45.6M additional GMV/year**

---

## 7. Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| Feature engineering | textstat, textblob, scikit-learn | Lightweight, no GPU required |
| Scoring model | RandomForestRegressor (200 trees) | Interpretable, fast inference |
| LLM generation | Groq (llama-3.3-70b-versatile) | 500 tok/s, cost-efficient |
| Evaluation | ROUGE-L + custom LLM judge | Objective + semantic quality |
| Orchestration | Python async pipeline | Scalable to 5K deals/day |
| Storage | JSON + CSV audit trail | Simple now, BigQuery-ready |
| Monitoring | Structured logging → Datadog (Phase 2) | Full observability |

---

*This blueprint is a living document. Revise quarterly based on CVR lift data and model performance.*