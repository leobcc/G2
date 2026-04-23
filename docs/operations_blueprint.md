# Operations Blueprint: AI-Driven Deal Content Pipeline

## 1. Executive Summary
Currently, Groupon relies on 100 FTEs to write, vet, and edit deal content. This operation is designed to optimize for throughput rather than conversion rate (CVR). The proposed pipeline fully automates content optimization using an LLM-based intelligent rewriting system that actively maximizes variables observed to correlate with CVR—specifically, title length and detailed description word counts. 

By implementing this AI-driven approach, we will replace manual drafting, reduce costs dramatically, and guarantee statistically improved conversion metrics.

## 2. System Architecture at Scale
The PoC successfully demonstrates the process on single rows. To scale to handles all deals:

### A. Data Ingestion & Scoring
- **Automated Ingestion**: Inbound deals flow from merchants to an initial staging database.
- **Pre-Flight Scoring**: Our EDA algorithm assesses the raw deal. If it falls below a threshold (e.g., description length < 50 words, title < 40 characters), the deal is flagged for optimization.

### B. The Optimization Engine (LLM)
- **Model**: Utilizing high-throughput, low-latency LLMs (e.g., Llama-3.3-70b-versatile via Groq).
- **Prompt Architecture**: Strict constraints on hallucinations (No price tampering, no new fine-print rules) while adhering to our CVR data principles (longer compelling copy, bullet points).
- **JSON Formatting**: Outputs strict structured data to be piped directly into the CMS.

### C. The Evaluator (LLM-as-a-Judge)
- An independent evaluator LLM acts as the QA gate.
- Evaluates the `new_content` vs `original` to ensure increased conversion likelihood and checks heuristic metrics (e.g., `desc_length_delta > 0`).

## 3. Human-in-the-Loop Change Management
What is the path from 100 FTE to the target state?

- **Phase 1 (Month 1-2): "Copilot" Mode** 
  - Engine proposes rewrites. 50% of the FTEs act as approvers, ensuring brand voice and safety.
- **Phase 2 (Month 3-5): "Autopilot with Exceptions"** 
  - System automatically approves 90% of deals that pass the Evaluator score > 90/100.
  - FTE team is reduced to ~10 "Quality Auditors" who resolve the 10% flagged by the LLM Evaluator for edge-case errors or policy violations.
- **Target State (Month 6+)**
  - Fully automated. The former content team shifts roles from writing copy to managing strategy, running deep A/B tests, or is phased out to realize absolute cost efficiency.

## 4. Weekly KPIs to Monitor
1. **Automated Conversion Lift (Delta-CVR)**: Track CVR of AI-written vs Original Baseline.
2. **Human Intervention Rate (HIR)**: Percentage of AI rewrites flagged for auditor review (Target: < 5%).
3. **Execution Latency**: Time from merchant submission to live published deal (Target: < 10 seconds).
4. **LLM Cost per Deal**: API overhead spend relative to traditional human wage cost.