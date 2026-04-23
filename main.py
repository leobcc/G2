import pandas as pd
from src.optimizer import optimize_deal
from src.evaluator import evaluate_deal_content
import time

def process_deals(limit=5):
    filepath = "data/deals.csv"
    df = pd.read_csv(filepath).head(limit)
    results = []

    print(f"Starting PoC on {limit} deals to test optimization engine.")

    for i, row in df.iterrows():
        deal_dict = row.to_dict()
        
        # Step 1: Optimize the content
        new_content = optimize_deal(deal_dict)
        
        # Step 2: Evaluate the optimization using an LLM-as-a-judge
        eval_metrics = evaluate_deal_content(deal_dict, new_content)
        
        results.append({
            "original": deal_dict,
            "new_content": new_content,
            "eval_metrics": eval_metrics
        })
        time.sleep(1) # rate limiting for API
        
    # Save the PoC run
    import os
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(results).to_json("results/poc_results.json", orient='records', indent=4)
    print(f"\nProcessing complete. Saved {len(results)} evaluated optimizations to results/poc_results.json.\n")

if __name__ == "__main__":
    process_deals()
