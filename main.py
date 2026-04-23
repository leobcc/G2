import pandas as pd
import asyncio
import os
import json
from src.optimizer import optimize_deal_async
from src.evaluator import evaluate_deal_content_async
import time

async def process_single_deal(deal_dict):
    """Async pipeline for optimizing and evaluating a single deal"""
    try:
        # Step 1: Optimize the content using Chain-of-Thought & Pydantic Validation
        new_content_obj = await optimize_deal_async(deal_dict)
        new_content = new_content_obj.model_dump()
        
        # Step 2: Multi-Metric NLP & LLM-Judge Evaluation
        eval_metrics = await evaluate_deal_content_async(deal_dict, new_content)
        
        return {
            "original": deal_dict,
            "new_content": new_content,
            "eval_metrics": eval_metrics
        }
    except Exception as e:
        print(f"Error processing deal {deal_dict.get('deal_id', 'Unknown')}: {e}")
        return None

async def process_deals_bulk(limit=5):
    filepath = "data/deals.csv"
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    df = pd.read_csv(filepath).head(limit)
    print(f"Starting CUTTING-EDGE ASYNC PoC on {limit} deals.")
    
    start_time = time.time()
    
    # Process concurrently using asyncio gather
    tasks = [process_single_deal(row.to_dict()) for i, row in df.iterrows()]
    results = await asyncio.gather(*tasks)
    
    # Filter out any failed processing
    results = [r for r in results if r is not None]
    
    elapsed_time = time.time() - start_time
    print(f"Processed {len(results)} deals concurrently in {elapsed_time:.2f} seconds.")
    
    # Save Output
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(results).to_json("results/poc_results.json", orient='records', indent=4)
    print(f"Saved highly robust evaluations to results/poc_results.json.\n")

if __name__ == "__main__":
    import platform
    if os.name == 'nt' or platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(process_deals_bulk(limit=5))
