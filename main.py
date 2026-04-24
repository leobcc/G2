"""
main.py
Production-grade deal content optimization pipeline.

Features:
- Triages deal by worst-performing (lowest CVR) first
- Uses category-aware few-shot optimizer
- Rigorous multi-metric evaluation per deal
- Retry logic with exponential backoff
- Structured JSON+CSV output with full audit trail
- Run summary statistics
"""

import os
import sys
import json
import time
import logging
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

# ── local modules ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.analyzer import engineer_features, run_full_analysis
from src.scorer   import score_deal, score_dataframe
from src.optimizer import optimize_deal
from src.evaluator import evaluate_rewrite

# ── logging config ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout,  # stdout flushes line-by-line in a terminal; also avoids CP1252 stderr issues
    force=True,
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
DATA_PATH   = Path("data/deals.csv")


def load_and_prepare(filepath: Path, run_analysis: bool = True) -> pd.DataFrame:
    """Load deals, engineer features, run statistical analysis."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"  Loaded {len(df)} deals across {df['category'].nunique()} categories")

    logger.info("Engineering NLP features...")
    df = engineer_features(df)

    if run_analysis:
        logger.info("Running full statistical analysis...")
        try:
            run_full_analysis(str(filepath))
        except Exception as e:
            logger.warning(f"Analysis run encountered non-critical error: {e}")

    return df


def triage(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """
    Priority-score and sort deals for processing.
    Priority = low CVR + high opportunity (bad content that CAN be improved).
    """
    logger.info("Scoring all deals and triaging by priority...")
    score_results = score_dataframe(df)
    df = df.copy()
    df['content_score']        = score_results['composite_score'].values
    df['rewrite_needed']       = score_results['rewrite_needed'].values
    df['improvement_opportunity'] = score_results['improvement_opportunity'].values

    # True priority: very low CVR AND low content score = highest opportunity
    df['triage_priority'] = (1 - df['cvr'].rank(pct=True)) * 0.6 + \
                            df['improvement_opportunity'].rank(pct=True) * 0.4

    prioritized = df.sort_values('triage_priority', ascending=False)
    logger.info(f"  {df['rewrite_needed'].sum()} deals flagged as rewrite-needed")
    logger.info(f"  Processing top {limit} by priority")
    return prioritized.head(limit).copy()


def run_pipeline(
    limit: int = 10,
    run_eda: bool = True,
    output_csv: bool = True,
) -> pd.DataFrame:
    """
    Master pipeline: load → analyse → triage → optimize → evaluate → save.
    """
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── 1. Load & prepare ────────────────────────
    logger.info("─" * 60)
    logger.info("STEP 1/4  Load & Analyse - feature engineering + statistical analysis")
    df_full = load_and_prepare(DATA_PATH, run_analysis=run_eda)
    logger.info("STEP 1/4  Complete")

    # ── 2. Triage ────────────────────────────────
    logger.info("-" * 60)
    logger.info("STEP 2/4  Triage - scoring all deals and selecting priority batch")
    df_batch = triage(df_full, limit)

    # ── 3. Optimize & Evaluate ───────────────────
    logger.info("-" * 60)
    logger.info(f"STEP 3/4  Optimize & Evaluate - processing {limit} deals via LLM")
    logger.info("-" * 60)
    results = []
    stats = {'pass': 0, 'marginal': 0, 'fail': 0, 'error': 0}

    for idx, (_, row) in enumerate(df_batch.iterrows(), start=1):
        deal = row.to_dict()
        deal_id = deal.get('deal_id', f'deal_{idx}')[:8]

        logger.info(f"[{idx:02d}/{limit}] {deal_id}...  CVR={deal.get('cvr','?'):6}  "
                    f"ContentScore={deal.get('content_score', '?'):.1f}")

        try:
            logger.info(f"         [1/2] LLM rewriting...")
            _t0 = time.time()
            # Optimize (with category-aware few-shot context)
            optimized = optimize_deal(deal, df_reference=df_full)
            logger.info(f"         [1/2] done ({time.time()-_t0:.1f}s)")

            logger.info(f"         [2/2] LLM judging (blinded A/B)...")
            _t0 = time.time()
            # Evaluate
            evaluation = evaluate_rewrite(deal, optimized, scorer_fn=score_deal)
            logger.info(f"         [2/2] done ({time.time()-_t0:.1f}s)")
            verdict = evaluation.get('verdict', 'FAIL')
            stats[verdict.lower()] += 1

            judge_score_delta = evaluation.get('llm_judge', {}).get('score_delta', 0)
            comp_delta        = evaluation.get('composite', {}).get('delta', 0)
            words_added       = evaluation.get('length', {}).get('words_added', 0)

            logger.info(f"         >> {verdict}  | CompositeD={comp_delta:+.1f}  "
                        f"JudgeD={judge_score_delta:+.0f}  WordsAdded={words_added:+d}")

            results.append({
                'deal_id':          deal.get('deal_id'),
                'category':         deal.get('category'),
                'geo':              deal.get('geo'),
                'merchant_name':    deal.get('merchant_name'),
                'cvr_original':     deal.get('cvr'),
                'content_score_original': deal.get('content_score'),
                'original_title':   deal.get('title'),
                'original_desc':    deal.get('description'),
                'improved_title':   optimized.get('improved_title'),
                'improved_desc':    optimized.get('improved_description'),
                'improved_options': optimized.get('improved_option_names'),
                'reasoning':        optimized.get('reasoning'),
                'verdict':          verdict,
                'composite_orig':   evaluation['composite']['original'],
                'composite_new':    evaluation['composite']['optimized'],
                'composite_delta':  evaluation['composite']['delta'],
                'words_added':      words_added,
                'title_chars_added': evaluation['length']['new_title_chars'] - evaluation['length']['orig_title_chars'],
                'specificity_delta': evaluation['specificity']['delta'],
                'rouge_l':          evaluation['rouge_l'],
                'judge_new_score':  evaluation['llm_judge'].get('new_score'),
                'judge_orig_score': evaluation['llm_judge'].get('orig_score'),
                'judge_score_delta': judge_score_delta,
                'judge_reasoning':  evaluation['llm_judge'].get('reasoning'),
                'guardrail_passed': optimized.get('_meta', {}).get('guardrail_passed', False),
                'full_evaluation':  json.dumps(evaluation),
                'run_ts':           run_ts,
            })

        except Exception as e:
            logger.error(f"  Pipeline error for deal {deal_id}: {e}")
            stats['error'] += 1

        # Polite rate limiting
        time.sleep(0.8)

    # ── 4. Save results ───────────────────────────
    logger.info("─" * 60)
    logger.info("STEP 4/4  Saving results...")
    results_df = pd.DataFrame(results)

    json_path = RESULTS_DIR / f"pipeline_results_{run_ts}.json"
    csv_path  = RESULTS_DIR / f"pipeline_results_{run_ts}.csv"
    results_df.to_json(json_path, orient='records', indent=2)
    if output_csv:
        results_df.to_csv(csv_path, index=False)

    # Always overwrite "latest" for convenience
    results_df.to_json(RESULTS_DIR / "latest_results.json", orient='records', indent=2)
    if output_csv:
        results_df.to_csv(RESULTS_DIR / "latest_results.csv", index=False)

    # ── 5. Summary report ─────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("═" * 60)
    logger.info(f"  Deals processed : {len(results)}")
    logger.info(f"  PASS            : {stats['pass']}")
    logger.info(f"  MARGINAL        : {stats['marginal']}")
    logger.info(f"  FAIL            : {stats['fail']}")
    logger.info(f"  ERROR           : {stats['error']}")
    if not results_df.empty:
        logger.info(f"  Avg composite Δ : {results_df['composite_delta'].mean():+.2f}")
        logger.info(f"  Avg words added : {results_df['words_added'].mean():+.1f}")
        logger.info(f"  Avg judge Δ     : {results_df['judge_score_delta'].mean():+.2f}")
        logger.info(f"  Guardrail pass  : {results_df['guardrail_passed'].mean() * 100:.0f}%")
    logger.info(f"  Results saved   : {json_path}")
    logger.info("═" * 60)

    return results_df


if __name__ == "__main__":
    run_pipeline(limit=10, run_eda=True)

