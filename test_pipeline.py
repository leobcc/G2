from src.analyzer import run_full_analysis
from src.scorer import score_deal, score_dataframe
import pandas as pd

print("=" * 60)
print("TEST: Analyzer + Scorer")
print("=" * 60)

findings, df = run_full_analysis('data/deals.csv')

print("\nTop 10 RF Feature Importances:")
for k, v in list(findings['rf_importance_top10'].items())[:10]:
    print(f"  {k:<40} {v:.4f}")

print(f"\nRF CV R2: {findings['rf_cv_r2']}")

print("\nTop positive correlations (p<0.05):")
for k, v in list(findings['top_positive_correlations'].items())[:8]:
    print(f"  {k:<40} r={v['r']:+.4f}  p={v['p']:.4f}")

print("\nTop negative correlations:")
for k, v in list(findings['top_negative_correlations'].items())[:5]:
    print(f"  {k:<40} r={v['r']:+.4f}  p={v['p']:.4f}")

# Test scorer on first deal
deal = df.iloc[0].to_dict()
score = score_deal(deal)
print(f"\nScorer test on deal 0 (cvr={deal.get('cvr','N/A')}):")
print(f"  Composite score: {score['composite_score']}")
print(f"  Rewrite needed: {score['rewrite_needed']}")
print(f"  Breakdown: {score['breakdown']}")

# Batch score a sample
scores_df = score_dataframe(df.head(10))
print(f"\nBatch scores (first 10):")
print(scores_df[['composite_score', 'rewrite_needed']].to_string())
