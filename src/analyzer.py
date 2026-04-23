"""
analyzer.py
Deep exploratory data analysis and NLP feature engineering for Groupon deal content.
Extracts 20+ features and models their relationship with conversion rate (CVR).
"""

import pandas as pd
import numpy as np
import re
import os
import json
import warnings
warnings.filterwarnings('ignore')

import textstat
from textblob import TextBlob
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


# ─────────────────────────────────────────────
# CONSTANTS — derived from domain knowledge
# ─────────────────────────────────────────────
SOCIAL_PROOF_PATTERNS = [
    r'\d[\d,]+\s*(customers?|clients?|homes?|reviews?|ratings?|stars?)',
    r'\d+[\-\s]?star',
    r'top[- ]rated',
    r'most popular',
    r'award.?winning',
    r'certified',
    r'licensed',
    r'background.?check',
    r'satisfaction guarantee',
    r'100%',
]

URGENCY_PATTERNS = [
    r'\blimited\b',
    r'\bonly\b',
    r'\btoday\b',
    r'\bnow\b',
    r'\bexclusive\b',
    r'\bact fast\b',
    r'\bdon.t miss\b',
    r'\bgrab\b',
    r'\bsave\b',
]

SPECIFICITY_PATTERNS = [
    r'\b\d+\s*(minutes?|hours?|days?|weeks?|months?|sessions?|visits?)\b',
    r'\bup to\b',
    r'\bincludes?\b',
    r'\bwhat.s included\b',
    r'\bwhat we offer\b',
    r'\bnot included\b',
    r'\b(good to know|important)\b',
    r'\$\d+',
    r'\d+%',
]

STRUCTURE_PATTERNS = [
    r'what (we offer|is included|you.ll (get|receive))',
    r'why (you should|choose|book)',
    r'good to know',
    r'what is not included',
    r'important (note|info|details)',
]

GENERIC_OPTION_NAMES = ['option 1', 'option 2', 'option 3', 'option 4', 'session']


def count_pattern_matches(text: str, patterns: list) -> int:
    """Count how many patterns match in lowercased text."""
    text_lower = str(text).lower()
    return sum(1 for p in patterns if re.search(p, text_lower))


def has_generic_options(option_names: str) -> int:
    """Return 1 if options are generic placeholders like 'Option 1'."""
    if pd.isna(option_names):
        return 1
    names = [n.strip().lower() for n in str(option_names).split(',')]
    return int(all(any(g in n for g in GENERIC_OPTION_NAMES) for n in names))


def count_fine_print_restrictions(fine_print: str) -> int:
    """Count the number of semicolon-separated restrictions."""
    if pd.isna(fine_print) or not fine_print:
        return 0
    return len(str(fine_print).split(';'))


def get_sentiment(text: str) -> tuple:
    """Return (polarity, subjectivity) via TextBlob."""
    try:
        blob = TextBlob(str(text))
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except Exception:
        return 0.0, 0.0


def get_readability(text: str) -> dict:
    """Get multiple readability scores for a piece of text."""
    text = str(text)
    if len(text.split()) < 5:
        return {'flesch_ease': 50.0, 'flesch_kincaid_grade': 8.0, 'automated_readability': 8.0}
    return {
        'flesch_ease': textstat.flesch_reading_ease(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'automated_readability': textstat.automated_readability_index(text),
    }


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full suite of 20+ content features on a deals DataFrame.
    """
    df = df.copy()

    # --- Basic length features ---
    df['title_length'] = df['title'].fillna('').apply(len)
    df['title_word_count'] = df['title'].fillna('').apply(lambda x: len(str(x).split()))
    df['desc_length'] = df['description'].fillna('').apply(len)
    df['desc_word_count'] = df['description'].fillna('').apply(lambda x: len(str(x).split()))
    df['fine_print_length'] = df['fine_print'].fillna('').apply(len)
    df['fine_print_restriction_count'] = df['fine_print'].apply(count_fine_print_restrictions)

    # --- Readability ---
    desc_read = df['description'].fillna('').apply(get_readability).apply(pd.Series)
    df['desc_flesch_ease'] = desc_read['flesch_ease']
    df['desc_fk_grade'] = desc_read['flesch_kincaid_grade']

    title_read = df['title'].fillna('').apply(get_readability).apply(pd.Series)
    df['title_flesch_ease'] = title_read['flesch_ease']

    # --- Sentiment ---
    desc_sent = df['description'].fillna('').apply(get_sentiment)
    df['desc_polarity'] = desc_sent.apply(lambda x: x[0])
    df['desc_subjectivity'] = desc_sent.apply(lambda x: x[1])

    title_sent = df['title'].fillna('').apply(get_sentiment)
    df['title_polarity'] = title_sent.apply(lambda x: x[0])

    # --- Pattern-based signals ---
    df['social_proof_count'] = df['description'].fillna('').apply(
        lambda x: count_pattern_matches(x, SOCIAL_PROOF_PATTERNS))
    df['urgency_count'] = (
        df['title'].fillna('').apply(lambda x: count_pattern_matches(x, URGENCY_PATTERNS)) +
        df['description'].fillna('').apply(lambda x: count_pattern_matches(x, URGENCY_PATTERNS))
    )
    df['specificity_count'] = df['description'].fillna('').apply(
        lambda x: count_pattern_matches(x, SPECIFICITY_PATTERNS))
    df['structure_section_count'] = df['description'].fillna('').apply(
        lambda x: count_pattern_matches(x, STRUCTURE_PATTERNS))

    # --- Option quality ---
    df['has_generic_options'] = df['option_names'].apply(has_generic_options)
    df['num_options_available'] = df['num_options'].fillna(1)

    # --- Pricing features ---
    df['price_to_value_ratio'] = df['price'] / df['value'].replace(0, np.nan)
    df['log_price'] = np.log1p(df['price'].fillna(0))

    # --- Fine print trap score: many restrictions = friction ---
    df['fine_print_friction'] = df['fine_print_restriction_count'].clip(upper=15) / 15.0

    # --- Content richness ratio ---
    df['desc_to_fine_print_ratio'] = df['desc_word_count'] / (
        df['fine_print_restriction_count'].replace(0, 1))

    return df


def run_statistical_analysis(df: pd.DataFrame) -> dict:
    """
    Run Pearson + Spearman correlations and OLS regression against CVR.
    Returns a dict of findings.
    """
    FEATURE_COLS = [
        'title_length', 'title_word_count', 'desc_word_count', 'desc_length',
        'fine_print_restriction_count', 'fine_print_friction',
        'desc_flesch_ease', 'desc_fk_grade', 'title_flesch_ease',
        'desc_polarity', 'desc_subjectivity', 'title_polarity',
        'social_proof_count', 'urgency_count', 'specificity_count', 'structure_section_count',
        'has_generic_options', 'num_options_available',
        'price_to_value_ratio', 'log_price', 'discount_pct',
        'image_quality_score', 'desc_to_fine_print_ratio',
    ]

    # Filter to available columns
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    sub = df[available_cols + ['cvr']].dropna()

    pearson_results = {}
    spearman_results = {}
    for col in available_cols:
        pr, pp = stats.pearsonr(sub[col], sub['cvr'])
        sr, sp = stats.spearmanr(sub[col], sub['cvr'])
        pearson_results[col] = {'r': round(pr, 4), 'p': round(pp, 4)}
        spearman_results[col] = {'r': round(sr, 4), 'p': round(sp, 4)}

    # Sort by absolute Pearson r
    sorted_pearson = dict(sorted(
        pearson_results.items(), key=lambda x: abs(x[1]['r']), reverse=True
    ))

    # ─── Random Forest importance ───
    X = sub[available_cols].copy()
    y = sub['cvr']
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_importance = dict(zip(available_cols, rf.feature_importances_.round(4)))
    rf_importance = dict(sorted(rf_importance.items(), key=lambda x: x[1], reverse=True))

    rf_cv = cross_val_score(rf, X, y, cv=5, scoring='r2')

    return {
        'pearson': sorted_pearson,
        'spearman': spearman_results,
        'rf_importance': rf_importance,
        'rf_cv_r2_mean': round(rf_cv.mean(), 4),
        'rf_cv_r2_std': round(rf_cv.std(), 4),
        'n_deals': len(sub),
    }


def get_top_bottom_performers(df: pd.DataFrame, n: int = 20) -> dict:
    """Split deals into top and bottom CVR performers and compute mean feature diffs."""
    df_sorted = df.sort_values('cvr', ascending=False).dropna(subset=['cvr'])
    top = df_sorted.head(n)
    bottom = df_sorted.tail(n)

    feature_cols = [c for c in df.columns if c not in [
        'deal_id', 'title', 'description', 'fine_print', 'option_names',
        'merchant_name', 'geo', 'category', 'subcategory', 'cvr', 'aov', 'refund_rate'
    ] and df[c].dtype in [np.float64, np.int64]]

    comparison = {}
    for col in feature_cols:
        comparison[col] = {
            'top_mean': round(top[col].mean(), 3),
            'bottom_mean': round(bottom[col].mean(), 3),
            'delta': round(top[col].mean() - bottom[col].mean(), 3),
        }
    return comparison


def run_full_analysis(filepath: str = "data/deals.csv", save_dir: str = "docs") -> dict:
    """
    Master analysis function. Runs everything and saves a JSON findings report.
    """
    print("Loading data...")
    df = pd.read_csv(filepath)
    print(f"  → {len(df)} deals loaded")

    print("Engineering features...")
    df = engineer_features(df)
    print(f"  → {len([c for c in df.columns if c not in pd.read_csv(filepath).columns])} new features created")

    print("Running statistical analysis...")
    stats_results = run_statistical_analysis(df)
    print(f"  → RF CV R² = {stats_results['rf_cv_r2_mean']:.4f} ± {stats_results['rf_cv_r2_std']:.4f}")

    print("Computing top vs bottom performer comparison...")
    perf_comparison = get_top_bottom_performers(df, n=min(50, len(df) // 5))

    # Category-level CVR stats
    cat_stats = df.groupby('category')['cvr'].agg(['mean', 'median', 'std', 'count']).round(5)

    # Key findings summary
    top_positive_features = {k: v for k, v in stats_results['pearson'].items()
                             if v['r'] > 0.1 and v['p'] < 0.05}
    top_negative_features = {k: v for k, v in stats_results['pearson'].items()
                             if v['r'] < -0.1 and v['p'] < 0.05}

    findings = {
        'n_deals': stats_results['n_deals'],
        'top_positive_correlations': top_positive_features,
        'top_negative_correlations': top_negative_features,
        'rf_importance_top10': dict(list(stats_results['rf_importance'].items())[:10]),
        'rf_cv_r2': stats_results['rf_cv_r2_mean'],
        'category_cvr': cat_stats.to_dict(),
        'top_vs_bottom': perf_comparison,
        'all_pearson': stats_results['pearson'],
    }

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'analysis_findings.json'), 'w') as f:
        json.dump(findings, f, indent=2)

    print(f"\n✓ Analysis complete. Findings saved to {save_dir}/analysis_findings.json")
    print("\nTop positive features (Pearson r, p-value):")
    for feat, vals in sorted(top_positive_features.items(), key=lambda x: x[1]['r'], reverse=True):
        print(f"  {feat:35s} r={vals['r']:+.4f}  p={vals['p']:.4f}")

    print("\nTop negative features:")
    for feat, vals in sorted(top_negative_features.items(), key=lambda x: x[1]['r']):
        print(f"  {feat:35s} r={vals['r']:+.4f}  p={vals['p']:.4f}")

    return findings, df


if __name__ == "__main__":
    run_full_analysis()
