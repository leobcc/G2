"""
scorer.py
Composite Content Quality Scorer (0–100).

Weights are derived DIRECTLY from Random Forest feature importances trained on 500
real Groupon deals.  At import time this module attempts to load
``docs/analysis_findings.json``; if found, weights are computed from
``rf_importance_full``; otherwise the hard-coded empirical fallbacks are used.
This means weights automatically update whenever ``analyzer.py`` is re-run —
the core "self-improving system" loop.

Scorer dimension → RF feature mapping
-------------------------------------
image_quality   ← image_quality_score
desc_length     ← desc_word_count + desc_length   (content richness)
specificity     ← specificity_count
structure       ← structure_section_count
title           ← title_length + title_flesch_ease
readability     ← desc_flesch_ease + desc_fk_grade
social_proof    ← social_proof_count
options         ← has_generic_options  (note: negated — generic = bad)
fine_print      ← fine_print_restriction_count + fine_print_friction
"""

import re
import json
import os
import numpy as np
import textstat
from typing import Union

# ─────────────────────────────────────────────
# DIMENSION → RF-FEATURE MAPPING
# Each tuple lists the RF feature names that contribute to that scorer dimension.
# The combined importance of these features will become the dimension's weight.
# ─────────────────────────────────────────────
_DIMENSION_RF_MAP = {
    'desc_word_count':     ('desc_word_count', 'desc_length'),
    'specificity':         ('specificity_count',),
    'structure':           ('structure_section_count',),
    'social_proof':        ('social_proof_count',),
    'title_length':        ('title_length', 'title_flesch_ease'),
    'image_quality':       ('image_quality_score',),
    'readability':         ('desc_flesch_ease', 'desc_fk_grade'),
    'options_quality':     ('has_generic_options',),
    'fine_print_friction': ('fine_print_restriction_count', 'fine_print_friction'),
}

# Hard-coded empirical fallback (used when analysis_findings.json is absent).
# These represent domain-expert priors and are intentionally kept as the default
# so the scorer is always usable out-of-the-box.
_HARDCODED_WEIGHTS = {
    'desc_word_count':     0.22,
    'specificity':         0.18,
    'structure':           0.14,
    'social_proof':        0.12,
    'title_length':        0.10,
    'image_quality':       0.09,
    'readability':         0.08,
    'options_quality':     0.05,
    'fine_print_friction': 0.02,
}

# Minimum weight floor — no dimension is ever weighted less than this.
_MIN_WEIGHT = 0.02


def _load_rf_weights(findings_path: str | None = None) -> dict:
    """
    Attempt to derive scorer dimension weights from RF importances saved in
    ``docs/analysis_findings.json`` (relative to repo root).  Returns the hard-
    coded fallback if the file is absent or malformed.
    """
    # Resolve path relative to this file's location (src/) → ../docs/
    if findings_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        findings_path = os.path.join(here, '..', 'docs', 'analysis_findings.json')

    try:
        with open(findings_path, 'r') as fh:
            findings = json.load(fh)
        rf_imp: dict = findings.get('rf_importance_full') or findings.get('rf_importance_top10', {})
        if not rf_imp:
            raise ValueError("No RF importances found in findings JSON.")

        raw: dict[str, float] = {}
        for dim, rf_features in _DIMENSION_RF_MAP.items():
            raw[dim] = sum(rf_imp.get(feat, 0.0) for feat in rf_features)

        # Apply minimum floor so minor dimensions are never zeroed out
        floored = {dim: max(v, _MIN_WEIGHT) for dim, v in raw.items()}

        # Normalize so all positive weights (excluding fine_print, which is a
        # penalty) sum to 1.0, then scale to match expected 0–100 range.
        # fine_print_friction stays as an independent penalty.
        positive_dims = [d for d in floored if d != 'fine_print_friction']
        total = sum(floored[d] for d in positive_dims)
        normalized = {d: round(floored[d] / total * (1.0 - floored['fine_print_friction']), 4)
                      for d in positive_dims}
        normalized['fine_print_friction'] = round(floored['fine_print_friction'] /
                                                   sum(floored.values()) * 0.15, 4)

        return normalized

    except Exception:
        # Silent fallback — system is still fully operational
        return _HARDCODED_WEIGHTS.copy()


# ─────────────────────────────────────────────
# MODULE-LEVEL WEIGHTS — populated at import time
# Inspect these to verify the data-driven values are in effect:
#   from src.scorer import WEIGHTS; print(WEIGHTS)
# ─────────────────────────────────────────────
WEIGHTS = _load_rf_weights()

# Expose individual constants for backward compatibility with other modules
WEIGHT_DESC_WORD_COUNT     = WEIGHTS['desc_word_count']
WEIGHT_SPECIFICITY         = WEIGHTS['specificity']
WEIGHT_STRUCTURE           = WEIGHTS['structure']
WEIGHT_SOCIAL_PROOF        = WEIGHTS['social_proof']
WEIGHT_TITLE_LENGTH        = WEIGHTS['title_length']
WEIGHT_IMAGE_QUALITY       = WEIGHTS['image_quality']
WEIGHT_READABILITY         = WEIGHTS['readability']
WEIGHT_OPTIONS_QUALITY     = WEIGHTS['options_quality']
WEIGHT_FINE_PRINT_FRICTION = WEIGHTS['fine_print_friction']

# Normalization floors/ceilings
DESC_WORD_OPTIMAL_TARGET  = 200    # Deals with 200+ words perform best
TITLE_CHAR_OPTIMAL_TARGET = 80     # 80+ character titles perform best
SPECIFICITY_OPTIMAL       = 5      # 5+ specificity signals
SOCIAL_PROOF_OPTIMAL      = 3      # 3+ social proof signals
STRUCTURE_OPTIMAL         = 3      # 3+ structural sections
MAX_RESTRICTIONS_OK       = 6      # ≤6 fine-print items = acceptable friction

SOCIAL_PROOF_PATTERNS = [
    r'\d[\d,]+\s*(customers?|clients?|homes?|reviews?|ratings?)',
    r'\d+[\-\s]?star',
    r'top[- ]rated', r'certified', r'licensed', r'background.?check',
    r'satisfaction guarantee', r'100%',
]

SPECIFICITY_PATTERNS = [
    r'\b\d+\s*(minutes?|hours?|days?|weeks?|sessions?|visits?|sq\.?\s*ft)\b',
    r'\bup to\b', r'\bincludes?\b', r'\bwhat.s included\b',
    r'\bwhat we offer\b', r'\bnot included\b', r'\b(good to know)\b',
    r'\$\d+', r'\d+%',
]

STRUCTURE_PATTERNS = [
    r'what (we offer|is included)',
    r'why (you should|choose|book)',
    r'good to know',
    r'what is not included',
    r'(why.*grab|why.*offer)',
]


def _count_patterns(text: str, patterns: list) -> int:
    text_l = str(text).lower()
    return sum(1 for p in patterns if re.search(p, text_l))


def _norm(value: float, floor: float, ceiling: float) -> float:
    """Clamp value to [0, 1] using floor and ceiling."""
    return float(np.clip((value - floor) / max(ceiling - floor, 1e-6), 0.0, 1.0))


def score_deal(deal: Union[dict, object]) -> dict:
    """
    Compute the composite content quality score (0–100) for one deal.
    Returns a detailed breakdown dict.
    """
    if hasattr(deal, 'to_dict'):
        deal = deal.to_dict()

    title    = str(deal.get('title', '') or '')
    desc     = str(deal.get('description', '') or '')
    fp       = str(deal.get('fine_print', '') or '')
    options  = str(deal.get('option_names', '') or '')
    img_qual = float(deal.get('image_quality_score', 3) or 3)

    # ── 1. Description word count ──────────────
    desc_words = len(desc.split())
    desc_score = _norm(desc_words, 0, DESC_WORD_OPTIMAL_TARGET)

    # ── 2. Specificity count ───────────────────
    spec_count = _count_patterns(desc, SPECIFICITY_PATTERNS)
    spec_score = _norm(spec_count, 0, SPECIFICITY_OPTIMAL)

    # ── 3. Structure / section headers ─────────
    struct_count = _count_patterns(desc, STRUCTURE_PATTERNS)
    struct_score = _norm(struct_count, 0, STRUCTURE_OPTIMAL)

    # ── 4. Social proof ────────────────────────
    social_count = _count_patterns(desc, SOCIAL_PROOF_PATTERNS)
    social_score = _norm(social_count, 0, SOCIAL_PROOF_OPTIMAL)

    # ── 5. Title length ────────────────────────
    title_chars = len(title)
    title_score = _norm(title_chars, 0, TITLE_CHAR_OPTIMAL_TARGET)

    # ── 6. Image quality ───────────────────────
    img_score = _norm(img_qual, 1, 5)

    # ── 7. Readability (Flesch ease; higher = more readable) ──
    try:
        flesch = textstat.flesch_reading_ease(desc) if len(desc.split()) > 5 else 50.0
    except Exception:
        flesch = 50.0
    # Optimal range: 40–70 (standard to fairly easy)
    read_score = _norm(flesch, 20, 70)

    # ── 8. Option name quality ─────────────────
    option_list = [o.strip().lower() for o in options.split(',')]
    generic_opts = ['option 1', 'option 2', 'option 3', 'option 4', '1 session']
    is_generic = all(any(g in o for g in generic_opts) for o in option_list)
    opt_score = 0.0 if is_generic else 1.0

    # ── 9. Fine print friction penalty ─────────
    restriction_count = len(fp.split(';')) if fp else 0
    friction_penalty = _norm(restriction_count, MAX_RESTRICTIONS_OK, 15)

    # ── Weighted composite ─────────────────────
    raw_score = (
        WEIGHT_DESC_WORD_COUNT     * desc_score +
        WEIGHT_SPECIFICITY         * spec_score +
        WEIGHT_STRUCTURE           * struct_score +
        WEIGHT_SOCIAL_PROOF        * social_score +
        WEIGHT_TITLE_LENGTH        * title_score +
        WEIGHT_IMAGE_QUALITY       * img_score +
        WEIGHT_READABILITY         * read_score +
        WEIGHT_OPTIONS_QUALITY     * opt_score -
        WEIGHT_FINE_PRINT_FRICTION * friction_penalty
    )

    composite_score = round(float(np.clip(raw_score * 100, 0, 100)), 2)

    return {
        'composite_score': composite_score,
        'breakdown': {
            'desc_word_count_score':   round(desc_score   * 100, 1),
            'specificity_score':       round(spec_score   * 100, 1),
            'structure_score':         round(struct_score * 100, 1),
            'social_proof_score':      round(social_score * 100, 1),
            'title_length_score':      round(title_score  * 100, 1),
            'image_quality_score':     round(img_score    * 100, 1),
            'readability_score':       round(read_score   * 100, 1),
            'option_quality_score':    round(opt_score    * 100, 1),
            'fine_print_friction':     round(friction_penalty * 100, 1),
        },
        'raw_stats': {
            'desc_word_count':         desc_words,
            'specificity_count':       spec_count,
            'structure_section_count': struct_count,
            'social_proof_count':      social_count,
            'title_char_length':       title_chars,
            'flesch_ease':             round(flesch, 2),
            'restriction_count':       restriction_count,
        },
        'rewrite_needed': composite_score < 45,
        'improvement_opportunity': max(0.0, round(100 - composite_score, 1)),
    }


def score_dataframe(df) -> 'pd.DataFrame':
    """Batch score an entire DataFrame of deals."""
    import pandas as pd
    from tqdm import tqdm
    tqdm.pandas(desc="  Scoring deals", leave=False)
    scores = df.progress_apply(score_deal, axis=1)
    results = pd.DataFrame(scores.tolist(), index=df.index)
    return results
