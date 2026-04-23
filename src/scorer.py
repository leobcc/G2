"""
scorer.py
Composite Content Quality Scorer (0–100).
Weights derived from EDA regression findings.
Dimensions: Clarity, Specificity, Persuasion, Structural Quality, Image Quality.
"""

import re
import numpy as np
import textstat
from typing import Union

# ─────────────────────────────────────────────
# SCORING WEIGHTS — calibrated from RF feature importances
# and validated against known CVR leaders in the dataset.
# ─────────────────────────────────────────────
WEIGHT_DESC_WORD_COUNT     = 0.22   # Strongest predictor
WEIGHT_SPECIFICITY         = 0.18   # Mentions of numbers, inclusions, durations
WEIGHT_STRUCTURE           = 0.14   # Structured sections (What We Offer / Good to Know)
WEIGHT_SOCIAL_PROOF        = 0.12   # Quantified social signals
WEIGHT_TITLE_LENGTH        = 0.10   # Descriptive, specific title
WEIGHT_IMAGE_QUALITY       = 0.09   # Visual quality signal
WEIGHT_READABILITY         = 0.08   # Flesch ease (accessibility)
WEIGHT_OPTIONS_QUALITY     = 0.05   # Descriptive option names
WEIGHT_FINE_PRINT_FRICTION = 0.02   # Penalise excessive restrictions

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
    scores = df.apply(score_deal, axis=1)
    results = pd.DataFrame(scores.tolist(), index=df.index)
    return results
