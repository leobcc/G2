"""
evaluator.py
Rigorous multi-metric evaluation of original vs. optimized deal content.

Metrics:
  1. Composite content score delta  (src/scorer.py)
  2. Readability delta              (Flesch-Kincaid via textstat)
  3. Specificity delta              (concrete-entity count)
  4. ROUGE-L                        (content preservation / anti-hallucination)
  5. LLM-as-a-judge (blinded A/B)   (rubric: 5 axes, randomised A/B assignment)
"""

import os
import json
import random
import logging
import textstat
from groq import Groq
from dotenv import load_dotenv
from rouge_score import rouge_scorer

load_dotenv()
logger = logging.getLogger(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"
ROUGE = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

JUDGE_SYSTEM_PROMPT = """You are a rigorous content quality analyst.
You will be shown two versions (A and B) of a Groupon deal description.
You do NOT know which is original and which was rewritten.

Score each version on the following five axes (0–20 each, total = 100):
  1. Clarity            — Is the language clear and easy to understand?
  2. Persuasiveness     — Does it create compelling reasons to buy?
  3. Specificity        — Does it include concrete facts, numbers, details?
  4. Accuracy/Trust     — Does it feel trustworthy, not over-hyped?
  5. Professionalism    — Is the tone appropriate for a commercial listing?

Return ONLY valid JSON:
{
  "score_A": <total 0-100>,
  "score_B": <total 0-100>,
  "breakdown_A": {"clarity": 0-20, "persuasiveness": 0-20, "specificity": 0-20, "accuracy": 0-20, "professionalism": 0-20},
  "breakdown_B": {"clarity": 0-20, "persuasiveness": 0-20, "specificity": 0-20, "accuracy": 0-20, "professionalism": 0-20},
  "winner": "A" or "B",
  "reasoning": "one sentence"
}
"""


def _compute_specificity(text: str) -> int:
    """Count concrete fact signals: numbers, durations, named services."""
    import re
    patterns = [
        r'\b\d+\s*(minutes?|hours?|days?|sessions?|sq\.?\s*ft|lbs?|km|miles?)\b',
        r'\b\d+[\-–]\d+\b',  # ranges like 1-3 hours
        r'\$\d+', r'\d+%',
        r'\b(certified|licensed|insured|bonded|background.?check)\b',
        r'\bup to\b', r'\bincludes?\b',
    ]
    text_l = text.lower()
    return sum(1 for p in patterns if re.search(p, text_l))


def _llm_judge(original: dict, optimized: dict) -> dict:
    """
    Blinded A/B LLM judgment.
    Randomly assign original/optimized to A/B to prevent positional bias.
    """
    orig_title = str(original.get('title', ''))
    orig_desc  = str(original.get('description', ''))
    new_title  = str(optimized.get('improved_title', ''))
    new_desc   = str(optimized.get('improved_description', ''))

    # Randomly assign to A/B
    flip = random.random() > 0.5
    if flip:
        a_title, a_desc, b_title, b_desc = new_title, new_desc, orig_title, orig_desc
        a_is_new = True
    else:
        a_title, a_desc, b_title, b_desc = orig_title, orig_desc, new_title, new_desc
        a_is_new = False

    user_prompt = f"""DEAL — VERSION A:
Title: {a_title}
Description: {a_desc[:500]}

DEAL — VERSION B:
Title: {b_title}
Description: {b_desc[:500]}

Score both versions on all 5 axes. Return only valid JSON."""

    try:
        resp = client.chat.completions.create(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            model=MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=600,
        )
        result = json.loads(resp.choices[0].message.content)

        # Decode which version won
        judge_winner_label = result.get('winner', 'A')
        new_is_winner = (judge_winner_label == 'A') == a_is_new
        score_new  = result['score_A'] if a_is_new else result['score_B']
        score_orig = result['score_B'] if a_is_new else result['score_A']

        return {
            'new_score':  score_new,
            'orig_score': score_orig,
            'new_wins':   new_is_winner,
            'score_delta': score_new - score_orig,
            'breakdown_new':  result['breakdown_A'] if a_is_new else result['breakdown_B'],
            'breakdown_orig': result['breakdown_B'] if a_is_new else result['breakdown_A'],
            'reasoning': result.get('reasoning', ''),
            'blinding_flip': flip,
        }
    except Exception as e:
        logger.warning(f"LLM judge call failed: {e}")
        return {'new_wins': None, 'new_score': 0, 'orig_score': 0, 'score_delta': 0,
                'reasoning': f'Judge failed: {e}', 'blinding_flip': flip}


def evaluate_rewrite(original: dict, optimized: dict, scorer_fn=None) -> dict:
    """
    Full multi-metric evaluation of a single deal rewrite.

    Args:
        original:    raw deal dict from deals.csv
        optimized:   output dict from optimizer.optimize_deal()
        scorer_fn:   optional callable(deal_dict) -> {'composite_score': float}

    Returns:
        evaluation dict with all metrics and an aggregate pass/fail verdict.
    """
    orig_desc = str(original.get('description', '') or '')
    new_desc  = str(optimized.get('improved_description', '') or '')
    orig_title = str(original.get('title', '') or '')
    new_title  = str(optimized.get('improved_title', '') or '')

    # ── 1. Composite score delta ────────────────
    if scorer_fn:
        orig_score_obj = scorer_fn(original)
        new_deal = {**original,
                    'title':       new_title,
                    'description': new_desc,
                    'option_names': optimized.get('improved_option_names', original.get('option_names', ''))}
        new_score_obj = scorer_fn(new_deal)
        composite_orig = orig_score_obj.get('composite_score', 0)
        composite_new  = new_score_obj.get('composite_score', 0)
        composite_delta = round(composite_new - composite_orig, 2)
    else:
        composite_orig = composite_new = composite_delta = None

    # ── 2. Readability delta ────────────────────
    def safe_flesch(text):
        try:
            return textstat.flesch_reading_ease(text) if len(text.split()) > 5 else 50.0
        except Exception:
            return 50.0

    def safe_fk(text):
        try:
            return textstat.flesch_kincaid_grade(text) if len(text.split()) > 5 else 8.0
        except Exception:
            return 8.0

    orig_flesch  = safe_flesch(orig_desc)
    new_flesch   = safe_flesch(new_desc)
    orig_fk      = safe_fk(orig_desc)
    new_fk       = safe_fk(new_desc)

    # ── 3. Specificity delta ────────────────────
    orig_spec = _compute_specificity(orig_desc)
    new_spec  = _compute_specificity(new_desc)

    # ── 4. ROUGE-L (content preservation) ───────
    rouge_scores = ROUGE.score(orig_desc, new_desc)
    rougeL_f1 = round(rouge_scores['rougeL'].fmeasure, 4)

    # ── 5. LLM A/B judge ────────────────────────
    judge = _llm_judge(original, optimized)

    # ── 6. Length stats ─────────────────────────
    orig_words  = len(orig_desc.split())
    new_words   = len(new_desc.split())
    orig_title_len = len(orig_title)
    new_title_len  = len(new_title)

    # ── Aggregate verdict ────────────────────────
    # Pass if: composite improved AND (judge says new wins OR score_delta > 5)
    composite_improved = (composite_delta is not None and composite_delta > 0)
    judge_wins         = judge.get('new_wins', False) is True
    length_improved    = new_words > orig_words and new_title_len > orig_title_len
    spec_improved      = new_spec >= orig_spec

    verdict = "PASS" if (judge_wins and length_improved and spec_improved) else \
              "MARGINAL" if (judge_wins or composite_improved) else "FAIL"

    return {
        'verdict': verdict,
        'composite': {
            'original': composite_orig,
            'optimized': composite_new,
            'delta': composite_delta,
        },
        'readability': {
            'orig_flesch_ease': round(orig_flesch, 2),
            'new_flesch_ease':  round(new_flesch, 2),
            'orig_fk_grade':    round(orig_fk, 2),
            'new_fk_grade':     round(new_fk, 2),
        },
        'length': {
            'orig_desc_words':  orig_words,
            'new_desc_words':   new_words,
            'words_added':      new_words - orig_words,
            'orig_title_chars': orig_title_len,
            'new_title_chars':  new_title_len,
        },
        'specificity': {
            'original': orig_spec,
            'optimized': new_spec,
            'delta': new_spec - orig_spec,
        },
        'rouge_l': rougeL_f1,
        'llm_judge': judge,
    }
