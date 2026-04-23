"""
optimizer.py
Multi-dimensional, category-aware LLM deal optimizer.
Uses chain-of-thought reasoning + few-shot in-context examples drawn from
the top-performing deals in the same category (or globally if category 
has too few deals).
Includes post-generation guardrail validation.
"""

import os
import re
import json
import time
import logging
from typing import Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────────────────────────────────────
# DATA-DERIVED RULES — compiled from rigorous EDA findings.
# These become the LLM's "constitution" for rewriting.
# ─────────────────────────────────────────────────────────────────────────────
EDA_RULES = """
CONVERSION RULES (derived from statistical analysis of 500 deals):

1. DESCRIPTION LENGTH: Top-converting deals have 150–250 words. Under 50 = severe underperformer.
   Action: Expand with specific, factual detail drawn from the merchant's service.

2. STRUCTURAL SECTIONS: Deals with 3 named sections ("What We Offer", "Why You Should Grab This",
   "Good to Know") convert significantly better. Add these headers if absent.

3. SPECIFICITY: High-CVR deals include: concrete durations (e.g. "45-minute"), named equipment,
   credential details, named services. Do NOT use vague language like "premium service" or
   "amazing experience" without backing it up with facts.

4. SOCIAL PROOF: Quantify trust signals. If any are implied in the original, surface them explicitly.
   Examples: "4.7-star rating", "over 2,000 clients served", "background-checked technicians".

5. TITLE QUALITY: Optimal titles are 60–100 characters and include: service name, merchant name,
   a specific hook (outcome, key feature, or offer detail). Avoid vague filler words.

6. OPTION NAMES: Generic names like "Option 1" severely hurt CVR. Rename to benefit-first labels
   e.g. "Single Session", "Family Package", "Premium Bundle".

7. FINE PRINT: Do NOT add, remove, or change any restriction. Only reformat for clarity if needed.

8. ANTI-HALLUCINATION: Never invent facts (prices, percentages, dates, credentials) that are not
   implicit or stated in the original content. If unsure, omit rather than fabricate.
"""


def _build_system_prompt() -> str:
    return f"""You are a specialist e-commerce conversion copywriter for Groupon.
Your rewrites are grounded in data, not guesswork.

{EDA_RULES}

You will receive a deal and must:
1. First, in "reasoning": briefly identify the 2–3 biggest conversion weaknesses.
2. Then produce improved versions of: title, description, option_names.
3. Return fine_print UNCHANGED (copy it verbatim).

Output must be valid JSON with these EXACT keys:
{{
  "reasoning": "...",
  "improved_title": "...",
  "improved_description": "...",
  "improved_option_names": "...",
  "improved_fine_print": "..."
}}
"""


def _get_fewshot_examples(df_reference, category: str, n_examples: int = 2) -> str:
    """
    Pull n top-CVR deals from the same category as few-shot positive examples.
    Falls back to global top deals if category has too few.
    """
    if df_reference is None or df_reference.empty:
        return ""

    cat_df = df_reference[df_reference['category'] == category].copy()
    if len(cat_df) < n_examples:
        cat_df = df_reference.copy()

    top = cat_df.nlargest(n_examples, 'cvr')
    examples = []
    for _, row in top.iterrows():
        cvr_pct = f"{row['cvr'] * 100:.1f}%"
        ex = (
            f"EXAMPLE (CVR: {cvr_pct} — top performer in this category):\n"
            f"  Title: {row['title']}\n"
            f"  Description (first 200 chars): {str(row['description'])[:200]}...\n"
        )
        examples.append(ex)
    return "\nHere are high-performing deals in the same category for reference:\n" + "\n".join(examples)


def _build_user_prompt(deal: dict, fewshot_examples: str) -> str:
    return f"""{fewshot_examples}

DEAL TO REWRITE:
Deal ID: {deal.get('deal_id', 'N/A')}
Category: {deal.get('category', '')} / {deal.get('subcategory', '')}
Geo: {deal.get('geo', '')}
Merchant: {deal.get('merchant_name', '')}
Title: {deal.get('title', '')}
Description: {deal.get('description', '')}
Fine Print (DO NOT CHANGE): {deal.get('fine_print', '')}
Current CVR: {deal.get('cvr', 'unknown')}
Price: ${deal.get('price', '')}
Options: {deal.get('option_names', '')}

Rewrite this deal following all conversion rules. Return ONLY valid JSON.
"""


def _validate_output(original: dict, output: dict) -> tuple:
    """
    Post-generation guardrail validation.
    Returns (is_valid: bool, violations: list[str])
    """
    violations = []

    # 1. Fine print must be unchanged
    orig_fp = str(original.get('fine_print', '')).strip()
    new_fp  = str(output.get('improved_fine_print', '')).strip()
    if orig_fp and new_fp and new_fp != orig_fp:
        # Allow minor whitespace differences — check core content
        orig_items = set(i.strip().lower() for i in orig_fp.split(';'))
        new_items  = set(i.strip().lower() for i in new_fp.split(';'))
        if not orig_items.issubset(new_items):
            violations.append(f"Fine print restrictions removed or altered")

    # 2. Price must NOT appear in description (we can't guarantee accuracy)
    orig_price = str(original.get('price', ''))
    new_desc = str(output.get('improved_description', ''))
    if orig_price and f'${orig_price}' in new_desc and f'${orig_price}' not in str(original.get('description', '')):
        violations.append("Price injected into description — not in original")

    # 3. Title/description must not be empty
    if not output.get('improved_title', '').strip():
        violations.append("Empty improved_title")
    if len(output.get('improved_description', '').split()) < 30:
        violations.append("Description too short post-rewrite — likely truncated")

    # 4. Required JSON keys present
    for key in ['improved_title', 'improved_description', 'improved_fine_print', 'improved_option_names']:
        if key not in output:
            violations.append(f"Missing key: {key}")

    return (len(violations) == 0), violations


def optimize_deal(
    deal: dict,
    df_reference=None,
    max_retries: int = 3,
) -> dict:
    """
    Optimize a single deal using the LLM with few-shot category context.
    Validates the output and retries on failure or guardrail violations.
    """
    category = str(deal.get('category', ''))
    fewshot = _get_fewshot_examples(df_reference, category)

    system_prompt = _build_system_prompt()
    user_prompt   = _build_user_prompt(deal, fewshot)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                model=MODEL,
                temperature=0.5,
                response_format={"type": "json_object"},
                max_tokens=1500,
            )
            output = json.loads(response.choices[0].message.content)
            is_valid, violations = _validate_output(deal, output)

            if is_valid:
                output['_meta'] = {
                    'attempt': attempt,
                    'guardrail_passed': True,
                    'model': MODEL,
                }
                return output
            else:
                logger.warning(f"Attempt {attempt}: guardrail violations — {violations}")
                if attempt < max_retries:
                    user_prompt += f"\n\nPREVIOUS ATTEMPT FAILED VALIDATION:\n" + "\n".join(violations) + "\nPlease fix these issues."
                    time.sleep(1)

        except json.JSONDecodeError as e:
            logger.warning(f"Attempt {attempt}: JSON decode error — {e}")
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"Attempt {attempt}: API error — {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    # Final fallback: return original content marked as failed
    logger.error(f"All {max_retries} attempts failed for deal {deal.get('deal_id', '?')}. Returning original.")
    return {
        'reasoning': 'Optimization failed after all retries.',
        'improved_title': deal.get('title', ''),
        'improved_description': deal.get('description', ''),
        'improved_option_names': deal.get('option_names', ''),
        'improved_fine_print': deal.get('fine_print', ''),
        '_meta': {'attempt': max_retries, 'guardrail_passed': False, 'model': MODEL},
    }
