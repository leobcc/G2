import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def evaluate_deal_content(original_deal_data, new_content):
    """
    LLM-as-a-judge evaluator comparing the original to the newly optimized content.
    Returns:
       - AI preference score (1 if the rewrite is better, 0 if original is better).
       - Length comparisons (desc_word_count, title_length)
    """
    prompt = f"""
You are a content evaluation expert. Data shows longer, descriptive titles & descriptions increase conversion.
Rate whether the New Content is more likely to convert than the Original Content.

Original Content:
Title: {original_deal_data.get('title', '')}
Description: {original_deal_data.get('description', '')}

New Content:
Title: {new_content.get('improved_title', '')}
Description: {new_content.get('improved_description', '')}

Evaluate the two and reply ONLY with a valid JSON containing:
"winner": "original" or "new"
"reason": "Brief explanation of your choice"
"score": A float between 0 and 100 on how confident you are.
"""

    orig_title_len = len(str(original_deal_data.get('title', '')))
    orig_desc_len = len(str(original_deal_data.get('description', '')).split())
    
    new_title_len = len(str(new_content.get('improved_title', '')))
    new_desc_len = len(str(new_content.get('improved_description', '')).split())

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        eval_result = json.loads(response.choices[0].message.content)
        
        # Calculate the heuristic features from the EDA findings:
        title_length_delta = new_title_len - orig_title_len
        desc_length_delta = new_desc_len - orig_desc_len
        
        return {
            "eval_result": eval_result,
            "heuristic_metrics": {
                "title_length_increased": title_length_delta > 0,
                "desc_word_count_increased": desc_length_delta > 0,
                "title_length_delta": title_length_delta,
                "desc_length_delta": desc_length_delta
            }
        }
    except Exception as e:
        print(f"Error evaluating deal: {e}")
        return {}