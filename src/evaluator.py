import os
import instructor
from groq import AsyncGroq
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt
import textstat

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
if api_key:
    eval_client = instructor.from_groq(AsyncGroq(api_key=api_key), mode=instructor.Mode.JSON)
else:
    eval_client = None

class MultiMetricEvaluation(BaseModel):
    chain_of_reasoning: str = Field(description="Step by step reasoning judging the persuasiveness, clarity, and safety of the new copy vs original.")
    persuasiveness_score: int = Field(ge=1, le=10, description="Rate the pure persuasive and emotional appeal of the deal copy out of 10.")
    clarity_score: int = Field(ge=1, le=10, description="Rate how clear the deal, price, and value are out of 10.")
    hallucination_penalty: int = Field(ge=0, le=10, description="Rate the hallucination risk. 0 means 100% truthful. 10 means extremely fake restrictions or promises were added.")
    judge_verdict: str = Field(pattern='^(NEW|ORIGINAL)$', description="Must output strictly NEW or ORIGINAL depending on which variant will likely convert better.")

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
async def evaluate_deal_content_async(original_deal_data, new_content):
    if not eval_client:
        raise ValueError("GROQ API Key missing.")

    prompt = f"""
You are a Lead Editor and QA Auditor for a multi-million dollar e-commerce platform.
Your job is to strictly evaluate the new generated content vs the original baseline.

Original Deal:
TITLE: {original_deal_data.get('title', '')}
DESC: {original_deal_data.get('description', '')}
STR: {original_deal_data.get('fine_print', '')}

Candidate AI Deal:
TITLE: {new_content.get('improved_title', '')}
DESC: {new_content.get('improved_description', '')}
STR: {new_content.get('improved_fine_print', '')}

Evaluate across:
1. Persuasiveness (Emotional hook, urgency, clear value)
2. Clarity (Are the inclusions / exclusions easily digestible?)
3. Hallucination Risk (Did the new deal invent new rules not present in the original STR? e.g. "Includes 2 glasses of wine" when not mentioned).
Return strict JSON aligning with the required schema.
"""

    orig_title_len = len(str(original_deal_data.get('title', '')))
    orig_desc_len = len(str(original_deal_data.get('description', '')).split())
    new_title_len = len(str(new_content.get('improved_title', '')))
    new_desc_len = len(str(new_content.get('improved_description', '')).split())
    
    response = await eval_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_model=MultiMetricEvaluation,
        temperature=0.0,
        max_tokens=2048
    )

    # Calculate heuristic classical features (NLP Readability)
    orig_readability = textstat.flesch_reading_ease(str(original_deal_data.get('description', '')))
    new_readability = textstat.flesch_reading_ease(str(new_content.get('improved_description', '')))

    return {
        "multi_agent_eval": response.model_dump(),
        "nlp_heuristic_metrics": {
            "title_len_delta": new_title_len - orig_title_len,
            "desc_len_delta": new_desc_len - orig_desc_len,
            "readability_delta": new_readability - orig_readability
        }
    }

def evaluate_deal_content(original_deal_data, new_content):
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        return loop.run_until_complete(evaluate_deal_content_async(original_deal_data, new_content))
    except Exception as e:
        print(f"Evaluation Error: {e}")
        return {"multi_agent_eval": {"judge_verdict": "ERROR", "persuasiveness_score":0}, "nlp_heuristic_metrics": {}}