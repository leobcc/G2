import os
import instructor
from groq import AsyncGroq
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
if api_key:
    client = instructor.from_groq(AsyncGroq(api_key=api_key), mode=instructor.Mode.JSON)
else:
    client = None

class OptimizedDeal(BaseModel):
    conversion_strategy_rationale: str = Field(description="Chain-of-Thought: Detail precisely why you are choosing this wording to boost CVR based on the category and target audience.")
    improved_title: str = Field(description="Extended, explicit, and highly readable title.")
    improved_description: str = Field(description="Comprehensive and highly detailed description leveraging urgency, clear value, and excellent readability.")
    improved_fine_print: str = Field(description="Exact same restrictions and semantic rules as the original fine print, reformatted strictly for readability, absolutely no hallucinations.")

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
async def optimize_deal_async(deal_data: dict) -> OptimizedDeal:
    """
    Advanced async optimizer using Instructor/Pydantic validation, Chain of Thought,
    and NLP-driven conversion rules parsed from the Random Forest.
    """
    if not client:
        raise ValueError("GROQ API Key not found.")

    prompt = f'''
You are an elite conversion rate optimization (CRO) AI copywriter.
Based on our Random Forest feature importance, the primary drivers of Conversion Rate (CVR) are:
1. High Description Word Count (Depth/Detail)
2. Value Semantics & Persuasiveness
3. Good Readability (Flesch-Kincaid)

Original Deal Context:
Target Category: {deal_data.get('category', 'General')}
Subcategory: {deal_data.get('subcategory', '')}
Current Title: {deal_data.get('title', '')}
Current Description: {deal_data.get('description', '')}
Strict Fine Print: {deal_data.get('fine_print', '')}
Selling Price: {deal_data.get('price', '')}
Discount %: {deal_data.get('discount_pct', '')}

Your Task: 
Think step-by-step. First, outline your conversion strategy (Chain-of-Thought). Then rewrite the title and description to maximize conversion (long, clear, persuasive, value-centric). You MUST NOT invent fake legal fine-print or alter the price. You must return VALID structured output against the designated schema.
'''

    response = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a specialized JSON data extractor and world-class copywriter."},
            {"role": "user", "content": prompt}
        ],
        response_model=OptimizedDeal,
        temperature=0.7,
        max_tokens=2048
    )
    
    return response

# Synchronous wrapper for back-compatibility if needed
def optimize_deal(deal_data: dict) -> dict:
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        result = loop.run_until_complete(optimize_deal_async(deal_data))
        return result.model_dump()
    except Exception as e:
        print(f"Error optimizing deal: {e}")
        return {
            "conversion_strategy_rationale": "Error connecting to model. Fallback triggered.",
            "improved_title": deal_data.get('title', ''),
            "improved_description": deal_data.get('description', ''),
            "improved_fine_print": deal_data.get('fine_print', '')
        }