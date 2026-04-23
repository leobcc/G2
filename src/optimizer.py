import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def optimize_deal(deal_data):
    """
    Optimizes a deal content based on our EDA findings:
    1. Longer, detailed descriptions convert better (cvr corr: 0.548).
    2. Longer, descriptive titles convert better (cvr corr: 0.356).
    """
    prompt = f'''
You are a top-tier e-commerce copywriter. Your goal is to maximize the conversion rate of a deal.
Data shows that:
1. Deals with longer, more detailed descriptions convert significantly better.
2. Deals with longer, more specific titles convert better.

Here is the current deal:
Title: {deal_data.get('title', '')}
Description: {deal_data.get('description', '')}
Fine Print: {deal_data.get('fine_print', '')}
Price: {deal_data.get('price', '')}
Category: {deal_data.get('category', '')}

Your task:
Rewrite the Title and Description to make them much more appealing, longer, detailed, and clear.
Do NOT invent new fine print restrictions or change the price/offer mechanics.
Return ONLY valid JSON with the exact keys: "improved_title", "improved_description", "improved_fine_print"
'''

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error optimizing deal: {e}")
        return {
            "improved_title": deal_data.get('title', ''),
            "improved_description": deal_data.get('description', ''),
            "improved_fine_print": deal_data.get('fine_print', '')
        }