import json
from config import gemini_client

def route_query(query: str, has_image: bool = False) -> str:
    """
    Analyzes the user request and decides which agent should handle it.
    Routes:
      - 'VISION': For queries with uploaded medical scans/images.
      - 'RAG': For technical, clinical, research, or fact-checking questions that require consulting medical documents.
      - 'CHAT': For conversational greetings, triage assistance, patient care guidance, and general medical dialogue.
    """
    # 1. If an image is provided, we must use the Vision Agent
    if has_image:
        print("[Router Agent] Routing to: VISION (Image detected)")
        return "VISION"
        
    # 2. Use Gemini to classify text-only queries
    system_prompt = (
        "You are an expert medical routing agent. Your job is to classify the user's query into one of two routes:\n"
        "1. 'RAG': Use this route for queries asking for specific factual information, clinical research, medical literature findings, "
        "dosing guidelines, drug interactions, disease details, or information likely stored in medical databases/documents.\n"
        "2. 'CHAT': Use this route for conversational greetings, basic triage ('my throat hurts, what should I do?'), general queries "
        "about patient care guidance, general health advice, empathy statements, or conversational follow-up questions.\n\n"
        "Return your answer as a JSON object with a single key 'route' that must be either 'RAG' or 'CHAT'. Do not explain your choice."
    )
    
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=query,
            config={
                "system_instruction": system_prompt,
                "response_mime_type": "application/json",
                "temperature": 0.0, # High determinism
            }
        )
        # Parse JSON output from model
        result = json.loads(response.text.strip())
        route = result.get("route", "CHAT").upper()
        
        # Guard against unexpected outputs
        if route not in ["RAG", "CHAT"]:
            route = "CHAT"
            
        print(f"[Router Agent] Routing to: {route} (Reason: Classified by LLM)")
        return route
    except Exception as e:
        print(f"[Router Agent] Error routing query: {e}. Defaulting to CHAT.")
        return "CHAT"
