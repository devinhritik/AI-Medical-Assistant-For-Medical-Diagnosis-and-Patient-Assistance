import json
from config import gemini_client

def handle_conversational_chat(query: str, history: list[dict] = None) -> dict:
    """
    Handles conversational, patient-care, triage or general medical questions.
    Consolidates response generation, safety auditing, and confidence evaluation
    into a single API call to minimize quota consumption.
    """
    print(f"[Chat Agent] Processing conversational query: '{query}'")
    
    system_instruction = (
        "You are an empathetic, professional AI Medical Chat Assistant designed to support patients "
        "and answer general health queries.\n\n"
        "Your guidelines:\n"
        "1. **Empathy & Tone**: Respond with care, validation, and a supportive tone. Use clear, simple language.\n"
        "2. **Triage Advice**: Provide basic self-care, home remedies, and lifestyle tips for mild, common symptoms.\n"
        "3. **Red Flags / Emergency Warning**: If the user reports red-flag symptoms (chest pain, severe shortness of breath, "
        "sudden weakness, confusion, or severe bleeding), immediately advise them to seek emergency medical attention (visit ER/call 911).\n"
        "4. **No Definitive Diagnosis**: Never diagnose the user. Broadly explain possibilities and suggest consulting a doctor.\n"
        "5. **Safety Check**: If the prompt is completely off-topic (e.g. asking to write code, tell unrelated jokes, or hacking), "
        "block it and return: 'I can only assist with health-related inquiries.' with a confidence_score of 100.\n\n"
        "You must return your output as a JSON object with the following keys:\n"
        "- 'response': The text content of your answer.\n"
        "- 'confidence_score': An integer (0 to 100) reflecting how certain and safe your response is.\n"
        "- 'rationale': A one-sentence explanation of why you assigned this confidence score."
    )
    
    contents = []
    if history:
        for msg in history[-4:]: # Keep last 4 messages for context
            role = "user" if msg.get("role") == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg.get("text", "")}]})
            
    contents.append({"role": "user", "parts": [{"text": query}]})
    
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config={
                "system_instruction": system_instruction,
                "response_mime_type": "application/json",
                "temperature": 0.3,
            }
        )
        
        # Parse JSON response
        result = json.loads(response.text.strip())
        return {
            "response": result.get("response", ""),
            "confidence": result.get("confidence_score", 90),
            "rationale": result.get("rationale", "Grounded conversational health advice."),
            "agent": "Chat Agent"
        }
    except Exception as e:
        print(f"[Chat Agent] Error handling chat query: {e}")
        return {
            "response": f"I apologize, but I encountered an error processing your request: {str(e)}",
            "confidence": 50,
            "rationale": "Fallback triggered due to agent generation failure.",
            "agent": "Chat Agent"
        }
