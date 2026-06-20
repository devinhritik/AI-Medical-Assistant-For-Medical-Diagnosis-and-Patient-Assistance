import json
from PIL import Image
from config import gemini_client

def analyze_medical_image(image: Image.Image, query: str = "") -> dict:
    """
    Analyzes a medical image (X-ray, MRI, CT, skin lesion) using Gemini Multimodal Vision.
    Consolidates image analysis, safety checks, and confidence rating in a single call.
    """
    print("[Vision Agent] Running medical image analysis...")
    
    system_instruction = (
        "You are an AI Medical Imaging Research Assistant. Your role is to analyze medical scans "
        "(such as X-rays, MRIs, CT scans, or clinical photos) and provide a detailed research and educational report.\n\n"
        "Please follow this structure in your analysis:\n"
        "1. **Scan Type & Anatomy**: Identify the type of imaging and anatomical region.\n"
        "2. **Key Observations**: Describe the visible findings and anomalies clearly.\n"
        "3. **Clinical Research Context**: Discuss common conditions associated with such findings.\n"
        "4. **Recommendations for Verification**: Suggest follow-up tests a clinician would order.\n\n"
        "Safety instructions:\n"
        "- Do not provide a formal diagnosis. State this is for educational purposes.\n"
        "- If the image is NOT a medical scan or clinical photograph, block it and return: "
        "'I can only analyze medical scans or clinical photos.' with a confidence_score of 100.\n\n"
        "You must return your output as a JSON object with the following keys:\n"
        "- 'analysis': The text containing the structured report.\n"
        "- 'confidence_score': An integer (0 to 100) reflecting how certain and safe your response is.\n"
        "- 'rationale': A one-sentence explanation of why you assigned this confidence score."
    )
    
    user_prompt = query if query.strip() else "Analyze this medical scan image and describe the key findings."
    
    try:
        from pydantic import BaseModel
        
        class VisionResponse(BaseModel):
            analysis: str
            confidence_score: int
            rationale: str

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image, user_prompt],
            config={
                "system_instruction": system_instruction,
                "response_mime_type": "application/json",
                "response_schema": VisionResponse,
                "temperature": 0.1,
            }
        )
        
        result = json.loads(response.text.strip())
        return {
            "analysis": result.get("analysis", ""),
            "confidence": result.get("confidence_score", 85),
            "rationale": result.get("rationale", "Multimodal visual diagnostic analysis."),
            "agent": "Vision Agent"
        }
    except Exception as e:
        print(f"[Vision Agent] Error in image analysis: {e}")
        return {
            "analysis": f"Error: Failed to process medical image: {str(e)}",
            "confidence": 40,
            "rationale": "Fallback triggered due to image analysis generation failure.",
            "agent": "Vision Agent"
        }
