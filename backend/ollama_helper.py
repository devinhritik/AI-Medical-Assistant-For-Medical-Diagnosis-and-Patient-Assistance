import requests
import json
import base64
import io
from PIL import Image

OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "gemma2:2b"
EMBED_MODEL = "nomic-embed-text"
VISION_MODEL = "moondream"

def get_ollama_embeddings(text: str) -> list[float]:
    """
    Calls the local Ollama embeddings endpoint using the nomic-embed-text model.
    Returns a vector of 768 dimensions.
    """
    url = f"{OLLAMA_URL}/api/embeddings"
    payload = {
        "model": EMBED_MODEL,
        "prompt": text
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        print(f"[Ollama Helper] Embedding error: {e}")
        raise e

def generate_ollama_chat(messages: list[dict], system_instruction: str = None, json_mode: bool = False) -> str:
    """
    Calls the local Ollama chat endpoint using gemma2:2b.
    Allows optional system instructions and JSON response formatting constraint.
    """
    url = f"{OLLAMA_URL}/api/chat"
    
    # Construct message history list
    formatted_messages = []
    
    if system_instruction:
        formatted_messages.append({"role": "system", "content": system_instruction})
        
    for msg in messages:
        formatted_messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content") or msg.get("text") or ""
        })
        
    payload = {
        "model": CHAT_MODEL,
        "messages": formatted_messages,
        "stream": False
    }
    
    if json_mode:
        payload["format"] = "json"
        
    try:
        response = requests.post(url, json=payload, timeout=90)
        response.raise_for_status()
        return response.json()["message"]["content"]
    except Exception as e:
        print(f"[Ollama Helper] Chat generation error: {e}")
        raise e

def pil_to_base64(image: Image.Image) -> str:
    """
    Converts a Pillow image to a base64 encoded string required by Ollama's vision API.
    """
    buffered = io.BytesIO()
    # Save as JPEG for compression and speed
    image.convert("RGB").save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_ollama_vision(image: Image.Image, prompt: str, system_instruction: str = None, json_mode: bool = False) -> str:
    """
    Calls the local Ollama visual model generate endpoint using moondream.
    Sends the visual prompt and base64 encoded image string.
    """
    url = f"{OLLAMA_URL}/api/generate"
    base64_image = pil_to_base64(image)
    
    full_prompt = prompt
    if system_instruction:
        full_prompt = f"{system_instruction}\n\nUser Query: {prompt}"
        
    payload = {
        "model": VISION_MODEL,
        "prompt": full_prompt,
        "images": [base64_image],
        "stream": False
    }
    
    if json_mode:
        payload["format"] = "json"
        
    try:
        response = requests.post(url, json=payload, timeout=90)
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        print(f"[Ollama Helper] Vision generation error: {e}")
        raise e
