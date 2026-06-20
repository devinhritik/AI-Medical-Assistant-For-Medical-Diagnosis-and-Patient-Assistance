import os
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "medical_knowledge")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment or .env file.")

# Initialize the Gemini GenAI client
# The new google-genai SDK uses genai.Client
try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    gemini_client = None

def get_qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
