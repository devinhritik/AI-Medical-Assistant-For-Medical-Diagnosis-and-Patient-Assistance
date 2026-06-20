from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

print("Listing models that support embedding...")
try:
    # Attempt to list models or see if there is another model
    for model in client.models.list():
        # Check supported methods in model or just print
        print(f"Model Name: {model.name}, DisplayName: {model.display_name}")
except Exception as e:
    print(f"Error listing models: {e}")

try:
    print("\nTesting gemini-embedding-2...")
    response = client.models.embed_content(
        model='gemini-embedding-2',
        contents='test text'
    )
    print(f"Success with gemini-embedding-2! Vector size: {len(response.embeddings[0].values)}")
except Exception as e:
    print(f"Failed with gemini-embedding-2: {e}")

try:
    print("\nTesting gemini-embedding-001...")
    response = client.models.embed_content(
        model='gemini-embedding-001',
        contents='test text'
    )
    print(f"Success with gemini-embedding-001! Vector size: {len(response.embeddings[0].values)}")
except Exception as e:
    print(f"Failed with gemini-embedding-001: {e}")
