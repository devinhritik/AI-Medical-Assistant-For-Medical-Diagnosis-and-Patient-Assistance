from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

try:
    print("Testing gemini-2.5-flash...")
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents='Hello, respond with test.',
    )
    print(f"Success with gemini-2.5-flash: {response.text}")
except Exception as e:
    print(f"Failed with gemini-2.5-flash: {e}")

try:
    print("Testing gemini-1.5-flash...")
    response = client.models.generate_content(
        model='gemini-1.5-flash',
        contents='Hello, respond with test.',
    )
    print(f"Success with gemini-1.5-flash: {response.text}")
except Exception as e:
    print(f"Failed with gemini-1.5-flash: {e}")
