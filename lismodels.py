import os
from dotenv import load_dotenv
from google import genai

load_dotenv(dotenv_path=".env", override=True)

# 🔍 Debug print
api_key = os.getenv("GEMINI_API_KEY")
print("API KEY:", api_key)

if not api_key:
    raise ValueError("API key not found. Check your .env file!")

client = genai.Client(api_key=api_key)

models = client.models.list()

for m in models:
    print(m.name)