import os
from dotenv import load_dotenv
from google import genai

# 🔥 FORCE load .env from current folder
load_dotenv(dotenv_path=".env", override=True)

# 🔍 Debug print
api_key = os.getenv("GEMINI_API_KEY")
print("API KEY:", api_key)

# ❌ Stop if not found
if not api_key:
    raise ValueError("API key not found. Check your .env file!")

# ✅ Create client
client = genai.Client(api_key=api_key)

# ✅ List models
models = client.models.list()

for m in models:
    print(m.name)