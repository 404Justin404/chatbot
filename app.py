from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import requests
import os

app = FastAPI()

# Store your OpenRouter API key as an environment variable for security
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "sk-or-your-key-here")

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
async def chat(input: ChatInput):
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openchat/openchat-3.5",  # a solid, fast free model
                "messages": [
                    {"role": "system", "content": "You are a helpful chatbot."},
                    {"role": "user", "content": input.message}
                ]
            },
            timeout=30
        )

        response.raise_for_status()
        data = response.json()
        reply = data["choices"][0]["message"]["content"].strip()
        return {"reply": reply}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
