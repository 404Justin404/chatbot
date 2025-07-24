from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import pipeline
import torch

app = FastAPI()

# Mount static assets and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Select device (GPU if available)
device = 0 if torch.cuda.is_available() else -1

# Load a smarter local model (Falcon-1B)
print("Loading model... (this may take a few seconds)")
chatbot = pipeline("text-generation", model="tiiuae/falcon-rw-1b", device=device)
print("Model loaded successfully.")

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(message: str = Form(...), personality: str = Form("")):
    # Combine personality and user message into a prompt
    prompt = f"{personality.strip()}\nUser: {message.strip()}\nAI:"
    
    # Generate a response with controlled settings
    response = chatbot(
        prompt,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=chatbot.tokenizer.eos_token_id
    )[0]["generated_text"]

    # Extract and clean the bot's answer
    answer = response.split("AI:")[-1].strip().split("User:")[0].strip()

    return JSONResponse({"response": answer})
