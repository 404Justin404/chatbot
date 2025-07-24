from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import pipeline
import torch

app = FastAPI()

# Mount static and template folders
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Choose GPU if available
device = 0 if torch.cuda.is_available() else -1

# Load GPT-2 Medium
print("Loading GPT-2 Medium model...")
chatbot = pipeline("text-generation", model="gpt2-medium", device=device)
print("Model loaded.")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(message: str = Form(...), personality: str = Form("")):
    # Construct a prompt that includes personality and the user's message
    prompt = f"{personality.strip()}\n\nUser: {message.strip()}\nBot:"

    # Generate a response
    output = chatbot(
        prompt,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=chatbot.tokenizer.eos_token_id
    )[0]["generated_text"]

    # Extract the bot's part of the response
    answer = output.split("Bot:")[-1].strip().split("User:")[0].strip()

    return JSONResponse({"response": answer})
