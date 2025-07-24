from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import pipeline
import torch

app = FastAPI()

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Use GPU if available
device = 0 if torch.cuda.is_available() else -1

# Load GPT2-XL model (a.k.a GPT-X1)
print("Loading GPT2-XL... please wait")
chatbot = pipeline("text-generation", model="gpt2-xl", device=device)
print("Model loaded.")

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(message: str = Form(...), personality: str = Form("")):
    # Construct the prompt with personality and message
    prompt = f"{personality.strip()}\n\nUser: {message.strip()}\nBot:"
    
    response = chatbot(
        prompt,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=chatbot.tokenizer.eos_token_id
    )[0]["generated_text"]

    # Extract only the bot's reply
    answer = response.split("Bot:")[-1].strip().split("User:")[0].strip()

    return JSONResponse({"response": answer})
