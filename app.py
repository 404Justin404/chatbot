from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import pipeline, set_seed

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load a small local model
chatbot = pipeline("text-generation", model="distilgpt2")
set_seed(42)

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(message: str = Form(...), personality: str = Form("")):
    prompt = f"{personality.strip()}\nUser: {message.strip()}\nAI:"
    response = chatbot(prompt, max_length=200, num_return_sequences=1)[0]["generated_text"]
    answer = response.split("AI:")[-1].strip().split("User:")[0].strip()
    return JSONResponse({"response": answer})
