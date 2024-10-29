from fastapi import FastAPI, Request
from transformers import MarianMTModel, MarianTokenizer
import torch
import asyncio

# Initialize FastAPI application
app = FastAPI()

# Load MarianMT model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-fr"  # English to French example
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

@app.on_event("startup")
async def load_model():
    """
    If you have multiple language pairs, load all necessary models
    or do heavy lifting during app startup for efficiency.
    """
    print("Model loaded and ready.")

@app.get("/")
async def hello_world():
    return "Hello World"

@app.post("/translate/")
async def translate_text(request: Request):
    """
    Endpoint for translating text asynchronously.
    """
    data = await request.json()
    source_text = data.get("text")
    
    if not source_text:
        return {"error": "No text provided"}
    
    target_language = data.get("target_language", "fr")  # default is French

    # Asynchronous translation process
    translated_text = await asyncio.to_thread(process_translation, source_text, target_language)

    return {
        "original_text": source_text,
        "translated_text": translated_text,
        "target_language": target_language
    }

def process_translation(source_text: str, target_language: str):
    """
    Tokenizes the input text, performs translation using MarianMT, and decodes the output.
    """
    # Tokenize input text
    inputs = tokenizer(source_text, return_tensors="pt", padding=True)

    # Perform translation (on CPU or GPU)
    with torch.no_grad():
        translated_tokens = model.generate(**inputs)

    # Decode the translated tokens
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

if __name__ == "__main__":
    # Uvicorn will run the FastAPI application on a server capable of handling many requests
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
