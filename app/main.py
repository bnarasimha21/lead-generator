from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import openai
import uuid
import json
import re

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.post("/upload-audio", response_class=HTMLResponse)
async def upload_audio(request: Request, file: UploadFile = File(...)):
    # Unique name per request so concurrent uploads do not overwrite each other
    temp_path = os.path.join(BASE_DIR, f"temp_audio_{uuid.uuid4().hex}.wav")
    transcript = ""
    transcribed = False
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        # Transcribe using OpenAI Whisper
        with open(temp_path, "rb") as audio_file:
            transcript_response = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            transcript = transcript_response.text
            transcribed = True
    except Exception as e:
        transcript = f"Transcription failed: {e}"
    finally:
        # Remove temp file even if transcription failed
        if os.path.exists(temp_path):
            os.remove(temp_path)
    # Extract info with LLM (now also extract company)
    extraction_prompt = f"""
    Given the following user requirement, extract:
    1. Who is the user?
    2. What is the company (if mentioned)?
    3. What is the query/requirement?
    4. What action needs to be taken?
    
    Requirement: '''{transcript}'''
    
    Respond in JSON with keys: user, company, query, action.
    """
    extracted = {"user": "", "company": "", "query": "", "action": ""}
    if not transcribed:
        # Do not send the error text to the LLM as if it were the user's requirement
        return templates.TemplateResponse(request, "index.html", {
            "transcript": transcript,
            "extracted": extracted
        })
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": extraction_prompt}]
        )
        content = response.choices[0].message.content or ""
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # Model wrapped the JSON in prose or a code fence: take the outermost {...}
            match = re.search(r'\{.*\}', content, re.DOTALL)
            parsed = json.loads(match.group(0)) if match else None
        if isinstance(parsed, dict):
            extracted = {k: str(parsed.get(k, "")) for k in extracted}
    except Exception as e:
        extracted = {"user": "Extraction failed", "company": "", "query": str(e), "action": ""}
    return templates.TemplateResponse(request, "index.html", {
        "transcript": transcript,
        "extracted": extracted
    }) 