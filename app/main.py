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

def _field_to_text(value):
    """Render a JSON value for display: null -> "", lists joined, others str()."""
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(_field_to_text(v) for v in value if v is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _first_balanced_object(text):
    """Return the first balanced {...} substring, honouring JSON strings, or None."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def _parse_json_object(content):
    """Parse a JSON object out of an LLM reply; None if there is none."""
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    candidates = []
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        candidates.append(match.group(0))
    balanced = _first_balanced_object(content)
    if balanced:
        candidates.append(balanced)
    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


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
        parsed = _parse_json_object(content)
        if parsed is None:
            extracted = {"user": "Extraction failed", "company": "",
                         "query": "The model reply did not contain a JSON object", "action": ""}
        else:
            extracted = {k: _field_to_text(parsed.get(k)) for k in extracted}
    except Exception as e:
        extracted = {"user": "Extraction failed", "company": "", "query": str(e), "action": ""}
    return templates.TemplateResponse(request, "index.html", {
        "transcript": transcript,
        "extracted": extracted
    }) 