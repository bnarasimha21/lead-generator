import json
import logging
import os

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAIError

from app import config
from app.providers import ProviderError, build_extractor, build_transcriber
from app.request_guard import BODY_TOO_LARGE_DETAIL, RECEIVED_AT_SCOPE_KEY, RateLimitAndBodySizeMiddleware
from app.schema import Lead
from app.timing import Timing
from app.uploads import UploadTooLarge, read_upload_limited, sniff_audio_format
from app.verify import verify_lead

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lead_generator")

app = FastAPI()
# Rate limiting / daily budget and the body-size cap are enforced here, at
# the ASGI layer, before Starlette ever reads the request body (issue 5 /
# findings 5, 6): see app/request_guard.py for why this has to happen
# before the route handler, not inside it.
app.add_middleware(RateLimitAndBodySizeMiddleware)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

_transcriber = None
_extractor = None

_FIXED_ERRORS = {
    400: "The uploaded file could not be read as audio.",
    413: BODY_TOO_LARGE_DETAIL,
    502: "The transcription or extraction service failed. Please try again.",
}


def _error_response(status_code: int, detail: str = None, headers: dict = None) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": detail or _FIXED_ERRORS[status_code]}, headers=headers)


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    # Every other error path in this app replies with {"error": ...}
    # (see _error_response above). Without this handler, an
    # HTTPException raised from inside a Depends() (as get_transcriber/
    # get_extractor do below) would instead get FastAPI's own default
    # {"detail": ...} shape, a different contract for the same kind of
    # failure depending on which line happened to raise it.
    #
    # Verification-review finding 9: this used to call
    # _error_response(exc.status_code, exc.detail) and nothing else,
    # silently dropping exc.headers whenever an HTTPException set any.
    # No current call site inside this app actually sets headers on an
    # HTTPException it raises (the 429/Retry-After path is handled
    # entirely inside RateLimitAndBodySizeMiddleware's own send(), never
    # reaching this handler at all), so this was latent rather than an
    # observed failure. Still fixed, since it is a one-line, contained
    # change: any future HTTPException(..., headers={...}) raised from
    # inside this app (a Depends(), a route body, etc.) now keeps its
    # headers intact through this handler rather than silently losing
    # them.
    return _error_response(exc.status_code, exc.detail, headers=exc.headers)


def get_transcriber():
    global _transcriber
    if _transcriber is None:
        try:
            _transcriber = build_transcriber()
        except OpenAIError:
            # Most commonly a missing OPENAI_API_KEY: without this, the
            # SDK's own configuration error propagates out of this
            # dependency unhandled, producing an opaque 500 instead of
            # the same defined 502 a provider call failure gets.
            logger.exception("failed to construct transcriber (likely missing OPENAI_API_KEY)")
            raise HTTPException(status_code=502, detail=_FIXED_ERRORS[502])
    return _transcriber


def get_extractor():
    global _extractor
    if _extractor is None:
        try:
            _extractor = build_extractor()
        except OpenAIError:
            logger.exception("failed to construct extractor (likely missing OPENAI_API_KEY)")
            raise HTTPException(status_code=502, detail=_FIXED_ERRORS[502])
    return _extractor


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload-audio")
async def upload_audio(
    request: Request,
    file: UploadFile = File(...),
    transcriber=Depends(get_transcriber),
    extractor=Depends(get_extractor),
):
    received_at = request.scope.get(RECEIVED_AT_SCOPE_KEY)
    timing = Timing(start=received_at)
    timing.mark("received")

    # Rate limiting, the daily budget, and the raw body-size cap are all
    # already enforced by RateLimitAndBodySizeMiddleware before this
    # handler ever runs (findings 5, 6): by the time we get here, this
    # request's slot is already reserved.
    try:
        audio_bytes = await read_upload_limited(file)
    except UploadTooLarge:
        logger.info(json.dumps({"event": "upload_rejected", "reason": "too_large"}))
        return _error_response(413)

    sniffed = sniff_audio_format(audio_bytes[:64])
    if sniffed is None:
        logger.info(json.dumps({"event": "upload_rejected", "reason": "unrecognised_format", "bytes": len(audio_bytes)}))
        return _error_response(400)

    timing.mark("audio_validated")

    try:
        transcription = await transcriber.transcribe(audio_bytes, f"upload.{sniffed.extension}", sniffed.content_type)
    except ProviderError:
        logger.info(json.dumps({"event": "transcription_failed", "bytes": len(audio_bytes)}))
        return _error_response(502)
    timing.mark("transcribed")

    transcript = transcription.text or ""
    word_count = len(transcript.split())

    usage = {"audio_seconds": transcription.audio_seconds, "input_tokens": 0, "output_tokens": 0}
    message = None
    rejected = []

    if word_count < config.MIN_WORDS_FOR_EXTRACTION:
        lead = Lead.empty()
        message = "Transcript too short to extract a lead."
        timing.mark("extracted")
        timing.mark("verified")
    else:
        try:
            extraction = await extractor.extract(transcript)
        except ProviderError:
            logger.info(json.dumps({"event": "extraction_failed", "words": word_count}))
            return _error_response(502)
        timing.mark("extracted")

        usage["input_tokens"] = extraction.usage.input_tokens
        usage["output_tokens"] = extraction.usage.output_tokens

        if extraction.status == "ok":
            lead, rejected = verify_lead(extraction.lead, transcript)
        else:
            lead = Lead.empty()
            message = extraction.message
        timing.mark("verified")

    estimated_cost = _estimate_cost(usage)
    timing.mark("responded")

    body = {
        "transcript": transcript,
        "lead": lead.to_display_dict(rejected),
        "rejected": rejected,
        "message": message,
        "timing": timing.as_dict(),
        "usage": {**usage, "estimated_cost": estimated_cost},
    }
    logger.info(json.dumps({
        "event": "upload_complete",
        "bytes": len(audio_bytes),
        "words": word_count,
        "rejected_count": len(rejected),
        "timing_ms": timing.as_dict(),
    }))
    return JSONResponse(content=body, headers={"Server-Timing": timing.server_timing_header()})


def _estimate_cost(usage: dict):
    if config.PRICE_PER_INPUT_TOKEN is None or config.PRICE_PER_OUTPUT_TOKEN is None:
        return None
    cost = usage["input_tokens"] * config.PRICE_PER_INPUT_TOKEN + usage["output_tokens"] * config.PRICE_PER_OUTPUT_TOKEN
    if usage.get("audio_seconds") and config.PRICE_PER_AUDIO_SECOND is not None:
        cost += usage["audio_seconds"] * config.PRICE_PER_AUDIO_SECOND
    return round(cost, 6)
