"""ASGI-level rate limiting and body-size enforcement for /upload-audio
(adversarial-review findings 5, 6, 8).

Finding 5: the old flow called limiter.check(client_ip) in the route
handler, then `await read_upload_limited(file)` (which, for a large
upload, awaits multiple times while FastAPI/Starlette read the multipart
body), and only called limiter.record(client_ip) afterwards. Any `await`
between check and record is a window where concurrently-scheduled requests
from the same IP can all see "allowed" before any of them records, letting
more requests through than RATE_LIMIT_PER_IP / DAILY_CALL_BUDGET allow.

Fixed by reserving the slot (RateLimiter.reserve, a single call with no
`await` inside it) as the very first thing this middleware does for the
route, before Starlette/FastAPI ever touches the request body.

Finding 6: the old size check only rejected an oversized upload after
app/uploads.py had already read (chunked, but still read) the whole body.
This middleware rejects based on the Content-Length header up front, and
also counts bytes as they stream in for a request that lies about (or
omits) Content-Length, aborting before the rest of an oversized body is
read into memory.

Finding 6 also requires that an oversized/wrong-format rejection still
counts against the budget, not dodge it: since the reservation above
happens before the size check, a 413 here already consumed a slot, exactly
like a 400 from the route handler's own format sniff does.

Finding 8 / verification-review finding 2: this middleware derives the
client IP itself, by reading the raw `X-Forwarded-For` header off the
ASGI scope, rather than trusting `scope["client"]` or uvicorn's own
--proxy-headers/--forwarded-allow-ips rewriting.

An earlier fix ran uvicorn with --proxy-headers --forwarded-allow-ips='*',
which rewrites scope["client"] from the LEFT-MOST entry of
X-Forwarded-For. That entry is fully client-controlled: an attacker can
send a fresh fake value on every request and get a brand-new per-IP
rate-limit bucket each time, draining the whole DAILY_CALL_BUDGET while
never once hitting their own per-IP limit. That is worse than the
original shared-bucket bug, so Procfile and the Dockerfile's CMD no
longer pass those flags at all.

This app is only ever reachable through its platform's own reverse proxy
(Railway's edge), never directly from the public internet (see the
README note on the assumption this relies on). That single trusted hop
appends its own observed peer address as the LAST entry of
X-Forwarded-For; it does not strip whatever the client put in front of
it. So `_client_ip` below reads the header itself and takes the
RIGHT-MOST entry, trusting exactly the one hop that can reach this
process at all and ignoring every entry the client could have prepended.
If no X-Forwarded-For header is present (e.g. local dev with no proxy in
front), it falls back to the raw ASGI peer address.
"""

import json
import time

from fastapi import HTTPException

from app import config
from app import ratelimit as ratelimit_module

UPLOAD_PATH = "/upload-audio"

# Key stashed directly on the ASGI scope dict (shared by reference with
# the Request object the route handler sees) so the "received" timing
# mark can reflect when this middleware first saw the request, before
# Starlette/FastAPI parse the multipart body to build the UploadFile.
# Marking "received" from inside the route handler, as this app used to
# do, happens after that parsing, so it silently excluded multipart
# parsing time from the "received" to "audio_validated" span.
RECEIVED_AT_SCOPE_KEY = "lead_generator_received_at"

# The exact body sent by _send_json(send, 413, ...) for the Content-Length
# based early rejection below; the streaming-body rejection (_BodyTooLarge)
# must produce the identical {"error": ...} shape and status, whichever
# code path catches it.
BODY_TOO_LARGE_DETAIL = "The uploaded file is too large."


class _BodyTooLarge(HTTPException):
    """Raised from guarded_receive() when a streaming (chunked, or
    Content-Length-lying) upload exceeds MAX_UPLOAD_BYTES.

    Verification-review finding 4: this used to be a plain Exception.
    When it was raised from inside FastAPI's own multipart body parsing
    (fastapi/routing.py's request-body handling), FastAPI's
    `except HTTPException: raise` / `except Exception as e: raise
    HTTPException(400, "There was an error parsing the body") from e`
    only re-raises an exception unchanged if it already IS an
    HTTPException; a plain Exception instead got rewritten into a
    generic 400 "There was an error parsing the body", and Starlette's
    own exception-handling middleware (always present inside the FastAPI
    app, underneath this middleware) sent that 400 response before this
    middleware's own `except _BodyTooLarge:` below ever saw anything
    propagate - so the client got a 400, never the intended 413.

    Making _BodyTooLarge itself an HTTPException(413, ...) means
    FastAPI's `except HTTPException: raise` branch matches first and
    re-raises it completely unchanged, so it reaches the app's own
    registered `@app.exception_handler(HTTPException)` (app/main.py) and
    renders as the correct 413 with the correct {"error": ...} body,
    without this middleware needing to intercept it at all for that
    code path. The `except _BodyTooLarge:` handling further down is kept
    for the other code path exercised by this module's own tests: a
    bare ASGI app (not FastAPI) reading the body directly, where there is
    no FastAPI/Starlette exception-handling layer to catch it first.
    """

    def __init__(self):
        super().__init__(status_code=413, detail=BODY_TOO_LARGE_DETAIL)


def _client_ip(scope) -> str:
    """The IP this request is rate-limited and budgeted under: the
    RIGHT-MOST entry of X-Forwarded-For (appended by the single trusted
    reverse-proxy hop in front of this process), never the left-most
    client-supplied entry. Falls back to the raw ASGI peer address when no
    X-Forwarded-For header is present at all.
    """
    xff_values = [value for name, value in scope.get("headers", []) if name == b"x-forwarded-for"]
    if xff_values:
        raw = b", ".join(xff_values).decode("latin1")
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if parts:
            return parts[-1]
    client = scope.get("client")
    return client[0] if client else "unknown"


def _content_length(scope):
    for name, value in scope.get("headers", []):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


async def _send_json(send, status: int, body: dict, extra_headers=None) -> None:
    payload = json.dumps(body).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())]
    if extra_headers:
        headers.extend(extra_headers)
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": payload})


class RateLimitAndBodySizeMiddleware:
    """Pure ASGI middleware. Only touches POST /upload-audio; everything
    else passes straight through unmodified.
    """

    def __init__(self, app, upload_path: str = UPLOAD_PATH):
        self.app = app
        self.upload_path = upload_path

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") != self.upload_path or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        scope[RECEIVED_AT_SCOPE_KEY] = time.monotonic()

        client_ip = _client_ip(scope)

        # Reserve atomically, before anything about the body is read: this
        # is the fix for finding 5, and it runs even earlier than the old
        # in-handler check did, so there is no window at all for the body
        # read to create a race.
        allowed, retry_after = ratelimit_module.limiter.reserve(client_ip)
        if not allowed:
            await _send_json(
                send,
                429,
                {"error": "Rate limit exceeded. Please try again shortly."},
                extra_headers=[(b"retry-after", str(int(retry_after)).encode())],
            )
            return

        max_bytes = config.MAX_UPLOAD_BYTES

        content_length = _content_length(scope)
        if content_length is not None and content_length > max_bytes:
            # Already reserved above: a declared-oversized request still
            # consumes its slot (finding 6), it just never reaches the
            # route handler or buffers a single byte of body.
            await _send_json(send, 413, {"error": BODY_TOO_LARGE_DETAIL})
            return

        seen = 0

        async def guarded_receive():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body") or b"")
                if seen > max_bytes:
                    raise _BodyTooLarge()
            return message

        response_started = False

        async def guarded_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, guarded_receive, guarded_send)
        except _BodyTooLarge:
            if response_started:
                # The route handler already started a response through some
                # other path; nothing sane to do but propagate.
                raise
            await _send_json(send, 413, {"error": BODY_TOO_LARGE_DETAIL})
