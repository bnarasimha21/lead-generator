"""Verification-review finding 9: app.main's shared
@app.exception_handler(HTTPException) handler used to call
_error_response(exc.status_code, exc.detail) and nothing else, which
silently dropped exc.headers whenever an HTTPException carried any. No
current call site inside this app sets headers on an HTTPException it
raises itself (the 429/Retry-After path is handled entirely inside
RateLimitAndBodySizeMiddleware's own send(), never reaching this
handler), so this was latent, not an observed failure; it is fixed
regardless since the change is one line and fully contained.

This test adds a throwaway route to the real app (removed at the end of
the test) that raises an HTTPException with a custom header, and confirms
the header survives through the real handler. No real OpenAI call is
involved.
"""

from fastapi import HTTPException

from app.main import app


def test_http_exception_with_custom_headers_keeps_them_through_the_shared_handler(client):
    route_path = "/__test_only_header_check"

    async def _raise_with_header():
        raise HTTPException(status_code=409, detail="conflict", headers={"X-Test-Reason": "because-test"})

    app.add_api_route(route_path, _raise_with_header, methods=["GET"])
    try:
        response = client.get(route_path)
    finally:
        app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != route_path]

    assert response.status_code == 409
    assert response.json() == {"error": "conflict"}
    assert response.headers.get("x-test-reason") == "because-test"


def test_http_exception_without_headers_still_works_as_before(client):
    """Regression guard: exc.headers defaults to None on a plain
    HTTPException(status_code, detail), and passing headers=None through
    to _error_response/JSONResponse must not change any existing
    response (e.g. the 502 provider-construction-failure path)."""
    route_path = "/__test_only_header_check_none"

    async def _raise_without_header():
        raise HTTPException(status_code=502, detail="upstream failed")

    app.add_api_route(route_path, _raise_without_header, methods=["GET"])
    try:
        response = client.get(route_path)
    finally:
        app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != route_path]

    assert response.status_code == 502
    assert response.json() == {"error": "upstream failed"}
