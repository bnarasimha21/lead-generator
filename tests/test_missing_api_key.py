"""LOW item: a missing OPENAI_API_KEY used to surface as an unhandled
500 from inside a Depends() call (the openai SDK raises OpenAIError at
client construction time, straight out of get_transcriber/get_extractor,
with no try/except anywhere between it and FastAPI's generic error
handler). That is a stack trace leaking to the caller instead of the
same defined 502 a provider call failure gets.

This exercises the real get_transcriber/get_extractor dependency
functions (no fake_providers override), with LEAD_PROVIDERS switched to
"openai" and no API key configured, and asserts a clean 502 JSON body.
"""

import app.main as main_module
from app import config


def test_missing_api_key_maps_to_502_not_500(client, monkeypatch, sample_wav_bytes):
    monkeypatch.setattr(config, "LEAD_PROVIDERS", "openai")
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # The transcriber/extractor are built lazily and cached on these
    # module globals; force a rebuild attempt for this test regardless
    # of whichever earlier test already populated them.
    monkeypatch.setattr(main_module, "_transcriber", None)
    monkeypatch.setattr(main_module, "_extractor", None)

    response = client.post(
        "/upload-audio",
        files={"file": ("sample.wav", sample_wav_bytes, "audio/wav")},
    )

    assert response.status_code == 502
    assert response.json() == {"error": "The transcription or extraction service failed. Please try again."}
