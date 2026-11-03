"""Per-request timing marks (issue 5).

Every successful response carries all six marks, in order, as milliseconds
since the request started: received, audio_validated, transcribed,
extracted, verified, responded. They go into the JSON response, a
Server-Timing header, and one structured log line per request that never
contains transcript text or audio bytes, only sizes and counts.
"""

import time
from collections import OrderedDict

MARK_NAMES = (
    "received",
    "audio_validated",
    "transcribed",
    "extracted",
    "verified",
    "responded",
)


class Timing:
    def __init__(self, start: float = None):
        # `start`, when given, is a time.monotonic() value captured earlier
        # than this object's construction (e.g. by the ASGI middleware, as
        # soon as the request arrived), so the "received" mark reflects
        # when the request actually came in rather than when this route
        # handler's body began running, after FastAPI has already parsed
        # the multipart upload to build the UploadFile.
        self._start = start if start is not None else time.monotonic()
        self.marks = OrderedDict()

    def mark(self, name: str) -> None:
        self.marks[name] = round((time.monotonic() - self._start) * 1000, 2)

    def as_dict(self) -> dict:
        return dict(self.marks)

    def server_timing_header(self) -> str:
        parts = []
        prev = 0.0
        for name in self.marks:
            dur = max(self.marks[name] - prev, 0.0)
            parts.append(f'{name};dur={dur:.2f}')
            prev = self.marks[name]
        return ", ".join(parts)

    def all_marks_present_in_order(self) -> bool:
        names = list(self.marks.keys())
        if names != [n for n in MARK_NAMES if n in names]:
            return False
        values = list(self.marks.values())
        return values == sorted(values)
