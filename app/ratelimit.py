"""In-memory per-IP rate limiting and a daily call budget (issue 5). The
endpoint spends money per call, so both exist to put a ceiling on spend.
Single-process, in-memory: fine for this app's scale, documented as a
limitation (won't survive a restart or coordinate across workers).
"""

import time
from collections import defaultdict, deque
from datetime import date
from typing import Deque, Dict, Optional, Tuple

from app import config


class RateLimiter:
    def __init__(self, limit: int = None, window_seconds: int = None, daily_budget: int = None):
        self.limit = limit if limit is not None else config.RATE_LIMIT_PER_IP
        self.window_seconds = window_seconds if window_seconds is not None else config.RATE_LIMIT_WINDOW_SECONDS
        self.daily_budget = daily_budget if daily_budget is not None else config.DAILY_CALL_BUDGET
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._day: Optional[date] = None
        self._day_count = 0

    def _reset_day_if_needed(self) -> None:
        today = date.today()
        if self._day != today:
            self._day = today
            self._day_count = 0

    def check(self, client_ip: str, now: Optional[float] = None) -> Tuple[bool, Optional[float]]:
        """Returns (allowed, retry_after_seconds). Call record() only after
        the request is allowed and actually proceeds.
        """
        now = now if now is not None else time.monotonic()
        self._reset_day_if_needed()
        if self._day_count >= self.daily_budget:
            return False, float(self.window_seconds)

        hits = self._hits[client_ip]
        while hits and now - hits[0] > self.window_seconds:
            hits.popleft()
        if len(hits) >= self.limit:
            retry_after = self.window_seconds - (now - hits[0])
            return False, max(retry_after, 1.0)
        return True, None

    def record(self, client_ip: str, now: Optional[float] = None) -> None:
        now = now if now is not None else time.monotonic()
        self._reset_day_if_needed()
        self._day_count += 1
        self._hits[client_ip].append(now)


limiter = RateLimiter()
