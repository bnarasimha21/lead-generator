"""In-memory per-IP rate limiting and a daily call budget (issue 5). The
endpoint spends money per call, so both exist to put a ceiling on spend.
Single-process, in-memory: fine for this app's scale, documented as a
limitation (won't survive a restart or coordinate across workers).

Verification-review finding 8: the per-IP hit table (self._hits) is
pruned of any IP whose hits have all aged out of the window (see
RateLimiter._trim), rather than keeping a permanent entry for every
distinct IP this process has ever observed, which would otherwise grow
without bound for the lifetime of a long-running process. Per-IP lazy
trimming alone is not enough: an IP that is only ever seen once is never
looked up again, so its entry would never get the chance to be trimmed.
RateLimiter._sweep_if_due adds a full-table sweep, triggered once the
table grows past a size threshold, that walks every entry and drops the
ones that are stale regardless of whether that IP is ever checked again.
"""

import time
from collections import deque
from datetime import date, datetime, timedelta
from typing import Deque, Dict, Optional, Tuple

from app import config


def _seconds_until_local_midnight(now_dt: Optional[datetime] = None) -> float:
    """How long until the daily call budget actually resets (local
    midnight, matching _reset_day_if_needed's use of date.today()).

    The daily-budget-exceeded case used to return a fixed
    window_seconds (60 by default) as Retry-After, which is nowhere
    near right: the budget doesn't reset in another minute, it resets
    at midnight. A client that waited the advertised 60 seconds and
    retried would just get rejected again, every minute, until actual
    midnight.
    """
    now_dt = now_dt if now_dt is not None else datetime.now()
    midnight = datetime.combine(now_dt.date() + timedelta(days=1), datetime.min.time())
    return max((midnight - now_dt).total_seconds(), 1.0)


DEFAULT_SWEEP_THRESHOLD = 10_000


class RateLimiter:
    def __init__(
        self,
        limit: int = None,
        window_seconds: int = None,
        daily_budget: int = None,
        sweep_threshold: int = DEFAULT_SWEEP_THRESHOLD,
    ):
        self.limit = limit if limit is not None else config.RATE_LIMIT_PER_IP
        self.window_seconds = window_seconds if window_seconds is not None else config.RATE_LIMIT_WINDOW_SECONDS
        self.daily_budget = daily_budget if daily_budget is not None else config.DAILY_CALL_BUDGET
        self.sweep_threshold = sweep_threshold
        self._hits: Dict[str, Deque[float]] = {}
        self._day: Optional[date] = None
        self._day_count = 0

    def _reset_day_if_needed(self) -> None:
        today = date.today()
        if self._day != today:
            self._day = today
            self._day_count = 0

    def _sweep_if_due(self, now: float) -> None:
        """Full-table sweep, run once self._hits holds more than
        self.sweep_threshold distinct IPs: drops every entry whose hits
        have all aged out of the window, regardless of whether that IP's
        own _trim is ever invoked again. Without this, a flood of one-off
        IPs that each make exactly one request (and are never looked up
        again) would grow this table without bound, since _trim's lazy
        per-IP eviction only runs when that same IP is checked again.
        """
        if len(self._hits) <= self.sweep_threshold:
            return
        stale_ips = [ip for ip, hits in self._hits.items() if not hits or now - hits[-1] > self.window_seconds]
        for ip in stale_ips:
            del self._hits[ip]

    def _trim(self, client_ip: str, now: float) -> Deque[float]:
        """Returns client_ip's hits, trimmed of anything outside the
        current window, WITHOUT leaving a dead entry in self._hits.

        Finding 8: this used to be a plain `self._hits[client_ip]` read
        against a defaultdict(deque). That autocreates an entry for every
        distinct IP this process has ever seen on its very first request,
        whether or not it was ever rate limited, and never removes it even
        once every hit in that deque has aged out of the window: every
        unique client IP a long-running process ever observes becomes a
        permanent entry, which is unbounded growth for an in-memory,
        single-process table (the daily call budget caps total requests,
        not distinct IPs).

        This reads with plain dict.get (no autocreation), trims expired
        entries, and deletes the dict entry outright once nothing remains,
        so the table only ever holds IPs with at least one hit still
        inside the current window.
        """
        hits = self._hits.get(client_ip)
        if hits is None:
            return deque()
        while hits and now - hits[0] > self.window_seconds:
            hits.popleft()
        if not hits:
            del self._hits[client_ip]
        return hits

    def check(self, client_ip: str, now: Optional[float] = None) -> Tuple[bool, Optional[float]]:
        """Returns (allowed, retry_after_seconds). Call record() only after
        the request is allowed and actually proceeds.

        Deprecated for request handling: check() then record() leaves a gap
        for an await to land in between (see reserve()), which is exactly
        the race adversarial review finding 5 flagged: concurrent requests
        could all see "allowed" before any of them recorded. Kept only for
        any caller that genuinely wants to peek without committing.
        """
        now = now if now is not None else time.monotonic()
        self._reset_day_if_needed()
        self._sweep_if_due(now)
        if self._day_count >= self.daily_budget:
            return False, _seconds_until_local_midnight()

        hits = self._trim(client_ip, now)
        if len(hits) >= self.limit:
            retry_after = self.window_seconds - (now - hits[0])
            return False, max(retry_after, 1.0)
        return True, None

    def record(self, client_ip: str, now: Optional[float] = None) -> None:
        now = now if now is not None else time.monotonic()
        self._reset_day_if_needed()
        self._day_count += 1
        self._hits.setdefault(client_ip, deque()).append(now)

    def reserve(self, client_ip: str, now: Optional[float] = None) -> Tuple[bool, Optional[float]]:
        """Atomically check-and-record in one call with no `await` in
        between (issue 5 / finding 5): the request handler calls this
        exactly once, before reading anything from the request body. Since
        this method contains no `await`, the event loop cannot switch to a
        different request's coroutine partway through, so concurrent
        requests from the same IP cannot all observe "allowed" before any
        of them books the slot. Returns (allowed, retry_after_seconds), the
        same shape as check().

        Deliberately has no matching "release"/rollback: a request that
        reserves a slot and is later rejected for being oversized or the
        wrong format still consumes it (finding 6). Otherwise a flood of
        garbage uploads could dodge the rate limit and the daily budget
        entirely, since every single one of them would be "rolled back"
        before ever counting.
        """
        now = now if now is not None else time.monotonic()
        self._reset_day_if_needed()
        self._sweep_if_due(now)
        if self._day_count >= self.daily_budget:
            return False, _seconds_until_local_midnight()

        hits = self._trim(client_ip, now)
        if len(hits) >= self.limit:
            retry_after = self.window_seconds - (now - hits[0])
            return False, max(retry_after, 1.0)

        # _trim returns either the live deque already stored in self._hits
        # (non-empty after trimming) or a fresh orphan deque() it chose not
        # to store because it was empty (finding 8); only the latter case
        # needs to be (re)inserted before appending the new hit.
        if client_ip not in self._hits:
            self._hits[client_ip] = hits
        hits.append(now)
        self._day_count += 1
        return True, None


limiter = RateLimiter()
