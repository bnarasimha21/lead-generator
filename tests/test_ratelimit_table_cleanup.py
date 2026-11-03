"""Verification-review finding 8: RateLimiter._hits used to be a
defaultdict(deque), so a bare read (`self._hits[client_ip]`) autocreated a
permanent entry for every distinct IP ever seen by this process, and
nothing ever removed an entry once its hits aged out of the window. For
an in-memory, single-process table meant to live for the lifetime of a
long-running process, that is unbounded growth keyed by distinct client
IP, independent of the daily call budget (which caps total requests, not
distinct IPs).

These tests exercise RateLimiter directly (no HTTP layer) to confirm the
table only ever holds IPs with at least one hit still inside the current
window.
"""

from app.ratelimit import RateLimiter


def test_check_on_a_never_seen_ip_does_not_create_a_table_entry():
    limiter = RateLimiter(limit=3, window_seconds=60, daily_budget=1000)
    allowed, _ = limiter.check("1.2.3.4")
    assert allowed is True
    assert "1.2.3.4" not in limiter._hits


def test_hit_that_ages_out_of_the_window_is_evicted_from_the_table():
    limiter = RateLimiter(limit=3, window_seconds=60, daily_budget=1000)
    limiter.reserve("1.2.3.4", now=0.0)
    assert "1.2.3.4" in limiter._hits

    # Well past the window: the next check/reserve for this IP must trim
    # the stale hit and drop the now-empty entry entirely, not just empty
    # the deque and leave it sitting in the dict forever.
    allowed, _ = limiter.check("1.2.3.4", now=1000.0)
    assert allowed is True
    assert "1.2.3.4" not in limiter._hits


def test_table_does_not_grow_without_bound_across_many_one_off_ips():
    """Each of these IPs is seen exactly once, so RateLimiter._trim's
    per-IP lazy eviction never gets a second chance to run for any of
    them (nothing ever looks that IP up again). Only the full-table sweep
    (_sweep_if_due), triggered once the table passes sweep_threshold, can
    bound growth in this scenario."""
    limiter = RateLimiter(limit=3, window_seconds=60, daily_budget=100000, sweep_threshold=50)
    # Each request is from a brand-new IP, 1000 seconds after the last:
    # well outside the 60s window by the time later requests happen, so
    # every earlier entry is stale by the time the sweep runs.
    for i in range(500):
        limiter.reserve(f"10.0.{i // 256}.{i % 256}", now=float(i) * 1000.0)
    # The table was swept repeatedly (500 one-off IPs vastly exceeds the
    # threshold of 50), and each sweep's "now" is far past every earlier
    # IP's single hit, so the table never holds more than roughly
    # sweep_threshold entries at any point, nowhere near all 500 distinct
    # IPs ever seen.
    assert len(limiter._hits) <= limiter.sweep_threshold + 2


def test_sweep_removes_stale_entries_once_threshold_crossed():
    limiter = RateLimiter(limit=3, window_seconds=60, daily_budget=100000, sweep_threshold=2)
    limiter.reserve("1.1.1.1", now=0.0)
    limiter.reserve("2.2.2.2", now=0.0)
    limiter.reserve("3.3.3.3", now=0.0)
    assert len(limiter._hits) == 3
    # Crossing the threshold (4th distinct IP, table size 3 > threshold 2)
    # with "now" far past the 60s window triggers a full sweep before this
    # call's own hit is recorded; all three earlier stale entries are
    # dropped, leaving only the new one.
    limiter.reserve("4.4.4.4", now=10_000.0)
    assert set(limiter._hits.keys()) == {"4.4.4.4"}


def test_reserve_still_rate_limits_correctly_after_table_cleanup_changes():
    """Regression guard: the cleanup must not break the actual limiting
    behavior it sits alongside."""
    limiter = RateLimiter(limit=2, window_seconds=60, daily_budget=1000)
    allowed1, _ = limiter.reserve("1.2.3.4", now=0.0)
    allowed2, _ = limiter.reserve("1.2.3.4", now=1.0)
    allowed3, _ = limiter.reserve("1.2.3.4", now=2.0)
    assert (allowed1, allowed2, allowed3) == (True, True, False)
    assert len(limiter._hits["1.2.3.4"]) == 2
