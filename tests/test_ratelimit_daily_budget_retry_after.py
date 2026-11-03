"""LOW item: the daily-budget-exceeded case used to return a fixed
window_seconds (60 by default) as Retry-After, as if the budget would
reset in another minute. It actually resets at local midnight
(RateLimiter._reset_day_if_needed uses date.today()), so a client that
waited the advertised 60 seconds and retried would just get rejected
again, every minute, until actual midnight. Retry-After should instead
reflect the real seconds remaining until that reset.
"""

from datetime import datetime, timedelta

from app.ratelimit import RateLimiter, _seconds_until_local_midnight


def test_seconds_until_local_midnight_matches_hand_calculation():
    now_dt = datetime(2026, 11, 3, 23, 0, 0)  # 1 hour before midnight
    assert _seconds_until_local_midnight(now_dt) == 3600.0


def test_seconds_until_local_midnight_just_after_midnight_is_almost_a_full_day():
    now_dt = datetime(2026, 11, 3, 0, 0, 1)
    seconds = _seconds_until_local_midnight(now_dt)
    assert 86390 < seconds <= 86399


def test_reserve_daily_budget_exceeded_retry_after_is_not_the_fixed_window():
    """The window is 60s by default; the real reset is hours away. If
    this ever regresses back to `float(self.window_seconds)`, retry_after
    would come back as exactly 60.0, which this asserts against.
    """
    limiter = RateLimiter(limit=1000, window_seconds=60, daily_budget=1)
    allowed, retry_after = limiter.reserve("1.2.3.4")
    assert allowed is True
    allowed, retry_after = limiter.reserve("5.6.7.8")  # different IP, same shared daily budget
    assert allowed is False
    assert retry_after != 60.0
    # Bounded between "a few seconds from now" and "at most a full day
    # away", i.e. clearly computed from wall-clock midnight rather than
    # a small fixed window.
    assert 1.0 <= retry_after <= 86400.0


def test_check_daily_budget_exceeded_retry_after_is_not_the_fixed_window():
    limiter = RateLimiter(limit=1000, window_seconds=60, daily_budget=1)
    limiter.record("1.2.3.4")
    allowed, retry_after = limiter.check("5.6.7.8")
    assert allowed is False
    assert retry_after != 60.0
    assert 1.0 <= retry_after <= 86400.0


def test_daily_budget_retry_after_is_stable_within_the_same_day(monkeypatch):
    """Two rejections a few seconds apart on the same day should both
    point at (roughly) the same midnight, not drift back toward a fixed
    window each time."""
    import app.ratelimit as ratelimit_module

    fixed_now = datetime(2026, 11, 3, 10, 0, 0)

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr(ratelimit_module, "datetime", _FixedDatetime)

    limiter = RateLimiter(limit=1000, window_seconds=60, daily_budget=1)
    limiter.record("1.2.3.4")
    _, first_retry = limiter.check("5.6.7.8")
    _, second_retry = limiter.check("9.9.9.9")
    assert first_retry == second_retry
    expected = (datetime(2026, 11, 4, 0, 0, 0) - fixed_now).total_seconds()
    assert first_retry == expected
