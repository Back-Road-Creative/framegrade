"""Facebook recovery: SUPPRESSED -> RECOVERING -> HEALTHY against a frozen baseline.

Every series is synthetic: 28 days at 1000 views, then the scenario under test.
"""

from __future__ import annotations

import datetime

from gomoveshift.tracking.suppression import classify_facebook_reel_reach

DAY0 = datetime.date(2000, 1, 1)


def _day(i: int) -> str:
    return (DAY0 + datetime.timedelta(days=i)).isoformat()


def _series(*segments):
    """segments: (n_days, level) pairs appended after 28 baseline days of 1000."""
    levels = [1000.0] * 28
    for n, level in segments:
        levels += [float(level)] * n
    return {_day(i): v for i, v in enumerate(levels)}


def _first(status, state, after=""):
    return min(d for d, s in status.items() if s == state and d > after)


def test_collapse_then_full_return_walks_suppressed_recovering_healthy():
    # collapse on day 28 (suppressed day 32), back to 1000 on day 50.
    status = classify_facebook_reel_reach(_series((22, 100), (30, 1000)))
    assert status[_day(31)] == "HEALTHY"
    assert status[_day(32)] == "SUPPRESSED"
    assert status[_day(53)] == "SUPPRESSED"
    assert status[_day(54)] == "RECOVERING"  # 4th high day in the 7-day window
    assert status[_day(59)] == "RECOVERING"
    assert status[_day(60)] == "HEALTHY"  # 7th consecutive day at >=0.7x baseline
    assert status[_day(79)] == "HEALTHY"


def test_partial_rebound_stays_recovering():
    # 600 is >=0.5x and <0.7x the frozen 1000 baseline.
    status = classify_facebook_reel_reach(_series((22, 100), (40, 600)))
    assert status[_day(54)] == "RECOVERING"
    assert all(status[_day(i)] == "RECOVERING" for i in range(54, 90))


def test_relapse_during_recovery_returns_to_suppressed():
    status = classify_facebook_reel_reach(_series((22, 100), (6, 1000), (20, 100)))
    assert status[_day(54)] == "RECOVERING"
    # 7-day median drops under 0.3x the frozen baseline again.
    assert _first(status, "SUPPRESSED", _day(54)) < _day(70)
    assert status[_day(75)] == "SUPPRESSED"


def test_long_suppression_with_decaying_live_median_does_not_recover():
    status = classify_facebook_reel_reach(_series((120, 100)))
    assert all(status[_day(i)] == "SUPPRESSED" for i in range(32, 148))


def test_baseline_clears_on_healthy_so_a_new_collapse_flags_again():
    status = classify_facebook_reel_reach(_series((22, 100), (40, 1000), (14, 250)))
    assert status[_day(79)] == "HEALTHY"
    assert status[_day(93)] == "HEALTHY"
    assert status[_day(94)] == "SUPPRESSED"  # 4th collapsed day in the 7-day window
