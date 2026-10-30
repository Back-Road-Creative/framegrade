"""Backtest: the Facebook detector flags the real August 2026 reel-reach collapse.

The series is synthetic but built from the eval numbers: daily reel views of
about 414/day for June, ~350 for July (filled so the series is continuous),
249/day Aug 1-13, 30.5/day Aug 14-26, 33/day September, 36/day October, with
small deterministic jitter.

Known gap: ``classify_facebook_reel_reach`` has no recovery branch, so once
SUPPRESSED the series never returns to HEALTHY or RECOVERING even though
October is flat at a low level. This test pins that behavior; it does not add
recovery logic.
"""

from __future__ import annotations

from datetime import date, timedelta

from gomoveshift.tracking.suppression import classify_facebook_reel_reach

# The first day the 7-day median (strictly prior days) drops below 0.3x the
# 28-day median: observed with this fixture, not assumed.
FIRST_SUPPRESSED = "2026-08-18"


def _level(d: date) -> float:
    if d < date(2026, 7, 1):
        return 414.0
    if d < date(2026, 8, 1):
        return 350.0
    if d <= date(2026, 8, 13):
        return 249.0
    if d <= date(2026, 8, 26):
        return 30.5
    if d < date(2026, 10, 1):
        return 33.0
    return 36.0


def _series() -> dict:
    out = {}
    d = date(2026, 6, 1)
    while d <= date(2026, 10, 31):
        jitter = ((d.toordinal() * 7) % 5 - 2) * 0.02  # -4%..+4%, deterministic
        out[d.isoformat()] = _level(d) * (1 + jitter)
        d += timedelta(days=1)
    return out


def test_healthy_before_the_collapse():
    status = classify_facebook_reel_reach(_series())
    assert all(s == "HEALTHY" for day, s in status.items() if day <= "2026-08-13")


def test_flags_the_collapse_by_aug_21():
    status = classify_facebook_reel_reach(_series())
    first = min(day for day, s in status.items() if s == "SUPPRESSED")
    assert first == FIRST_SUPPRESSED
    assert first <= "2026-08-21"
    assert all(s == "SUPPRESSED" for day, s in status.items() if day >= first)


def test_still_suppressed_in_october():
    assert classify_facebook_reel_reach(_series())["2026-10-15"] == "SUPPRESSED"
