"""``facebook_reel_plays_series``: account-level daily reel plays from the
nightly cumulative observation log (interim series, no per-day FB API pull)."""

from __future__ import annotations

from types import SimpleNamespace

from gomoveshift.tracking.facebook_series import facebook_reel_plays_series


def _obs(fetched, total, *, post="p1", event="2026-08-01T12:00:00-04:00", status="measured"):
    return {
        "status": status,
        "post_id": post,
        "post_format": "reel",
        "event_at": event,
        "fetched_at": fetched,
        "cumulative": {"reel_total_plays": total},
    }


def _rec(*logs):
    shorts = [SimpleNamespace(observation_log={"facebook": log}) for log in logs]
    return SimpleNamespace(shorts=shorts)


def test_empty_records():
    assert facebook_reel_plays_series([]) == {}


def test_two_shorts_three_nights_sum_per_day():
    a = [
        _obs("2026-08-01T22:00:00-04:00", 100, post="a"),
        _obs("2026-08-02T22:00:00-04:00", 130, post="a"),
        _obs("2026-08-03T22:00:00-04:00", 140, post="a"),
    ]
    b = [
        _obs("2026-08-01T23:00:00-04:00", 10, post="b"),
        _obs("2026-08-02T23:00:00-04:00", 25, post="b"),
        _obs("2026-08-03T23:00:00-04:00", 30, post="b"),
    ]
    assert facebook_reel_plays_series([_rec(a), _rec(b)]) == {
        "2026-08-01": 110.0,
        "2026-08-02": 45.0,
        "2026-08-03": 15.0,
    }


def test_counter_reset_interval_skipped():
    log = [
        _obs("2026-08-01T22:00:00-04:00", 100),
        _obs("2026-08-02T22:00:00-04:00", 50),
        _obs("2026-08-03T22:00:00-04:00", 60),
    ]
    assert facebook_reel_plays_series([_rec(log)]) == {"2026-08-01": 100.0, "2026-08-03": 10.0}


def test_non_measured_ignored():
    log = [
        _obs("2026-08-01T22:00:00-04:00", 100),
        _obs("2026-08-02T22:00:00-04:00", 999, status="not_configured"),
        _obs("2026-08-03T22:00:00-04:00", 130),
    ]
    assert facebook_reel_plays_series([_rec(log)]) == {"2026-08-01": 100.0, "2026-08-03": 30.0}


def test_old_single_observation_is_baseline_only():
    log = [_obs("2026-08-10T22:00:00-04:00", 500, event="2026-07-01T12:00:00-04:00")]
    assert facebook_reel_plays_series([_rec(log)]) == {}


def test_fetch_day_uses_new_york_calendar_day():
    # 02:30 UTC Aug 2 is 22:30 Aug 1 in New York.
    log = [_obs("2026-08-02T02:30:00+00:00", 40, event="2026-08-01T20:00:00-04:00")]
    assert facebook_reel_plays_series([_rec(log)]) == {"2026-08-01": 40.0}
