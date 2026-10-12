"""A short never spans a county line.

A clip that straddles two counties passes the post-selection place gate with
either county burned. The planner's veto mask closes that by construction: every
second within one window length of a county change is vetoed.
"""

from __future__ import annotations

import numpy as np

from gomoveshift.pipeline.day_pipeline import _shorts_planning_mask
from gomoveshift.pipeline.short_planner import DEFAULT_TARGET_DURATION_SEC, plan_shorts
from gomoveshift.pipeline.stop_mask import COUNTY_MASK_SAMPLE_S, build_county_change_mask

_LAT0, _LON0 = 39.83, -77.23
_STEP_DEG = 0.0002  # ~22 m/s: always moving


def _scores(n):
    """Planner score array: column 0 timestamp, column 1 composite."""
    return np.column_stack([np.arange(n, dtype=float), np.ones(n)])


def _trace(n):
    return [(float(t), _LAT0 + t * _STEP_DEG, _LON0) for t in range(n)]


class _Geo:
    """County A before ``switch_t`` seconds, county B after; ``miss_t`` -> {}."""

    def __init__(self, switch_t=None, miss_t=None):
        self.switch_t, self.miss_t, self.calls = switch_t, miss_t, 0

    def reverse_geocode_detailed(self, lat, lon):
        self.calls += 1
        t = round((lat - _LAT0) / _STEP_DEG)
        if self.miss_t is not None and abs(t - self.miss_t) < 15:
            return {}
        county = "A" if self.switch_t is None or t < self.switch_t else "B"
        return {"county": county, "state": "PA"}


def test_planner_never_picks_a_window_across_the_county_line():
    n = 1200
    scores = _scores(n)
    mask = _shorts_planning_mask(None, scores, _trace(n), "2025-09-14", geocoder=_Geo(switch_t=600))
    ranges = plan_shorts(scores, stop_mask=mask)
    assert ranges
    lo, hi = 600 - DEFAULT_TARGET_DURATION_SEC, 600 + DEFAULT_TARGET_DURATION_SEC
    for r in ranges:
        assert r.end_sec <= lo or r.start_sec >= hi, (r, lo, hi)


def test_no_geocoder_or_no_gps_leaves_the_stop_mask_unchanged():
    n = 600
    scores = _scores(n)
    stop = np.zeros(n, dtype=bool)
    stop[100:120] = True
    out = _shorts_planning_mask(stop, scores, _trace(n), "d", geocoder=None)
    assert np.array_equal(out, stop)
    out = _shorts_planning_mask(stop, scores, None, "d", geocoder=_Geo(switch_t=300))
    assert np.array_equal(out, stop)
    assert build_county_change_mask(0, _trace(10), _Geo(5), half_window_s=38).size == 0
    assert not build_county_change_mask(50, [], _Geo(5), half_window_s=38).any()
    assert not build_county_change_mask(50, _trace(50), None, half_window_s=38).any()


def test_a_geocode_miss_in_a_single_county_day_vetoes_nothing():
    mask = build_county_change_mask(900, _trace(900), _Geo(miss_t=450), half_window_s=38)
    assert not mask.any()


def test_veto_interval_is_exact():
    mask = build_county_change_mask(1200, _trace(1200), _Geo(switch_t=600), half_window_s=38)
    # samples every COUNTY_MASK_SAMPLE_S: last A sample 570, first B sample 600
    assert COUNTY_MASK_SAMPLE_S == 30
    lo, hi = 570 - 38, 600 + 38
    assert mask[lo : hi + 1].all()
    assert not mask[:lo].any()
    assert not mask[hi + 1 :].any()
