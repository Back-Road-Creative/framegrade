"""Shorts landmark bonus: a window inside a measured landmark polygon ranks
higher. Soft, ranking-only; never admits a window the quality bar or opening
refuses, and any failure to measure a landmark means no bonus."""

import logging

import numpy as np
import pytest

from gomoveshift.pipeline import day_pipeline as dp
from gomoveshift.pipeline.destination_resolver import landmark_seconds_mask
from gomoveshift.pipeline.short_planner import LANDMARK_BONUS, ShortRange, plan_shorts

N = 500


def _scores(fn) -> np.ndarray:
    arr = np.zeros((N, 9), dtype=np.float64)
    arr[:, 0] = np.arange(N, dtype=np.float64)
    arr[:, 1] = [fn(t) for t in range(N)]
    return arr


def _two_equal_moments(t):
    # Two identical 40 s moments; the later one is the landmark in the tests.
    return 0.8 if (100 <= t < 140 or 300 <= t < 340) else 0.3


def _mask(lo, hi):
    m = np.zeros(N, dtype=bool)
    m[lo:hi] = True
    return m


def _picks(**kw):
    return [(r.start_sec, r.end_sec) for r in plan_shorts(_scores(_two_equal_moments), **kw)]


def test_landmark_window_outranks_an_equal_window():
    assert LANDMARK_BONUS > 0
    plain = plan_shorts(_scores(_two_equal_moments), target_count=1)
    boosted = plan_shorts(
        _scores(_two_equal_moments), target_count=1, landmark_mask=_mask(295, 345)
    )
    assert plain[0].start_sec < 200  # tie falls to the earlier moment
    assert boosted[0].start_sec >= 290  # the landmark moment wins
    assert boosted[0].landmark is True and plain[0].landmark is False


def test_grazing_below_quorum_gets_no_bonus():
    # Only ~26% of the later window's seconds sit in the polygon.
    graze = _mask(300, 310)
    assert _picks(target_count=1, landmark_mask=graze) == _picks(target_count=1)
    r = plan_shorts(_scores(_two_equal_moments), target_count=2, landmark_mask=graze)
    assert all(not s.landmark for s in r)


def test_none_and_all_false_mask_reproduce_plain_picks():
    plain = _picks(target_count=3)
    assert _picks(target_count=3, landmark_mask=None) == plain
    assert _picks(target_count=3, landmark_mask=np.zeros(N, dtype=bool)) == plain
    # A mask that does not line up with the timeline is ignored, never guessed at.
    assert _picks(target_count=3, landmark_mask=np.ones(N - 1, dtype=bool)) == plain


def test_bonus_never_admits_a_below_bar_window():
    # The landmark window is flat 0.5, below the explicit bar of 0.6.
    def fn(t):
        return 0.9 if 100 <= t < 140 else 0.5

    r = plan_shorts(_scores(fn), target_count=5, quality_bar=0.6, landmark_mask=_mask(300, 345))
    assert r and all(s.start_sec < 200 for s in r)


def test_bonus_never_admits_a_weak_opening():
    # The landmark window's body clears the bar but its first 4 s do not.
    def fn(t):
        if 100 <= t < 140:
            return 0.9
        if 300 <= t < 304:
            return 0.0
        return 0.9 if 304 <= t < 345 else 0.5

    r = plan_shorts(_scores(fn), target_count=1, quality_bar=0.6, landmark_mask=_mask(290, 350))
    assert r and all(not (295 <= s.start_sec < 304) for s in r)


def test_bonus_does_not_override_the_stop_mask_or_gap():
    stop = _mask(295, 345)
    r = plan_shorts(
        _scores(_two_equal_moments), target_count=1, stop_mask=stop, landmark_mask=_mask(295, 345)
    )
    assert r and r[0].start_sec < 200


def _poly(name, lat0, lon0, lat1, lon1, **tags):
    ring = [
        {"lat": lat0, "lon": lon0},
        {"lat": lat0, "lon": lon1},
        {"lat": lat1, "lon": lon1},
        {"lat": lat1, "lon": lon0},
        {"lat": lat0, "lon": lon0},
    ]
    return {
        "type": "way",
        "geometry": ring,
        "tags": {"name": name, **(tags or {"leisure": "park"})},
    }


def _fixes(n, inside_until):
    # one fix a second heading north; inside the polygon while lat < inside_until
    return [(float(t), 32.0 + t * 1e-4, -80.0) for t in range(n)]


PARK = _poly("Hunting Island", 31.99, -80.01, 32.02, -79.99)


def test_mask_marks_only_seconds_inside_a_measured_polygon():
    pts = [(float(t), 32.0 + t * 1e-4, -80.0) for t in range(60)]
    # fixes cross lat 32.0030 (t=30); polygon covers lat <= 32.003.
    park = _poly("Hunting Island", 31.99, -80.01, 32.003, -79.99)
    m = landmark_seconds_mask(60, pts, poi_source=lambda _b: [park])
    assert m is not None and m[:30].all() and not m[31:].any()


def test_point_pois_and_bounds_only_areas_never_count():
    pts = [(float(t), 32.0, -80.0) for t in range(30)]
    node = {"type": "node", "lat": 32.0, "lon": -80.0, "tags": {"name": "Pier", "leisure": "park"}}
    bounds = {
        "type": "relation",
        "bounds": {"minlat": 31.9, "minlon": -80.1, "maxlat": 32.1, "maxlon": -79.9},
        "tags": {"name": "Big Park", "leisure": "park"},
    }
    m = landmark_seconds_mask(30, pts, poi_source=lambda _b: [node, bounds])
    # Nothing measured: no mask at all, never an invented landmark.
    assert m is None


def test_poi_source_failure_or_no_gps_gives_no_mask_and_a_warning(caplog):
    def boom(_b):
        raise OSError("overpass down")

    with caplog.at_level(logging.WARNING):
        assert landmark_seconds_mask(30, [(0.0, 32.0, -80.0)], poi_source=boom) is None
    assert "landmark bonus disabled" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert landmark_seconds_mask(30, [], poi_source=lambda _b: [PARK]) is None
    assert "landmark bonus disabled" in caplog.text


def test_day_pipeline_poi_failure_logs_and_plans_with_no_bonus(caplog):
    def boom(_b):
        raise OSError("overpass down")

    scores = _scores(_two_equal_moments)
    pts = [(float(t), 32.0, -80.0) for t in range(N)]
    with caplog.at_level(logging.WARNING):
        mask = dp._landmark_planning_mask(scores, pts, "2026-04-22", poi_source=boom)
    assert mask is None and "landmark bonus disabled" in caplog.text
    plain = [(r.start_sec, r.end_sec) for r in plan_shorts(scores, target_count=3)]
    got = [
        (r.start_sec, r.end_sec) for r in plan_shorts(scores, target_count=3, landmark_mask=mask)
    ]
    assert got == plain


def test_day_pipeline_mask_is_aligned_with_the_day_scores():
    scores = _scores(_two_equal_moments)
    pts = [(float(t), 32.0, -80.0) for t in range(N)]
    mask = dp._landmark_planning_mask(scores, pts, "2026-04-22", poi_source=lambda _b: [PARK])
    assert mask is not None and mask.shape == (N,) and mask.all()


def test_refill_forwards_the_landmark_mask_to_plan_shorts(monkeypatch):
    # The refill passes plan_kwargs through untouched, so the day's mask reaches it.
    seen = {}

    def fake_plan(scores, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(dp, "plan_shorts", fake_plan)
    lf = type("LF", (), {"index": 1, "segments": [object()]})()
    monkeypatch.setattr(dp, "to_cut_segment", lambda s: s)
    monkeypatch.setattr(dp, "segments_to_master_ranges", lambda segs, fb: [(0.0, 400.0)])
    monkeypatch.setattr(dp, "assign_shorts_to_lfs_by_source_window", lambda *a, **k: {1: []})
    mask = _mask(0, 10)
    dp.fill_shorts_to_per_lf_capacity(
        [], [lf], [(0.0, 400.0)], 1, _scores(_two_equal_moments), None, {"landmark_mask": mask}
    )
    assert seen["landmark_mask"] is mask


def test_shortrange_landmark_defaults_false_and_is_not_in_the_audit_hash():
    from gomoveshift.pipeline.ost_audit_sidecar import (
        compute_shorts_ost_audit_hash,
        format_shorts_ost_audit,
    )

    assert ShortRange(start_sec=0, end_sec=20, integrated_score=1.0, rank=1).landmark is False
    base = {
        "rank": 1,
        "parent_lf_index": 1,
        "ost_text": "Somewhere",
        "start_sec": 10.0,
        "end_sec": 40.0,
        "regions": [],
    }
    plain = format_shorts_ost_audit("2025-09-14", [base])
    rich = format_shorts_ost_audit("2025-09-14", [dict(base, landmark=True)])
    assert rich["shorts"][0]["landmark"] is True and plain["shorts"][0]["landmark"] is None
    assert compute_shorts_ost_audit_hash(rich) == compute_shorts_ost_audit_hash(plain)
    # Pinned: the existing hash for this payload is unchanged by the new key.
    assert compute_shorts_ost_audit_hash(plain) == compute_shorts_ost_audit_hash(
        format_shorts_ost_audit("2025-09-14", [base])
    )
