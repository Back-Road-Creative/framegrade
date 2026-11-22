"""Shorts landmark bonus: a window inside a measured landmark polygon ranks
higher. Soft, ranking-only; never admits a window the quality bar or opening
refuses, and any failure to measure a landmark means no bonus."""

import logging

import numpy as np
import pytest

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
    # Window mean is far above the bar (0.5) but its first 4 s are silent. The bonus
    # must not let it in: with the opening check bypassed this window would win on
    # its large sum, boosted or not. The plain 0.8 moment at 100 is the only valid pick.
    def fn(t):
        if 100 <= t < 140:
            return 0.8
        if 300 <= t < 304:
            return 0.0
        return 1.0 if 304 <= t < 340 else 0.3

    scores = _scores(fn)
    bar = 0.5
    plain = plan_shorts(scores, target_count=1, quality_bar=bar)
    boosted = plan_shorts(scores, target_count=1, quality_bar=bar, landmark_mask=_mask(290, 350))
    assert [(r.start_sec, r.end_sec) for r in boosted] == [(r.start_sec, r.end_sec) for r in plain]
    assert boosted and all(r.opening_mean > bar for r in boosted)
    assert all(not (295 <= r.start_sec < 304) for r in boosted)


def test_bonus_does_not_override_the_stop_mask():
    stop = _mask(295, 345)
    r = plan_shorts(
        _scores(_two_equal_moments), target_count=1, stop_mask=stop, landmark_mask=_mask(295, 345)
    )
    assert r and r[0].start_sec < 200


def test_bonus_does_not_override_the_hard_gap():
    # Two landmark moments 30 s apart (inside the 60 s gap): never both picked.
    def fn(t):
        return 0.8 if (100 <= t < 138 or 168 <= t < 206) else 0.3

    r = plan_shorts(_scores(fn), target_count=5, min_gap_sec=60.0, landmark_mask=_mask(95, 210))
    spans = sorted((x.start_sec, x.end_sec) for x in r)
    assert all(b[0] - a[1] >= 60.0 for a, b in zip(spans, spans[1:]))


def test_the_bonus_never_costs_a_short():
    # Reviewer repro: a boosted middle window blocks both outer ones under the gap.
    n = 200
    arr = np.zeros((n, 9), dtype=np.float64)
    arr[:, 0] = np.arange(n, dtype=np.float64)
    arr[:, 1] = 0.1
    arr[0:38, 1] = 0.9
    arr[160:198, 1] = 0.9
    arr[80:118, 1] = 0.85
    kw = dict(target_count=5, min_gap_sec=60.0, stride_sec=5.0)
    plain = [(r.start_sec, r.end_sec) for r in plan_shorts(arr, **kw)]
    mask = np.zeros(n, dtype=bool)
    mask[80:118] = True
    boosted = plan_shorts(arr, landmark_mask=mask, **kw)
    assert len(plain) == 2
    assert [(r.start_sec, r.end_sec) for r in boosted] == plain
    assert all(not r.landmark for r in boosted)  # the plain picks ranked them


def test_polygon_rows_must_each_hold_the_quorum():
    # Park A covers 30% of the later window and park B another 30%: no single
    # polygon holds half, so no bonus (a union of 60% would wrongly qualify).
    a = _mask(300, 311)
    b = _mask(311, 322)
    two = np.vstack([a, b])
    assert _picks(target_count=1, landmark_mask=two) == _picks(target_count=1)
    one = np.vstack([_mask(295, 345), b])
    assert _picks(target_count=1, landmark_mask=one)[0][0] >= 290


def test_misaligned_mask_warns(caplog):
    with caplog.at_level(logging.WARNING):
        _picks(target_count=1, landmark_mask=np.ones(N - 1, dtype=bool))
    assert "landmark bonus disabled" in caplog.text


# ---- resolver mask, day_pipeline wiring, audit hash --------------------------

import logging as _lg  # noqa: E402

from gomoveshift.pipeline import day_pipeline as dp  # noqa: E402
from gomoveshift.pipeline.destination_resolver import landmark_seconds_mask  # noqa: E402


def _poly(name, lat0, lon0, lat1, lon1, closed=True, **tags):
    ring = [
        {"lat": lat0, "lon": lon0},
        {"lat": lat0, "lon": lon1},
        {"lat": lat1, "lon": lon1},
        {"lat": lat1, "lon": lon0},
    ]
    if closed:
        ring.append({"lat": lat0, "lon": lon0})
    return {
        "type": "way",
        "geometry": ring,
        "tags": {"name": name, **(tags or {"leisure": "park"})},
    }


PARK = _poly("Hunting Island", 31.99, -80.01, 32.02, -79.99)


def _fix_line(n, step=1):
    return [(float(t), 32.0 + t * 1e-4, -80.0) for t in range(0, n, step)]


def test_mask_marks_only_seconds_inside_a_measured_polygon():
    pts = _fix_line(60)
    park = _poly("Hunting Island", 31.99, -80.01, 32.003, -79.99)  # lat 32.003 == t=30
    m = landmark_seconds_mask(np.arange(60.0), pts, poi_source=lambda _b: [park])
    assert m.shape == (1, 60) and m[0, :30].all() and not m[0, 31:].any()


def test_point_pois_bounds_only_areas_and_open_lines_never_count(caplog):
    pts = [(float(t), 32.0, -80.0) for t in range(30)]
    node = {"type": "node", "lat": 32.0, "lon": -80.0, "tags": {"name": "Pier", "leisure": "park"}}
    bounds = {
        "type": "relation",
        "bounds": {"minlat": 31.9, "minlon": -80.1, "maxlat": 32.1, "maxlon": -79.9},
        "tags": {"name": "Big Park", "leisure": "park"},
    }
    open_l = _poly("Trail Park", 31.99, -80.01, 32.02, -79.99, closed=False)
    with caplog.at_level(_lg.WARNING):
        m = landmark_seconds_mask(
            np.arange(30.0), pts, poi_source=lambda _b: [node, bounds, open_l]
        )
    assert m is None and "landmark bonus disabled" in caplog.text


def test_poi_source_failure_or_no_gps_gives_no_mask_and_a_warning(caplog):
    def boom(_b):
        raise OSError("overpass down")

    ts = np.arange(30.0)
    with caplog.at_level(_lg.WARNING):
        assert landmark_seconds_mask(ts, [(0.0, 32.0, -80.0)], poi_source=boom) is None
    assert "landmark bonus disabled" in caplog.text
    caplog.clear()
    with caplog.at_level(_lg.WARNING):
        assert landmark_seconds_mask(ts, [], poi_source=lambda _b: [PARK]) is None
    assert "landmark bonus disabled" in caplog.text


def test_a_bad_fix_never_reaches_the_poi_bbox():
    seen = []

    def src(bbox):
        seen.append(bbox)
        return [PARK]

    pts = _fix_line(60) + [(61.0, 0.0, 0.0)]
    landmark_seconds_mask(np.arange(60.0), pts, poi_source=src)
    s, w, n, e = seen[0]
    assert 31.9 < s and n < 32.1 and -80.1 < w and e < -79.9


def test_a_day_too_wide_for_one_query_asks_no_poi_source(caplog):
    def src(_b):
        raise AssertionError("must not query")

    pts = [(0.0, 30.0, -80.0), (10.0, 36.0, -80.0)]
    with caplog.at_level(_lg.WARNING):
        assert landmark_seconds_mask(np.arange(11.0), pts, poi_source=src) is None
    assert "landmark bonus disabled" in caplog.text


def test_mask_follows_score_row_times_and_never_credits_across_a_gps_gap():
    # Row times start at 100 (a multi-file day offset); fixes every 3 s inside the
    # park, then a 20 s hole, then fixes again.
    ts = np.arange(100.0, 160.0)
    pts = [(float(t), 32.0, -80.0) for t in range(100, 130, 3)] + [
        (float(t), 32.0, -80.0) for t in range(150, 160, 3)
    ]
    m = landmark_seconds_mask(ts, pts, poi_source=lambda _b: [PARK])[0]
    assert m[:27].all()  # 3 s cadence is inside the 5 s gap bound
    assert not m[30:49].any()  # inside the hole: no credit
    assert m[50:57].all()


@pytest.mark.real_overpass
def test_overpass_fast_budget_and_partial_replies(monkeypatch, tmp_path):
    from gomoveshift import pipeline  # noqa: F401
    from gomoveshift.pipeline import overpass_source as op

    calls = []

    def fake_post(query, **kw):
        calls.append((query, kw))
        return {"elements": [], "remark": "runtime error: Query timed out"}

    monkeypatch.setenv("GMS_OVERPASS_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(op, "post_overpass_query", fake_post)
    with pytest.raises(RuntimeError, match="partial"):
        op._query_overpass((32.0, -80.0, 32.1, -79.9), fast=True)
    assert f"[timeout:{op.OVERPASS_FAST_TIMEOUT}]" in calls[0][0] and calls[0][1] == {"fast": True}
    assert list(tmp_path.iterdir()) == []  # a partial reply is never cached


def test_post_overpass_fast_tries_one_endpoint_once(monkeypatch):
    from gomoveshift.pipeline import overpass_source as op

    urls = []

    def boom(req, timeout=None):
        urls.append((req.full_url, timeout))
        raise OSError("down")

    monkeypatch.setattr(op.urllib.request, "urlopen", boom)
    monkeypatch.setattr(op.time, "sleep", lambda _s: None)
    with pytest.raises(RuntimeError):
        op.post_overpass_query("q", fast=True)
    assert len(urls) == 1 and urls[0][1] == op.OVERPASS_FAST_CLIENT_TIMEOUT


def test_day_pipeline_poi_failure_logs_and_plans_with_no_bonus(caplog):
    def boom(_b):
        raise OSError("overpass down")

    scores = _scores(_two_equal_moments)
    pts = [(float(t), 32.0, -80.0) for t in range(N)]
    with caplog.at_level(_lg.WARNING):
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
    assert mask is not None and mask.shape == (1, N) and mask.all()


def test_refill_forwards_the_landmark_mask_to_plan_shorts(monkeypatch):
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


def _run_day(tmp_path, monkeypatch, short_ost_text_fn):
    """run_day_pipeline with rendering stubbed; returns (plan kwargs, refill kwargs)."""
    from dataclasses import dataclass

    from gomoveshift.pipeline.day_grouper import DayBatch

    @dataclass
    class _Ep:
        index: int
        concat_path: object
        total_duration: float
        date: str
        creation_time: object = None

    date = "2026-04-22"
    ep = _Ep(1, tmp_path / "ep0.mp4", float(N), date)
    ep.concat_path.write_bytes(b"x")
    master = tmp_path / "out" / f"day_{date}" / "day_master.mp4"

    def _master(*a, **kw):
        master.parent.mkdir(parents=True, exist_ok=True)
        master.write_bytes(b"x")
        return master

    seen = {"plan": [], "refill": []}

    def _plan(scores, **kw):
        seen["plan"].append(kw)
        return [ShortRange(start_sec=300.0, end_sec=330.0, integrated_score=1.0, rank=1)]

    def _fill(shorts, lfs, fb, cap, scores, mask, plan_kw):
        seen["refill"].append(plan_kw)
        return shorts, []

    monkeypatch.setattr(dp, "encode_day_master", _master)
    monkeypatch.setattr(dp, "compute_day_file_boundaries", lambda *a, **kw: [(0.0, float(N))])
    monkeypatch.setattr(dp, "extract_and_concat_segments", lambda *a, **kw: None)
    monkeypatch.setattr(dp, "prepend_montage", lambda *a, **kw: None)
    monkeypatch.setattr(dp, "purge_day_intermediates", lambda *a, **kw: None)
    monkeypatch.setattr(dp, "plan_shorts", _plan)
    monkeypatch.setattr(dp, "cap_shorts_to_per_lf_capacity", lambda s, *a, **k: (s, []))
    monkeypatch.setattr(dp, "fill_shorts_to_per_lf_capacity", _fill)

    def _apply(input_path, output_path, location, layout, **kw):
        output_path.write_bytes(b"x")
        return output_path

    monkeypatch.setattr(dp, "apply_ost", _apply)

    def _render(source_video, output_path, start_time, duration, location, **kw):
        output_path.write_bytes(b"x")
        return True

    monkeypatch.setattr(dp, "_render_short_ffmpeg", _render)
    pts = [(float(t), 32.0 + t * 0.0001, -80.0) for t in range(0, N, 2)]
    dp.run_day_pipeline(
        day_batch=DayBatch(date=date, episodes=[ep]),
        day_scores=_scores(lambda t: 0.8),
        output_dir=tmp_path / "out",
        location="Somewhere, TestCounty",
        config=None,
        lf_plans=[],
        gps_points_by_date={date: pts},
        short_ost_text_fn=short_ost_text_fn,
        state={"ost_audit_confirmed": True, "ost_audit_confirmed_this_run": True},
    )
    return seen


def test_run_day_pipeline_hands_the_mask_to_the_plan_and_the_refill(tmp_path, monkeypatch):
    def fn(pts, sr):
        return "Beaufort County"

    fn.has_landmark_poi_source = True
    fn.landmark_poi_source = lambda _b: [_poly("Wide Park", 31.9, -80.1, 32.2, -79.9)]
    seen = _run_day(tmp_path, monkeypatch, fn)
    assert seen["plan"] and isinstance(seen["plan"][0].get("landmark_mask"), np.ndarray)
    assert (
        seen["refill"]
        and seen["refill"][0].get("landmark_mask") is seen["plan"][0]["landmark_mask"]
    )


def test_a_closure_without_the_poi_source_never_queries_it(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    def boom(*_a, **_k):
        raise AssertionError("POI source must not be called")

    monkeypatch.setattr(dp, "landmark_seconds_mask", boom)
    for fn in (MagicMock(return_value="Beaufort County"), lambda pts, sr: "Beaufort County"):
        seen = _run_day(tmp_path, monkeypatch, fn)
        assert "landmark_mask" not in seen["plan"][-1]


def test_make_short_ost_text_fn_exposes_the_landmark_poi_source():
    from gomoveshift.pipeline.gps_enrichment import _make_short_ost_text_fn

    src = lambda _b: []  # noqa: E731
    fn = _make_short_ost_text_fn(object(), poi_source=src)
    assert fn.has_landmark_poi_source is True and fn.landmark_poi_source is src


# Hash of this payload computed at cd03920 (before the `landmark` key existed).
_CD03920_SHORTS_AUDIT_HASH = "26816a99bc8b3c61ce0000ac3a82fcbadbcd0c988db3d4bb0b12b0bed30feff8"


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
    assert compute_shorts_ost_audit_hash(plain) == _CD03920_SHORTS_AUDIT_HASH
