"""A short never borrows a place for a clip with no GPS of its own.

Regression: shorts must not name places their clips were not filmed at.
One hole: on a day WITH GPS, a short planned over a GPS dropout (a lost
lock, a tunnel, an episode whose GPX never matched) had no fix to vote its own
place, so it burned the parent LF's destination POI or county instead.

The fix is structural: the shorts planner honours a GPS-gap veto, so it cannot
pick a window without the fixes the place vote needs; a gap short that still
reaches the render hard-fails before any burn. A GPS-less DAY keeps its
operator/county anchor (``test_gpsless_short_fallback``), never a destination.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from gomoveshift.models import ResolvedLocation
from gomoveshift.pipeline import day_pipeline
from gomoveshift.pipeline.gps_quality import trim_gps_to_clip_timeline, usable_fix_times
from gomoveshift.pipeline.short_planner import DEFAULT_MIN_DURATION_SEC, plan_shorts
from gomoveshift.pipeline.stop_mask import GPS_MAX_FIX_GAP_S, build_gps_gap_mask
from _state_helpers import make_state

_DATE = "2025-09-14"
_LAT0, _LON0 = 39.83, -77.23
_STEP_DEG = 0.0002  # ~22 m/s: every span is moving, never a stop


def _trace(*spans):
    """1 Hz moving fixes ``(t, lat, lon)`` over each inclusive ``(t0, t1)`` span."""
    return [
        (float(t), _LAT0 + t * _STEP_DEG, _LON0)
        for t0, t1 in spans
        for t in range(int(t0), int(t1) + 1)
    ]


# ── the veto itself ────────────────────────────────────────────────────────


def test_usable_fix_times_are_sorted_and_skip_fixes_without_coordinates():
    pts = [(5.0, 1.0, 2.0), (1.0, 1.0, 2.0), (3.0, None, 2.0), (4.0, 1.0, None)]
    assert usable_fix_times(pts) == [1.0, 5.0]
    assert usable_fix_times(None) == []


def test_a_gps_less_day_vetoes_nothing():
    for points in (None, []):
        mask = build_gps_gap_mask(600, points)
        assert mask.shape == (600,)
        assert not mask.any()


def test_full_coverage_vetoes_nothing():
    assert not build_gps_gap_mask(300, _trace((0, 299))).any()


def test_a_dropout_is_vetoed_edge_to_edge():
    mask = build_gps_gap_mask(300, _trace((0, 100), (161, 299)))
    assert mask[101:162].all()
    assert not mask[:101].any()
    assert not mask[162:].any()


def test_seconds_before_the_first_fix_are_vetoed():
    mask = build_gps_gap_mask(1200, _trace((600, 1199)))
    assert mask[:601].all()
    assert not mask[601:].any()


def test_a_short_lock_delay_is_not_vetoed():
    assert not build_gps_gap_mask(300, _trace((3, 296))).any()


def test_fixes_without_coordinates_do_not_count():
    pts = [(t, None if 50 <= t <= 80 else lat, lon) for t, lat, lon in _trace((0, 299))]
    mask = build_gps_gap_mask(300, pts)
    assert mask[50:82].all()
    assert not mask[:50].any()
    assert not mask[82:].any()


def test_a_sparse_logger_within_the_gap_limit_is_not_vetoed():
    step = int(GPS_MAX_FIX_GAP_S)
    pts = [(float(t), _LAT0 + t * _STEP_DEG, _LON0) for t in range(0, 300, step)]
    assert not build_gps_gap_mask(300, pts).any()


def test_every_window_the_veto_leaves_open_holds_two_fixes():
    """The place vote needs a quorum of two fixes; every window of the planner's
    minimum short length that the veto leaves open must hold them."""
    rng = np.random.default_rng(7)
    times, t = [], 0.0
    while t < 3600:
        times.append(t)
        t += float(rng.choice([1, 1, 1, 2, 5, 7, 8, 12, 40, 150]))
    pts = [(tt, _LAT0 + tt * _STEP_DEG, _LON0) for tt in times]
    mask = build_gps_gap_mask(3600, pts)
    width = int(DEFAULT_MIN_DURATION_SEC)
    open_windows = 0
    for s in range(0, 3600 - width + 1):
        if mask[s : s + width].any():
            continue
        open_windows += 1
        inside = trim_gps_to_clip_timeline(pts, [(s, s + width)])
        assert len(inside) >= 2, f"window {s}-{s + width}s holds {len(inside)} fix(es)"
    assert open_windows > 0  # not vacuous
    assert mask.any()  # the trace really has vetoed gaps


def test_the_planner_never_picks_a_window_over_a_dropout():
    n = 600
    scores = np.zeros((n, 11), dtype=np.float32)
    scores[:, 0] = np.arange(n)
    scores[:, 1] = 0.2
    scores[:, 2] = 0.5
    scores[250:350, 1] = 1.0  # the day's best moment sits in the dropout
    pts = _trace((0, 240), (360, 599))
    unvetoed = plan_shorts(scores, target_count=3, min_gap_sec=10.0)
    assert any(r.start_sec < 350 and r.end_sec > 250 for r in unvetoed)  # would pick it
    picked = plan_shorts(
        scores, target_count=3, min_gap_sec=10.0, stop_mask=build_gps_gap_mask(n, pts)
    )
    assert picked
    for r in picked:
        assert r.end_sec <= 241 or r.start_sec >= 361, (r.start_sec, r.end_sec)
        assert len(trim_gps_to_clip_timeline(pts, [(r.start_sec, r.end_sec)])) >= 2


# ── the day pipeline wiring ────────────────────────────────────────────────


class _PlanReachedError(Exception):
    """Stops the day pipeline once ``plan_shorts`` has seen its inputs."""


@pytest.fixture
def burned():
    return []


@pytest.fixture
def render_patches(tmp_path, burned):
    def _capture_short(*, source_video, output_path, start_time, duration, location, **kwargs):
        burned.append(location)
        Path(output_path).touch()
        return True

    patches = [
        patch.object(day_pipeline, "encode_day_master", return_value=tmp_path / "master.mp4"),
        patch.object(
            day_pipeline, "extract_and_concat_segments", return_value=tmp_path / "seg.mp4"
        ),
        patch.object(day_pipeline, "prepend_montage", return_value=tmp_path / "wm.mp4"),
        patch.object(day_pipeline, "apply_ost", side_effect=lambda **kw: kw["output_path"]),
        patch.object(day_pipeline, "_render_short_ffmpeg", side_effect=_capture_short),
    ]
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


def _run_day(tmp_path, gps_points_by_date, **state_kwargs):
    lf_plan = SimpleNamespace(
        index=1,
        segments=[SimpleNamespace(source_file_index=0, start_time=0.0, end_time=1800.0)],
        montage_peaks=[],
        total_duration=1800.0,
    )
    day_batch = SimpleNamespace(
        date=_DATE,
        episodes=[SimpleNamespace(index=1, concat_path=None, total_duration=1800.0)],
    )
    per_lf_ost_text = state_kwargs.pop("per_lf_ost_text", None)
    per_lf_locations = state_kwargs.pop("per_lf_locations", None)
    day_pipeline.run_day_pipeline(
        day_batch=day_batch,
        day_scores=np.array([[0.5] * 11] * 1800, dtype=np.float32),
        output_dir=tmp_path / "out",
        location="Adams County, PA",
        config=None,
        lf_plans=[lf_plan],
        per_lf_locations=per_lf_locations,
        per_lf_ost_text=per_lf_ost_text,
        gps_points_by_date=gps_points_by_date,
        state=make_state(ost_audit_confirmed=True, output_base=str(tmp_path), **state_kwargs),
    )


def test_run_day_pipeline_plans_shorts_under_the_gps_gap_veto(tmp_path, render_patches):
    seen = {}

    def _spy(day_scores, *, stop_mask=None, **kwargs):
        seen["mask"] = stop_mask
        raise _PlanReachedError

    with patch.object(day_pipeline, "plan_shorts", side_effect=_spy):
        with pytest.raises(_PlanReachedError):
            _run_day(tmp_path, {_DATE: _trace((0, 20), (200, 1799))})
    mask = np.asarray(seen["mask"], dtype=bool)
    assert mask.shape == (1800,)
    assert mask[21:201].all()  # the dropout: no short may cover it
    assert not mask[:21].any()
    assert not mask[201:].any()
    # The persisted STOP mask stays a stop mask; the veto is planning-only.
    persisted = np.load(tmp_path / "_stages" / f"stop_mask_{_DATE}.npz")["mask"]
    assert not persisted.any()


def test_a_gap_short_on_a_gps_day_fails_before_any_burn(tmp_path, render_patches, burned):
    """A short in the dropout (planner bypassed) never falls back to the
    operator label, destination or county — the day fails before any burn."""
    gap_short = SimpleNamespace(rank=1, start_sec=30.0, end_sec=68.0)
    with patch.object(day_pipeline, "plan_shorts", return_value=[gap_short]):
        with pytest.raises(RuntimeError, match="no GPS fix of its own"):
            _run_day(
                tmp_path,
                {_DATE: _trace((0, 20), (200, 1799))},
                per_lf_ost_text={1: "Gettysburg to Hanover via PA-116, Adams County, PA"},
                per_lf_locations={1: "Adams County, PA"},
                per_lf_resolved_location={1: ResolvedLocation(destination="Gettysburg")},
                per_lf_overrides={1: "Gettysburg Battlefield, PA"},
            )
    assert burned == []
