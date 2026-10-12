"""Regression: trim-ratio source durations must index in the DAY-LOCAL
episode space, not the drive-wide global space.

``state.file_durations`` is DRIVE-WIDE (a flat list indexed by the global
episode index). ``_partition_lf_plans_by_date`` remaps every LF segment's
``source_file_index`` from GLOBAL to DAY-LOCAL. So for any LF on day 2+ of a
multi-day drive, indexing the drive-wide ``file_durations`` with a day-local
``source_file_index`` reads a DIFFERENT day's episode duration — inflating the
denominator and producing the wrong trim ratio (hence the wrong title verb,
wrong ``🎬 Final cut`` disclosure, wrong shorts lead band).

Single-day drives are coincidentally correct (day-local == global), which is
why the older single-day tests miss this. These tests use a 2-day drive with
DIFFERENT per-day episode durations so the day-2 denominator is provably wrong
under a drive-wide read.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from _state_helpers import make_state

DAY1 = "2025-09-14"
DAY2 = "2025-09-15"


def _seg(source_file_index: int):
    return SimpleNamespace(
        source_file_index=source_file_index,
        source_path=f"/tmp/ep_{source_file_index}.mp4",
        start_time=0.0,
        end_time=10.0,
        original_start_time=0.0,
        original_end_time=10.0,
    )


def _lf(index: int, segments, total_duration: float):
    return SimpleNamespace(
        index=index,
        segments=segments,
        montage_peaks=[],
        total_duration=total_duration,
    )


def _multiday_state():
    # Drive-wide (global) index space:
    #   global 0,1 → day 1 episodes (1000s, 2000s)
    #   global 2,3 → day 2 episodes ( 500s,  800s)
    return make_state(
        file_durations=[1000.0, 2000.0, 500.0, 800.0],
        episode_dates=[DAY1, DAY1, DAY2, DAY2],
        per_lf_recording_date={1: DAY1, 2: DAY2},
        per_lf_gpx_path={},  # GPS-less → distance/elev/regions invariants skip
        per_lf_locations={},
        location="",
    )


def test_fresh_compute_uses_day_local_durations(tmp_path):
    """Fresh-encode path: a day-2 LF's segments carry DAY-LOCAL
    ``source_file_index`` (post-partition). Its trim ratio must divide by
    its OWN day-2 source durations (500 + 800 = 1300), NOT day-1's
    (1000 + 2000 = 3000) that share the same day-local indices 0 and 1.
    """
    from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
        _assert_post_encoding_invariants,
    )

    state = _multiday_state()
    skeleton_lfs = [
        # Day 1 LF: day-local sfi 0 → 1000s. 300/1000 = 0.30 (also correct today).
        _lf(1, [_seg(0)], total_duration=300.0),
        # Day 2 LF: day-local sfi 0,1 → 500+800 = 1300s. 260/1300 = 0.20.
        _lf(2, [_seg(0), _seg(1)], total_duration=260.0),
    ]

    _assert_post_encoding_invariants(
        state=state,
        skeleton_lfs=skeleton_lfs,
        regular_lf_paths=[tmp_path / "lf_01.mp4", tmp_path / "lf_02.mp4"],
    )

    assert state.per_lf_trim_ratio[1] == pytest.approx(0.30, abs=1e-4)
    # Drive-wide read gives 260/3000 ≈ 0.0867 — the aliasing bug.
    assert state.per_lf_trim_ratio[2] == pytest.approx(260.0 / 1300.0, abs=1e-4)


def test_day_local_source_durations_slices_by_day():
    """The helper buckets the drive-wide durations by day and returns each
    LF's OWN day's slice in day-local order."""
    from gomoveshift.pipeline.post_encoding_invariants import (
        _day_local_source_durations,
    )

    out = _day_local_source_durations(
        per_lf_recording_date={1: DAY1, 2: DAY2},
        episode_dates=[DAY1, DAY1, DAY2, DAY2],
        file_durations=[1000.0, 2000.0, 500.0, 800.0],
    )
    assert out[1] == [1000.0, 2000.0]
    assert out[2] == [500.0, 800.0]


def test_day_local_source_durations_empty_when_inputs_missing():
    """No episode_dates / recording_date → empty map so the compute helpers
    fall back to the drive-wide list (single-day / legacy resume)."""
    from gomoveshift.pipeline.post_encoding_invariants import (
        _day_local_source_durations,
    )

    assert _day_local_source_durations({}, [DAY1], [10.0]) == {}
    assert _day_local_source_durations({1: DAY1}, [], [10.0]) == {}


def test_compute_helper_honors_per_lf_source_durations():
    """Direct: a day-2 LF whose per-LF slice is supplied divides by that
    slice, not the drive-wide list at the same day-local indices."""
    from gomoveshift.pipeline.post_encoding_invariants import (
        _compute_per_lf_trim_ratio,
    )

    lfs = [_lf(2, [_seg(0), _seg(1)], total_duration=260.0)]
    file_durations = [1000.0, 2000.0, 500.0, 800.0]  # drive-wide
    result = _compute_per_lf_trim_ratio(
        lfs, file_durations, per_lf_source_durations={2: [500.0, 800.0]}
    )
    assert result[2] == pytest.approx(260.0 / 1300.0, abs=1e-4)


def test_resume_backstop_uses_day_local_durations(tmp_path):
    """Resume backstop: same multi-day shape. Fresh compute drops both LFs
    (total_duration=0 → simulated resume-rehydrated skeleton), the ffprobe
    backstop fills them from finals on disk, and the day-2 LF must still
    divide by its OWN day-2 source durations.
    """
    from gomoveshift.pipeline.drive_phases import phase_bc_day_first as _mod

    lf1 = tmp_path / f"day_{DAY1}" / "regular_lfs" / "lf_01.mp4"
    lf2 = tmp_path / f"day_{DAY2}" / "regular_lfs" / "lf_02.mp4"
    lf1.parent.mkdir(parents=True)
    lf2.parent.mkdir(parents=True)
    lf1.write_bytes(b"x")
    lf2.write_bytes(b"x")

    state = _multiday_state()
    # total_duration=0 → fresh compute omits both → backstop path runs.
    skeleton_lfs = [_lf(1, [_seg(0)], 0.0), _lf(2, [_seg(0), _seg(1)], 0.0)]

    original_probe = _mod._probe_duration
    _mod._probe_duration = lambda p: 300.0 if "lf_01" in str(p) else 260.0
    try:
        _mod._assert_post_encoding_invariants(
            state=state,
            skeleton_lfs=skeleton_lfs,
            regular_lf_paths=[lf1, lf2],
        )
    finally:
        _mod._probe_duration = original_probe

    assert state.per_lf_trim_ratio[1] == pytest.approx(0.30, abs=1e-4)
    assert state.per_lf_trim_ratio[2] == pytest.approx(260.0 / 1300.0, abs=1e-4)
