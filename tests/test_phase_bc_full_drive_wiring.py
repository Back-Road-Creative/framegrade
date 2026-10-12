"""phase_bc → LongFormAssembler full-drive wiring integration guard.

The bug this closes: ``config.drive.lf_selection_mode == "full_drive"``
was documented but never read by ``_build_drive_plan_from_state``. The day-
first phase called ``LongFormAssembler.build_drive_plan_per_day`` which
unconditionally ran the adaptive-threshold (best-moments) search.

A follow-up bug: once wired, that full_drive branch
called ``LongFormAssembler.build_drive_plan_per_day``/``build_drive_plan``
with NO ``stop_mask``, so the planner (which requires one) raised and every
full_drive run loud-failed at ``_assert_drive_plan_built`` (a full-drive
listing returned 0 LFs).
``_build_drive_plan_from_state`` now builds the drive-wide stop mask itself
(GPS speed, optical-motion fallback — same knobs as ``run_day_pipeline``)
and threads it into both assembler paths, and the full-drive segments honor
the planner's unmasked ranges so parked spans are physically cut.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gomoveshift.analysis.shorts_models import GPSData, GPSTrackpoint
from gomoveshift.config import Config, DriveConfig
from gomoveshift.pipeline.day_grouper import DayBatch
from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    _assert_drive_plan_built,
    _build_drive_plan_from_state,
)

from _state_helpers import make_state

_PHASE_BC_LOGGER = "gomoveshift.pipeline.drive_phases.phase_bc_day_first"


def _dense_scores(duration_sec: int, start_offset: float = 0.0) -> np.ndarray:
    n = int(duration_sec)
    arr = np.zeros((n, 11), dtype=np.float64)
    arr[:, 0] = np.arange(n, dtype=np.float64) + start_offset
    arr[:, 1] = 0.5 + 0.3 * np.sin(np.linspace(0, 6.28, n))
    arr[:, 2] = 0.6
    return arr


def _full_drive_config(*, adaptive: bool) -> Config:
    cfg = Config()
    cfg.drive = DriveConfig(
        lf_selection_mode="full_drive",
        full_drive_max_duration_sec=7200,
        adaptive_threshold_enabled=adaptive,
        min_lf_duration=480.0,
        min_segment_duration=30.0,
        min_motion_energy=0.0,
    )
    return cfg


def _trackpoint(t: float, lat: float, lon: float) -> GPSTrackpoint:
    return GPSTrackpoint(lat=lat, lon=lon, elevation=None, timestamp=t)


def _moving_trace_with_stop(*, duration: int, stop_start: int, stop_end: int, t0: float) -> GPSData:
    """1 Hz trace moving ~15 m/s except a parked span [stop_start, stop_end)."""
    pts: list[GPSTrackpoint] = []
    lat = 51.5
    for i in range(duration):
        if not (stop_start <= i < stop_end):
            lat += 0.000135  # ~15 m per second northward
        pts.append(_trackpoint(t0 + i, lat, -9.5))
    return GPSData(trackpoints=pts)


@pytest.mark.parametrize("adaptive", [True, False])
def test_full_drive_wiring_builds_mask_and_plans_one_lf(adaptive: bool, tmp_path: Path):
    """No GPS → optical fallback mask (all moving here) → one whole-timeline LF.
    Both the per-day path (adaptive=True) and the legacy path must plan."""
    cfg = _full_drive_config(adaptive=adaptive)
    ep = tmp_path / "ep_00.mp4"
    ep.write_bytes(b"")
    state = make_state(
        drive_scores=_dense_scores(1800),
        file_boundaries=[(0.0, 1800.0)],
        episode_files=[str(ep)],
        episode_dates=["2025-09-14"],
        region="ireland",
        gps_per_episode=None,
    )
    plan = _build_drive_plan_from_state(cfg, state)
    assert plan is not None, "full_drive wiring must build the stop mask itself and plan"
    assert len(plan.regular_lfs) == 1
    assert plan.regular_lfs[0].total_duration == pytest.approx(1799.5, abs=1.0)
    day_batches = [DayBatch(date="2025-09-14", episodes=[SimpleNamespace(index=1)])]
    _assert_drive_plan_built(plan, state, day_batches)  # must not raise


def test_full_drive_wiring_cuts_parked_span_from_gps(tmp_path: Path):
    """A 120 s parked span in the GPS trace is masked and physically cut out of
    the LF: two segments, the kept duration ~ 1800 - 120, second not contiguous."""
    cfg = _full_drive_config(adaptive=True)
    ep = tmp_path / "ep_00.mp4"
    ep.write_bytes(b"")
    t0 = 1_700_000_000.0
    gps = _moving_trace_with_stop(duration=1800, stop_start=600, stop_end=720, t0=t0)
    state = make_state(
        drive_scores=_dense_scores(1800),
        file_boundaries=[(0.0, 1800.0)],
        episode_files=[str(ep)],
        episode_dates=["2025-09-14"],
        region="ireland",
        gps_per_episode=[(gps, t0)],
    )
    plan = _build_drive_plan_from_state(cfg, state)
    assert plan is not None
    assert len(plan.regular_lfs) == 1
    lf = plan.regular_lfs[0]
    assert len(lf.segments) == 2, [
        (s.original_start_time, s.original_end_time) for s in lf.segments
    ]
    first, second = lf.segments
    assert first.original_end_time == pytest.approx(600.0, abs=5.0)
    assert second.original_start_time == pytest.approx(720.0, abs=5.0)
    assert not second.contiguous_with_previous
    kept = sum(s.duration for s in lf.segments)
    assert kept == pytest.approx(1680.0, abs=10.0)


def test_stop_mask_kwargs_reads_top_level_stop_detection():
    """``stop_detection`` lives on the top-level Config, not ``config.drive``;
    the shared kwargs helper must read it there (day_pipeline used to look on
    ``config.drive`` and silently fell back to the hard-coded defaults)."""
    from gomoveshift.config import StopDetectionConfig
    from gomoveshift.pipeline.stop_mask import stop_mask_kwargs

    cfg = Config()
    cfg.stop_detection = StopDetectionConfig(min_stop_duration=7.0, motion_threshold=0.02)
    cfg.stop_detection.gps_speed_corroboration.stop_speed_kmh = 12.0
    kw = stop_mask_kwargs(cfg)
    assert kw == {
        "stop_speed_kmh": 12.0,
        "min_stop_duration_s": 7.0,
        "optical_motion_threshold": 0.02,
    }


def test_best_moments_default_does_not_trigger_full_drive_path(tmp_path: Path):
    """Regression: default (best_moments) must NOT hit the full_drive branch."""
    cfg = Config()
    cfg.drive = DriveConfig(
        lf_selection_mode="best_moments",
        adaptive_threshold_enabled=True,
        min_lf_duration=480.0,
        min_segment_duration=30.0,
        min_motion_energy=0.0,
    )
    ep = tmp_path / "ep_00.mp4"
    ep.write_bytes(b"")
    state = make_state(
        drive_scores=_dense_scores(1800),
        file_boundaries=[(0.0, 1800.0)],
        episode_files=[str(ep)],
        episode_dates=["2025-09-14"],
        region="ireland",
        gps_per_episode=None,
    )
    plan = _build_drive_plan_from_state(cfg, state)
    assert plan is not None
    # best_moments path goes through adaptive threshold + group_into_regular_lfs;
    # the precise LF count depends on scoring but it must NOT be the single
    # full-timeline LF the full_drive path produces.
    if len(plan.regular_lfs) == 1:
        lf = plan.regular_lfs[0]
        # If best_moments happens to produce one LF, it must be a highlights cut
        # (sum of high-score segments) — strictly less than the full timeline.
        assert lf.total_duration < 1799.0, (
            f"best_moments must not emit a full-timeline LF: total={lf.total_duration}"
        )
