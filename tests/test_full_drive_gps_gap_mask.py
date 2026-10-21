"""full_drive planning never retains a span the route-claim gate would refuse.

A place named in viewer text needs every retained second provably inside it, so
spans with no GPS fix longer than ``route_claims.GAP_S`` are masked out of the
full_drive LF ranges, upstream of the gate (Ace Basin 2026-10-21: episode_01 had
no GPS for its first ~718 s and the gate refused every label).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gomoveshift.config import DriveConfig
from gomoveshift.pipeline.longform_assembler import LongFormAssembler


def _scores(n: int) -> np.ndarray:
    arr = np.zeros((n, 11), dtype=np.float64)
    arr[:, 0] = np.arange(n, dtype=np.float64)
    arr[:, 1] = 0.6
    arr[:, 2] = 0.6
    return arr


def _gps(times) -> tuple:
    tps = [
        {"timestamp": float(t), "lat": 32.5 + t * 1e-5, "lon": -80.4, "elevation": 5.0}
        for t in times
    ]
    return ({"trackpoints": tps}, 0.0)


def _cfg() -> DriveConfig:
    return DriveConfig(
        lf_selection_mode="full_drive",
        full_drive_max_duration_sec=7200,
        min_lf_duration=480.0,
        min_segment_duration=30.0,
        min_motion_energy=0.0,
    )


def _plan(gps_times, n=1800):
    return LongFormAssembler(_cfg()).build_drive_plan_per_day(
        scores=_scores(n),
        processed_files=[Path("/tmp/ep_00.mp4")],
        file_boundaries=[(0.0, float(n))],
        episode_dates=["2026-02-21"],
        gps_per_episode=[_gps(gps_times)] if gps_times is not None else None,
        stop_mask=np.zeros(n, dtype=bool),
    )


def _spans(plan):
    return [(s.start_time, s.end_time) for s in plan.regular_lfs[0].segments]


def test_leading_gps_less_span_is_not_retained():
    plan = _plan(range(700, 1800, 2))
    assert min(a for a, _ in _spans(plan)) >= 700.0
    assert plan.regular_lfs[0].total_duration <= 1100.0


def test_mid_drive_gap_over_allowance_is_cut_out():
    times = [t for t in range(0, 1800, 2) if not 600 <= t < 900]
    for a, b in _spans(_plan(times)):
        assert b <= 601.0 or a >= 899.0


def test_dense_gps_keeps_whole_timeline():
    assert _plan(range(0, 1800, 2)).regular_lfs[0].total_duration >= 1790.0


def test_gpsless_day_is_unchanged():
    assert _plan(None).regular_lfs[0].total_duration >= 1790.0
