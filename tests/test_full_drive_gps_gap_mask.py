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
from gomoveshift.pipeline.ingest.day_gpx_slicer import MARKER
from gomoveshift.pipeline.longform_assembler import LongFormAssembler


def _scores(n: int) -> np.ndarray:
    arr = np.zeros((n, 11), dtype=np.float64)
    arr[:, 0] = np.arange(n, dtype=np.float64)
    arr[:, 1] = 0.6
    arr[:, 2] = 0.6
    return arr


def _gps(times, creator=None) -> tuple:
    tps = [
        {"timestamp": float(t), "lat": 32.5 + t * 1e-5, "lon": -80.4, "elevation": 5.0}
        for t in times
    ]
    data = {"trackpoints": tps}
    if creator is not None:
        data["creator"] = creator
    return (data, 0.0)


def _cfg() -> DriveConfig:
    return DriveConfig(
        lf_selection_mode="full_drive",
        full_drive_max_duration_sec=7200,
        min_lf_duration=480.0,
        min_segment_duration=30.0,
        min_motion_energy=0.0,
    )


def _plan(gps_times, n=1800, creator=None):
    return _plan_episodes([_gps(gps_times, creator)] if gps_times is not None else None, n)


def _plan_episodes(gps_per_episode, n=1800):
    """One day of ``n`` seconds; ``gps_per_episode`` may hold several episodes
    in sequence, each ``n // len`` seconds long."""
    k = len(gps_per_episode) if gps_per_episode else 1
    span = n // k
    return LongFormAssembler(_cfg()).build_drive_plan_per_day(
        scores=_scores(n),
        processed_files=[Path(f"/tmp/ep_{i:02d}.mp4") for i in range(k)],
        file_boundaries=[(float(i * span), float((i + 1) * span)) for i in range(k)],
        episode_dates=["2026-02-21"] * k,
        gps_per_episode=gps_per_episode,
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


# --- a day-GPX-sliced track is judged at the day-track gap -------------------
#
# Clonakilty day 12 (2026-09-17): the sliced sidecar had 58 holes of 5-20 s
# (phone logger asleep) and the 5 s rule cut 1085 s, under the part floor. A
# track the slicer wrote (``creator`` = ``day_gpx_slicer.MARKER``) is judged at
# ``route_claims.DAY_TRACK_GAP_S`` (one 30 s evidence bin); the claim gate
# applies the same tolerance to the same track, so nothing passes here that it
# would refuse.


def _phone_holes(hole_len, every=120, n=1800):
    """One fix a second, with a ``hole_len`` s silence every ``every`` s."""
    return [t for t in range(0, n) if t % every >= hole_len]


def test_marked_day_track_with_phone_holes_keeps_the_timeline():
    plan = _plan(_phone_holes(20), creator=MARKER)
    assert plan.regular_lfs[0].total_duration >= 1790.0


def test_the_same_holes_on_an_unmarked_track_are_cut():
    plan = _plan(_phone_holes(20))
    assert plan.regular_lfs[0].total_duration <= 1800.0 - 14 * 20 + 1.0


def test_a_hole_longer_than_the_day_track_gap_is_still_cut():
    plan = _plan(_phone_holes(40, every=600), creator=MARKER)
    for a, b in _spans(plan):
        assert all(b <= h + 1.0 or a >= h + 40 - 1.0 for h in (0, 600, 1200))


def test_a_day_mixing_gpmf_and_day_track_episodes_keeps_the_gpmf_rule():
    # Episode 0 is a day-track with 20 s holes, episode 1 a dense GPMF track:
    # the stricter rule applies to the whole day (fail closed), so the holes cut.
    holes = _gps(_phone_holes(20, n=900), creator=MARKER)
    dense = _gps(range(900, 1800))
    plan = _plan_episodes([holes, dense])
    assert sum(lf.total_duration for lf in plan.regular_lfs) <= 1800.0 - 7 * 20 + 1.0
