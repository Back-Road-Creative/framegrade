"""A full_drive day too short for a full-drive long-form ships as a best_moments long-form.

rec0060-0061 (job 9ee99a65): 2889 s kept against full_drive_part_min_sec=3600 failed the whole
drive forever. The day is now planned with the best_moments selection (same scores and stop
mask, within the day), logged, and recorded on the day's LF entry in drive_plan.json.
"""

from pathlib import Path

import numpy as np

from gomoveshift.config import DriveConfig
from gomoveshift.models import RegularLFPlan
from gomoveshift.pipeline.longform_assembler import LongFormAssembler

MIN, MAX = 3600, 5400


def _scores(n):
    arr = np.zeros((n, 11))
    arr[:, 0] = np.arange(n)
    arr[:, 1] = 0.5
    arr[:, 2] = 0.6
    return arr


def _cfg():
    return DriveConfig(
        lf_selection_mode="full_drive",
        full_drive_max_duration_sec=7200,
        min_lf_duration=480.0,
        min_segment_duration=30.0,
        min_motion_energy=0.0,
        full_drive_part_min_sec=MIN,
        full_drive_part_max_sec=MAX,
    )


def _build(lengths, mask=None):
    bounds, off = [], 0.0
    for n in lengths:
        bounds.append((off, off + n))
        off += n
    total = int(off)
    return LongFormAssembler(_cfg()).build_drive_plan_per_day(
        scores=_scores(total),
        processed_files=[Path(f"/tmp/ep_{i:02d}.mp4") for i in range(len(lengths))],
        file_boundaries=bounds,
        episode_dates=[f"2026-03-0{i + 4}" for i in range(len(lengths))],
        stop_mask=np.zeros(total, dtype=bool) if mask is None else mask,
    )


def test_a_short_day_is_a_best_moments_lf_recorded_in_the_plan(caplog):
    with caplog.at_level("WARNING"):
        plan = _build([2889])
    assert len(plan.regular_lfs) == 1
    lf = plan.regular_lfs[0]
    assert lf.lf_mode == "best_moments"
    assert lf.lf_mode_reason == "kept 2889s under full_drive_part_min_sec=3600s"
    assert 480 <= lf.total_duration <= 2889
    assert "day 2026-03-04: kept 2889s under full_drive_part_min_sec=3600s -> best_moments" in (
        caplog.text
    )
    d = lf.to_dict()
    assert d["lf_mode"] == "best_moments" and "3600s" in d["lf_mode_reason"]
    assert RegularLFPlan.from_dict(d).lf_mode == "best_moments"


def test_the_fallback_lf_stays_inside_its_own_day():
    plan = _build([2889, 4000])
    short, whole = plan.regular_lfs
    assert short.lf_mode == "best_moments"
    assert all(0.0 <= s.original_start_time and s.original_end_time <= 2889 for s in short.segments)
    # the day that meets the floor is unchanged
    assert whole.lf_mode is None and "lf_mode" not in whole.to_dict()
    assert all(s.original_start_time >= 2889 for s in whole.segments)


def test_the_fallback_cuts_stopped_seconds():
    mask = np.zeros(2889, dtype=bool)
    mask[500:900] = True
    lf = _build([2889], mask).regular_lfs[0]
    assert lf.lf_mode == "best_moments"
    for s in lf.segments:
        assert not mask[int(s.original_start_time) : int(s.original_end_time)].any()
