"""``_synthetic_score_rows`` motion energy must not fabricate stops.

The labels-only synthetic score chunk left ``COL_MOTION_ENERGY`` at 0.
``build_stop_mask`` falls back to optical motion < threshold when GPS speed
is unmeasurable (fewer than 2 fixes per 5s window), so every GPS-less second
of a synthetic run was masked as a stop -- the plan builder then raised or
returned no LF, and a ``--label-dry-run``/``--list-lfs`` invocation printed
"no LFs" for a drive that has real LFs once actually scored.

Pinned here: a synthetic full_drive plan (GPS-less drive, one GPS-less
episode among others, and a GPS tunnel/fix-loss gap) must match the plan a
motion-bearing scoring pass would produce.
"""

from types import SimpleNamespace

import numpy as np

from gomoveshift.analysis.shorts_models import GPSData, GPSTrackpoint
from gomoveshift.analysis.unified_scorer import COL_MOTION_ENERGY, UnifiedSegmentScorer
from gomoveshift.config import Config
from gomoveshift.pipeline.drive_phases.phase_a import _synthetic_score_rows
from gomoveshift.pipeline.drive_phases.phase_bc_day_first import _build_drive_plan_from_state

T0 = 1_790_000_000.0


def _gps(duration, gap=None, lat0=35.0, lon0=-80.0):
    tps = []
    for t in range(int(duration)):
        if gap and gap[0] <= t < gap[1]:
            continue
        tps.append(
            GPSTrackpoint(lat=lat0 + t * 0.00018, lon=lon0, elevation=None, timestamp=T0 + t)
        )
    return GPSData(trackpoints=tps)


def _state(durs, gps, motion):
    scorer = UnifiedSegmentScorer(Config().drive.scoring_weights)
    rows = [_synthetic_score_rows(d) for d in durs]
    for r in rows:
        r[:, COL_MOTION_ENERGY] = motion
    scores = scorer.merge_file_scores(rows, durs)
    fb, off = [], 0.0
    for d in durs:
        fb.append((off, off + d))
        off += d
    return SimpleNamespace(
        drive_scores=scores,
        file_boundaries=fb,
        episode_files=[f"/x/ep{i}.mp4" for i in range(len(durs))],
        episode_dates=["2026-09-26"] * len(durs),
        region="usa",
        gps_per_episode=gps,
    )


def _summ(plan):
    if plan is None:
        return None
    return [
        (
            lf.index,
            [(s.source_file_index, round(s.start_time), round(s.end_time)) for s in lf.segments],
        )
        for lf in plan.regular_lfs
    ]


def _assert_synthetic_matches_real(durs, gps):
    cfg = Config()
    cfg.drive.lf_selection_mode = "full_drive"
    real = _summ(_build_drive_plan_from_state(cfg, _state(durs, gps, 0.2)))
    synthetic = _summ(_build_drive_plan_from_state(cfg, _state(durs, gps, 1.0)))
    assert synthetic == real


class TestSyntheticScoreRowsMotionEnergy:
    def test_pinned_to_one(self):
        rows = _synthetic_score_rows(10.0)
        assert np.all(rows[:, COL_MOTION_ENERGY] == 1.0)

    def test_gps_less_drive_matches_real_plan(self):
        durs = [900.0, 900.0]
        gps = [(None, 0.0), (None, 0.0)]
        _assert_synthetic_matches_real(durs, gps)

    def test_one_gps_less_episode_matches_real_plan(self):
        durs = [900.0, 900.0]
        gps = [(_gps(900), T0), (None, 0.0)]
        _assert_synthetic_matches_real(durs, gps)

    def test_gps_tunnel_gaps_match_real_plan(self):
        durs = [900.0, 900.0]
        gps = [(_gps(900, gap=(500, 620)), T0), (_gps(900, gap=(300, 330)), T0)]
        _assert_synthetic_matches_real(durs, gps)
