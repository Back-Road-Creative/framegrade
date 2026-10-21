"""Resume must keep source-window shorts↔LF attribution.

The documented workflow is a ``--skip-upload`` audit run followed by a
re-run with ``--youtube-upload``: the shipped run is therefore a RESUME.
On resume, ``phase_bc_day_first`` reconstructs each completed day's
``DayArtifacts`` from disk. If that reconstruction drops ``short_ranges``,
the source-window shorts→LF assignment branch is skipped and the code
falls back to the legacy ``divmod`` equal-split — which yields the right
COUNT per LF but arbitrary source attribution (a short whose footage
lives in LF #2 gets filed under LF #1 by rank position).

This locks the fix: on the finals-only resume path, the reconstructed
``short_ranges`` (rebuilt from the ``shorts_ost_audit_<date>.json``
sidecar) drive the source-window bucketing, so a day whose shorts sit
3-in-LF1 / 1-in-LF2 by source window buckets {3, 1}, NOT the {2, 2}
equal-split — and the "legacy equal-split" fallback WARN never fires.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from _state_helpers import make_state
from gomoveshift.pipeline.drive_phases.phase_bc_day_first import PHASE_BC_CODE_VERSION

DATE = "2026-02-20"
EP_DURATION = 1200.0


def _segment(file_idx: int, start_time: float, end_time: float):
    """LFSegment-shaped dict for drive_plan.json (source-time window)."""
    return {
        "source_file_index": file_idx,
        "source_path": f"/fake/ep_{file_idx:03d}.mp4",
        "start_time": start_time,
        "end_time": end_time,
        "duration": end_time - start_time,
        "original_start_time": start_time,
        "original_end_time": end_time,
        "composite_score": 0.5,
        "gps_center_lat": 35.0,
        "gps_center_lon": -80.0,
        "terrain_type": "countryside",
        "contiguous_with_previous": False,
        "avg_motion_energy": 0.0,
    }


def _lf_plan_dict(index: int, start_time: float, end_time: float):
    seg = _segment(0, start_time, end_time)
    return {
        "index": index,
        "segments": [seg],
        "montage_peaks": [seg],
        "shorts_target": 2,
        "region": "usa",
        "total_duration": end_time - start_time,
        "undersized": False,
    }


def _build_resume_state(tmp_path: Path):
    """One day, one episode (0-1200s), two LFs split at 600s:
    LF1 owns master [0, 600), LF2 owns master [600, 1200).

    Four finished shorts on disk; their sidecar windows place three
    inside LF1's source window and one inside LF2's — so source-window
    buckets {3, 1} while equal-split would give {2, 2}.
    """
    output_base = tmp_path / "out"
    stages_dir = output_base / "_stages"
    stages_dir.mkdir(parents=True)

    ep_path = "/fake/ep_000.mp4"
    episode = SimpleNamespace(
        index=0,
        concat_path=Path(ep_path),
        date=DATE,
        total_duration=EP_DURATION,
        gpx_path=None,
        creation_time=None,
    )
    ingest_plan = SimpleNamespace(episodes=[episode])

    plan_dict = {
        "source_files": [ep_path],
        "processed_files": [ep_path],
        "regular_lfs": [
            _lf_plan_dict(1, 0.0, 600.0),
            _lf_plan_dict(2, 600.0, 1200.0),
        ],
        "score_threshold": 0.0,
        "total_source_duration": EP_DURATION,
        "total_highlight_duration": 0.0,
        "best_moments_filtered": False,
        "plan_code_version": PHASE_BC_CODE_VERSION,
    }
    (stages_dir / "drive_plan.json").write_text(json.dumps(plan_dict))

    # On-disk finals for the finals-only resume fast-path.
    day_dir = stages_dir / f"day_{DATE}"
    lf_dir = day_dir / "regular_lfs"
    sh_dir = day_dir / "shorts"
    lf_dir.mkdir(parents=True)
    sh_dir.mkdir(parents=True)
    (lf_dir / "lf_01.mp4").write_bytes(b"lf1")
    (lf_dir / "lf_02.mp4").write_bytes(b"lf2")
    for rank in (1, 2, 3, 4):
        (sh_dir / f"short_{rank:02d}.mp4").write_bytes(b"short")

    # Sidecar carrying each short's source window (day-master timeline).
    # ranks 1,2,4 → LF1 window; rank 3 → LF2 window. Note rank 4 lands in
    # LF1 even though equal-split would file it under LF2 by position.
    sidecar = {
        "date": DATE,
        "generated_at": "2026-02-20T00:00:00+00:00",
        "shorts": [
            {"rank": 1, "parent_lf_index": 1, "ost_text": "", "start_sec": 80.0, "end_sec": 120.0},
            {"rank": 2, "parent_lf_index": 1, "ost_text": "", "start_sec": 280.0, "end_sec": 320.0},
            {"rank": 3, "parent_lf_index": 2, "ost_text": "", "start_sec": 880.0, "end_sec": 920.0},
            {"rank": 4, "parent_lf_index": 1, "ost_text": "", "start_sec": 480.0, "end_sec": 520.0},
        ],
    }
    (stages_dir / f"shorts_ost_audit_{DATE}.json").write_text(json.dumps(sidecar))

    state = make_state(
        output_base=output_base,
        stages_base=stages_dir,
        ingest_plan=ingest_plan,
        episode_files=[ep_path],
        episode_dates=[DATE],
        episode_locations=[""],
        file_durations=[EP_DURATION],
        region="usa",
        location="Test Area",
        drive_scores=None,
        file_boundaries=[(0.0, EP_DURATION)],
    )
    # GPS-less synthetic plan — blanket per-LF overrides so locations resolve.
    state.per_lf_overrides = {i: f"Synthetic Route {i}, Test" for i in range(0, 21)}
    # master_path=None → finals-only resume fast-path (#2).
    state.per_day_completed = {DATE: {"master_path": None, "master_sha": ""}}
    return state


def _run_phase(state):
    from gomoveshift.config import Config
    from gomoveshift.pipeline.drive_phases.phase_bc_day_first import PhaseBCDayFirst

    geocoder = MagicMock()
    geocoder.reverse_geocode.return_value = None

    def _run_all_days_impl(**kwargs):
        return []  # every day is a resume-skip; no fresh work

    cfg = Config()
    phase = PhaseBCDayFirst(cfg)
    with (
        patch(
            "gomoveshift.pipeline.gps_enrichment._build_shared_geocoder",
            return_value=geocoder,
        ),
        patch(
            "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
            side_effect=_run_all_days_impl,
        ),
    ):
        phase.run(state, probe_duration_fn=lambda p: 100.0)
    return phase


def test_resume_keeps_source_window_buckets_not_equal_split(tmp_path, caplog):
    state = _build_resume_state(tmp_path)

    with caplog.at_level(logging.WARNING):
        _run_phase(state)

    shorts_per_lf = state.get("shorts_per_lf")
    assert isinstance(shorts_per_lf, dict)
    assert set(shorts_per_lf.keys()) == {1, 2}

    lf1 = {Path(p).name for p in shorts_per_lf[1]}
    lf2 = {Path(p).name for p in shorts_per_lf[2]}

    # Source-window assignment: 3 shorts under LF1, 1 under LF2.
    # Equal-split (the resume regression) would give {2, 2}.
    assert lf1 == {"short_01.mp4", "short_02.mp4", "short_04.mp4"}, lf1
    assert lf2 == {"short_03.mp4"}, lf2

    # The legacy equal-split fallback must NOT have fired on resume.
    assert not any(
        "falling back to legacy equal-split" in rec.getMessage() for rec in caplog.records
    ), [rec.getMessage() for rec in caplog.records if "equal-split" in rec.getMessage()]


def test_resume_without_sidecar_hard_fails_instead_of_equal_split(tmp_path):
    """Ambiguous shorts↔LF
    attribution (no short_ranges, >1 LF) must hard-fail — the equal-split
    guess linked shorts to the wrong parent LF (wrong first-comment link,
    inherited recording date, trim ratio)."""
    import pytest

    state = _build_resume_state(tmp_path)
    (Path(state.stages_base) / f"shorts_ost_audit_{DATE}.json").unlink()

    with pytest.raises(RuntimeError, match="cannot attribute"):
        _run_phase(state)
