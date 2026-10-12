"""Resume must preserve the already-burned per-LF OST text.

On resume, a day recorded in ``per_day_completed`` (its LFs already NVENC-burned
to disk) has FIXED OST pixels. Phase BC nonetheless re-derives ``per_lf_ost_text``
for EVERY LF from a fresh geocode pass and overwrote ``state.per_lf_ost_text``
wholesale. After a geocode-cache roll or a code change the fresh text can DIFFER
from what was burned — and Phase D reads ``state.per_lf_ost_text`` for both the
YouTube title and the thumbnail, so a resume shipped titles/thumbnails that
diverged from the burned frames.

Fix: for LFs on a completed (skipped, not re-encoded) day, PREFER the persisted
(confirmed-burned) OST text; only un-encoded days re-derive.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from _state_helpers import make_state

# Reuse the drive_plan / segment fixture shape from the phase's own suite.
from test_phase_bc_day_first import _detailed_by_lat, _make_real_lf_dict, _seg_dict

BURNED_SENTINEL = "Killarney to Dingle via N86, Kerry"


def _write_drive_plan(stages_dir: Path, lfs: list) -> Path:
    payload = {
        "source_files": [],
        "processed_files": [],
        "regular_lfs": lfs,
        "score_threshold": 0.0,
        "total_source_duration": 0.0,
        "total_highlight_duration": 0.0,
        "highlight_ratio": 0.0,
        "best_moments_filtered": False,
    }
    path = stages_dir / "drive_plan.json"
    path.write_text(json.dumps(payload))
    return path


def _one_lf_day_state(tmp_path, *, persisted_ost, per_day_completed=None):
    output_base = tmp_path / "out"
    stages_dir = output_base / "_stages"
    stages_dir.mkdir(parents=True)

    # One LF (index 1) whose footage lives on episode 0 (day 2026-02-20).
    _write_drive_plan(stages_dir, [_make_real_lf_dict(1, [_seg_dict(0, 100, 200, 35.0, -80.0)])])

    ep = SimpleNamespace(
        index=0,
        concat_path=Path("/fake/ep_000.mp4"),
        date="2026-02-20",
        total_duration=1500.0,
        gpx_path=None,
        creation_time=None,
    )
    state = make_state(
        output_base=output_base,
        ingest_plan=SimpleNamespace(episodes=[ep]),
        episode_files=["/fake/ep_000.mp4"],
        episode_dates=["2026-02-20"],
        episode_locations=[""],
        file_durations=[1500.0],
        region="usa",
        location="Test Trip",
        drive_scores=None,
        file_boundaries=[(0.0, 1500.0)],
        per_lf_ost_text=dict(persisted_ost),
    )
    if per_day_completed is not None:
        state.per_day_completed = per_day_completed
    return state, output_base, stages_dir


def _geocoder():
    # reverse_geocode → None so build_lf_title falls back to the derived
    # county label; the vote path uses reverse_geocode_detailed. The fresh
    # label is county-based ("Buncombe County, NC"), never the burned sentinel.
    geocoder = MagicMock()
    geocoder.reverse_geocode.return_value = None
    geocoder.reverse_geocode_detailed.side_effect = _detailed_by_lat
    return geocoder


def _run_phase(state, run_all_days_impl):
    from gomoveshift.config import Config
    from gomoveshift.pipeline.drive_phases.phase_bc_day_first import PhaseBCDayFirst

    phase = PhaseBCDayFirst(Config())
    with (
        patch(
            "gomoveshift.pipeline.gps_enrichment._build_shared_geocoder",
            return_value=_geocoder(),
        ),
        # Destination POI resolution hits Overpass (network) — stub it out so
        # the test is deterministic and offline; it never affects OST preference.
        patch(
            "gomoveshift.pipeline.gps_enrichment._resolve_lf_destination",
            return_value=None,
        ),
        patch(
            "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
            side_effect=run_all_days_impl,
        ),
    ):
        phase.run(state, probe_duration_fn=lambda p: 100.0)


def test_resume_preserves_burned_ost_for_completed_day(tmp_path):
    """A day already in per_day_completed (finals on disk, NOT re-encoded)
    keeps its persisted burned OST text — the fresh divergent re-derivation
    must NOT overwrite it, so Phase D's title/thumbnail match the burned frames.
    """
    per_day_completed = {
        "2026-02-20": {
            "master_path": str(tmp_path / "out" / "_stages" / "day_2026-02-20_dropped.mp4")
        }
    }
    state, _output_base, stages_dir = _one_lf_day_state(
        tmp_path, persisted_ost={1: BURNED_SENTINEL}, per_day_completed=per_day_completed
    )

    # Finals-only resume fast-path: master dropped, LF + short finals on disk.
    day_dir = stages_dir / "day_2026-02-20"
    (day_dir / "regular_lfs").mkdir(parents=True)
    (day_dir / "shorts").mkdir(parents=True)
    (day_dir / "regular_lfs" / "lf_01.mp4").write_bytes(b"lf" * 100)
    (day_dir / "shorts" / "short_01.mp4").write_bytes(b"short" * 50)

    calls = {"batches": []}

    def _fake_run_all_days(**kwargs):
        calls["batches"].extend(b.date for b in kwargs.get("day_batches") or [])
        return []

    _run_phase(state, _fake_run_all_days)

    # The completed day was resumed from finals — never re-encoded.
    assert calls["batches"] == []
    # Burned OST text is preserved; the fresh county-level re-derivation did NOT
    # overwrite it (that would ship a divergent title/thumbnail).
    assert state.per_lf_ost_text[1] == BURNED_SENTINEL


def test_resume_re_derives_ost_for_unencoded_day(tmp_path):
    """A day NOT in per_day_completed (re-encoded this run) still re-derives:
    a stale persisted OST value must be replaced by the fresh derivation, so
    the fix never freezes text for genuinely re-burned LFs.
    """
    state, _output_base, stages_dir = _one_lf_day_state(
        tmp_path, persisted_ost={1: BURNED_SENTINEL}, per_day_completed=None
    )

    def _fake_run_all_days(**kwargs):
        artifacts = []
        for batch in kwargs["day_batches"]:
            day_dir = stages_dir / f"day_{batch.date}"
            (day_dir / "regular_lfs").mkdir(parents=True, exist_ok=True)
            lf = day_dir / "regular_lfs" / "lf_01.mp4"
            lf.write_bytes(b"lf" * 100)
            master = stages_dir / f"day_{batch.date}_master.mp4"
            master.write_bytes(b"m")
            artifacts.append(
                SimpleNamespace(
                    date=batch.date,
                    master_path=master,
                    regular_lf_paths=[lf],
                    regular_lf_indices=[1],
                    short_paths=[],
                    short_ranges=[],
                )
            )
        return artifacts

    _run_phase(state, _fake_run_all_days)

    # Re-encoded day: fresh derivation wins, stale burned sentinel is gone.
    assert state.per_lf_ost_text[1] != BURNED_SENTINEL
    assert state.per_lf_ost_text[1]  # non-empty fresh label
