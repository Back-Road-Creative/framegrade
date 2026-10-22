"""Clip ordering contract.

Required behaviour: everything chronological by capture time. LFs in
day-temporal order, segments within each LF in source-time order, and shorts
within each LF in source-time order (NOT rank order).

Long-form ordering was already chronological (day-partition + segment
ordering). Shorts shipped in rank order (short_01 = highest score); the pipeline
now sorts each LF's shorts bucket by source-time start_sec at the
``PhaseBCDayFirst`` bucketing site so downstream ``short_index`` assignment
publishes earliest-first.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from _state_helpers import make_state
from gomoveshift.pipeline.drive_phases.phase_bc_day_first import PHASE_BC_CODE_VERSION


def _segment(file_idx: int, start: float = 0.0, end: float = 10.0):
    return SimpleNamespace(
        source_file_index=file_idx,
        source_path=Path(f"/fake/ep_{file_idx:03d}.mp4"),
        start_time=start,
        end_time=end,
        original_start_time=start,
        original_end_time=end,
        composite_score=0.5,
        gps_center_lat=35.0,
        gps_center_lon=-80.0,
        terrain_type="countryside",
        contiguous_with_previous=False,
        avg_motion_energy=0.0,
    )


def _write_finals(day_dir: Path, num_lfs: int, num_shorts: int) -> None:
    lf_dir = day_dir / "regular_lfs"
    sh_dir = day_dir / "shorts"
    lf_dir.mkdir(parents=True, exist_ok=True)
    sh_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, num_lfs + 1):
        (lf_dir / f"lf_{i:02d}.mp4").write_bytes(b"final-lf")
    for i in range(1, num_shorts + 1):
        (sh_dir / f"short_{i:02d}.mp4").write_bytes(b"final-short")


def _build_state(tmp_path: Path, days_to_lf_indices: dict[str, list[int]]):
    """Multi-day drive state. One episode per day, one segment per LF on that day."""
    output_base = tmp_path / "out"
    stages_dir = output_base / "_stages"
    stages_dir.mkdir(parents=True)

    dates = list(days_to_lf_indices.keys())
    episodes: list = []
    episode_files: list[str] = []
    episode_dates: list[str] = []
    file_durations: list[float] = []
    file_boundaries: list[tuple[float, float]] = []
    cursor = 0.0
    per_day = 1500.0
    for i, d in enumerate(dates):
        ep_path = f"/fake/ep_{i:03d}.mp4"
        episode_files.append(ep_path)
        episode_dates.append(d)
        file_durations.append(per_day)
        file_boundaries.append((cursor, cursor + per_day))
        cursor += per_day
        episodes.append(
            SimpleNamespace(
                index=i,
                concat_path=Path(ep_path),
                date=d,
                total_duration=per_day,
                gpx_path=None,
                creation_time=None,
            )
        )
    ingest_plan = SimpleNamespace(episodes=episodes)

    plan_dict = {
        "source_files": episode_files,
        "processed_files": episode_files,
        "regular_lfs": [],
        "score_threshold": 0.0,
        "total_source_duration": sum(file_durations),
        "total_highlight_duration": 0.0,
        "best_moments_filtered": False,
        "plan_code_version": PHASE_BC_CODE_VERSION,
    }
    for ep_idx, d in enumerate(dates):
        for lf_idx in days_to_lf_indices[d]:
            seg = _segment(ep_idx).__dict__.copy()
            seg["source_path"] = str(seg["source_path"])
            seg["duration"] = seg["end_time"] - seg["start_time"]
            plan_dict["regular_lfs"].append(
                {
                    "index": lf_idx,
                    "segments": [seg],
                    "montage_peaks": [seg],
                    "shorts_target": 2,
                    "region": "usa",
                    "total_duration": 10.0,
                    "undersized": False,
                }
            )
    (stages_dir / "drive_plan.json").write_text(json.dumps(plan_dict))

    state = make_state(
        output_base=output_base,
        stages_base=stages_dir,
        ingest_plan=ingest_plan,
        episode_files=episode_files,
        episode_dates=episode_dates,
        episode_locations=[""] * len(dates),
        file_durations=file_durations,
        region="usa",
        location="Test Area",
        drive_scores=None,
        file_boundaries=file_boundaries,
    )
    state.per_lf_overrides = {i: f"Synthetic Route {i}, Test" for i in range(0, 21)}
    return state, output_base, stages_dir


def _run_phase(state, run_all_days_impl):
    from gomoveshift.config import Config
    from gomoveshift.pipeline.drive_phases.phase_bc_day_first import PhaseBCDayFirst

    geocoder = MagicMock()
    geocoder.reverse_geocode.return_value = None

    cfg = Config()
    phase = PhaseBCDayFirst(cfg)
    with (
        patch(
            "gomoveshift.pipeline.gps_enrichment._build_shared_geocoder",
            return_value=geocoder,
        ),
        patch(
            "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
            side_effect=run_all_days_impl,
        ),
    ):
        phase.run(state, probe_duration_fn=lambda p: 100.0)
    return phase


class TestShortsChronologicalWithinLF:
    """Shorts inside an LF's bucket are sorted by source-time start_sec.

    Constructs a fresh-encode run where rank order is the REVERSE of
    chronological order (rank=1 short starts latest, rank=N short starts
    earliest). The contract: ``state['shorts_per_lf'][lf_idx]`` after
    ``PhaseBCDayFirst.run`` must be in ascending ``start_sec`` order, NOT
    rank order.
    """

    def test_single_lf_shorts_sorted_by_start_sec_not_rank(self, tmp_path):
        from gomoveshift.pipeline.day_pipeline import DayArtifacts

        state, _, stages_dir = _build_state(tmp_path, {"2026-02-20": [1]})
        day_dir = stages_dir / "day_2026-02-20"
        _write_finals(day_dir, num_lfs=1, num_shorts=3)

        # Rank order ≠ chronological order:
        # rank=1 (short_01) — latest in day at start_sec=500
        # rank=2 (short_02) — earliest at start_sec=100
        # rank=3 (short_03) — middle at start_sec=300
        short_paths = [
            day_dir / "shorts" / "short_01.mp4",
            day_dir / "shorts" / "short_02.mp4",
            day_dir / "shorts" / "short_03.mp4",
        ]
        short_ranges = [
            SimpleNamespace(rank=1, start_sec=500.0, end_sec=530.0),
            SimpleNamespace(rank=2, start_sec=100.0, end_sec=130.0),
            SimpleNamespace(rank=3, start_sec=300.0, end_sec=330.0),
        ]

        def run_all_days_impl(**kwargs):
            return [
                DayArtifacts(
                    date="2026-02-20",
                    master_path=None,
                    regular_lf_paths=[day_dir / "regular_lfs" / "lf_01.mp4"],
                    short_paths=short_paths,
                    short_ranges=short_ranges,
                )
            ]

        _run_phase(state, run_all_days_impl)

        shorts_per_lf = state["shorts_per_lf"]
        actual_names = [Path(p).name for p in shorts_per_lf[1]]
        # Earliest start_sec first: short_02 (100), short_03 (300), short_01 (500).
        assert actual_names == ["short_02.mp4", "short_03.mp4", "short_01.mp4"], (
            f"Shorts within LF 1 not in chronological order. Got {actual_names}; "
            "The ordering contract requires ascending start_sec, not rank."
        )

    def test_multi_lf_day_each_bucket_independently_sorted(self, tmp_path):
        """Two LFs on the same day each get their own bucket sorted by start_sec."""
        from gomoveshift.pipeline.day_pipeline import DayArtifacts

        state, _, stages_dir = _build_state(tmp_path, {"2026-02-20": [1, 2]})
        day_dir = stages_dir / "day_2026-02-20"
        _write_finals(day_dir, num_lfs=2, num_shorts=4)

        # Need real LF segments that map shorts → LFs by source window.
        # Override plan_dict with non-overlapping LF segment intervals.
        plan_path = stages_dir / "drive_plan.json"
        plan = json.loads(plan_path.read_text())
        # LF 1 spans master [0, 600); LF 2 spans master [600, 1500).
        for lf in plan["regular_lfs"]:
            seg = lf["segments"][0]
            if lf["index"] == 1:
                seg["start_time"] = 0.0
                seg["end_time"] = 600.0
            else:
                seg["start_time"] = 600.0
                seg["end_time"] = 1500.0
            seg["original_start_time"] = seg["start_time"]
            seg["original_end_time"] = seg["end_time"]
            seg["duration"] = seg["end_time"] - seg["start_time"]
            lf["montage_peaks"] = [dict(seg)]
            lf["total_duration"] = seg["end_time"] - seg["start_time"]
        plan_path.write_text(json.dumps(plan))

        short_paths = [day_dir / "shorts" / f"short_{i:02d}.mp4" for i in range(1, 5)]
        # rank=1 → LF2 late (start=1000), rank=2 → LF1 early (start=50),
        # rank=3 → LF2 early (start=700), rank=4 → LF1 late (start=400).
        short_ranges = [
            SimpleNamespace(rank=1, start_sec=1000.0, end_sec=1030.0),
            SimpleNamespace(rank=2, start_sec=50.0, end_sec=80.0),
            SimpleNamespace(rank=3, start_sec=700.0, end_sec=730.0),
            SimpleNamespace(rank=4, start_sec=400.0, end_sec=430.0),
        ]

        def run_all_days_impl(**kwargs):
            return [
                DayArtifacts(
                    date="2026-02-20",
                    master_path=None,
                    regular_lf_paths=[
                        day_dir / "regular_lfs" / "lf_01.mp4",
                        day_dir / "regular_lfs" / "lf_02.mp4",
                    ],
                    short_paths=short_paths,
                    short_ranges=short_ranges,
                )
            ]

        _run_phase(state, run_all_days_impl)

        shorts_per_lf = state["shorts_per_lf"]
        # LF 1: shorts at start_sec 50 then 400 → short_02, short_04.
        lf1_names = [Path(p).name for p in shorts_per_lf[1]]
        assert lf1_names == ["short_02.mp4", "short_04.mp4"], f"LF 1 not chronological: {lf1_names}"
        # LF 2: shorts at start_sec 700 then 1000 → short_03, short_01.
        lf2_names = [Path(p).name for p in shorts_per_lf[2]]
        assert lf2_names == ["short_03.mp4", "short_01.mp4"], f"LF 2 not chronological: {lf2_names}"


class TestLFsChronologicalAcrossDays:
    """Multi-day drive: LF order tracks calendar order, not rank.

    Day-first partition assigns each LF to exactly one date. The drive plan
    keeps LFs in plan-index order, which (per the day-partition step) tracks
    calendar order. This test locks that contract end-to-end via a 3-day
    drive: LF indices 1,2,3 attached to dates 2026-02-20, -21, -22.
    """

    def test_lf_paths_appear_in_day_temporal_order(self, tmp_path):
        from gomoveshift.pipeline.day_pipeline import DayArtifacts

        state, _, stages_dir = _build_state(
            tmp_path,
            {"2026-02-20": [1], "2026-02-21": [2], "2026-02-22": [3]},
        )
        for d in ("2026-02-20", "2026-02-21", "2026-02-22"):
            _write_finals(stages_dir / f"day_{d}", num_lfs=1, num_shorts=1)

        def run_all_days_impl(**kwargs):
            out = []
            for d in ("2026-02-20", "2026-02-21", "2026-02-22"):
                day_dir = stages_dir / f"day_{d}"
                out.append(
                    DayArtifacts(
                        date=d,
                        master_path=None,
                        regular_lf_paths=[day_dir / "regular_lfs" / "lf_01.mp4"],
                        short_paths=[day_dir / "shorts" / "short_01.mp4"],
                        short_ranges=[SimpleNamespace(rank=1, start_sec=10.0, end_sec=40.0)],
                    )
                )
            return out

        _run_phase(state, run_all_days_impl)

        # Each LF lands in its own day-dir; the LF list keys 1→day20,
        # 2→day21, 3→day22 reflects chronological assembly.
        shorts_per_lf = state["shorts_per_lf"]
        assert "day_2026-02-20" in str(shorts_per_lf[1][0])
        assert "day_2026-02-21" in str(shorts_per_lf[2][0])
        assert "day_2026-02-22" in str(shorts_per_lf[3][0])
