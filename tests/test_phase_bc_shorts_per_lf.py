"""Regression tests locking the ``state["shorts_per_lf"]`` contract.

Bug context:
``phase_d.py`` looks for shorts under ``<output_base>/shorts/lf_NN/``, but the
day-first architecture writes them to ``<cache>/_stages/day_<DATE>/shorts/
short_NN.mp4``. phase_d READS the canonical
path list from ``state["shorts_per_lf"]`` instead of globbing a non-existent
LF-keyed directory. These tests lock the WRITE side: after
``PhaseBCDayFirst.run`` returns, ``state["shorts_per_lf"]`` must be a
``dict[int, list[Path]]`` mapping every LF index in the drive plan to the
day-first canonical short paths.

The implementation populates the field at the bottom of
``run``. These tests are the regression-prevention contract — if
the future-tense refactors break the contract, these tests fire first.

Cases covered:

1. **Fast-path #2 (finals-only reconstruction)** — when ``per_day_completed``
   has an entry but the master is gone and finals are on disk: every LF in
   the drive plan gets a non-empty list of Paths, each pointing at
   ``_stages/day_<DATE>/shorts/short_NN.mp4`` and existing on disk.

2. **Multi-LF day distinct buckets** — two LFs assigned to the same day
   receive disjoint short-path lists (no cross-contamination of the two LFs'
   buckets on a single day).

3. **Empty input early-return** — no episodes yields
   ``state["shorts_per_lf"] == {}`` without raising.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from _state_helpers import make_state
from gomoveshift.pipeline.drive_phases.phase_bc_day_first import PHASE_BC_CODE_VERSION


# ── Fixtures: build minimal plans + on-disk finals ────────────────────


def _segment(file_idx: int, lat: float = 35.0, lon: float = -80.0):
    """Return a LFSegment-shaped SimpleNamespace.

    Matches the attribute surface ``_partition_lf_plans_by_date`` and
    ``derive_per_lf_locations`` actually read. We don't need a real
    LFSegment because the partition mutates ``source_file_index`` and the
    OST-derivation code only reads ``gps_center_*`` + bounds.
    """
    return SimpleNamespace(
        source_file_index=file_idx,
        source_path=Path(f"/fake/ep_{file_idx:03d}.mp4"),
        start_time=0.0,
        end_time=10.0,
        original_start_time=0.0,
        original_end_time=10.0,
        composite_score=0.5,
        gps_center_lat=lat,
        gps_center_lon=lon,
        terrain_type="countryside",
        contiguous_with_previous=False,
        avg_motion_energy=0.0,
    )


def _lf_plan(index: int, file_idx: int, lat: float = 35.0, lon: float = -80.0):
    """Return a RegularLFPlan-shaped SimpleNamespace pinned to one episode."""
    seg = _segment(file_idx, lat, lon)
    return SimpleNamespace(
        index=index,
        segments=[seg],
        montage_peaks=[seg],
        shorts_target=2,
        region="usa",
        total_duration=10.0,
        undersized=False,
    )


def _write_finals(day_dir: Path, lf_indices, num_shorts: int) -> None:
    """Create ``regular_lfs/lf_NN.mp4`` and ``shorts/short_NN.mp4`` files
    matching what ``_day_dir_has_finals`` requires + what
    ``_reconstruct_day_artifacts_from_disk`` will glob.

    LF finals are named by their DRIVE-GLOBAL index, exactly as
    ``run_day_pipeline`` writes them (``lf_{lf_plan.index:02d}.mp4``). The
    finals-only reconstruction parses that stem back into
    ``regular_lf_indices``; a per-day 1..N numbering here would mis-key a
    later day's LF (global LF 3 written as ``lf_01`` collapses onto LF 1).
    """
    lf_dir = day_dir / "regular_lfs"
    sh_dir = day_dir / "shorts"
    lf_dir.mkdir(parents=True, exist_ok=True)
    sh_dir.mkdir(parents=True, exist_ok=True)
    for idx in lf_indices:
        (lf_dir / f"lf_{int(idx):02d}.mp4").write_bytes(b"final-lf-bytes")
    for i in range(1, num_shorts + 1):
        (sh_dir / f"short_{i:02d}.mp4").write_bytes(b"final-short-bytes")


def _build_state_with_drive_plan(
    tmp_path: Path,
    *,
    days_to_lf_indices: dict[str, list[int]],
    shorts_per_day: int,
    pre_record_days: bool,
):
    """Build a phase_bc-ready state with a drive_plan.json pre-written and
    optional ``per_day_completed`` markers for the fast-path tests.

    ``days_to_lf_indices`` maps date strings → list of GLOBAL LF indices for
    that day. One episode per day, one segment per LF (segment's
    ``source_file_index`` points at that day's episode index). LFs of the
    same day are stacked (multi-LF day case).
    """
    output_base = tmp_path / "out"
    stages_dir = output_base / "_stages"
    stages_dir.mkdir(parents=True)

    dates = list(days_to_lf_indices.keys())

    # One episode per day — keeps the partition trivial.
    episodes: list = []
    episode_files: list[str] = []
    episode_dates: list[str] = []
    file_durations: list[float] = []
    file_boundaries: list[tuple[float, float]] = []
    cursor = 0.0
    per_day_total = 1500.0
    for i, d in enumerate(dates):
        ep_path = f"/fake/ep_{i:03d}.mp4"
        episode_files.append(ep_path)
        episode_dates.append(d)
        file_durations.append(per_day_total)
        file_boundaries.append((cursor, cursor + per_day_total))
        cursor += per_day_total
        episodes.append(
            SimpleNamespace(
                index=i,
                concat_path=Path(ep_path),
                date=d,
                total_duration=per_day_total,
                gpx_path=None,
                creation_time=None,
            )
        )
    ingest_plan = SimpleNamespace(episodes=episodes)

    # Build the real drive_plan.json on disk so PhaseBCDayFirst loads it
    # and threads it into the partition + OST loops. Each LF's segment
    # uses the GLOBAL episode index (matches how a real DrivePlan stores
    # LFs before partition).
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
        for pos, lf_idx in enumerate(days_to_lf_indices[d]):
            seg = _segment(ep_idx).__dict__.copy()
            # Same-day LFs get DISTINCT source windows (pos*10..pos*10+10)
            # so source-window shorts attribution can tell them apart —
            # the ambiguous equal-split fallback is a hard-fail.
            seg["start_time"] = pos * 10.0
            seg["end_time"] = (pos + 1) * 10.0
            seg["original_start_time"] = seg["start_time"]
            seg["original_end_time"] = seg["end_time"]
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

    # Write per-day finals (one LF per real LF on that day, N shorts/day).
    per_day_completed: dict = {}
    for d in dates:
        day_dir = stages_dir / f"day_{d}"
        _write_finals(day_dir, lf_indices=days_to_lf_indices[d], num_shorts=shorts_per_day)
        # Sidecar carrying each short's day-master source window, round-robin
        # across the day's LF windows — real runs always persist this, and
        # the finals-only resume rebuilds short_ranges from it (the legacy
        # equal-split fallback hard-fails).
        lf_list = days_to_lf_indices[d]
        sidecar = {
            "date": d,
            "shorts": [
                {
                    "rank": i,
                    "parent_lf_index": lf_list[(i - 1) % len(lf_list)],
                    "ost_text": "",
                    "start_sec": ((i - 1) % len(lf_list)) * 10.0
                    + 1.0
                    + ((i - 1) // len(lf_list)) * 2.0,
                    "end_sec": ((i - 1) % len(lf_list)) * 10.0
                    + 2.0
                    + ((i - 1) // len(lf_list)) * 2.0,
                }
                for i in range(1, shorts_per_day + 1)
            ],
        }
        (stages_dir / f"shorts_ost_audit_{d}.json").write_text(json.dumps(sidecar))
        if pre_record_days:
            # master_path=None → forces fast-path #2 (finals-only path).
            per_day_completed[d] = {"master_path": None, "master_sha": ""}

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
    # Blanket per-LF overrides for synthetic GPS-less plans.
    state.per_lf_overrides = {i: f"Synthetic Route {i}, Test" for i in range(0, 21)}
    if pre_record_days:
        state.per_day_completed = per_day_completed
    return state, output_base, stages_dir


def _run_phase(state, run_all_days_impl=None):
    """Invoke ``PhaseBCDayFirst.run`` with geocoder + run_all_days patched.

    Default ``run_all_days_impl`` returns ``[]`` — the fast-path tests
    write all finals up front and ``per_day_completed`` is set so all days
    take the skip path; ``run_all_days`` should NOT be called at all (or
    if called, sees no remaining batches).
    """
    from gomoveshift.config import Config
    from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
        PhaseBCDayFirst,
    )

    geocoder = MagicMock()
    geocoder.reverse_geocode.return_value = None

    if run_all_days_impl is None:

        def run_all_days_impl(**kwargs):  # noqa: E306
            # No remaining batches expected; if invoked anyway, return
            # nothing so we exercise the skipped-only artifact path.
            return []

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


# ── Cases ────────────────────────────────────────────────────────────


class TestShortsPerLfFastPath2:
    """Fast-path #2 (finals-only reconstruction) populates ``shorts_per_lf``.

    Fixture: 4 days × 1 LF/day, 3 shorts/day = 12 shorts total.
    Without this test, a future
    refactor that only populates ``shorts_per_lf`` from
    ``run_all_days`` output would silently regress the finals-only resume
    path back to the original bug.
    """

    def test_every_lf_in_drive_plan_has_non_empty_paths_list(self, tmp_path):
        # 4 days, 1 LF each (single-LF-per-day shape).
        state, _, stages_dir = _build_state_with_drive_plan(
            tmp_path,
            days_to_lf_indices={
                "2026-02-20": [1],
                "2026-02-21": [2],
                "2026-02-22": [3],
                "2026-02-23": [4],
            },
            shorts_per_day=3,
            pre_record_days=True,
        )

        _run_phase(state)

        shorts_per_lf = state.get("shorts_per_lf")
        assert isinstance(shorts_per_lf, dict), (
            "phase_bc must write state['shorts_per_lf'] as a dict — phase_d "
            "reads this field to find day-first canonical short paths"
        )
        # Every LF in the plan is keyed.
        assert set(shorts_per_lf.keys()) == {1, 2, 3, 4}
        # Each list is non-empty (3 shorts/day × 1 LF/day = 3 shorts/LF).
        for lf_idx, paths in shorts_per_lf.items():
            assert paths, (
                f"LF {lf_idx} has empty shorts list — phase_d will report no Drive uploads"
            )
            assert len(paths) == 3, f"LF {lf_idx} expected 3 shorts, got {len(paths)}"

    def test_paths_point_at_day_first_canonical_disk_locations(self, tmp_path):
        state, _, stages_dir = _build_state_with_drive_plan(
            tmp_path,
            days_to_lf_indices={
                "2026-02-20": [1],
                "2026-02-21": [2],
            },
            shorts_per_day=2,
            pre_record_days=True,
        )

        _run_phase(state)

        shorts_per_lf = state["shorts_per_lf"]
        for lf_idx, paths in shorts_per_lf.items():
            for p in paths:
                p = Path(p)
                # Each path must exist on disk where day-first wrote it.
                assert p.exists(), f"LF {lf_idx} short {p} missing on disk"
                # Must live under _stages/day_<DATE>/shorts/.
                parts = p.parts
                assert "shorts" in parts, f"path {p} not under shorts/"
                day_dir_name = p.parent.parent.name
                assert day_dir_name.startswith("day_"), (
                    f"path {p} parent {day_dir_name} must be day_<DATE>"
                )

    def test_multi_lf_day_assigns_distinct_buckets(self, tmp_path):
        """Two LFs on the same day must receive disjoint short lists.

        4 shorts on a 2-LF day, attributed by SOURCE WINDOW from the
        persisted sidecar (2 per LF here). The exact split is an
        implementation detail; the contract is no overlap and no
        empty list.
        """
        state, _, stages_dir = _build_state_with_drive_plan(
            tmp_path,
            days_to_lf_indices={
                # Day with 2 LFs
                "2026-02-20": [1, 2],
                # Day with 1 LF (control — ensures keying is per-LF)
                "2026-02-21": [3],
            },
            shorts_per_day=4,
            pre_record_days=True,
        )

        _run_phase(state)

        shorts_per_lf = state["shorts_per_lf"]
        assert set(shorts_per_lf.keys()) == {1, 2, 3}
        lf1_paths = {str(p) for p in shorts_per_lf[1]}
        lf2_paths = {str(p) for p in shorts_per_lf[2]}
        assert lf1_paths and lf2_paths, "Both LFs on a multi-LF day must get shorts"
        assert lf1_paths.isdisjoint(lf2_paths), (
            "LFs on the same day must receive disjoint short buckets — "
            "cross-contamination produces duplicate Metricool rows"
        )
        # And the third LF (different day) must get its own day's shorts only.
        lf3_paths = {str(p) for p in shorts_per_lf[3]}
        for p in lf3_paths:
            assert "day_2026-02-21" in p, f"LF 3 short {p} must come from its own day, not another"


class TestShortsPerLfEmpty:
    """No-input early-return preserves the empty-dict contract."""

    def test_no_episodes_yields_empty_dict_no_error(self, tmp_path):
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            PhaseBCDayFirst,
        )

        # State with no ingest_plan and no episode_files → early-return.
        output_base = tmp_path / "out"
        output_base.mkdir(parents=True)
        state = make_state(
            output_base=output_base,
            ingest_plan=None,
        )

        cfg = Config()
        phase = PhaseBCDayFirst(cfg)
        # Must not raise.
        phase.run(state, probe_duration_fn=lambda p: 100.0)

        assert state.get("shorts_per_lf") == {}, (
            "Empty input must leave state['shorts_per_lf'] as an empty dict, "
            "not unset (phase_d's canonical-read path expects the key present)"
        )
