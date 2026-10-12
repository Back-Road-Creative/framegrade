"""Per-LF disk-derivable migration tests.

The state model declares ``per_lf_recording_date`` and ``per_lf_trim_ratio`` on
``PipelineState`` and adds load-time disk-derivation in
``PipelineState.from_dict``. These tests pin that contract — the
in-phase ``_derive_per_lf_*_from_paths`` backstops in phase_bc / phase_d
that used to fire on every resume are now unreachable by construction
from any legitimate state file.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gomoveshift.pipeline.drive_state_model import PipelineState


def _make_skeleton_lf(idx: int, segments=None, total_duration: float = 300.0):
    return SimpleNamespace(
        index=idx,
        segments=segments or [],
        total_duration=total_duration,
    )


def _make_drive_plan(skeleton_lfs):
    return SimpleNamespace(regular_lfs=skeleton_lfs)


def _write_lf_file(stages_dir: Path, date: str, lf_idx: int) -> Path:
    day_dir = stages_dir / f"day_{date}" / "regular_lfs"
    day_dir.mkdir(parents=True, exist_ok=True)
    p = day_dir / f"lf_{lf_idx:02d}.mp4"
    p.write_bytes(b"fake-lf-bytes")
    return p


def test_legacy_state_migrates_per_lf_recording_date_from_disk(tmp_path):
    """JSON state with ``regular_lf_paths`` under ``day_<DATE>/...`` but
    no ``per_lf_recording_date`` key → from_dict populates the field
    from the parent directory regex.
    """
    stages = tmp_path / "stages"
    path1 = _write_lf_file(stages, "2026-02-20", 1)
    path2 = _write_lf_file(stages, "2026-02-21", 2)

    skeleton_lfs = [_make_skeleton_lf(1), _make_skeleton_lf(2)]
    drive_plan = _make_drive_plan(skeleton_lfs)

    raw = {
        "output_base": str(tmp_path / "out"),
        "regular_lf_paths": [str(path1), str(path2)],
        # NOTE: no per_lf_recording_date key — the bug class this closes.
        "drive_plan": drive_plan,
    }

    instance = PipelineState.from_dict(raw)
    assert instance.per_lf_recording_date == {1: "2026-02-20", 2: "2026-02-21"}


def test_legacy_state_migrates_per_lf_trim_ratio_from_disk(tmp_path):
    """JSON state with regular_lf_paths + file_durations but missing
    per_lf_trim_ratio → from_dict ffprobes finals and fills the map.
    """
    stages = tmp_path / "stages"
    path1 = _write_lf_file(stages, "2026-02-20", 1)

    seg = SimpleNamespace(source_file_index=0)
    skeleton_lfs = [_make_skeleton_lf(1, segments=[seg], total_duration=300.0)]
    drive_plan = _make_drive_plan(skeleton_lfs)

    raw = {
        "output_base": str(tmp_path / "out"),
        "regular_lf_paths": [str(path1)],
        "file_durations": [1000.0],
        "drive_plan": drive_plan,
    }

    # ffprobe → 300.0 → trim_ratio = 300 / 1000 = 0.3
    with patch(
        "gomoveshift.pipeline.drive_phases.phase_bc_day_first._probe_duration",
        return_value=300.0,
    ):
        instance = PipelineState.from_dict(raw)
    assert instance.per_lf_trim_ratio is not None
    assert instance.per_lf_trim_ratio[1] == 0.3


def test_trim_ratio_migration_uses_day_local_durations_multiday(tmp_path):
    """The load-time trim-ratio migration must index source
    durations in the DAY-LOCAL episode space, like the fresh-encode path
    and the resume backstop.

    ``source_file_index`` is day-local after ``_partition_lf_plans_by_date``;
    ``file_durations`` is drive-wide. A day-2 LF whose day-local indices 0,1
    collide with day-1's global indices must divide by its OWN day's
    durations (500 + 800 = 1300), not day-1's (1000 + 2000 = 3000).
    """
    stages = tmp_path / "stages"
    path1 = _write_lf_file(stages, "2026-02-20", 1)
    path2 = _write_lf_file(stages, "2026-02-21", 2)

    seg = SimpleNamespace(source_file_index=0)
    seg2a = SimpleNamespace(source_file_index=0)
    seg2b = SimpleNamespace(source_file_index=1)
    skeleton_lfs = [
        _make_skeleton_lf(1, segments=[seg]),
        _make_skeleton_lf(2, segments=[seg2a, seg2b]),
    ]
    drive_plan = _make_drive_plan(skeleton_lfs)

    raw = {
        "output_base": str(tmp_path / "out"),
        "regular_lf_paths": [str(path1), str(path2)],
        # Drive-wide (global) index space: global 0,1 → day 1; global 2,3 → day 2.
        "file_durations": [1000.0, 2000.0, 500.0, 800.0],
        "episode_dates": ["2026-02-20", "2026-02-20", "2026-02-21", "2026-02-21"],
        "drive_plan": drive_plan,
        # NOTE: no per_lf_trim_ratio and no per_lf_recording_date — the
        # recording-date migration runs first and feeds the trim migration.
    }

    with patch(
        "gomoveshift.pipeline.drive_phases.phase_bc_day_first._probe_duration",
        side_effect=lambda p: 300.0 if "lf_01" in str(p) else 260.0,
    ):
        instance = PipelineState.from_dict(raw)

    assert instance.per_lf_trim_ratio is not None
    assert instance.per_lf_trim_ratio[1] == pytest.approx(300.0 / 1000.0, abs=1e-4)
    # Drive-wide read gives 260/3000 ≈ 0.0867 — the aliasing bug.
    assert instance.per_lf_trim_ratio[2] == pytest.approx(260.0 / 1300.0, abs=1e-4)


def test_trim_ratio_migration_single_day_with_episode_dates_unchanged(tmp_path):
    """Single-day state WITH episode_dates: day-local == global, so the
    migration's result must match the plain drive-wide read."""
    stages = tmp_path / "stages"
    path1 = _write_lf_file(stages, "2026-02-20", 1)

    seg = SimpleNamespace(source_file_index=0)
    skeleton_lfs = [_make_skeleton_lf(1, segments=[seg])]
    drive_plan = _make_drive_plan(skeleton_lfs)

    raw = {
        "output_base": str(tmp_path / "out"),
        "regular_lf_paths": [str(path1)],
        "file_durations": [1000.0],
        "episode_dates": ["2026-02-20"],
        "drive_plan": drive_plan,
    }

    with patch(
        "gomoveshift.pipeline.drive_phases.phase_bc_day_first._probe_duration",
        return_value=300.0,
    ):
        instance = PipelineState.from_dict(raw)
    assert instance.per_lf_trim_ratio[1] == pytest.approx(0.3, abs=1e-4)


def test_already_populated_state_is_not_overwritten(tmp_path):
    """If the state file already carries ``per_lf_recording_date`` /
    ``per_lf_trim_ratio``, the migration must be a no-op.
    """
    stages = tmp_path / "stages"
    # Note: writing to a path that DOES NOT match day_<DATE> regex — so
    # if the migration were to run, it would clobber to {} (and we'd
    # never know). The instance value must survive.
    path1 = stages / "regular_lfs" / "lf_01.mp4"
    path1.parent.mkdir(parents=True, exist_ok=True)
    path1.write_bytes(b"x")

    skeleton_lfs = [_make_skeleton_lf(1)]
    drive_plan = _make_drive_plan(skeleton_lfs)

    raw = {
        "output_base": str(tmp_path / "out"),
        "regular_lf_paths": [str(path1)],
        "drive_plan": drive_plan,
        "per_lf_recording_date": {1: "2025-12-31"},
        "per_lf_trim_ratio": {1: 0.42},
    }

    instance = PipelineState.from_dict(raw)
    assert instance.per_lf_recording_date == {1: "2025-12-31"}
    assert instance.per_lf_trim_ratio == {1: 0.42}


def test_migration_idempotent(tmp_path):
    """Load → save → load again must produce stable values."""
    stages = tmp_path / "stages"
    path1 = _write_lf_file(stages, "2026-02-20", 1)
    skeleton_lfs = [_make_skeleton_lf(1)]
    drive_plan = _make_drive_plan(skeleton_lfs)

    raw = {
        "output_base": str(tmp_path / "out"),
        "regular_lf_paths": [str(path1)],
        "drive_plan": drive_plan,
    }
    first = PipelineState.from_dict(raw)
    assert first.per_lf_recording_date == {1: "2026-02-20"}

    # Round-trip back through dict (drive_plan is a SimpleNamespace
    # which to_dict won't serialize cleanly — drop it before the
    # second load so we exercise the "already populated, no
    # skeleton_lfs available" branch).
    raw2 = {
        "output_base": str(tmp_path / "out"),
        "regular_lf_paths": [str(path1)],
        "per_lf_recording_date": dict(first.per_lf_recording_date),
    }
    second = PipelineState.from_dict(raw2)
    assert second.per_lf_recording_date == first.per_lf_recording_date


def test_partial_state_with_no_lf_paths_skips_migration(tmp_path):
    """Phase-a-only resume (no regular_lf_paths) → migration is a safe
    no-op, no crash. ``per_lf_recording_date`` / ``per_lf_trim_ratio``
    stay ``None``.
    """
    raw = {
        "output_base": str(tmp_path / "out"),
        # No regular_lf_paths, no drive_plan, no per_lf_* keys.
    }
    instance = PipelineState.from_dict(raw)
    assert instance.per_lf_recording_date is None
    assert instance.per_lf_trim_ratio is None


def test_migration_skips_when_drive_plan_absent(tmp_path):
    """regular_lf_paths present but no drive_plan → no skeleton_lfs →
    migration cannot map paths to LF indices → safe no-op.
    """
    stages = tmp_path / "stages"
    path1 = _write_lf_file(stages, "2026-02-20", 1)
    raw = {
        "output_base": str(tmp_path / "out"),
        "regular_lf_paths": [str(path1)],
        # No drive_plan.
    }
    instance = PipelineState.from_dict(raw)
    assert instance.per_lf_recording_date is None
    assert instance.per_lf_trim_ratio is None


def test_recording_date_migration_skips_unparseable_paths(tmp_path):
    """LFs whose paths don't match ``day_<YYYY-MM-DD>`` are omitted; the
    rest are populated.
    """
    stages = tmp_path / "stages"
    good = _write_lf_file(stages, "2026-02-20", 1)
    bad_dir = stages / "no_day_prefix" / "regular_lfs"
    bad_dir.mkdir(parents=True, exist_ok=True)
    bad = bad_dir / "lf_02.mp4"
    bad.write_bytes(b"x")

    skeleton_lfs = [_make_skeleton_lf(1), _make_skeleton_lf(2)]
    drive_plan = _make_drive_plan(skeleton_lfs)
    raw = {
        "output_base": str(tmp_path / "out"),
        "regular_lf_paths": [str(good), str(bad)],
        "drive_plan": drive_plan,
    }
    instance = PipelineState.from_dict(raw)
    assert instance.per_lf_recording_date == {1: "2026-02-20"}
