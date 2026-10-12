"""Contract enforcement closure.

Three closures on top of the base per-LF contract work:

1. ``per_lf_gpx_path`` has a load-time migration parallel to its
   siblings ``per_lf_recording_date`` and ``per_lf_trim_ratio``. A state
   file written by an older binary (which did not declare the field) loads with the field missing;
   the migration rebuilds it from ``regular_lf_paths`` + ``episode_dates`` + ``gpx_files``.

2. The per-LF completeness gate uses an index-intersection check, not a
   length-only check. A mapping with the right length but offset keys
   fails the gate rather than silently passing.

3. ``per_lf_gpx_path`` is in ``_INT_KEYED_DICTS`` so the dict-key
   invariant fires on a corrupted-on-disk state file before any consumer
   indexes it with the wrong key type.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.pipeline.drive_state_model import PipelineState
from gomoveshift.pipeline.exceptions import StateInvariantError
from gomoveshift.pipeline.state_invariants import (
    _INT_KEYED_DICTS,
    _check_int_keyed_dicts,
    _check_per_lf_completeness,
)
from _state_helpers import make_state


def _lf(index: int):
    return SimpleNamespace(index=index, segments=[SimpleNamespace(source_file_index=0)])


def _drive_plan(indices):
    return SimpleNamespace(regular_lfs=[_lf(i) for i in indices])


# ──────────────────────────────────────────────────────────────────────
# 1. Migration: per_lf_gpx_path rebuilt from regular_lf_paths + episode_dates
# ──────────────────────────────────────────────────────────────────────


def test_migration_rebuilds_per_lf_gpx_path_from_day_dirs(tmp_path):
    """A legacy state lacks per_lf_gpx_path. Migration (now run from
    ``__post_init__``) rebuilds it by matching each LF's
    day_<DATE> directory against ``episode_dates``."""
    instance = PipelineState(
        source_dir=tmp_path,
        output_base=tmp_path,
        per_lf_gpx_path=None,
        regular_lf_paths=[
            Path("/stages/day_2026-02-20/regular_lfs/lf_00.mp4"),
            Path("/stages/day_2026-02-21/regular_lfs/lf_01.mp4"),
        ],
        gpx_files=["/src/gopro_00.gpx", "/src/gopro_01.gpx"],
        episode_dates=["2026-02-20", "2026-02-21"],
        drive_plan=_drive_plan([0, 1]),
    )

    assert instance.per_lf_gpx_path == {
        0: "/src/gopro_00.gpx",
        1: "/src/gopro_01.gpx",
    }


def test_migration_noop_when_field_already_populated(tmp_path):
    """If state already has the field, ``__post_init__`` migration leaves
    it untouched even when episode_dates would point to different files."""
    instance = PipelineState(
        source_dir=tmp_path,
        output_base=tmp_path,
        per_lf_gpx_path={0: "/preserved.gpx"},
        regular_lf_paths=[Path("/stages/day_2026-02-20/regular_lfs/lf_00.mp4")],
        gpx_files=["/other.gpx"],
        episode_dates=["2026-02-20"],
        drive_plan=_drive_plan([0]),
    )
    assert instance.per_lf_gpx_path == {0: "/preserved.gpx"}


def test_migration_noop_when_inputs_missing(tmp_path):
    """Missing any of regular_lf_paths / gpx_files / episode_dates leaves
    the field None (Phase D falls back to its segment-index path)."""
    instance = PipelineState(
        source_dir=tmp_path,
        output_base=tmp_path,
        regular_lf_paths=[Path("/stages/day_2026-02-20/regular_lfs/lf_00.mp4")],
        gpx_files=None,
        episode_dates=["2026-02-20"],
        drive_plan=_drive_plan([0]),
    )
    assert instance.per_lf_gpx_path is None


def test_migration_picks_first_match_on_duplicate_dates(tmp_path):
    """Multi-source-file-per-day: migration picks the first GPX for that
    date, mirroring the pre-partition behaviour of
    ``_build_per_lf_gpx_path_map`` which uses ``segments[0]``."""
    instance = PipelineState(
        source_dir=tmp_path,
        output_base=tmp_path,
        regular_lf_paths=[Path("/stages/day_2026-02-21/regular_lfs/lf_00.mp4")],
        gpx_files=["/src/a.gpx", "/src/b.gpx"],
        episode_dates=["2026-02-21", "2026-02-21"],
        drive_plan=_drive_plan([0]),
    )
    assert instance.per_lf_gpx_path == {0: "/src/a.gpx"}


# ──────────────────────────────────────────────────────────────────────
# 2. Completeness uses index-intersection, not length
# ──────────────────────────────────────────────────────────────────────


def test_completeness_fails_on_offset_keys_same_length():
    """The mapping has the same LENGTH as expected_indices but the keys
    are offset by 1. Length-only check would pass; intersection check
    fails as it should."""
    state = make_state(
        phases_completed=["phase_bc_day_first"],
        regular_lf_paths=["/a.mp4", "/b.mp4", "/c.mp4"],
        drive_plan=_drive_plan([1, 2, 3]),
        per_lf_recording_date={0: "2026-02-20", 1: "2026-02-20", 2: "2026-02-21"},
        per_lf_trim_ratio={1: 0.7, 2: 0.7, 3: 0.7},
        per_lf_gpx_path={1: "/a.gpx", 2: "/b.gpx", 3: "/c.gpx"},
    )
    with pytest.raises(StateInvariantError, match=r"per_lf_recording_date is missing index"):
        _check_per_lf_completeness(state)


def test_completeness_passes_on_matching_indices():
    state = make_state(
        phases_completed=["phase_bc_day_first"],
        regular_lf_paths=["/a.mp4", "/b.mp4"],
        drive_plan=_drive_plan([0, 1]),
        per_lf_recording_date={0: "2026-02-20", 1: "2026-02-21"},
        per_lf_trim_ratio={0: 0.7, 1: 0.7},
        per_lf_gpx_path={0: "/a.gpx", 1: "/b.gpx"},
    )
    _check_per_lf_completeness(state)  # no raise


def test_completeness_fails_on_missing_per_lf_gpx_path():
    # NOTE: dict-shape fixture intentional — PipelineState.__post_init__
    # runs _migrate_per_lf_gpx_path_legacy_state which auto-fills missing
    # indices, defeating the test's intent. Dict-shape bypasses the
    # migration and exercises _check_per_lf_completeness in isolation.
    state = {
        "phases_completed": ["phase_bc_day_first"],
        "regular_lf_paths": ["/a.mp4", "/b.mp4"],
        "drive_plan": _drive_plan([0, 1]),
        "per_lf_recording_date": {0: "2026-02-20", 1: "2026-02-21"},
        "per_lf_trim_ratio": {0: 0.7, 1: 0.7},
        "per_lf_gpx_path": {0: "/a.gpx"},
    }
    with pytest.raises(StateInvariantError, match=r"per_lf_gpx_path is missing index"):
        _check_per_lf_completeness(state)


# ──────────────────────────────────────────────────────────────────────
# 3. _INT_KEYED_DICTS includes per_lf_gpx_path
# ──────────────────────────────────────────────────────────────────────


def test_per_lf_gpx_path_in_int_keyed_dicts():
    assert "per_lf_gpx_path" in _INT_KEYED_DICTS
    assert "per_lf_recording_date" in _INT_KEYED_DICTS
    assert "per_lf_trim_ratio" in _INT_KEYED_DICTS


def test_int_key_invariant_catches_str_key_in_per_lf_gpx_path():
    state = make_state(per_lf_gpx_path={"0": "/a.gpx"})
    with pytest.raises(StateInvariantError, match=r"per_lf_gpx_path has non-int key"):
        _check_int_keyed_dicts(state)
