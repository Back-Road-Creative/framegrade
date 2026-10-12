"""Phase D entry-point reconciliation: ``state["shorts_per_lf"]`` vs. disk.

Pins the fix for the split-brain between
``state["shorts_per_lf"]`` and the day-first cache shorts on disk.

Background:
    The day-first writer puts shorts at
    ``<stages_base>/day_<YYYY-MM-DD>/shorts/short_NN.mp4`` (day-keyed,
    NOT LF-keyed). Phase D is standardized to consume the
    state dict, but a stale or empty ``state["shorts_per_lf"]`` while
    the files are present on disk silently produces a 0-shorts phase_d
    run. The reconciliation below detects that split-brain at phase_d
    entry, WARN-logs both counts + the file list, and rewrites the
    state dict to match disk reality.

Reconciliation is ADDITIVE — never deletes files based on state.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from gomoveshift.pipeline.drive_phases.phase_d import (
    _reconcile_shorts_per_lf_against_disk,
)

from _state_helpers import make_state


def _seed_day(
    stages: Path,
    date: str,
    n_lfs: int,
    n_shorts: int,
    lf_index_start: int = 1,
) -> Path:
    """Seed a day dir using drive-global LF numbering — phase_bc names
    every ``lf_NN.mp4`` file after ``lf_plan.index`` (drive-global, never
    repeats across days), so ``lf_index_start`` lets multi-day tests
    construct realistic disk layouts.
    """
    day = stages / f"day_{date}"
    (day / "regular_lfs").mkdir(parents=True)
    (day / "shorts").mkdir(parents=True)
    for offset in range(n_lfs):
        idx = lf_index_start + offset
        (day / "regular_lfs" / f"lf_{idx:02d}.mp4").write_bytes(b"x")
    for r in range(1, n_shorts + 1):
        (day / "shorts" / f"short_{r:02d}.mp4").write_bytes(b"x")
    return day


class TestSplitBrainReconciliation:
    """The genuine split-brain the guard was built for: disk has shorts
    the state dict never attributed. It must HARD-FAIL upstream of any
    consumer/upload — never silently equal-split them into wrong LFs."""

    def test_empty_state_dict_with_files_on_disk_hard_fails(self, tmp_path):
        """Empty state dict + 9 mp4s on disk → raises rather than guessing
        an equal-split, and never mutates the dict."""
        stages = tmp_path / "_stages"
        stages.mkdir()
        _seed_day(stages, "2026-04-21", n_lfs=1, n_shorts=9)

        state = make_state(
            stages_base=str(stages),
            shorts_per_lf={},
            regular_lf_paths=[
                str(stages / "day_2026-04-21" / "regular_lfs" / "lf_01.mp4"),
            ],
        )

        with pytest.raises(RuntimeError, match="SPLIT-BRAIN"):
            _reconcile_shorts_per_lf_against_disk(state)

        # Never silently repopulated with an equal-split guess.
        assert state["shorts_per_lf"] == {}


class TestNoMutationWhenStateMatchesDisk:
    """State already in sync with disk → silent no-op."""

    def test_state_matches_disk_no_warning_no_mutation(self, tmp_path, caplog):
        stages = tmp_path / "_stages"
        stages.mkdir()
        day = _seed_day(stages, "2026-04-21", n_lfs=1, n_shorts=3)

        existing_paths = [day / "shorts" / f"short_{r:02d}.mp4" for r in range(1, 4)]
        state = make_state(
            stages_base=str(stages),
            shorts_per_lf={1: list(existing_paths)},
            regular_lf_paths=[str(day / "regular_lfs" / "lf_01.mp4")],
        )
        snapshot = {k: list(v) for k, v in state["shorts_per_lf"].items()}

        with caplog.at_level(logging.WARNING):
            _reconcile_shorts_per_lf_against_disk(state)

        # Same paths still there (additive, no deletion)
        assert {k: list(v) for k, v in state["shorts_per_lf"].items()} == snapshot
        # No WARN about reconciliation
        msgs = " ".join(r.getMessage() for r in caplog.records)
        assert "split-brain" not in msgs.lower()


class TestPopulatedStateNotClobbered:
    """A populated, count-correct
    ``shorts_per_lf`` reflecting the REAL source-window split (almost never
    an equal split) must NEVER be overwritten by an equal-split guess."""

    def test_non_equal_split_state_is_preserved(self, tmp_path):
        stages = tmp_path / "_stages"
        stages.mkdir()
        # 2 LFs, 4 shorts; true source-window split 3/1, NOT equal-split 2/2.
        day = _seed_day(stages, "2026-04-21", n_lfs=2, n_shorts=4, lf_index_start=1)
        s = [day / "shorts" / f"short_{r:02d}.mp4" for r in range(1, 5)]
        state = make_state(
            stages_base=str(stages),
            shorts_per_lf={1: [s[0], s[1], s[2]], 2: [s[3]]},
            regular_lf_paths=[
                str(day / "regular_lfs" / "lf_01.mp4"),
                str(day / "regular_lfs" / "lf_02.mp4"),
            ],
        )
        snapshot = {k: list(v) for k, v in state["shorts_per_lf"].items()}

        _reconcile_shorts_per_lf_against_disk(state)

        # Untouched: not rewritten to 2/2, membership order preserved.
        assert {k: list(v) for k, v in state["shorts_per_lf"].items()} == snapshot
        assert len(state["shorts_per_lf"][1]) == 3
        assert len(state["shorts_per_lf"][2]) == 1


class TestStalePartialState:
    """State knows SOME shorts for an LF but disk has more (6 unattributed)
    → genuine split-brain → hard-fail, no silent equal-split rewrite."""

    def test_partial_state_hard_fails(self, tmp_path):
        stages = tmp_path / "_stages"
        stages.mkdir()
        day = _seed_day(stages, "2026-04-21", n_lfs=1, n_shorts=9)

        # State only knows about 3 of the 9 shorts on disk.
        partial = [day / "shorts" / f"short_{r:02d}.mp4" for r in range(1, 4)]
        state = make_state(
            stages_base=str(stages),
            shorts_per_lf={1: partial},
            regular_lf_paths=[str(day / "regular_lfs" / "lf_01.mp4")],
        )

        with pytest.raises(RuntimeError, match="unattributed"):
            _reconcile_shorts_per_lf_against_disk(state)

        # State left as-is (not force-repopulated to disk truth).
        assert state["shorts_per_lf"] == {1: partial}


class TestMultiDayMultiLF:
    """Two days, multiple LFs each — a multi-day empty-state split-brain
    still hard-fails; the guard never guesses an equal-split."""

    def test_two_days_three_lfs_split_brain_hard_fails(self, tmp_path):
        stages = tmp_path / "_stages"
        stages.mkdir()
        # Day 1: 2 LFs (global indices 1, 2), 6 shorts
        day1 = _seed_day(stages, "2026-04-20", n_lfs=2, n_shorts=6, lf_index_start=1)
        # Day 2: 1 LF (global index 3), 5 shorts
        day2 = _seed_day(stages, "2026-04-21", n_lfs=1, n_shorts=5, lf_index_start=3)

        state = make_state(
            stages_base=str(stages),
            shorts_per_lf={},
            regular_lf_paths=[
                str(day1 / "regular_lfs" / "lf_01.mp4"),
                str(day1 / "regular_lfs" / "lf_02.mp4"),
                str(day2 / "regular_lfs" / "lf_03.mp4"),
            ],
        )

        with pytest.raises(RuntimeError, match="SPLIT-BRAIN"):
            _reconcile_shorts_per_lf_against_disk(state)

        assert state["shorts_per_lf"] == {}


class TestMissingStagesBaseSafe:
    """No stages_base + no day artifacts → reconciler is a no-op."""

    def test_missing_stages_base_no_op(self, tmp_path, caplog):
        state = make_state(
            shorts_per_lf={},
            regular_lf_paths=[],
        )
        with caplog.at_level(logging.WARNING):
            _reconcile_shorts_per_lf_against_disk(state)
        assert state["shorts_per_lf"] == {}

    def test_stages_base_with_no_day_dirs_no_op(self, tmp_path):
        stages = tmp_path / "_stages"
        stages.mkdir()
        state = make_state(
            stages_base=str(stages),
            shorts_per_lf={},
            regular_lf_paths=[],
        )
        _reconcile_shorts_per_lf_against_disk(state)
        assert state["shorts_per_lf"] == {}


class TestNeverDeletes:
    """If state has paths NOT on disk and disk has NO files, the
    reconciler must not silently empty the dict."""

    def test_state_only_no_disk_files_state_preserved(self, tmp_path, caplog):
        stages = tmp_path / "_stages"
        stages.mkdir()
        # Day dir exists but no shorts subdir
        day = stages / "day_2026-04-21"
        (day / "regular_lfs").mkdir(parents=True)
        (day / "regular_lfs" / "lf_01.mp4").write_bytes(b"x")

        ghost_paths = [Path("/tmp/missing/short_01.mp4")]
        state = make_state(
            stages_base=str(stages),
            shorts_per_lf={1: list(ghost_paths)},
            regular_lf_paths=[str(day / "regular_lfs" / "lf_01.mp4")],
        )
        _reconcile_shorts_per_lf_against_disk(state)
        # Reconciliation is additive; pre-existing state is NOT wiped
        # just because disk happens to be empty.
        assert state["shorts_per_lf"] == {1: ghost_paths}
