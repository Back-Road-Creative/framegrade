"""A ``--superlong`` resume must never call a day complete without its parts.

Failure case: a day with 4 LF finals and 56
shorts on disk and ZERO ``sl_NN.mp4``. The resume fast-path judged it complete
on LFs+shorts alone, skipped it, and the run walked into Phase D — so the only
producer of superlong parts (``day_pipeline._produce_superlong_parts``, reached
only inside ``run_day_pipeline``) was never called. Nothing downstream noticed:
the superlong gates key off a non-empty parts list, so zero parts fired zero
gates and the run would have reported success with no superlong at all.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from gomoveshift.pipeline.cleanup import finalized_superlong_count
from gomoveshift.pipeline.drive_phases.phase_bc_day_first import day_superlong_complete

_DATE = "2026-03-04"


def _day_dir(tmp_path: Path, *, parts: int = 0, empty_parts: int = 0) -> Path:
    """A day dir with LF + short finals, and ``parts`` non-empty sl_NN.mp4."""
    day_dir = tmp_path / f"day_{_DATE}"
    (day_dir / "regular_lfs").mkdir(parents=True)
    (day_dir / "shorts").mkdir(parents=True)
    (day_dir / "regular_lfs" / "lf_01.mp4").write_bytes(b"x")
    (day_dir / "shorts" / "short_01.mp4").write_bytes(b"x")
    sl = day_dir / "superlong"
    sl.mkdir()
    # A stray file: the dir exists but holds no part.
    (sl / "overpass_cache_cb3ca569c6b2.json").write_text("{}")
    for i in range(1, parts + 1):
        (sl / f"sl_{i:02d}.mp4").write_bytes(b"x")
    for i in range(parts + 1, parts + empty_parts + 1):
        (sl / f"sl_{i:02d}.mp4").write_bytes(b"")
    return day_dir


def _state(part_count=None):
    parts = {_DATE: {"part_count": part_count}} if part_count is not None else {}
    return SimpleNamespace(superlong_parts=parts)


# ── finalized_superlong_count ───────────────────────────────────────────────


def test_count_ignores_a_zero_byte_part(tmp_path):
    # A truncated part must never read as complete — same non-empty discipline
    # the master-retention gate already uses for shorts.
    assert finalized_superlong_count(_day_dir(tmp_path, parts=1, empty_parts=1)) == 1


def test_count_ignores_non_part_files(tmp_path):
    # A superlong/ dir held only an overpass cache file.
    assert finalized_superlong_count(_day_dir(tmp_path)) == 0


# ── day_superlong_complete ──────────────────────────────────────────────────


def test_no_state_record_is_incomplete_not_complete(tmp_path):
    """The failing case: LFs + shorts on disk, no parts, no state record.

    A missing record means superlong never ran for this day. It cannot mean
    "nothing was planned, so we are done" — the count is only derivable from
    the day's stop mask inside run_day_pipeline.
    """
    assert day_superlong_complete(_DATE, _day_dir(tmp_path), _state()) is False


def test_planned_but_unburned_parts_are_incomplete(tmp_path):
    assert day_superlong_complete(_DATE, _day_dir(tmp_path), _state(2)) is False


def test_partially_burned_parts_are_incomplete(tmp_path):
    assert day_superlong_complete(_DATE, _day_dir(tmp_path, parts=1), _state(2)) is False


def test_all_parts_on_disk_is_complete(tmp_path):
    # The guard against the regression this fix could introduce: a FINISHED
    # superlong drive must still skip, never re-encode forever.
    assert day_superlong_complete(_DATE, _day_dir(tmp_path, parts=2), _state(2)) is True


# ── resume re-entry ─────────────────────────────────────────────────────────
#
# The second half of the failure: after the skipping run, phases_completed named
# phase_bc_day_first, so the NEXT resume skips the phase wholesale and the
# per-day partition fix above never even runs.


def _resume(tmp_path, *, parts=0, part_count=None, enabled=True):
    from gomoveshift.pipeline.orchestrator_fingerprint import (
        invalidate_phase_bc_if_superlong_incomplete,
    )

    _day_dir(tmp_path, parts=parts)
    state = SimpleNamespace(
        superlong_enabled=enabled,
        superlong_parts=({_DATE: {"part_count": part_count}} if part_count else {}),
        phases_completed=["phase_a", "phase_bc_day_first", "phase_d"],
    )
    resumed = ["phase_a", "phase_bc_day_first", "phase_d"]
    stages_completed = list(resumed)
    out = invalidate_phase_bc_if_superlong_incomplete(
        state, resumed, stages_completed, stages_dir=tmp_path
    )
    return out, stages_completed, state


def test_resume_reenters_phase_bc_when_parts_missing(tmp_path):
    out, stages_completed, state = _resume(tmp_path)
    assert out == ["phase_a"]
    # phase_d goes too: it is what publishes the parts, and it declares
    # phase_bc as its prereq.
    assert stages_completed == ["phase_a"]
    assert state.phases_completed == ["phase_a"]


def test_resume_keeps_phase_bc_when_parts_complete(tmp_path):
    out, stages_completed, _ = _resume(tmp_path, parts=2, part_count=2)
    assert out == ["phase_a", "phase_bc_day_first", "phase_d"]
    assert stages_completed == ["phase_a", "phase_bc_day_first", "phase_d"]


def test_resume_is_a_noop_when_superlong_disabled(tmp_path):
    # The 99% path and the original recovery scenario: zero cost.
    out, stages_completed, _ = _resume(tmp_path, enabled=False)
    assert out == ["phase_a", "phase_bc_day_first", "phase_d"]
    assert stages_completed == ["phase_a", "phase_bc_day_first", "phase_d"]
