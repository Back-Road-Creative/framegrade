"""Tests for ``auto_populate_per_day_completed`` (prevention of lost resume state).

Guards against this failure: when
``pipeline_state.json`` is wiped or its ``per_day_completed`` field is
empty/corrupt but day directories with valid finals exist on disk,
neither resume fast-path fires and the pipeline re-encodes from
scratch (8+ hr loss).

The hook walks the stages dir for ``day_<YYYY-MM-DD>/`` directories
with valid finals (LF + short) and populates
``state["per_day_completed"][date]`` with a recovery marker so
``phase_bc_day_first``'s fast-path #2 (finals-only reconstruction)
fires on the next phase invocation.
"""

from pathlib import Path

from gomoveshift.pipeline.drive_orchestrator import auto_populate_per_day_completed


def _make_day_with_finals(stages_dir: Path, date: str) -> Path:
    """Build a day dir with at least one final LF and short on disk."""
    day_dir = stages_dir / f"day_{date}"
    (day_dir / "regular_lfs").mkdir(parents=True, exist_ok=True)
    (day_dir / "shorts").mkdir(parents=True, exist_ok=True)
    (day_dir / "regular_lfs" / "lf_01.mp4").write_bytes(b"x")
    (day_dir / "shorts" / "short_01.mp4").write_bytes(b"y")
    return day_dir


def _make_day_without_finals(stages_dir: Path, date: str) -> Path:
    """Build a day dir whose final subdirs exist but are empty."""
    day_dir = stages_dir / f"day_{date}"
    (day_dir / "regular_lfs").mkdir(parents=True, exist_ok=True)
    (day_dir / "shorts").mkdir(parents=True, exist_ok=True)
    return day_dir


def test_auto_populate_adds_recovery_marker_for_day_with_finals(tmp_path):
    """Empty per_day_completed + finals on disk → recovery marker added."""
    stages_dir = tmp_path / "_stages"
    stages_dir.mkdir()
    _make_day_with_finals(stages_dir, "2026-02-20")

    state: dict = {"per_day_completed": {}}
    auto_populate_per_day_completed(state, stages_dir, fresh=False)

    assert "2026-02-20" in state["per_day_completed"]
    entry = state["per_day_completed"]["2026-02-20"]
    assert entry == {"master_path": None}


def test_auto_populate_handles_missing_per_day_completed_field(tmp_path):
    """State without per_day_completed key at all → field gets created."""
    stages_dir = tmp_path / "_stages"
    stages_dir.mkdir()
    _make_day_with_finals(stages_dir, "2026-02-20")

    state: dict = {}
    auto_populate_per_day_completed(state, stages_dir, fresh=False)

    assert state["per_day_completed"]["2026-02-20"] == {
        "master_path": None,
    }


def test_auto_populate_does_not_overwrite_existing_entry(tmp_path):
    """Idempotent: existing per_day_completed entry is preserved."""
    stages_dir = tmp_path / "_stages"
    stages_dir.mkdir()
    _make_day_with_finals(stages_dir, "2026-02-20")

    existing = {
        "master_path": "/some/real/master.mp4",
        "master_sha": "deadbeef",
    }
    state: dict = {"per_day_completed": {"2026-02-20": dict(existing)}}
    auto_populate_per_day_completed(state, stages_dir, fresh=False)

    assert state["per_day_completed"]["2026-02-20"] == existing


def test_auto_populate_skips_day_without_finals(tmp_path):
    """Day dir with empty regular_lfs/shorts → no entry added."""
    stages_dir = tmp_path / "_stages"
    stages_dir.mkdir()
    _make_day_without_finals(stages_dir, "2026-02-21")

    state: dict = {"per_day_completed": {}}
    auto_populate_per_day_completed(state, stages_dir, fresh=False)

    assert state["per_day_completed"] == {}


def test_auto_populate_empty_stages_dir_is_noop(tmp_path):
    """No day_* dirs → no exception, no entries added."""
    stages_dir = tmp_path / "_stages"
    stages_dir.mkdir()

    state: dict = {"per_day_completed": {}}
    auto_populate_per_day_completed(state, stages_dir, fresh=False)

    assert state["per_day_completed"] == {}


def test_auto_populate_missing_stages_dir_is_noop(tmp_path):
    """Stages dir doesn't exist (nothing to walk) → no-op, no exception."""
    stages_dir = tmp_path / "does_not_exist"

    state: dict = {"per_day_completed": {}}
    auto_populate_per_day_completed(state, stages_dir, fresh=False)

    assert state["per_day_completed"] == {}


def test_auto_populate_mixed_days(tmp_path):
    """Multiple day dirs: finals-having gets marker, finals-lacking doesn't."""
    stages_dir = tmp_path / "_stages"
    stages_dir.mkdir()
    _make_day_with_finals(stages_dir, "2026-02-20")
    _make_day_with_finals(stages_dir, "2026-02-22")
    _make_day_without_finals(stages_dir, "2026-02-21")
    # Non-day directory should be ignored.
    (stages_dir / "scratch").mkdir()

    state: dict = {"per_day_completed": {}}
    auto_populate_per_day_completed(state, stages_dir, fresh=False)

    assert set(state["per_day_completed"].keys()) == {"2026-02-20", "2026-02-22"}


def test_auto_populate_ignores_non_date_dirs(tmp_path):
    """Dirs like day_foo or day_2026 (wrong format) are ignored."""
    stages_dir = tmp_path / "_stages"
    stages_dir.mkdir()
    (stages_dir / "day_foo" / "regular_lfs").mkdir(parents=True)
    (stages_dir / "day_foo" / "shorts").mkdir(parents=True)
    (stages_dir / "day_foo" / "regular_lfs" / "lf_01.mp4").write_bytes(b"x")
    (stages_dir / "day_foo" / "shorts" / "short_01.mp4").write_bytes(b"y")

    state: dict = {"per_day_completed": {}}
    auto_populate_per_day_completed(state, stages_dir, fresh=False)

    assert state["per_day_completed"] == {}


def test_auto_populate_is_noop_on_fresh_run(tmp_path):
    """``--fresh`` must re-encode: on-disk finals from a prior run are NOT
    recovered. Past failure: a ``--fresh`` re-run reused shorts from an earlier
    run that burned the wrong place name because this hook
    re-added the recovery marker after ``_try_resume`` purged the state."""
    stages_dir = tmp_path / "stages"
    _make_day_with_finals(stages_dir, "2026-03-04")
    state: dict = {"per_day_completed": {}}

    auto_populate_per_day_completed(state, stages_dir, fresh=True)

    assert state["per_day_completed"] == {}


def test_auto_populate_requires_explicit_fresh(tmp_path):
    """Robust by construction: every caller must say whether the run is
    ``--fresh``, so a new call site cannot silently resurrect stale finals."""
    import pytest

    with pytest.raises(TypeError):
        auto_populate_per_day_completed({}, tmp_path)  # type: ignore[call-arg]


def test_orchestrator_threads_fresh_into_auto_populate():
    """The drive run passes its ``fresh`` flag through, not a constant."""
    import inspect

    from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

    src = inspect.getsource(DrivePipelineOrchestrator.run)
    assert "auto_populate_per_day_completed(state, stages_dir, fresh=fresh)" in src
