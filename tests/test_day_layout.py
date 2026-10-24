"""One neutral definition of the ``day_<YYYY-MM-DD>`` day-directory layout.

The pattern used to be re-declared in three pipeline modules (the day-first
phase, the pipeline state model and the orchestrator's auto-populate hook),
and the copies disagreed: the state model searched the whole path string
while the others anchored on a directory name. These tests pin the shared
module and assert every former copy now answers identically.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift import day_layout
from gomoveshift.pipeline import drive_orchestrator, drive_state_model
from gomoveshift.pipeline.drive_phases import phase_bc_day_first
from gomoveshift.pipeline.drive_state_model import PipelineState


def test_day_dir_name_round_trips():
    assert day_layout.day_dir_name("2025-09-15") == "day_2025-09-15"
    assert day_layout.date_from_day_dir_name("day_2025-09-15") == "2025-09-15"


@pytest.mark.parametrize(
    "name",
    ["day_2025-09-15_master.mp4", "xday_2025-09-15", "day_2025-9-15", "day_", "regular_lfs", ""],
)
def test_date_from_day_dir_name_rejects_non_day_dirs(name):
    assert day_layout.date_from_day_dir_name(name) is None


_LF_PATH_CASES = [
    ("/stages/proj/day_2025-09-15/regular_lfs/lf_00.mp4", "2025-09-15"),
    # An ancestor day_* token must not skew the date.
    ("/archive/day_2024-01-01/restored/day_2025-09-15/regular_lfs/lf_03.mp4", "2025-09-15"),
    ("/stages/proj/regular_lfs/lf_00.mp4", None),
    ("/a/day_2024-01-01/b/day_2025-09-15/lf_00.mp4", None),
    ("/stages/day_2025-09-15/lf_00.mp4", "2025-09-15"),
    ("/stages/day_2025-09-15_master/regular_lfs/lf_00.mp4", None),
]


@pytest.mark.parametrize(("path", "expected"), _LF_PATH_CASES)
def test_day_date_from_lf_path(path, expected):
    assert day_layout.day_date_from_lf_path(path) == expected


@pytest.mark.parametrize(("path", "expected"), _LF_PATH_CASES)
def test_phase_bc_copy_matches_shared_module(path, expected):
    assert phase_bc_day_first._day_date_from_lf_path(path) == expected


@pytest.mark.parametrize(("path", "expected"), _LF_PATH_CASES)
def test_state_model_migration_matches_shared_module(path, expected):
    state = PipelineState()
    object.__setattr__(state, "regular_lf_paths", [path])
    object.__setattr__(state, "drive_plan", SimpleNamespace(regular_lfs=[SimpleNamespace(index=4)]))
    state._migrate_per_lf_recording_date_from_paths()
    assert (state.per_lf_recording_date or {}) == ({4: expected} if expected else {})


def test_orchestrator_autopopulate_matches_shared_module(tmp_path):
    stages = tmp_path / "stages"
    for name in ("day_2025-09-15", "day_2025-09-16_master", "regular_lfs"):
        (stages / name / "regular_lfs").mkdir(parents=True)
        (stages / name / "shorts").mkdir(parents=True)
        (stages / name / "regular_lfs" / "lf_00.mp4").write_bytes(b"x")
        (stages / name / "shorts" / "short_00.mp4").write_bytes(b"x")
    state = {"per_day_completed": {}}
    drive_orchestrator.auto_populate_per_day_completed(state, stages, fresh=False)
    assert set(state["per_day_completed"]) == {"2025-09-15"}


def test_no_module_redeclares_the_day_dir_pattern():
    for module in (drive_orchestrator, drive_state_model, phase_bc_day_first):
        for attr in ("_DAY_DIR_RE", "_DAY_DIR_DATE_RE"):
            assert not hasattr(module, attr), f"{module.__name__}.{attr} is a stale copy"
