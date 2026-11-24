"""Galley Head: phase_bc checkpointed, 0 of 2 ``lf_NN.mp4`` on disk, so ``phase_d`` raised
SPLIT-BRAIN on every retry. A resume must re-enter phase_bc; real orchestrator, fake phase
bodies."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from _state_helpers import make_state
from gomoveshift.config import Config
from gomoveshift.models import DrivePlan, LFSegment, RegularLFPlan
from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    PHASE_BC_CODE_VERSION,
    PhaseBCDayFirst,
    missing_planned_lf_indices,
)
from gomoveshift.pipeline.orchestrator_fingerprint import invalidate_phase_bc_if_lf_shortfall

DATE = "2026-02-20"
OTHER = "2026-02-21"


def _lf(index: int, file_index: int) -> RegularLFPlan:
    seg = LFSegment(
        source_file_index=file_index,
        source_path=Path(f"/fake/ep{file_index}.mp4"),
        start_time=0.0,
        end_time=60.0,
        original_start_time=0.0,
        original_end_time=60.0,
        composite_score=0.9,
        gps_center_lat=0.0,
        gps_center_lon=0.0,
        terrain_type="countryside",
        contiguous_with_previous=False,
    )
    return RegularLFPlan(
        index=index,
        segments=[seg],
        montage_peaks=[seg],
        shorts_target=1,
        region="usa",
        total_duration=60.0,
    )


def _write_plan(stages: Path, lfs, *, version=PHASE_BC_CODE_VERSION) -> None:
    stages.mkdir(parents=True, exist_ok=True)
    DrivePlan(
        source_files=[],
        processed_files=[],
        regular_lfs=lfs,
        score_threshold=0.5,
        total_source_duration=600.0,
        total_highlight_duration=120.0,
        plan_code_version=version,
    ).save(stages / "drive_plan.json")


def _finals(stages: Path, date: str, indices=(), empty=()) -> Path:
    day = stages / f"day_{date}"
    (day / "regular_lfs").mkdir(parents=True, exist_ok=True)
    (day / "shorts").mkdir(exist_ok=True)
    for i in indices:
        (day / "regular_lfs" / f"lf_{i:02d}.mp4").write_bytes(b"x")
    for i in empty:
        (day / "regular_lfs" / f"lf_{i:02d}.mp4").write_bytes(b"")
    (day / "shorts" / "short_01.mp4").write_bytes(b"x")
    return day


def _state(recorded=(DATE,)):
    return SimpleNamespace(
        episode_dates=[DATE, DATE, OTHER],
        per_day_completed={d: {"master_path": "/m"} for d in recorded},
        phases_completed=["phase_a", "phase_bc_day_first", "phase_d"],
    )


def _invalidate(stages, state, config=None):
    resumed = list(state.phases_completed)
    done = list(resumed)
    out = invalidate_phase_bc_if_lf_shortfall(
        state, resumed, done, stages_dir=stages, config=config or Config()
    )
    return out, done, state


# --- the shared rule: plan INDICES, not a file count --------------------------


def test_stale_extra_final_does_not_hide_a_missing_planned_lf(tmp_path):
    day = _finals(tmp_path, DATE, indices=(1, 3))
    assert missing_planned_lf_indices(day, [1, 2]) == [2]


def test_zero_byte_final_counts_as_missing(tmp_path):
    day = _finals(tmp_path, DATE, indices=(1,), empty=(2,))
    assert missing_planned_lf_indices(day, [1, 2]) == [2]


def test_every_planned_final_present_is_not_a_shortfall(tmp_path):
    day = _finals(tmp_path, DATE, indices=(1, 2))
    assert missing_planned_lf_indices(day, [1, 2]) == []


# --- the invalidator ---------------------------------------------------------


def test_galley_state_reenters_phase_bc_and_phase_d(tmp_path, caplog):
    """0 of 2 finals: both phases drop and the log names the day."""
    _write_plan(tmp_path, [_lf(1, 0), _lf(2, 1)])
    _finals(tmp_path, DATE)
    with caplog.at_level(logging.WARNING):
        out, done, state = _invalidate(tmp_path, _state())
    assert out == ["phase_a"]
    assert done == ["phase_a"]
    assert state.phases_completed == ["phase_a"]
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert DATE in msg and "missing 2 of 2" in msg


def test_a_stale_extra_final_does_not_keep_phase_bc(tmp_path):
    _write_plan(tmp_path, [_lf(1, 0), _lf(2, 1)])
    _finals(tmp_path, DATE, indices=(1, 3))
    out, _, _ = _invalidate(tmp_path, _state())
    assert out == ["phase_a"]


def test_a_second_day_short_of_its_own_lf_reenters(tmp_path):
    _write_plan(tmp_path, [_lf(1, 0), _lf(2, 2)])
    _finals(tmp_path, DATE, indices=(1,))
    _finals(tmp_path, OTHER)
    out, _, _ = _invalidate(tmp_path, _state(recorded=(DATE, OTHER)))
    assert out == ["phase_a"]


def test_no_plan_on_disk_leaves_phase_bc_and_says_it_cannot_verify(tmp_path, caplog):
    _finals(tmp_path, DATE)
    with caplog.at_level(logging.WARNING):
        out, _, _ = _invalidate(tmp_path, _state())
    assert out == ["phase_a", "phase_bc_day_first", "phase_d"]
    assert "cannot verify" in " ".join(r.getMessage() for r in caplog.records)


def test_a_plan_from_other_code_re_enters_phase_bc_where_it_is_rebuilt(tmp_path, caplog):
    """A present but unloadable plan is not "no verdict": phase_bc rebuilds it (a rebuilt plan
    is stamped with the current version, so it loads next time and this cannot loop)."""
    _write_plan(tmp_path, [_lf(1, 0), _lf(2, 1)], version=PHASE_BC_CODE_VERSION + 1)
    _finals(tmp_path, DATE)
    with caplog.at_level(logging.WARNING):
        out, done, state = _invalidate(tmp_path, _state())
    assert out == ["phase_a"]
    assert done == ["phase_a"]
    assert state.phases_completed == ["phase_a"]
    # Judge the invalidator's own WARNING by logger, level and args, not by words: the plan
    # loader (`_load_current_drive_plan`) also logs "drive_plan.json ... rebuilding", which
    # satisfied a text match with the invalidator's warning demoted to debug.
    own = [
        r
        for r in caplog.records
        if r.name == "gomoveshift.pipeline.drive_orchestrator" and r.levelno == logging.WARNING
    ]
    assert len(own) == 1 and own[0].args == (tmp_path,)


def test_a_corrupt_plan_re_enters_phase_bc(tmp_path):
    (tmp_path / "drive_plan.json").write_text("{not json")
    _finals(tmp_path, DATE)
    out, _, _ = _invalidate(tmp_path, _state())
    assert out == ["phase_a"]


def test_a_plan_rebuilt_by_phase_bc_loads_so_the_re_entry_cannot_loop(tmp_path):
    """phase_bc stamps a rebuilt plan with PHASE_BC_CODE_VERSION; with its finals present the
    next resume has a verdict and keeps phase_bc."""
    _write_plan(tmp_path, [_lf(1, 0), _lf(2, 1)])  # the stamp phase_bc gives a rebuilt plan
    _finals(tmp_path, DATE, indices=(1, 2))
    out, _, _ = _invalidate(tmp_path, _state())
    assert out == ["phase_a", "phase_bc_day_first", "phase_d"]


def test_every_planned_final_present_keeps_phase_bc(tmp_path):
    _write_plan(tmp_path, [_lf(1, 0), _lf(2, 1)])
    _finals(tmp_path, DATE, indices=(1, 2))
    out, _, _ = _invalidate(tmp_path, _state())
    assert out == ["phase_a", "phase_bc_day_first", "phase_d"]


def test_a_day_not_recorded_complete_is_never_judged(tmp_path):
    """The plan holds LFs for days this run never recorded complete: no re-entry loop."""
    _write_plan(tmp_path, [_lf(1, 0), _lf(2, 2)])
    _finals(tmp_path, DATE, indices=(1,))
    _finals(tmp_path, OTHER)
    out, _, _ = _invalidate(tmp_path, _state(recorded=(DATE,)))
    assert out == ["phase_a", "phase_bc_day_first", "phase_d"]


def test_phase_bc_not_resumed_is_a_noop(tmp_path):
    _write_plan(tmp_path, [_lf(1, 0)])
    state = _state()
    resumed = ["phase_a"]
    assert (
        invalidate_phase_bc_if_lf_shortfall(
            state, resumed, list(resumed), stages_dir=tmp_path, config=Config()
        )
        == resumed
    )


# --- minor 4: a day with no plan -----------------------------------------------


def _config(mode: str):
    cfg = Config()
    cfg.drive.lf_selection_mode = mode
    return cfg


@pytest.mark.parametrize(
    ("mode", "finals", "reenters"),
    [
        # full_drive synthesises >= lf_01 for a day with no plan (day_pipeline.py), so a
        # recorded day without it dropped a long-form; best_moments loops over no plans, so
        # an LF-less day is correct there.
        ("full_drive", (), True),
        ("full_drive", (1,), False),
        ("best_moments", (), False),
    ],
)
def test_planless_day_follows_the_lf_mode(tmp_path, mode, finals, reenters):
    _write_plan(tmp_path, [_lf(1, 2)])  # the plan covers only the other day
    _finals(tmp_path, DATE, indices=finals)
    out, _, _ = _invalidate(tmp_path, _state(), _config(mode))
    assert out == (["phase_a"] if reenters else ["phase_a", "phase_bc_day_first", "phase_d"])


def _partition(tmp_path, *, mode, plans, final_lfs, recorded=True, empty=()):
    stages = tmp_path / "_stages"
    day = _finals(stages, DATE, indices=final_lfs, empty=empty)
    master = day / f"day_{DATE}_master.mp4"
    master.write_bytes(b"m")
    state = make_state(
        per_day_completed={DATE: {"master_path": str(master)}} if recorded else {},
        allow_short_count=True,
    )
    batch = SimpleNamespace(date=DATE, total_duration=0.0, episodes=[])
    phase = PhaseBCDayFirst(_config(mode))
    _, skipped, remaining = phase._partition_batches_by_resume_state(
        state, [batch], stages, lf_plans_by_date=plans
    )
    return skipped, remaining


@pytest.mark.parametrize(
    ("mode", "plans", "finals", "empty", "reruns"),
    [
        ("full_drive", {}, (), (), True),  # plan-less full_drive owes lf_01
        ("best_moments", {}, (), (), False),  # plan-less best_moments is legitimately LF-less
        ("best_moments", {DATE: [_lf(1, 0), _lf(2, 1)]}, (1, 3), (), True),  # indices, not a count
        ("best_moments", {DATE: [_lf(1, 0), _lf(2, 1)]}, (1,), (2,), True),  # zero bytes = missing
    ],
)
def test_partition_guard_rule(tmp_path, mode, plans, finals, empty, reruns):
    skipped, remaining = _partition(tmp_path, mode=mode, plans=plans, final_lfs=finals, empty=empty)
    assert [b.date for b in remaining] == ([DATE] if reruns else [])
    assert [a.date for a in skipped] == ([] if reruns else [DATE])


def test_partition_guard_only_judges_a_recorded_day(tmp_path, caplog):
    """An unrecorded day is re-run by the legacy path and must not be reported as a
    day "recorded complete" that fell short."""
    plans = {DATE: [_lf(1, 0), _lf(2, 1)]}
    with caplog.at_level(logging.WARNING):
        skipped, remaining = _partition(
            tmp_path, mode="best_moments", plans=plans, final_lfs=(1,), recorded=False
        )
    assert skipped == [] and [b.date for b in remaining] == [DATE]
    assert "recorded complete" not in " ".join(r.getMessage() for r in caplog.records)


# --- the production call sites -------------------------------------------------


class _StopHereError(Exception):
    pass


def _resume_state(orch, tmp_path, *, finals):
    """A Galley-shaped checkpoint on disk: returns (source, output_base, stages)."""
    import numpy as np

    source = tmp_path / "source"
    source.mkdir()
    output_base = tmp_path / "out"
    _, stages = orch._resolve_paths(source, output_base)
    stages.mkdir(parents=True, exist_ok=True)
    np.savez(
        stages / "drive_scores.npz",
        scores=np.zeros((10, 9), dtype=np.float32),
        num_files=2,
        total_duration=700.0,
        file_durations=np.array([300.0, 400.0]),
    )
    _write_plan(stages, [_lf(1, 0), _lf(2, 1)])
    _finals(stages, DATE, indices=finals)
    master = stages / f"day_{DATE}" / f"day_{DATE}_master.mp4"
    master.write_bytes(b"m")
    (stages / "pipeline_state.json").write_text(
        json.dumps(
            {
                "source_dir": str(source),
                "output_base": str(output_base),
                "region": "usa",
                "publish_day": "wed",
                "location": "Test Route",
                "phases_completed": ["phase_a", "phase_bc_day_first"],
                "episode_locations": ["A, SC", "B, SC"],
                "episode_profiles": [{"title": "A"}, {"title": "B"}],
                "episode_files": ["/fake/ep0.mp4", "/fake/ep1.mp4"],
                "gpx_files": [None, None],
                "episode_dates": [DATE, DATE],
                "file_durations": [300.0, 400.0],
                "file_boundaries": [[0.0, 300.0], [300.0, 700.0]],
                "per_day_completed": {DATE: {"master_path": str(master)}},
                "regular_lf_paths": [],
                "shorts_per_lf": {},
                "run_start": "2026-04-19T08:00:00+00:00",
            }
        )
    )
    return source, output_base, stages


def _run_resume(tmp_path, *, finals):
    orch = DrivePipelineOrchestrator(config=Config())
    source, output_base, _ = _resume_state(orch, tmp_path, finals=finals)
    ran = []

    def _bc(state):
        ran.append("phase_bc_day_first")
        state["day_artifacts"] = [object()]
        state["regular_lf_paths"] = []
        state["shorts_per_lf"] = {}

    with (
        patch.object(DrivePipelineOrchestrator, "_phase_a_ingest_and_analyze"),
        patch.object(DrivePipelineOrchestrator, "_phase_bc_day_first", side_effect=_bc),
        patch.object(DrivePipelineOrchestrator, "_phase_d_metadata_and_publish"),
        patch("gomoveshift.pipeline.drive_orchestrator.ReviewGate"),
    ):
        orch.run(source_dir=source, output_base=output_base, region="usa", publish_day="wed")
    return ran


def test_orchestrator_resume_reenters_phase_bc_for_the_galley_state(tmp_path):
    assert _run_resume(tmp_path, finals=()) == ["phase_bc_day_first"]


def test_orchestrator_resume_reenters_phase_bc_when_the_plan_cannot_be_loaded(tmp_path):
    """Galley-shaped checkpoint whose drive_plan.json is from other code: phase_bc runs."""
    orch = DrivePipelineOrchestrator(config=Config())
    source, output_base, stages = _resume_state(orch, tmp_path, finals=())
    _write_plan(stages, [_lf(1, 0), _lf(2, 1)], version=PHASE_BC_CODE_VERSION + 1)
    ran = []

    def _bc(state):
        ran.append("phase_bc_day_first")
        state["day_artifacts"] = [object()]
        state["regular_lf_paths"] = []
        state["shorts_per_lf"] = {}

    with (
        patch.object(DrivePipelineOrchestrator, "_phase_a_ingest_and_analyze"),
        patch.object(DrivePipelineOrchestrator, "_phase_bc_day_first", side_effect=_bc),
        patch.object(DrivePipelineOrchestrator, "_phase_d_metadata_and_publish"),
        patch("gomoveshift.pipeline.drive_orchestrator.ReviewGate"),
    ):
        orch.run(source_dir=source, output_base=output_base, region="usa", publish_day="wed")
    assert ran == ["phase_bc_day_first"]


def test_orchestrator_resume_keeps_phase_bc_when_every_lf_is_on_disk(tmp_path):
    assert _run_resume(tmp_path, finals=(1, 2)) == []


@pytest.mark.overpass_answers_empty
def test_phase_bc_hands_the_plan_to_the_partition_guard(tmp_path):
    """The call site passes lf_plans_by_date (dropping it skips the short day)."""
    import numpy as np

    orch = DrivePipelineOrchestrator(config=Config())
    _, output_base, stages = _resume_state(orch, tmp_path, finals=())
    ep = [tmp_path / f"ep_{i}.mp4" for i in range(2)]
    for p in ep:
        p.write_bytes(b"fake")
    state = make_state(
        source_dir=tmp_path,
        output_base=output_base,
        stages_base=stages,
        region="usa",
        publish_day="sat",
        location="A to B, SC",
        episode_files=[str(p) for p in ep],
        episode_dates=[DATE, DATE],
        file_durations=[300.0, 400.0],
        file_boundaries=[(0.0, 300.0), (300.0, 700.0)],
        episode_locations=["A, SC", "A, SC"],
        gpx_files=[None, None],
        drive_scores=np.zeros((700, 9), dtype=np.float64),
        per_day_completed={
            DATE: {"master_path": str(stages / f"day_{DATE}" / f"day_{DATE}_master.mp4")}
        },
    )
    state.per_lf_overrides = {i: f"Route {i}, SC" for i in range(0, 21)}
    seen = {}

    def _capture(**kwargs):
        seen.update(kwargs)
        raise _StopHereError

    with (
        patch(
            "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
            side_effect=_capture,
        ),
        patch.object(DrivePipelineOrchestrator, "_probe_duration", return_value=700.0),
        # The recorded master passes its own integrity gate: only the LF shortfall can
        # send this day back to run_all_days.
        patch(
            "gomoveshift.pipeline.drive_phases.phase_bc_day_first._probe_master_integrity",
            return_value=True,
        ),
        pytest.raises(_StopHereError),
        patch.object(
            PhaseBCDayFirst,
            "_partition_batches_by_resume_state",
            autospec=True,
            side_effect=PhaseBCDayFirst._partition_batches_by_resume_state,
        ) as spy,
    ):
        orch._phase_bc_day_first(state)
    plans = spy.call_args.kwargs["lf_plans_by_date"]
    assert [lf.index for lf in plans[DATE]] == [1, 2]
    assert [b.date for b in seen["day_batches"]] == [DATE]


@pytest.fixture(autouse=True)
def _quiet(caplog):
    caplog.set_level(logging.WARNING)
