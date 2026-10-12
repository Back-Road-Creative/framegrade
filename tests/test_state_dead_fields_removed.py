"""Declared-but-never-written ``PipelineState`` fields must stay deleted.

``keep_segments_per_file``, ``lf_metadata``, ``early_plan`` and
``lf_drive_links`` were declared on ``PipelineState`` but no production
code path ever assigned them — no ``state.<field> =``, no constructor
kwarg, no ``flag_registry`` row. A declared-but-never-written field is
indistinguishable at runtime from a field that does not exist (it is
permanently its default), except that it passes an "is it declared?"
check and invites new readers. ``lf_drive_links`` had already grown one:
the orchestrator copied it into ``DrivePipelineResult.lf_drive_links``,
which was therefore always ``{}``.

These tests pin the removal and prove the plumbing that referenced the
dead names (``_INT_KEY_DICT_FIELDS``, the dataclass revive map) still
round-trips every surviving field.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from gomoveshift.models import DrivePipelineResult, DrivePlan
from gomoveshift.pipeline.drive_state_model import PipelineState

DEAD_FIELDS = (
    "keep_segments_per_file",
    "lf_metadata",
    "early_plan",
    "lf_drive_links",
)


def test_pipeline_state_does_not_declare_dead_fields():
    """None of the four never-written names may be declared again."""
    declared = {f.name for f in dataclasses.fields(PipelineState)}
    still_present = sorted(name for name in DEAD_FIELDS if name in declared)
    assert still_present == [], f"PipelineState re-declares never-written field(s): {still_present}"


def test_pipeline_state_json_round_trip_survives_the_removal():
    """Every surviving field still round-trips through JSON intact.

    Removing the dead names also removed an ``_INT_KEY_DICT_FIELDS``
    entry (``lf_metadata``) and a revive-map entry (``early_plan``);
    this is the proof neither removal disturbed a live field.
    """
    plan = DrivePlan(
        source_files=[],
        processed_files=[],
        regular_lfs=[],
        score_threshold=0.42,
        total_source_duration=1800.0,
        total_highlight_duration=600.0,
    )
    state = PipelineState(
        source_dir=Path("/tmp/src"),
        output_base=Path("/tmp/out"),
        region="ireland",
        publish_day="wed",
        drive_plan=plan,
        shorts_per_lf={0: ["a"], 1: ["b"]},
        route_profiles={"all": {"kind": "coastal"}},
        per_lf_titles={0: "Part 1"},
    )

    restored = PipelineState.from_dict(json.loads(json.dumps(state.to_dict())))

    # Every declared field survives — no silent drop.
    for f in dataclasses.fields(PipelineState):
        assert hasattr(restored, f.name), f"{f.name} vanished across the round trip"

    assert restored.source_dir == Path("/tmp/src")
    assert restored.output_base == Path("/tmp/out")
    assert restored.region == "ireland"
    assert restored.publish_day == "wed"
    assert isinstance(restored.drive_plan, DrivePlan)
    assert restored.drive_plan.score_threshold == 0.42
    # Int keys rehydrate for the surviving int-keyed dict fields.
    assert restored.shorts_per_lf == {0: ["a"], 1: ["b"]}
    assert restored.per_lf_titles == {0: "Part 1"}
    assert restored.route_profiles == {"all": {"kind": "coastal"}}


def test_drive_pipeline_result_drops_lf_drive_links():
    """The always-empty result field is gone, and the result still builds."""
    declared = {f.name for f in dataclasses.fields(DrivePipelineResult)}
    assert "lf_drive_links" not in declared, (
        "DrivePipelineResult.lf_drive_links was permanently {} — no writer exists"
    )

    result = DrivePipelineResult(
        success=True,
        source_dir=Path("/tmp/src"),
        output_dir=Path("/tmp/out"),
        drive_plan=None,
        regular_lf_paths=[],
        shorts_per_lf={},
        stages_completed=["A"],
        errors=[],
        timings={},
    )
    assert result.success is True
    assert result.stages_completed == ["A"]
