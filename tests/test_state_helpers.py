"""Sanity tests for ``tests/_state_helpers.make_state``.

Pins the contract that the fixture helper:

1. Returns a real ``PipelineState`` instance (not a dict).
2. Accepts dict-shaped compound fields and coerces them via ``from_dict``.
3. Passes already-constructed dataclass instances through unchanged.
4. Coerces ``upload_manifests`` dict values + int-key coercion.
5. Runs ``__post_init__`` (so disk-derivable migrations fire on construction).
"""

from __future__ import annotations

from pathlib import Path

from gomoveshift.models import DrivePlan
from gomoveshift.pipeline.drive_state_model import PipelineState
from gomoveshift.pipeline.uploader import UploadManifest

from _state_helpers import make_state


def test_returns_pipeline_state_instance(tmp_path: Path):
    state = make_state(output_base=tmp_path, region="kerry")
    assert isinstance(state, PipelineState)
    assert state.output_base == tmp_path
    assert state.region == "kerry"


def test_empty_kwargs_returns_default_state():
    state = make_state()
    assert isinstance(state, PipelineState)
    assert state.region == ""
    assert state.publish_day == "wed"


def test_drive_plan_dict_coerced_via_from_dict():
    state = make_state(
        drive_plan={
            "source_files": [],
            "processed_files": [],
            "regular_lfs": [],
            "score_threshold": 0.5,
            "total_source_duration": 0.0,
            "total_highlight_duration": 0.0,
        }
    )
    assert isinstance(state.drive_plan, DrivePlan)
    assert state.drive_plan.regular_lfs == []


def test_drive_plan_instance_passes_through():
    plan = DrivePlan(
        source_files=[],
        processed_files=[],
        regular_lfs=[],
        score_threshold=0.5,
        total_source_duration=0.0,
        total_highlight_duration=0.0,
    )
    state = make_state(drive_plan=plan)
    assert state.drive_plan is plan


def test_upload_manifests_dict_value_coerced():
    state = make_state(upload_manifests={0: {"project_name": "p0", "files": {"a.mp4": "link"}}})
    assert isinstance(state.upload_manifests[0], UploadManifest)
    assert state.upload_manifests[0].project_name == "p0"
    assert state.upload_manifests[0].files == {"a.mp4": "link"}


def test_upload_manifests_instance_passes_through():
    manifest = UploadManifest(project_name="p2", files={"b.mp4": "link"})
    state = make_state(upload_manifests={2: manifest})
    assert state.upload_manifests[2] is manifest


def test_upload_manifests_string_keys_coerced_to_int():
    # JSON round-trip leaves int keys as strings — make_state mirrors
    # PipelineState.from_dict by coercing them back to int.
    state = make_state(upload_manifests={"3": {"project_name": "p3"}})
    assert 3 in state.upload_manifests
    assert isinstance(state.upload_manifests[3], UploadManifest)


def test_dict_protocol_still_works_for_migration_phase():
    """While fixtures migrate, the dict-protocol shim is still
    present on PipelineState. Verify make_state output supports it so we
    can migrate fixtures incrementally without forcing same-PR rewrites
    of read-side test code."""
    state = make_state(output_base=Path("/tmp/x"), region="kerry")
    assert state["region"] == "kerry"
    assert state.get("publish_day") == "wed"
    assert "region" in state
