"""Compound-field re-hydration contract for ``PipelineState``.

Closes the bug class where plain dicts came back from a reload:
``state["upload_manifests"]`` and ``state["drive_schedule"]`` must
round-trip through ``to_dict()``/``from_dict()`` as their declared
dataclass shapes, not as plain dicts. Consumers reach for attribute
access (``manifest.files``, ``schedule.lf_slots``); the contract
enforces the shape at load time so the ``_manifest_files()`` shim is
unreachable by construction.
"""

from __future__ import annotations

import json
from pathlib import Path

from gomoveshift.pipeline.drive_state_model import PipelineState
from gomoveshift.pipeline.scheduler import (
    DriveReleaseSchedule,
    ScheduledSlot,
)
from gomoveshift.pipeline.uploader import UploadManifest


def _make_schedule() -> DriveReleaseSchedule:
    return DriveReleaseSchedule(
        region="ireland",
        lf_slots=[
            ScheduledSlot(
                short_index=0,
                date="2026-02-04",
                time="11:00:00",
                datetime_est="2026-02-04T11:00:00-05:00",
                day_of_week="Wednesday",
            ),
            ScheduledSlot(
                short_index=0,
                date="2026-02-11",
                time="11:00:00",
                datetime_est="2026-02-11T11:00:00-05:00",
                day_of_week="Wednesday",
            ),
        ],
        shorts_slots=[
            ScheduledSlot(
                short_index=1,
                date="2026-02-05",
                time="12:00:00",
                datetime_est="2026-02-05T12:00:00-05:00",
                day_of_week="Thursday",
            ),
        ],
    )


def _make_manifest(lf_idx: int) -> UploadManifest:
    return UploadManifest(
        project_name=f"proj_lf{lf_idx:02d}",
        folder_id=f"folder_{lf_idx}",
        files={
            f"short_{lf_idx:02d}_a.mp4": f"https://drive.google.com/lf{lf_idx}_a",
            f"short_{lf_idx:02d}_b.mp4": f"https://drive.google.com/lf{lf_idx}_b",
        },
        completed=True,
        created_at="2026-05-12T00:00:00",
        updated_at="2026-05-12T01:00:00",
    )


def test_drive_schedule_roundtrips_to_dataclass():
    """``drive_schedule`` must load back as a ``DriveReleaseSchedule``."""
    state = PipelineState(
        source_dir=Path("/tmp/src"),
        output_base=Path("/tmp/out"),
        drive_schedule=_make_schedule(),
    )
    restored = PipelineState.from_dict(state.to_dict())
    assert isinstance(restored.drive_schedule, DriveReleaseSchedule), (
        f"expected DriveReleaseSchedule, got {type(restored.drive_schedule).__name__}"
    )
    # Attribute access — NOT subscript. This is the contract.
    assert restored.drive_schedule.region == "ireland"
    assert len(restored.drive_schedule.lf_slots) == 2
    assert isinstance(restored.drive_schedule.lf_slots[0], ScheduledSlot)
    assert restored.drive_schedule.lf_slots[0].date == "2026-02-04"
    assert len(restored.drive_schedule.shorts_slots) == 1
    assert restored.drive_schedule.shorts_slots[0].short_index == 1


def test_drive_schedule_survives_json_serialization():
    """End-to-end JSON round-trip — the realistic resume path."""
    state = PipelineState(
        source_dir=Path("/tmp/src"),
        output_base=Path("/tmp/out"),
        drive_schedule=_make_schedule(),
    )
    payload = json.loads(json.dumps(state.to_dict()))
    restored = PipelineState.from_dict(payload)
    assert isinstance(restored.drive_schedule, DriveReleaseSchedule)
    assert restored.drive_schedule.lf_slots[0].datetime_est == "2026-02-04T11:00:00-05:00"


def test_upload_manifests_roundtrips_to_dataclass():
    """Each value of ``upload_manifests`` must rehydrate to ``UploadManifest``."""
    state = PipelineState(
        source_dir=Path("/tmp/src"),
        output_base=Path("/tmp/out"),
        upload_manifests={1: _make_manifest(1), 2: _make_manifest(2)},
    )
    restored = PipelineState.from_dict(state.to_dict())
    assert isinstance(restored.upload_manifests, dict)
    assert set(restored.upload_manifests.keys()) == {1, 2}
    for lf_idx, manifest in restored.upload_manifests.items():
        assert isinstance(manifest, UploadManifest), (
            f"LF {lf_idx} manifest is {type(manifest).__name__}, expected UploadManifest"
        )
        # Attribute access — the failure mode that the deleted
        # ``_manifest_files()`` shim used to paper over.
        assert manifest.files
        assert len(manifest.files) == 2


def test_upload_manifests_str_key_resume():
    """JSON-loaded keys come back as strings; both keys AND values must coerce."""
    state = PipelineState(
        source_dir=Path("/tmp/src"),
        output_base=Path("/tmp/out"),
        upload_manifests={0: _make_manifest(0), 1: _make_manifest(1)},
    )
    payload = json.loads(json.dumps(state.to_dict()))
    # Confirm the on-disk shape is what we expect (string keys).
    assert set(payload["upload_manifests"].keys()) == {"0", "1"}
    restored = PipelineState.from_dict(payload)
    assert set(restored.upload_manifests.keys()) == {0, 1}
    for manifest in restored.upload_manifests.values():
        assert isinstance(manifest, UploadManifest)
        assert manifest.files
        # And attribute access continues to work.
        assert manifest.completed is True


def test_pipelinestate_update_preserves_dataclass_shape():
    """``PipelineState.update`` from a raw dict must rehydrate compound fields."""
    state = PipelineState(source_dir=Path("/tmp/src"), output_base=Path("/tmp/out"))
    # Simulate the JSON-loaded payload path (string keys, dict values).
    raw_payload = {
        "upload_manifests": {
            "0": _make_manifest(0).to_dict(),
            "1": _make_manifest(1).to_dict(),
        },
        "drive_schedule": _make_schedule().to_dict(),
    }
    state.update(raw_payload)
    assert isinstance(state.drive_schedule, DriveReleaseSchedule)
    assert state.drive_schedule.region == "ireland"
    assert isinstance(state.upload_manifests, dict)
    for k, v in state.upload_manifests.items():
        assert isinstance(k, int)
        assert isinstance(v, UploadManifest)
        assert v.files


def test_manifest_files_shim_is_deleted():
    """The ``_manifest_files()`` dual-shape helper has been deleted.

    ``_coerce_dict_value_dataclasses`` guarantees every
    ``upload_manifests`` value is an ``UploadManifest`` dataclass at load
    time, so the dict-shape AttributeError the helper papered over cannot
    recur. Direct attribute reads (``prior_manifest.files``) are now the
    single contract path. This test guards against regression — if anyone
    re-introduces the helper, the load-time normalization is being
    bypassed somewhere upstream and that's the bug to fix.
    """
    # The upload loop was extracted into external_systems_coordinator.py;
    # check both modules to keep the dual-shape regression guard intact across the cut.
    drive_phases = Path(__file__).resolve().parent.parent / "gomoveshift/pipeline/drive_phases"
    src = (
        (drive_phases / "phase_d.py").read_text()
        + "\n"
        + (drive_phases / "external_systems_coordinator.py").read_text()
    )
    assert "def _manifest_files(" not in src, (
        "phase_d/coordinator still defines _manifest_files — the "
        "dual-shape helper was deleted. Load-time rehydration is the single contract path."
    )
    # The direct attribute-access form is now the contract.
    assert "prior_manifest.files" in src, (
        "phase_d/coordinator no longer reads prior_manifest.files directly — the "
        "conversion to attribute access on the load-normalized dataclass "
        "appears to have been reverted."
    )


def test_drive_schedule_none_roundtrips_cleanly():
    """When ``drive_schedule`` is unset, the round-trip must not raise."""
    state = PipelineState(source_dir=Path("/tmp/src"), output_base=Path("/tmp/out"))
    restored = PipelineState.from_dict(state.to_dict())
    assert restored.drive_schedule is None


def test_upload_manifests_empty_dict_roundtrips():
    state = PipelineState(
        source_dir=Path("/tmp/src"),
        output_base=Path("/tmp/out"),
        upload_manifests={},
    )
    restored = PipelineState.from_dict(state.to_dict())
    # An empty dict is preserved as-is (no rehydration needed).
    assert restored.upload_manifests == {} or restored.upload_manifests is None
