"""State-model contract round-trip tests.

Locks the structural close-out of the dual-shape contract bug class:

1. ``PipelineState.from_dict(state.to_dict())`` must rehydrate every
   compound field (``drive_plan``, ``drive_schedule``, ``upload_manifests``)
   to its declared dataclass shape — not the dict-form that JSON
   serialization produces. A past
   bug traced to ``upload_manifests`` coming back as ``dict[int, dict]``
   and ``getattr(manifest, "files", {})`` returning ``{}`` silently.
2. ``PipelineState.get(key, default)`` must distinguish field-absent
   from field-present-but-None — matching Python's own ``dict.get``.
3. ``UploadManifest.from_dict(m.to_dict())`` must round-trip cleanly.
4. If a real on-disk ``pipeline_state.json`` is available (developer
   workstation / CI fixture), confirm it loads with dataclass-shaped
   compound fields end-to-end. Skipped when no real state file is
   reachable from the test environment.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pytest

from gomoveshift.models import DrivePlan
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
        computed_at="2026-05-26T08:00:00",
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


def _make_drive_plan() -> DrivePlan:
    """DrivePlan with empty inner lists — enough to exercise from_dict."""
    return DrivePlan(
        source_files=[],
        processed_files=[],
        regular_lfs=[],
        score_threshold=0.5,
        total_source_duration=120.0,
        total_highlight_duration=60.0,
    )


# ─── Round-trip preserves dataclass shape ────────────────────────────────


def test_from_dict_to_dict_roundtrip_preserves_dataclass_shape():
    """All declared compound fields must rehydrate to their dataclass type."""
    state = PipelineState(
        source_dir=Path("/tmp/src"),
        output_base=Path("/tmp/out"),
        drive_plan=_make_drive_plan(),
        drive_schedule=_make_schedule(),
        upload_manifests={
            0: _make_manifest(0),
            1: _make_manifest(1),
        },
    )

    # Full JSON round-trip — the realistic resume path.
    payload = json.loads(json.dumps(state.to_dict()))
    restored = PipelineState.from_dict(payload)

    assert isinstance(restored.drive_plan, DrivePlan), (
        f"drive_plan should be DrivePlan, got {type(restored.drive_plan).__name__}"
    )
    assert isinstance(restored.drive_schedule, DriveReleaseSchedule), (
        f"drive_schedule should be DriveReleaseSchedule, "
        f"got {type(restored.drive_schedule).__name__}"
    )
    assert isinstance(restored.upload_manifests, dict)
    for k, v in restored.upload_manifests.items():
        assert isinstance(k, int), f"upload_manifests key {k!r} is {type(k).__name__}"
        assert isinstance(v, UploadManifest), (
            f"upload_manifests[{k}] is {type(v).__name__}, expected UploadManifest"
        )

    # Field values survive the round trip.
    assert restored.drive_schedule.region == "ireland"
    assert restored.drive_schedule.computed_at == "2026-05-26T08:00:00"
    assert restored.upload_manifests[0].project_name == "proj_lf00"
    assert restored.upload_manifests[1].files["short_01_a.mp4"].startswith("https://")


# ─── get() distinguishes missing vs None-valued ──────────────────────────


def test_get_distinguishes_missing_from_none():
    """``get`` mirrors ``dict.get`` semantics."""
    state = PipelineState()
    # phases_completed defaults to None on the dataclass.
    assert state.get("phases_completed", "default-value") is None, (
        "Contract: a None-valued attribute must return None even with a default."
    )
    assert state.get("nonexistent_field", "default-value") == "default-value"


def test_get_no_default_returns_none_for_missing():
    """``state.get('missing')`` returns ``None`` — no AttributeError, no raise."""
    state = PipelineState()
    assert state.get("nonexistent_field") is None
    # And for None-valued declared fields:
    assert state.get("phases_completed") is None


def test_get_returns_real_value_when_set():
    """Non-None values pass through unchanged regardless of default."""
    state = PipelineState(region="ireland")
    assert state.get("region") == "ireland"
    assert state.get("region", "fallback") == "ireland"


# ─── UploadManifest.from_dict / to_dict symmetry ─────────────────────────


def test_upload_manifest_from_dict_to_dict_roundtrip():
    """Symmetric round-trip for the dataclass behind that bug."""
    original = _make_manifest(7)
    restored = UploadManifest.from_dict(original.to_dict())
    assert restored == original
    # Attribute access (the failure mode the dict-shape bug surfaced).
    assert restored.files == original.files
    assert restored.completed is original.completed
    assert restored.folder_id == original.folder_id


# ─── Real on-disk pipeline_state.json (operational integration check) ────


def _find_real_state_file() -> Path | None:
    """Return a real pipeline_state.json if one is reachable, else None."""
    candidates: list[str] = []
    # Common test-fixture paths.
    fixture_dir = Path(__file__).resolve().parent / "fixtures"
    if fixture_dir.exists():
        candidates.extend(glob.glob(str(fixture_dir / "**/pipeline_state.json"), recursive=True))
    # Developer-workstation staging cache.
    home = os.environ.get("HOME") or os.path.expanduser("~")
    if home and Path(home).exists():
        candidates.extend(glob.glob(f"{home}/.cache/gomoveshift/*/_stages/pipeline_state.json"))
    for c in candidates:
        p = Path(c)
        if p.is_file():
            return p
    return None


def test_real_pipeline_state_json_roundtrips_cleanly():
    """Smoke-test against a real saved state file when available.

    The most operationally-important test: if a developer ran a pipeline
    recently, prove the saved state loads with dataclass-shaped compound
    fields. Skipped on CI / fresh environments where no real state exists.
    """
    state_path = _find_real_state_file()
    if state_path is None:
        pytest.skip("No real pipeline_state.json reachable from this environment")

    try:
        with state_path.open() as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        pytest.skip(f"Could not read {state_path}: {e}")

    restored = PipelineState.from_dict(payload)

    # Compound fields: dataclass shape OR None (field never populated).
    if restored.drive_plan is not None:
        assert isinstance(restored.drive_plan, DrivePlan), (
            f"drive_plan from {state_path} is {type(restored.drive_plan).__name__}"
        )
    if restored.drive_schedule is not None:
        assert isinstance(restored.drive_schedule, DriveReleaseSchedule), (
            f"drive_schedule from {state_path} is {type(restored.drive_schedule).__name__}"
        )
    if restored.upload_manifests:
        for lf_idx, manifest in restored.upload_manifests.items():
            assert isinstance(lf_idx, int), (
                f"upload_manifests key {lf_idx!r} from {state_path} "
                f"is {type(lf_idx).__name__}, expected int"
            )
            assert isinstance(manifest, UploadManifest), (
                f"upload_manifests[{lf_idx}] from {state_path} is "
                f"{type(manifest).__name__}, expected UploadManifest"
            )
