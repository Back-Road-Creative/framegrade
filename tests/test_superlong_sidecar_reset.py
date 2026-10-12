"""Superlong sidecar coverage for ``_run_sidecar_integrity_gate`` and its
``SidecarIntegrityError`` messaging.

Three defects, all on the superlong (``superlong_uploads.json``) path:

* ``--force-sidecar-reset`` wiped only ``youtube_uploads.json``, leaving a
  stale ``superlong_uploads.json`` in place.
* The gate returned early whenever ``drive_plan.regular_lfs`` was empty,
  so a superlong-only state got no integrity check at all.
* ``SidecarIntegrityError`` hardcoded ``youtube_uploads.json`` into its
  message even when the failing sidecar was ``superlong_uploads.json``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gomoveshift.pipeline.drive_phases.phase_d_gates import (
    _run_sidecar_integrity_gate,
)
from gomoveshift.pipeline.exceptions import SidecarIntegrityError
from gomoveshift.pipeline.youtube_sidecar import (
    assert_superlong_sidecar_integrity,
    load_superlong_sidecar,
    save_superlong_sidecar,
)

from _state_helpers import make_state


def _lf(idx, *, total_duration=1800.0):
    return SimpleNamespace(index=idx, total_duration=total_duration)


def _superlong_parts():
    return {
        "2026-01-01": {"records": [{"path": "/tmp/part0.mp4", "part_index": 0}]},
    }


def test_force_reset_wipes_superlong_sidecar_too(tmp_path):
    save_superlong_sidecar(tmp_path, {"2026-01-01:00": {"video_id": "old-part"}})
    plan = SimpleNamespace(regular_lfs=[_lf(1)])
    state = make_state(superlong_parts=_superlong_parts())

    _run_sidecar_integrity_gate(
        state=state,
        drive_plan=plan,
        output_base=tmp_path,
        force_sidecar_reset=True,
    )

    assert load_superlong_sidecar(tmp_path) == {}


def test_gate_runs_for_superlong_only_state_with_no_regular_lfs(tmp_path):
    # Sidecar disagrees with the plan: a phantom part key.
    save_superlong_sidecar(tmp_path, {"2026-01-01:99": {"video_id": "phantom"}})
    state = make_state(superlong_parts=_superlong_parts())

    with pytest.raises(SidecarIntegrityError):
        _run_sidecar_integrity_gate(
            state=state,
            drive_plan=None,
            output_base=tmp_path,
            force_sidecar_reset=False,
        )


def test_superlong_integrity_error_names_its_own_file(tmp_path):
    save_superlong_sidecar(tmp_path, {"2026-01-01:99": {"video_id": "phantom"}})
    state = make_state(superlong_parts=_superlong_parts())

    with pytest.raises(SidecarIntegrityError) as exc_info:
        assert_superlong_sidecar_integrity(tmp_path, state)

    message = str(exc_info.value)
    assert "superlong_uploads.json integrity check failed" in message
    assert "youtube_uploads.json integrity check failed" not in message
