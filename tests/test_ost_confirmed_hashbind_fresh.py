"""OST ``.confirmed`` staleness hardening.

Two robust guards so a leftover ``.confirmed`` marker can never confirm a
fresh run's never-reviewed burn text:

* ``--fresh`` purges ``*.confirmed`` alongside the other stale stage artifacts
  (``_purge_stale_stage_artifacts``) — a marker from a prior run cannot survive.
* A ``.confirmed`` is bound to the audit hash it confirmed (``stamp_confirmed``
  on pass), so ``exists()`` alone can't validate a materially different burn
  text: a stale binding reopens the gate. A legacy/hand-created marker (non-hash
  content) carries no binding and passes as a first-ever confirmation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
from gomoveshift.pipeline.ost_audit_sidecar import (
    OstAuditPendingError,
    compute_shorts_ost_audit_hash,
    format_shorts_ost_audit,
    read_confirmed_hash,
    shorts_gate_or_raise,
    stamp_confirmed,
)

DATE = "2025-09-14"


def _entries():
    return [
        {
            "rank": 1,
            "parent_lf_index": 0,
            "ost_text": "Killarney to Dingle",
            "start_sec": 10.0,
            "end_sec": 40.0,
        }
    ]


def test_fresh_purges_stale_ost_confirmed_markers(tmp_path):
    """``--fresh`` must delete leftover ``*.confirmed`` markers, not just the
    plan/state — otherwise a prior run's confirmation validates a fresh,
    never-reviewed burn (``confirmation_is_stale(None, x)`` is False)."""
    stages_dir = tmp_path / "stages"
    stages_dir.mkdir()
    output_base = tmp_path / "out"
    output_base.mkdir()

    ost_confirmed = stages_dir / f"ost_audit_{DATE}.confirmed"
    shorts_confirmed = stages_dir / f"shorts_ost_audit_{DATE}.confirmed"
    ost_confirmed.write_text("ok")
    shorts_confirmed.write_text("ok")
    (stages_dir / "drive_plan.json").write_text("{}")

    state = SimpleNamespace(output_base=output_base)
    DrivePipelineOrchestrator()._purge_stale_stage_artifacts(state, stages_dir)

    assert not ost_confirmed.exists()
    assert not shorts_confirmed.exists()
    # Existing behaviour preserved.
    assert not (stages_dir / "drive_plan.json").exists()


def test_read_confirmed_hash_only_binds_a_real_digest(tmp_path):
    """A hand-created marker (arbitrary content) carries no binding; a stamped
    sha256 digest is read back."""
    marker = tmp_path / f"ost_audit_{DATE}.confirmed"
    marker.write_text("ok")
    assert read_confirmed_hash(marker) is None

    stamp_confirmed(marker, "a" * 64)
    assert read_confirmed_hash(marker) == "a" * 64


def test_stale_confirmed_binding_reopens_shorts_gate(tmp_path):
    """A ``.confirmed`` bound to a DIFFERENT audit hash must reopen the gate —
    ``exists()`` alone can't pass a never-reviewed burn text."""
    confirmed = tmp_path / f"shorts_ost_audit_{DATE}.confirmed"
    stamp_confirmed(confirmed, "0" * 64)  # bound to a stale (different) burn text

    with pytest.raises(OstAuditPendingError):
        shorts_gate_or_raise(tmp_path, DATE, _entries(), {})


def test_first_pass_marker_stamps_binding_then_matches(tmp_path):
    """A legacy/empty marker passes (first-ever confirmation) and the gate
    stamps it with the current hash so future runs are bound."""
    confirmed = tmp_path / f"shorts_ost_audit_{DATE}.confirmed"
    confirmed.write_text("ok")

    shorts_gate_or_raise(tmp_path, DATE, _entries(), {})  # must NOT raise

    expected = compute_shorts_ost_audit_hash(format_shorts_ost_audit(DATE, _entries()))
    assert read_confirmed_hash(confirmed) == expected
