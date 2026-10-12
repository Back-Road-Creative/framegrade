"""``attach_master_identity_to_record`` — the shared cut-from-master
identity + day-attestation cross-check, tested directly on a plain dict.

``write_lf_provenance_sidecar`` has run this comparison for LFs.
This is the same code path, exposed so the thumbnail upload entries
(``drive_phases/phase_d_metadata``) reuse it instead of growing a second,
drifting copy. The contract under test is the LF sidecar's own: best-effort,
never gating, and absent evidence at ANY step is null-WITH-REASON — never
"verified".

The five cases below mirror
``test_lf_provenance_sidecar.py::TestMasterAttestationCrossCheckOnWrite``.
"""

from __future__ import annotations

import json

from gomoveshift.pipeline.lf_provenance_sidecar import attach_master_identity_to_record

DATE = "2026-02-24"


def _write_attestation(tmp_path, master_fingerprint=None):
    day_dir = tmp_path / "_stages" / f"day_{DATE}"
    day_dir.mkdir(parents=True)
    payload = {"schema": "gms.privacy_attestation/v1", "date": DATE}
    if master_fingerprint is not None:
        payload["master_fingerprint"] = master_fingerprint
    (day_dir / f"day_{DATE}_privacy_attestation.json").write_text(json.dumps(payload))


def _attach(tmp_path, *, date=DATE, per_day_completed=None) -> dict:
    record: dict = {}
    attach_master_identity_to_record(
        record,
        date=date,
        per_day_completed=per_day_completed,
        stages_base=tmp_path / "_stages",
    )
    return record


def test_matching_fingerprint_verifies_true(tmp_path):
    _write_attestation(tmp_path, master_fingerprint="123:456")
    record = _attach(tmp_path, per_day_completed={DATE: {"master_fingerprint": "123:456"}})
    assert record["cut_from_master_fingerprint"] == "123:456"
    assert record["master_attestation_verified"] is True
    assert "master_attestation_verified" not in record["null_reasons"]


def test_mismatched_fingerprint_verifies_false_and_records_the_reason(tmp_path):
    """A mismatch is recorded, never raised: the record says what it found
    and leaves enforcement to publication time."""
    _write_attestation(tmp_path, master_fingerprint="999:999")
    record = _attach(tmp_path, per_day_completed={DATE: {"master_fingerprint": "123:456"}})
    assert record["cut_from_master_fingerprint"] == "123:456"
    assert record["master_attestation_verified"] is False
    assert record["null_reasons"].get("master_attestation_verified")


def test_no_attestation_on_disk_is_none_with_reason(tmp_path):
    record = _attach(tmp_path, per_day_completed={DATE: {"master_fingerprint": "123:456"}})
    assert record["master_attestation_verified"] is None
    assert record["null_reasons"].get("master_attestation_verified")


def test_attestation_predating_the_field_is_none_with_reason(tmp_path):
    """An attestation written before ``master_fingerprint`` existed is damaged
    evidence, not agreement -- it must not read as verified."""
    _write_attestation(tmp_path, master_fingerprint=None)
    record = _attach(tmp_path, per_day_completed={DATE: {"master_fingerprint": "123:456"}})
    assert record["master_attestation_verified"] is None
    assert record["null_reasons"].get("master_attestation_verified")


def test_no_recording_date_is_none_with_reason(tmp_path):
    record = _attach(tmp_path, date=None, per_day_completed=None)
    assert record["cut_from_master_fingerprint"] is None
    assert record["master_attestation_verified"] is None
    assert record["null_reasons"].get("cut_from_master_fingerprint")
