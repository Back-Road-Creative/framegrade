"""A ``--skip-blur`` day writes an attestation that SAYS it shipped unblurred.

Before this, ``day_{date}_privacy_attestation.json`` had exactly one writer --
``_build_day_blur_mask`` -- and that function is gated behind
``blur_gate.blur_required``. So an operator who passed ``--skip-blur`` produced
no attestation at all, and the publish gate's fail-closed default
(``privacy_hold_reason(None)``) held the delivery for the *right* outcome with
the *wrong* record: an absent file is indistinguishable from a crashed run, a
permissions failure, or a directory nobody wrote to.

A ``--skip-blur`` run's LFs had no
attestation to check against at all.

So the skip path writes one too. It changes NO publication outcome -- held
before, held now -- and it is not a blur gate in disguise. What it changes is
that the hold now carries a cause an auditor can read, and every derived
surface's provenance cross-check has a real document to bind to.

**Why ``failed`` and not ``inconclusive``.** The module's own vocabulary makes
``inconclusive`` mean "nothing ran, so nobody knows". Under ``--skip-blur``
nothing ran, but we are not in the dark: the operator affirmatively turned blur
off, so the master is known to be uncovered. Recording the weaker state would
understate a certainty.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.pipeline import day_master_encoder as dme
from gomoveshift.publishing.dispatch import privacy_hold_reason

DATE = "2026-02-20"


def _cfg(*, blur_enabled=True, skip_blur=False):
    return SimpleNamespace(drive=SimpleNamespace(blur_enabled=blur_enabled, skip_blur=skip_blur))


def _doc(out_dir: Path) -> dict:
    path = out_dir / dme.PRIVACY_ATTESTATION_NAME.format(date=DATE)
    return json.loads(path.read_text())


def _master(tmp_path: Path) -> Path:
    m = tmp_path / f"day_{DATE}_master.mp4"
    m.write_bytes(b"not really an mp4, but it has a size and an mtime")
    return m


def test_a_skip_blur_day_writes_an_attestation_at_all(tmp_path):
    """The whole point: the file exists where it never used to."""
    dme.write_skip_blur_attestation(tmp_path, DATE)
    assert (tmp_path / dme.PRIVACY_ATTESTATION_NAME.format(date=DATE)).exists()


def test_it_names_every_required_class(tmp_path):
    dme.write_skip_blur_attestation(tmp_path, DATE)
    doc = _doc(tmp_path)
    assert tuple(doc["required_classes"]) == dme.REQUIRED_PRIVACY_CLASSES
    assert set(doc["classes"]) == set(dme.REQUIRED_PRIVACY_CLASSES)


def test_every_class_is_failed_not_inconclusive_and_never_verified(tmp_path):
    """``inconclusive`` would understate it: the operator turned blur OFF."""
    dme.write_skip_blur_attestation(tmp_path, DATE)
    for name, entry in _doc(tmp_path)["classes"].items():
        assert entry["state"] == dme.PRIVACY_FAILED, name
        assert entry["state"] != dme.PRIVACY_INCONCLUSIVE, name
        assert entry["state"] != dme.PRIVACY_VERIFIED, name


def test_each_reason_names_the_flag_that_caused_it(tmp_path):
    """An auditor reading only this file must learn WHY, not just that."""
    dme.write_skip_blur_attestation(tmp_path, DATE)
    for name, entry in _doc(tmp_path)["classes"].items():
        assert "skip-blur" in entry["reason"], (name, entry["reason"])


def test_the_written_document_is_what_the_publish_gate_reads(tmp_path):
    """Not a parallel format: the real gate, on the real bytes."""
    dme.write_skip_blur_attestation(tmp_path, DATE)
    reason = privacy_hold_reason(_doc(tmp_path))
    assert reason is not None
    assert "people" in reason or "faces" in reason


def test_it_carries_the_schema_and_date_the_other_writers_use(tmp_path):
    dme.write_skip_blur_attestation(tmp_path, DATE)
    doc = _doc(tmp_path)
    assert doc["schema"] == dme.PRIVACY_ATTESTATION_SCHEMA
    assert doc["date"] == DATE


def test_it_never_overwrites_a_real_detection_run(tmp_path):
    """A resume that reused a blurred master must keep its verified record."""
    path = tmp_path / dme.PRIVACY_ATTESTATION_NAME.format(date=DATE)
    real = {
        "schema": dme.PRIVACY_ATTESTATION_SCHEMA,
        "date": DATE,
        "required_classes": list(dme.REQUIRED_PRIVACY_CLASSES),
        "classes": {
            n: {"state": dme.PRIVACY_VERIFIED, "reason": "detector ran"}
            for n in dme.REQUIRED_PRIVACY_CLASSES
        },
    }
    path.write_text(json.dumps(real))
    dme.write_skip_blur_attestation(tmp_path, DATE)
    assert json.loads(path.read_text()) == real


def test_a_write_failure_is_logged_not_raised(tmp_path, monkeypatch):
    """By the time this runs the master is encoded; a record must not abort."""

    def boom(*_a, **_k):
        raise OSError("read-only")

    monkeypatch.setattr(dme, "save_cache_json", boom)
    dme.write_skip_blur_attestation(tmp_path, DATE)  # must not raise


def test_finalize_writes_the_skip_doc_and_stamps_the_master(tmp_path):
    """One helper owns both halves, so a new return path cannot stamp and
    forget the document that gives the stamp something to live in."""
    master = _master(tmp_path)
    dme.finalize_privacy_attestation(_cfg(skip_blur=True), tmp_path, DATE, master)
    doc = _doc(tmp_path)
    assert doc["master_fingerprint"] == dme.master_fingerprint(master)
    assert doc["classes"]["faces"]["state"] == dme.PRIVACY_FAILED


def test_finalize_on_a_blur_required_run_writes_no_skip_document(tmp_path):
    """The blur path writes its own; this must not invent a second one."""
    master = _master(tmp_path)
    dme.finalize_privacy_attestation(_cfg(skip_blur=False), tmp_path, DATE, master)
    assert not (tmp_path / dme.PRIVACY_ATTESTATION_NAME.format(date=DATE)).exists()


def test_finalize_on_a_blur_disabled_run_also_writes_the_skip_document(tmp_path):
    """``blur_enabled: false`` in config.yaml reaches the encode the same way
    ``--skip-blur`` does -- ``blur_required`` is the single question -- so it
    must leave the same record. Keying on the flag name instead would let one
    of the two ways of turning blur off go unrecorded."""
    master = _master(tmp_path)
    dme.finalize_privacy_attestation(
        _cfg(blur_enabled=False, skip_blur=False), tmp_path, DATE, master
    )
    assert _doc(tmp_path)["classes"]["plates"]["state"] == dme.PRIVACY_FAILED


def test_the_encoder_has_no_other_stamping_call_left(tmp_path):
    """Robust by construction: if a call site still stamps directly, a future
    ``--skip-blur`` return path silently goes back to writing nothing."""
    src = Path(dme.__file__).read_text()
    assert src.count("_stamp_master_fingerprint_into_attestation(") == 2, (
        "expected only the definition and the one call inside finalize_privacy_attestation"
    )
