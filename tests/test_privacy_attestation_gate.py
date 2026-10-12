"""Publication refuses media whose required privacy classes were not verified.

All four privacy classes -- people, faces, plates, vehicle occupants -- are
REQUIRED for a publishable export by policy. The day-master encoder
already recorded honestly what it could and could not detect, but nothing
consumed that record: a run whose person model would not resolve logged a
warning and shipped a faces-only mask, and every short cut from that master
published normally.

These tests cover the three pieces that close that loop:

  * the encoder writes a per-day PRIVACY ATTESTATION naming all four required
    classes and, per class, one of three states -- ``verified`` / ``failed`` /
    ``inconclusive`` -- with the reason;
  * ``dispatch.privacy_hold_reason`` treats anything short of ``verified`` on a
    required class as a HOLD, and an absent attestation as a hold too;
  * ``drain_queue`` holds such a delivery BEFORE taking the lease, never marks
    it ``failed_permanent``, and publishes it normally once the attestation
    says every class was verified.
"""

import json

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline import day_master_encoder as dme
from gomoveshift.publishing import dispatch as dsp
from gomoveshift.publishing.dispatch import drain_queue
from gomoveshift.publishing.queue import enqueue, open_queue_db

NOW = "2026-08-23T00:00:00+00:00"


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def _attestation(**states):
    """An attestation whose four required classes carry the given states."""
    return {
        "schema": dme.PRIVACY_ATTESTATION_SCHEMA,
        "date": "2026-02-24",
        "classes": {
            name: {"state": states.get(name, dsp.PRIVACY_VERIFIED), "reason": "test"}
            for name in dsp.PRIVACY_REQUIRED_CLASSES
        },
    }


def _build_mask(tmp_path, monkeypatch, cfg):
    for name in (
        "resolve_face_model",
        "resolve_plate_model",
        "resolve_vehicle_model",
        "resolve_person_model",
    ):
        monkeypatch.setattr(f"gomoveshift.analysis.{name}", lambda d: None)
    monkeypatch.setattr("redactcam.timeline.probe_video", lambda p: (30.0, 90, 1920, 1080))
    monkeypatch.setattr(
        "redactcam.detect_and_track", lambda *a, **k: ({0: [(10, 10, 20, 20)]}, 90, 1, 0)
    )
    monkeypatch.setattr(
        "redactcam.materialize_mask_video",
        lambda tl, p, **k: (p.write_bytes(b"mask"), p)[1],
    )
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"raw")
    sidecar = tmp_path / "r.json"
    dme._build_day_blur_mask(raw, cfg, tmp_path / "m.mkv", sidecar, "2026-02-24")
    return dme._privacy_attestation_path(sidecar, "2026-02-24")


class TestEncoderWritesAnAttestation:
    def test_every_required_class_is_named_with_a_state_and_a_reason(self, tmp_path, monkeypatch):
        cfg = Config()
        cfg.drive.blur_vehicle_cabin_frac = 0.0
        path = _build_mask(tmp_path, monkeypatch, cfg)
        doc = json.loads(path.read_text())
        assert set(doc["classes"]) == set(dsp.PRIVACY_REQUIRED_CLASSES)
        for name, entry in doc["classes"].items():
            assert entry["state"] in (
                dsp.PRIVACY_VERIFIED,
                dsp.PRIVACY_FAILED,
                dsp.PRIVACY_INCONCLUSIVE,
            ), name
            assert entry["reason"]

    def test_an_unresolvable_person_model_is_inconclusive_not_verified(self, tmp_path, monkeypatch):
        """This is the WARN that shipped a faces-only mask. It must
        never render as a privacy pass."""
        cfg = Config()
        cfg.drive.blur_vehicle_cabin_frac = 0.0
        doc = json.loads(_build_mask(tmp_path, monkeypatch, cfg).read_text())
        assert doc["classes"]["people"]["state"] == dsp.PRIVACY_INCONCLUSIVE

    def test_a_class_switched_off_by_config_is_inconclusive_not_absent(self, tmp_path, monkeypatch):
        """All four classes are required, so a config flag must not silently
        downgrade one to 'not required' -- it downgrades it to UNVERIFIED."""
        cfg = Config()
        cfg.drive.blur_vehicle_cabin_frac = 0.0  # occupant cabin blur OFF
        doc = json.loads(_build_mask(tmp_path, monkeypatch, cfg).read_text())
        entry = doc["classes"]["vehicle_occupants"]
        assert entry["state"] == dsp.PRIVACY_INCONCLUSIVE
        assert "blur_vehicle_cabin_frac" in entry["reason"]

    def test_a_reused_cached_mask_still_leaves_an_attestation(self, tmp_path, monkeypatch):
        """The attestation is what publication reads; a resume that reuses the
        cached mask must not leave publication with nothing to read."""
        cfg = Config()
        cfg.drive.blur_vehicle_cabin_frac = 0.0
        path = _build_mask(tmp_path, monkeypatch, cfg)
        path.unlink()
        assert _build_mask(tmp_path, monkeypatch, cfg).exists()

    def test_the_encoder_filename_matches_the_glob_dispatch_looks_for(self):
        """The writer is under ``pipeline/`` and the reader under
        ``publishing/``; nothing imports across that seam, so the filename
        convention is checked here instead of by the type system."""
        import fnmatch

        name = dme.PRIVACY_ATTESTATION_NAME.format(date="2026-02-24")
        assert fnmatch.fnmatch(name, dsp.PRIVACY_ATTESTATION_GLOB)


class TestPrivacyHoldReason:
    def test_all_four_verified_is_the_only_pass(self):
        assert dsp.privacy_hold_reason(_attestation()) is None

    def test_an_inconclusive_required_class_holds(self):
        reason = dsp.privacy_hold_reason(_attestation(people=dsp.PRIVACY_INCONCLUSIVE))
        assert reason and "people" in reason and dsp.PRIVACY_INCONCLUSIVE in reason

    def test_inconclusive_is_distinguishable_from_failed(self):
        """A detector that ran and found nothing is not the same claim as a
        detector that never ran; the hold must say which."""
        inconclusive = dsp.privacy_hold_reason(_attestation(plates=dsp.PRIVACY_INCONCLUSIVE))
        failed = dsp.privacy_hold_reason(_attestation(plates=dsp.PRIVACY_FAILED))
        assert inconclusive != failed
        assert dsp.PRIVACY_FAILED in failed

    def test_a_missing_class_holds(self):
        doc = _attestation()
        del doc["classes"]["faces"]
        assert "faces" in dsp.privacy_hold_reason(doc)

    def test_no_attestation_at_all_holds(self):
        assert dsp.privacy_hold_reason(None) is not None

    def test_merging_a_runs_days_keeps_the_worst_state_per_class(self):
        merged = dsp.merge_privacy_attestations(
            [_attestation(), _attestation(people=dsp.PRIVACY_INCONCLUSIVE)]
        )
        assert merged["classes"]["people"]["state"] == dsp.PRIVACY_INCONCLUSIVE
        assert merged["classes"]["faces"]["state"] == dsp.PRIVACY_VERIFIED

    def test_merging_nothing_is_no_attestation(self):
        assert dsp.merge_privacy_attestations([]) is None


class TestDrainHoldsUnverifiedMedia:
    def test_a_delivery_with_no_verified_attestation_is_not_published(self, conn):
        delivery_id = enqueue(conn, "proj", 0, "instagram", "hash-a")
        calls = []
        result = drain_queue(
            conn,
            {"instagram": calls.append},
            now=NOW,
            privacy_attestations=lambda row: _attestation(people=dsp.PRIVACY_INCONCLUSIVE),
        )
        assert calls == []
        assert result.attempted == 0
        assert result.held_privacy_unverified == 1
        assert result.held_for_privacy_review is True
        row = dict(conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone())
        assert row["status"] == "queued"  # untouched, still non-terminal

    def test_a_hold_never_marks_the_delivery_failed_permanent(self, conn):
        enqueue(conn, "proj", 0, "instagram", "hash-a")
        result = drain_queue(
            conn,
            {"instagram": lambda row: None},
            now=NOW,
            privacy_attestations=lambda row: None,
        )
        assert result.failed_permanent == 0
        assert result.held_privacy_unverified == 1

    def test_the_gate_runs_before_the_lease_is_taken(self, conn):
        """Same reasoning the no-handler check states: never lock a row for the
        full lease duration over a condition decidable without the lock."""
        delivery_id = enqueue(conn, "proj", 0, "instagram", "hash-a")
        drain_queue(
            conn,
            {"instagram": lambda row: None},
            now=NOW,
            privacy_attestations=lambda row: None,
        )
        row = dict(conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone())
        assert row["lease_owner"] is None
        assert row["attempt_count"] == 0

    def test_a_held_delivery_publishes_once_the_attestation_is_verified(self, conn):
        """Held for review, not destroyed: restoring the model and re-running
        the drain must publish the very same row."""
        delivery_id = enqueue(conn, "proj", 0, "instagram", "hash-a")
        drain_queue(
            conn,
            {"instagram": lambda row: None},
            now=NOW,
            privacy_attestations=lambda row: _attestation(people=dsp.PRIVACY_INCONCLUSIVE),
        )
        result = drain_queue(
            conn,
            {"instagram": lambda row: None},
            now=NOW,
            privacy_attestations=lambda row: _attestation(),
        )
        assert result.succeeded == 1
        assert result.held_privacy_unverified == 0
        status = conn.execute(
            "SELECT status FROM deliveries WHERE id=?", (delivery_id,)
        ).fetchone()[0]
        assert status == "published"

    def test_a_loader_that_blows_up_holds_rather_than_publishing(self, conn):
        """Fail closed: an unreadable attestation is not a privacy pass."""
        enqueue(conn, "proj", 0, "instagram", "hash-a")
        calls = []

        def _boom(row):
            raise OSError("drive unmounted")

        result = drain_queue(conn, {"instagram": calls.append}, now=NOW, privacy_attestations=_boom)
        assert calls == []
        assert result.held_privacy_unverified == 1

    def test_no_loader_injected_leaves_the_gate_inert(self, conn):
        """``drain_queue`` is injected its collaborators (handlers, and now the
        attestation loader); an un-wired caller must not silently change."""
        enqueue(conn, "proj", 0, "instagram", "hash-a")
        result = drain_queue(conn, {"instagram": lambda row: None}, now=NOW)
        assert result.succeeded == 1
        assert result.held_privacy_unverified == 0


class TestCliAttestationLoader:
    def test_it_reads_every_day_beside_the_rows_publish_plan_sidecar(self, tmp_path):
        from gomoveshift.cli.handlers.dispatch_queue import PrivacyAttestationLoader

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / dme.PRIVACY_ATTESTATION_NAME.format(date="2026-02-24")).write_text(
            json.dumps(_attestation())
        )
        (run_dir / dme.PRIVACY_ATTESTATION_NAME.format(date="2026-02-25")).write_text(
            json.dumps(_attestation(plates=dsp.PRIVACY_INCONCLUSIVE))
        )
        loader = PrivacyAttestationLoader(tmp_path / "unused")
        row = {"sidecar_path": str(run_dir / "proj_publish_plan.json")}
        assert dsp.privacy_hold_reason(loader(row)) is not None

    def test_an_output_dir_with_no_attestation_at_all_is_a_hold(self, tmp_path):
        from gomoveshift.cli.handlers.dispatch_queue import PrivacyAttestationLoader

        loader = PrivacyAttestationLoader(tmp_path)
        assert loader({"sidecar_path": None}) is None
