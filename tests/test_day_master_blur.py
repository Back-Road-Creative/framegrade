"""Privacy blur wired into the DRIVE day-master path.

The day master is the single source every LF / short / thumbnail is stream-cut
from, so blurring it DURING its one existing color+audio encode anonymizes every
downstream surface by construction. These tests cover:

  * a blur-enabled fresh encode builds a mask, threads it into the single-pass
    encode, and stamps the durable ``.blurred`` marker on the master;
  * ``--skip-blur`` skips the mask build and stamps no marker (the gate no-ops);
  * a resume REUSING a cached master that predates blur (no marker) HARD-FAILS
    (``BlurGateError``) before any LF/short cut — never a silent unblurred ship;
  * a resume reusing a MARKED master is trusted;
  * ``_build_day_blur_mask`` detection failure / a missing cabin model hard-fail;
  * ``run_all_days`` RE-RAISES ``BlurGateError`` instead of swallowing it into a
    per-day skip.

The ONNX detectors and ffmpeg are mocked — no real model files / encodes.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline import blur_gate
from gomoveshift.pipeline import day_master_encoder as dme


@pytest.fixture(autouse=True)
def _frame_pairing_proven(monkeypatch):
    """These tests use fake media files; the three-way frame-count gate has its own
    real-media tests in ``test_frame_index_pairing.py``."""
    monkeypatch.setattr(blur_gate, "frame_pairing_error", lambda *a, **k: None)
    monkeypatch.setattr(dme, "_counts_allow_reuse", lambda *a, **k: True)


def _batch(tmp_path, date="2026-02-24", n_ep=1, total_duration=600.0):
    from gomoveshift.pipeline.day_grouper import DayBatch

    episodes = []
    for i in range(1, n_ep + 1):
        ep_file = tmp_path / f"ep_{i:03d}.mp4"
        ep_file.touch()
        ep = MagicMock()
        ep.index = i
        ep.concat_path = ep_file
        ep.creation_time = datetime(2026, 2, 24, 10 + i, tzinfo=timezone.utc)
        ep.total_duration = total_duration / n_ep
        episodes.append(ep)
    # DayBatch.total_duration is a computed property summing episode durations.
    return DayBatch(date=date, episodes=episodes)


def _mock_encode(monkeypatch, *, success=True):
    """Mock the concat subprocess + SinglePassProcessor.process; capture process kwargs."""
    monkeypatch.setattr(
        "gomoveshift.pipeline.day_master_encoder.subprocess.run",
        lambda cmd, *a, **k: SimpleNamespace(returncode=0, stdout=b"", stderr=b""),
    )
    cap = {}

    def fake_process(self, video_path, output_path, location, **kwargs):
        cap["blur_mask_path"] = kwargs.get("blur_mask_path")
        cap["skip_ost"] = kwargs.get("skip_ost")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).touch()
        return SimpleNamespace(success=success, output_path=output_path)

    monkeypatch.setattr(
        "gomoveshift.pipeline.single_pass.SinglePassProcessor.process", fake_process
    )
    return cap


class TestFreshEncodeBlursMaster:
    def test_mask_built_threaded_and_master_marked(self, tmp_path, monkeypatch):
        cap = _mock_encode(monkeypatch)
        sentinel = tmp_path / "day_2026-02-24_blur_mask.mkv"
        built = {}
        monkeypatch.setattr(
            dme,
            "_build_day_blur_mask",
            lambda raw, cfg, mask, sidecar, date, **_k: built.setdefault("mask", mask) or sentinel,
        )
        # Cabin-coverage gate is exercised on its own in
        # TestCabinCoverageGateWiredIntoDriveMasterPath — no-op it here so this
        # test isolates mask-threading/marker-stamping.
        monkeypatch.setattr(blur_gate, "check_day_master_cabin_coverage", lambda c, v, m: None)
        out = dme.encode_day_master(_batch(tmp_path), tmp_path, config=Config())
        assert out == tmp_path / "day_2026-02-24_master.mp4"
        # The mask reached the single-pass encode (so the master IS blurred).
        assert cap["blur_mask_path"] == sentinel and cap["skip_ost"] is True
        # Durable proof-of-blur marker stamped beside the master.
        assert (tmp_path / "day_2026-02-24_master.mp4.blurred").exists()
        assert blur_gate.master_blur_proven(Config(), out) is True

    def test_skip_blur_skips_mask_and_marker(self, tmp_path, monkeypatch):
        cap = _mock_encode(monkeypatch)

        def _boom(*a, **k):
            raise AssertionError("mask build must not run under --skip-blur")

        monkeypatch.setattr(dme, "_build_day_blur_mask", _boom)
        cfg = Config()
        cfg.drive.skip_blur = True
        out = dme.encode_day_master(_batch(tmp_path), tmp_path, config=cfg)
        assert cap["blur_mask_path"] is None
        assert not (tmp_path / "day_2026-02-24_master.mp4.blurred").exists()


class TestPairedMaskPersistsItsSourceFrameCount:
    """The pairing check is the only moment the source container, the mask and the sidecar
    are all on disk and proven equal; the count is kept so a purged master can still prove
    its mask later (``mask_frame_gate``)."""

    def _rig(self, tmp_path, monkeypatch, *, pairing=None):
        cap = _mock_encode(monkeypatch)
        mask = tmp_path / "day_2026-02-24_blur_mask.mkv"
        sidecar = tmp_path / "day_2026-02-24_blur_regions.json"

        def build(raw, cfg, mask_target, sidecar_target, date, **_k):
            mask.write_bytes(b"mask")
            sidecar.write_text('{"source_frame_count": 120}')
            return mask

        monkeypatch.setattr(dme, "_build_day_blur_mask", build)
        monkeypatch.setattr(blur_gate, "check_day_master_cabin_coverage", lambda c, v, m: None)
        monkeypatch.setattr(blur_gate, "container_frame_count", lambda p: 120)
        monkeypatch.setattr(blur_gate, "sidecar_frame_count", lambda p: 120)
        monkeypatch.setattr(blur_gate, "redactcam_commit", lambda: "abc123")
        if pairing:
            monkeypatch.setattr(blur_gate, "frame_pairing_error", pairing)
        return cap, tmp_path / "day_2026-02-24_source_frames.json"

    def test_paired_mask_persists_source_frame_count(self, tmp_path, monkeypatch):
        _cap, out = self._rig(tmp_path, monkeypatch)
        dme.encode_day_master(_batch(tmp_path), tmp_path, config=Config())
        doc = json.loads(out.read_text())
        assert doc["counts"] == {"container": 120, "mask": 120, "sidecar": 120}
        sidecar = tmp_path / "day_2026-02-24_blur_regions.json"
        mask = tmp_path / "day_2026-02-24_blur_mask.mkv"
        import hashlib

        assert doc["sidecar_sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
        assert doc["mask"] == {"size": mask.stat().st_size, "mtime_ns": mask.stat().st_mtime_ns}
        assert doc["redactcam_commit"] == "abc123"
        assert doc["renderer"] == blur_gate.renderer_identity()
        assert "source_fingerprint" in doc
        assert not list(tmp_path.glob("*.tmp"))  # atomic write leaves no temp file

    def test_refused_pairing_writes_no_source_count(self, tmp_path, monkeypatch):
        _cap, out = self._rig(tmp_path, monkeypatch, pairing=lambda *a, **k: "BLUR GATE (pairing)")
        with pytest.raises(blur_gate.BlurGateError, match="pairing"):
            dme.encode_day_master(_batch(tmp_path), tmp_path, config=Config())
        assert not out.exists()

    def test_a_pairing_that_fails_after_the_cabin_repair_writes_nothing(
        self, tmp_path, monkeypatch
    ):
        answers = iter([None, "BLUR GATE (pairing) after repair"])
        _cap, out = self._rig(tmp_path, monkeypatch, pairing=lambda *a, **k: next(answers))
        with pytest.raises(blur_gate.BlurGateError, match="after repair"):
            dme.encode_day_master(_batch(tmp_path), tmp_path, config=Config())
        assert not out.exists()

    def test_uncountable_inputs_write_nothing_and_do_not_block(self, tmp_path, monkeypatch):
        _cap, out = self._rig(tmp_path, monkeypatch)
        monkeypatch.setattr(blur_gate, "container_frame_count", lambda p: -1)
        dme.encode_day_master(_batch(tmp_path), tmp_path, config=Config())
        assert not out.exists()

    def test_an_absent_redactcam_commit_is_recorded_as_null(self, tmp_path, monkeypatch):
        _cap, out = self._rig(tmp_path, monkeypatch)
        monkeypatch.setattr(blur_gate, "redactcam_commit", lambda: None)
        dme.encode_day_master(_batch(tmp_path), tmp_path, config=Config())
        assert json.loads(out.read_text())["redactcam_commit"] is None


class TestCabinCoverageGateWiredIntoDriveMasterPath:
    """The drive day-master path must run the same occupant-coverage check as the
    legacy per-episode path (``blur_gate.check_cabin_coverage``) — after the day
    mask is built, before the master encode composites it."""

    def test_leak_aborts_before_encode(self, tmp_path, monkeypatch):
        cap = _mock_encode(monkeypatch)
        mask_path = tmp_path / "day_2026-02-24_blur_mask.mkv"
        monkeypatch.setattr(
            dme, "_build_day_blur_mask", lambda raw, cfg, mask, sidecar, date, **_k: mask_path
        )
        seen = {}

        def _fake_check(config, video, mask):
            seen["args"] = (video, mask)
            return "BLUR GATE (cabin): 1 of 5 corroborated vehicle-frame(s) EXPOSED"

        monkeypatch.setattr(blur_gate, "check_day_master_cabin_coverage", _fake_check)
        with pytest.raises(blur_gate.BlurGateError, match="EXPOSED"):
            dme.encode_day_master(_batch(tmp_path), tmp_path, config=Config())
        # Verified against the raw concat + the just-built mask, and the encode
        # (irreversible) never ran.
        assert seen["args"] == (tmp_path / "day_2026-02-24_raw.mp4", mask_path)
        assert cap == {}
        assert not (tmp_path / "day_2026-02-24_master.mp4").exists()

    def test_no_leak_proceeds_to_encode(self, tmp_path, monkeypatch):
        cap = _mock_encode(monkeypatch)
        mask_path = tmp_path / "day_2026-02-24_blur_mask.mkv"
        monkeypatch.setattr(
            dme, "_build_day_blur_mask", lambda raw, cfg, mask, sidecar, date, **_k: mask_path
        )
        monkeypatch.setattr(blur_gate, "check_day_master_cabin_coverage", lambda c, v, m: None)
        out = dme.encode_day_master(_batch(tmp_path), tmp_path, config=Config())
        assert out == tmp_path / "day_2026-02-24_master.mp4"
        assert cap["blur_mask_path"] == mask_path


class TestMasterFingerprintStampedIntoAttestation:
    """A derived surface (LF/short/thumbnail) claims it was cut from the
    attested master. Nothing proved that until the attestation itself carries
    the master's identity — stamped here, right after promotion, using the
    same cheap ``size:mtime_ns`` fingerprint the resume-cache gate already
    trusts (never a full hash of the 60 GB master)."""

    def _no_network_models(self, monkeypatch):
        for name in (
            "resolve_face_model",
            "resolve_plate_model",
            "resolve_vehicle_model",
            "resolve_person_model",
        ):
            monkeypatch.setattr(f"gomoveshift.analysis.{name}", lambda d: None)

    def test_fresh_encode_stamps_master_fingerprint_into_attestation(self, tmp_path, monkeypatch):
        _mock_encode(monkeypatch)
        self._no_network_models(monkeypatch)
        monkeypatch.setattr("redactcam.timeline.probe_video", lambda p: (30.0, 90, 1920, 1080))
        monkeypatch.setattr(
            "redactcam.detect_and_track", lambda *a, **k: ({0: [(10, 10, 20, 20)]}, 90, 1, 0)
        )
        monkeypatch.setattr(
            "redactcam.materialize_mask_video",
            lambda tl, p, **k: (p.write_bytes(b"mask"), p)[1],
        )
        monkeypatch.setattr(blur_gate, "check_day_master_cabin_coverage", lambda c, v, m: None)
        cfg = Config()
        cfg.drive.blur_vehicle_cabin_frac = 0.0
        out = dme.encode_day_master(_batch(tmp_path), tmp_path, config=cfg)
        attestation_path = tmp_path / "day_2026-02-24_privacy_attestation.json"
        assert attestation_path.exists()
        doc = json.loads(attestation_path.read_text())
        assert doc["master_fingerprint"] == dme.master_fingerprint(out)

    def test_resume_shortcut_stamps_fingerprint_too(self, tmp_path, monkeypatch):
        """The cached-master resume path never calls ``_build_day_blur_mask``
        (the attestation already exists from the original run) — the stamp
        must still land so a resumed run's LFs can verify against it."""
        monkeypatch.setattr(dme, "assert_batch_duration_integrity", lambda *a, **k: None)

        class _Probe:
            def __init__(self, *a, **k):
                pass

            def probe_duration(self, p):
                return 600.0

        monkeypatch.setattr("gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient", _Probe)
        master = tmp_path / "day_2026-02-24_master.mp4"
        master.write_bytes(b"master-bytes")
        blur_gate.mark_applied(master)
        attestation_path = tmp_path / "day_2026-02-24_privacy_attestation.json"
        attestation_path.write_text(json.dumps({"schema": dme.PRIVACY_ATTESTATION_SCHEMA}))
        out = dme.encode_day_master(_batch(tmp_path), tmp_path, config=Config())
        assert out == master
        doc = json.loads(attestation_path.read_text())
        assert doc["master_fingerprint"] == dme.master_fingerprint(master)


class TestResumeGate:
    def _resume_env(self, tmp_path, monkeypatch, *, marked):
        monkeypatch.setattr(dme, "assert_batch_duration_integrity", lambda *a, **k: None)

        class _Probe:
            def __init__(self, *a, **k):
                pass

            def probe_duration(self, p):
                return 600.0

        monkeypatch.setattr("gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient", _Probe)
        master = tmp_path / "day_2026-02-24_master.mp4"
        master.write_bytes(b"master-bytes")
        if marked:
            blur_gate.mark_applied(master)
        return master

    def test_resume_stale_master_without_marker_hard_fails(self, tmp_path, monkeypatch):
        self._resume_env(tmp_path, monkeypatch, marked=False)
        with pytest.raises(blur_gate.BlurGateError, match="no .blurred marker"):
            dme.encode_day_master(_batch(tmp_path), tmp_path, config=Config())

    def test_resume_marked_master_is_trusted(self, tmp_path, monkeypatch):
        master = self._resume_env(tmp_path, monkeypatch, marked=True)
        out = dme.encode_day_master(_batch(tmp_path), tmp_path, config=Config())
        assert out == master

    def test_resume_skip_blur_no_ops_the_gate(self, tmp_path, monkeypatch):
        master = self._resume_env(tmp_path, monkeypatch, marked=False)
        cfg = Config()
        cfg.drive.skip_blur = True
        out = dme.encode_day_master(_batch(tmp_path), tmp_path, config=cfg)
        assert out == master  # no marker required when blur is skipped


class TestBuildDayBlurMask:
    def _no_network_models(self, monkeypatch):
        for name in (
            "resolve_face_model",
            "resolve_plate_model",
            "resolve_vehicle_model",
            "resolve_person_model",
        ):
            monkeypatch.setattr(f"gomoveshift.analysis.{name}", lambda d: None)

    def test_detects_and_materializes(self, tmp_path, monkeypatch):
        self._no_network_models(monkeypatch)
        monkeypatch.setattr("redactcam.timeline.probe_video", lambda p: (30.0, 90, 1920, 1080))
        monkeypatch.setattr(
            "redactcam.detect_and_track",
            lambda *a, **k: ({0: [(10, 10, 20, 20)]}, 90, 1, 0),
        )
        calls = []
        monkeypatch.setattr(
            "redactcam.materialize_mask_video",
            lambda tl, p, **k: (calls.append(p), p)[1],
        )
        raw = tmp_path / "raw.mp4"
        raw.write_bytes(b"raw")
        cfg = Config()
        cfg.drive.blur_vehicle_cabin_frac = 0.0  # no cabin → no vehicle-model hard-fail
        mask = tmp_path / "m.mkv"
        sidecar = tmp_path / "r.json"
        out = dme._build_day_blur_mask(raw, cfg, mask, sidecar, "2026-02-24")
        assert out == mask and calls == [mask] and sidecar.exists()

    def test_detection_failure_hard_fails(self, tmp_path, monkeypatch):
        self._no_network_models(monkeypatch)
        monkeypatch.setattr("redactcam.timeline.probe_video", lambda p: (30.0, 90, 1920, 1080))
        monkeypatch.setattr(
            "redactcam.detect_and_track",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model gone")),
        )
        cfg = Config()
        cfg.drive.blur_vehicle_cabin_frac = 0.0
        raw = tmp_path / "raw.mp4"
        raw.write_bytes(b"raw")
        with pytest.raises(blur_gate.BlurGateError, match="privacy detection failed"):
            dme._build_day_blur_mask(raw, cfg, tmp_path / "m.mkv", tmp_path / "r.json", "d")

    def test_cabin_on_but_no_vehicle_model_hard_fails(self, tmp_path, monkeypatch):
        self._no_network_models(monkeypatch)
        cfg = Config()
        cfg.drive.blur_vehicle_cabin_frac = 0.6  # cabin blur ON, model resolves to None
        raw = tmp_path / "raw.mp4"
        raw.write_bytes(b"raw")
        with pytest.raises(blur_gate.BlurGateError, match="vehicle model"):
            dme._build_day_blur_mask(raw, cfg, tmp_path / "m.mkv", tmp_path / "r.json", "d")

    def test_reuse_refuses_when_person_requirement_changes(self, tmp_path, monkeypatch):
        """A cached mask keyed only on the source fingerprint doesn't know what
        it proves: a first run that degraded to WARN (person model unavailable)
        must not be silently reused once the person model resolves, or people
        ship unblurred from a run that believed it required person blur."""
        self._no_network_models(monkeypatch)
        monkeypatch.setattr("redactcam.timeline.probe_video", lambda p: (30.0, 90, 1920, 1080))
        track_person_calls = []

        def _detect(*a, **k):
            track_person_calls.append(k.get("track_person"))
            return ({0: [(10, 10, 20, 20)]}, 90, 1, 0)

        monkeypatch.setattr("redactcam.detect_and_track", _detect)
        monkeypatch.setattr(
            "redactcam.materialize_mask_video",
            lambda tl, p, **k: (p.write_bytes(b"mask"), p)[1],
        )
        raw = tmp_path / "raw.mp4"
        raw.write_bytes(b"raw")
        cfg = Config()
        cfg.drive.blur_vehicle_cabin_frac = 0.0
        mask = tmp_path / "m.mkv"
        sidecar = tmp_path / "r.json"

        # First run: person model unresolvable (WARN degrade) — faces-only mask.
        dme._build_day_blur_mask(raw, cfg, mask, sidecar, "2026-02-24")
        assert track_person_calls == [False]

        # Second run, SAME source fingerprint, but the person model now
        # resolves — must re-detect with person tracking on, never reuse the
        # faces-only mask on a bare source-fingerprint match.
        person_model = tmp_path / "person.onnx"
        person_model.write_bytes(b"model")
        monkeypatch.setattr("gomoveshift.analysis.resolve_person_model", lambda d: person_model)
        dme._build_day_blur_mask(raw, cfg, mask, sidecar, "2026-02-24")
        assert track_person_calls == [False, True]


class TestBlurDetectionCheckpoint:
    """A multi-hour day blur that dies mid-pass must resume, not restart: one
    132.5 GB day lost ~8 h of detection to one CUDA
    crash because nothing was saved. redactcam checkpoints on request; these pin
    that the day master asks, and that a checkpoint made under different
    evidence or detector settings is never unioned into a new pass."""

    def _setup(self, tmp_path, monkeypatch):
        TestBuildDayBlurMask()._no_network_models(monkeypatch)
        monkeypatch.setattr("redactcam.timeline.probe_video", lambda p: (30.0, 90, 1920, 1080))
        monkeypatch.setattr(
            "redactcam.materialize_mask_video",
            lambda tl, p, **k: (p.write_bytes(b"mask"), p)[1],
        )
        raw = tmp_path / "raw.mp4"
        raw.write_bytes(b"raw")
        cfg = Config()
        cfg.drive.blur_vehicle_cabin_frac = 0.0
        return raw, cfg, tmp_path / "m.mkv", tmp_path / "day_d_blur_regions.json"

    def _crashing_detect(self, seen):
        """Writes a checkpoint the way redactcam does, then dies mid-pass."""

        def _detect(*a, **k):
            ckpt = Path(k["checkpoint_path"])
            seen.append(ckpt.exists())
            ckpt.write_text(json.dumps({"version": 1, "frames_done": 45}))
            raise RuntimeError("CUDA failure 999")

        return _detect

    def test_detection_checkpoints_and_cleans_up_on_success(self, tmp_path, monkeypatch):
        raw, cfg, mask, sidecar = self._setup(tmp_path, monkeypatch)
        calls = []

        def _detect(*a, **k):
            calls.append(k)
            Path(k["checkpoint_path"]).write_text("{}")
            return ({0: [(10, 10, 20, 20)]}, 90, 1, 0)

        monkeypatch.setattr("redactcam.detect_and_track", _detect)
        dme._build_day_blur_mask(raw, cfg, mask, sidecar, "d")

        ckpt = sidecar.with_name("day_d_blur_regions_detect_checkpoint.json")
        assert calls[0]["checkpoint_path"] == ckpt
        assert calls[0]["checkpoint_every_frames"] == int(30.0 * dme.BLUR_CHECKPOINT_EVERY_S)
        # A finished mask is reused through the sidecar; the checkpoint is spent.
        assert not ckpt.exists()

    def test_checkpoint_survives_a_crash_and_is_resumed(self, tmp_path, monkeypatch):
        raw, cfg, mask, sidecar = self._setup(tmp_path, monkeypatch)
        seen = []
        monkeypatch.setattr("redactcam.detect_and_track", self._crashing_detect(seen))
        for _ in range(2):
            with pytest.raises(blur_gate.BlurGateError, match="privacy detection failed"):
                dme._build_day_blur_mask(raw, cfg, mask, sidecar, "d")
        assert seen == [False, True]

    def test_checkpoint_from_other_evidence_is_discarded(self, tmp_path, monkeypatch):
        raw, cfg, mask, sidecar = self._setup(tmp_path, monkeypatch)
        seen = []
        monkeypatch.setattr("redactcam.detect_and_track", self._crashing_detect(seen))
        with pytest.raises(blur_gate.BlurGateError):
            dme._build_day_blur_mask(raw, cfg, mask, sidecar, "d")

        # The person model now resolves: boxes found without person tracking
        # must not be unioned into a pass that claims people are handled.
        person_model = tmp_path / "person.onnx"
        person_model.write_bytes(b"model")
        monkeypatch.setattr("gomoveshift.analysis.resolve_person_model", lambda d: person_model)
        with pytest.raises(blur_gate.BlurGateError):
            dme._build_day_blur_mask(raw, cfg, mask, sidecar, "d")
        assert seen == [False, False]

    def test_checkpoint_from_other_detector_settings_is_discarded(self, tmp_path, monkeypatch):
        raw, cfg, mask, sidecar = self._setup(tmp_path, monkeypatch)
        seen = []
        monkeypatch.setattr("redactcam.detect_and_track", self._crashing_detect(seen))
        with pytest.raises(blur_gate.BlurGateError):
            dme._build_day_blur_mask(raw, cfg, mask, sidecar, "d")

        cfg.drive.blur_face_conf = cfg.drive.blur_face_conf / 2
        with pytest.raises(blur_gate.BlurGateError):
            dme._build_day_blur_mask(raw, cfg, mask, sidecar, "d")
        assert seen == [False, False]


class TestRunAllDaysPropagatesBlurGate:
    def test_blur_gate_error_is_reraised_not_skipped(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline import day_pipeline

        monkeypatch.setattr(day_pipeline, "_derive_day_score_bounds", lambda *a, **k: (0.0, 10.0))

        def _raise(*a, **k):
            raise blur_gate.BlurGateError("master not provably blurred")

        monkeypatch.setattr(day_pipeline, "run_day_pipeline", _raise)
        batch = SimpleNamespace(date="2026-02-24", episodes=[])
        with pytest.raises(blur_gate.BlurGateError):
            day_pipeline.run_all_days(
                [batch],
                np.zeros((1, 9), dtype=np.float64),
                [(0.0, 10.0)],
                tmp_path,
                "County Kerry, Ireland",
            )


class TestCabinGateRepairsItsOwnLeaks:
    """A cabin-gate failure gets ONE targeted repair (``_repair_cabin_leaks``) before
    it stops the run: the cabins of the vehicles around each leaked frame go into the
    mask, and the gate then re-checks the WHOLE day. Only a leak that survives the
    repair stops the run, and blur is never switched off to get past it."""

    def _wire(self, tmp_path, monkeypatch, verdicts, added, failure=None):
        cap = _mock_encode(monkeypatch)
        mask_path = tmp_path / "day_2026-02-24_blur_mask.mkv"
        monkeypatch.setattr(
            dme, "_build_day_blur_mask", lambda raw, cfg, mask, sidecar, date, **_k: mask_path
        )
        calls = {"check": 0, "repair": [], "recorded": []}

        def _check(config, video, mask):
            calls["check"] += 1
            return verdicts[calls["check"] - 1]

        def _lookup(config, video, mask, repair_key=None):
            calls["lookup_key"] = repair_key
            return failure

        def _record(config, video, mask, rounds, repair_key=None):
            calls["recorded"].append(rounds)
            calls["record_key"] = repair_key

        monkeypatch.setattr(blur_gate, "cabin_repair_failure", _lookup)
        monkeypatch.setattr(blur_gate, "record_cabin_repair_failure", _record)

        def _repair(raw, cfg, mask, sidecar):
            calls["repair"].append((raw, mask, sidecar))
            return added

        monkeypatch.setattr(blur_gate, "check_day_master_cabin_coverage", _check)
        monkeypatch.setattr(dme, "_repair_cabin_leaks", _repair)
        return cap, mask_path, calls

    def test_repaired_leak_is_rechecked_then_encoded(self, tmp_path, monkeypatch):
        cap, mask_path, calls = self._wire(
            tmp_path, monkeypatch, ["BLUR GATE (cabin): 1 of 117 EXPOSED", None], added=4
        )
        out = dme.encode_day_master(_batch(tmp_path), tmp_path, config=Config())
        assert out == tmp_path / "day_2026-02-24_master.mp4"
        assert calls["check"] == 2
        assert calls["repair"] == [
            (
                tmp_path / "day_2026-02-24_raw.mp4",
                mask_path,
                tmp_path / "day_2026-02-24_blur_regions.json",
            )
        ]
        assert cap["blur_mask_path"] == mask_path

    def test_leak_that_survives_the_repair_still_stops_the_run(self, tmp_path, monkeypatch):
        cap, _, calls = self._wire(
            tmp_path,
            monkeypatch,
            ["BLUR GATE (cabin): EXPOSED", "BLUR GATE (cabin): MID", "BLUR GATE (cabin): STILL"],
            4,
        )
        with pytest.raises(blur_gate.BlurGateError, match="STILL"):
            dme.encode_day_master(_batch(tmp_path), tmp_path, config=Config())
        assert calls["check"] == 3
        assert cap == {}
        assert not (tmp_path / "day_2026-02-24_master.mp4").exists()

    def test_a_repair_that_adds_nothing_is_still_gated_then_stops(self, tmp_path, monkeypatch):
        """No repair ever skips a gate pass; a repair that changed nothing ends the
        rounds (another would repeat it) and the error is raised."""
        cap, _, calls = self._wire(
            tmp_path, monkeypatch, ["BLUR GATE (cabin): EXPOSED", "BLUR GATE (cabin): SAME"], 0
        )
        with pytest.raises(blur_gate.BlurGateError, match="SAME"):
            dme.encode_day_master(_batch(tmp_path), tmp_path, config=Config())
        assert calls["check"] == 2 and len(calls["repair"]) == 1
        assert cap == {}

    def test_a_repair_that_could_not_run_is_not_cached_as_failed(self, tmp_path, monkeypatch):
        """A repair that added nothing (an exception, a missing leak report) did not try the
        inputs, so a heal retry must run it again: it is never recorded as a failed repair.
        Only a repair that changed the mask every round and still leaked is recorded."""
        _cap, _, calls = self._wire(
            tmp_path, monkeypatch, ["BLUR GATE (cabin): EXPOSED", "BLUR GATE (cabin): SAME"], 0
        )
        with pytest.raises(blur_gate.BlurGateError, match="SAME"):
            dme.encode_day_master(_batch(tmp_path), tmp_path, config=Config())
        assert calls["recorded"] == []

    def test_a_repair_that_ran_every_round_and_still_leaks_is_cached(self, tmp_path, monkeypatch):
        _cap, _, calls = self._wire(
            tmp_path,
            monkeypatch,
            ["BLUR GATE (cabin): EXPOSED", "BLUR GATE (cabin): MID", "BLUR GATE (cabin): STILL"],
            4,
        )
        with pytest.raises(blur_gate.BlurGateError, match="STILL"):
            dme.encode_day_master(_batch(tmp_path), tmp_path, config=Config())
        assert calls["recorded"] == [2]

    def test_the_record_is_keyed_on_the_repair_settings_and_version(self, tmp_path, monkeypatch):
        _cap, _, calls = self._wire(
            tmp_path,
            monkeypatch,
            ["BLUR GATE (cabin): A", "BLUR GATE (cabin): B", "BLUR GATE (cabin): C"],
            4,
        )
        cfg = Config()
        cfg.drive.blur_cabin_repair_rounds = 2
        with pytest.raises(blur_gate.BlurGateError):
            dme.encode_day_master(_batch(tmp_path), tmp_path, config=cfg)
        key = {
            "version": dme.CABIN_REPAIR_VERSION,
            "rounds": 2,
            "window_s": dme.CABIN_REPAIR_WINDOW_S,
            "hold_frames": dme.CABIN_REPAIR_HOLD_FRAMES,
        }
        assert calls["lookup_key"] == key and calls["record_key"] == key

    def test_round_two_runs_when_round_one_still_leaks(self, tmp_path, monkeypatch):
        cap, mask_path, calls = self._wire(
            tmp_path,
            monkeypatch,
            ["BLUR GATE (cabin): A", "BLUR GATE (cabin): B", None],
            added=3,
        )
        out = dme.encode_day_master(_batch(tmp_path), tmp_path, config=Config())
        assert out == tmp_path / "day_2026-02-24_master.mp4"
        assert len(calls["repair"]) == 2 and calls["check"] == 3
        assert calls["recorded"] == []
        assert cap["blur_mask_path"] == mask_path

    def test_after_the_configured_rounds_the_gate_error_raises(self, tmp_path, monkeypatch):
        assert Config().drive.blur_cabin_repair_rounds == 2
        cap, _, calls = self._wire(
            tmp_path, monkeypatch, [f"BLUR GATE (cabin): L{i}" for i in range(6)], added=3
        )
        cfg = Config()
        cfg.drive.blur_cabin_repair_rounds = 3
        with pytest.raises(blur_gate.BlurGateError, match="L3"):
            dme.encode_day_master(_batch(tmp_path), tmp_path, config=cfg)
        assert len(calls["repair"]) == 3 and calls["check"] == 4  # a gate pass after each
        assert calls["recorded"] == [3]
        assert cap == {}

    def test_default_two_rounds_then_raise_records_the_failure(self, tmp_path, monkeypatch):
        cap, _, calls = self._wire(
            tmp_path, monkeypatch, [f"BLUR GATE (cabin): L{i}" for i in range(6)], added=3
        )
        with pytest.raises(blur_gate.BlurGateError, match="L2"):
            dme.encode_day_master(_batch(tmp_path), tmp_path, config=Config())
        assert len(calls["repair"]) == 2 and calls["check"] == 3
        assert calls["recorded"] == [2] and cap == {}

    def test_a_pass_at_round_one_encodes(self, tmp_path, monkeypatch):
        cap, _, calls = self._wire(tmp_path, monkeypatch, ["BLUR GATE (cabin): A", None], 3)
        dme.encode_day_master(_batch(tmp_path), tmp_path, config=Config())
        assert len(calls["repair"]) == 1 and "blur_mask_path" in cap

    def test_a_recorded_failure_raises_without_repeating_the_repair(self, tmp_path, monkeypatch):
        cap, _, calls = self._wire(
            tmp_path,
            monkeypatch,
            ["BLUR GATE (cabin): EXPOSED"],
            added=3,
            failure={"rounds": 2, "leaks": []},
        )
        with pytest.raises(blur_gate.BlurGateError, match="EXPOSED"):
            dme.encode_day_master(_batch(tmp_path), tmp_path, config=Config())
        assert calls["repair"] == [] and calls["check"] == 1 and cap == {}

    def test_zero_rounds_means_no_repair_and_no_pass(self, tmp_path, monkeypatch):
        cap, _, calls = self._wire(tmp_path, monkeypatch, ["BLUR GATE (cabin): EXPOSED"], 3)
        cfg = Config()
        cfg.drive.blur_cabin_repair_rounds = 0
        with pytest.raises(blur_gate.BlurGateError, match="EXPOSED"):
            dme.encode_day_master(_batch(tmp_path), tmp_path, config=cfg)
        assert calls["repair"] == [] and cap == {}


class TestRepairCabinLeaks:
    """``_repair_cabin_leaks`` reads the frames in order (the same frame numbers the
    gate reported; seeking can land a frame or two off), adds the cabin band of every
    vehicle seen within ``CABIN_REPAIR_WINDOW_S`` of a leaked frame, and swaps the new
    mask in only once it is fully written."""

    W, H, N, FPS = 64, 48, 100, 10.0
    CAR = (10, 10, 30, 20)
    # cabin_region(CAR, 0.5) = (10, 10, 30, 10); dilated by the timeline's 2 px.
    CABIN = (8, 8, 34, 14)

    def _setup(self, tmp_path, monkeypatch, leaks, materialize=None):
        from redactcam import timeline as br

        raw = tmp_path / "day_2026-02-24_raw.mp4"
        raw.write_bytes(b"raw")
        mask = tmp_path / "day_2026-02-24_blur_mask.mkv"
        mask.write_bytes(b"old-mask")
        sidecar = tmp_path / "day_2026-02-24_blur_regions.json"
        br.write_sidecar(
            sidecar,
            br.BlurTimeline(
                source_hash="1:2",
                sample_fps=8.0,
                conf=0.4,
                box_dilation_px=2,
                source_fps=self.FPS,
                source_frame_count=self.N,
                frame_width=self.W,
                frame_height=self.H,
                frame_boxes={0: [(1, 1, 4, 4)]},
            ),
        )
        if leaks is not None:
            blur_gate.cabin_leak_report_path(mask).write_text(
                json.dumps(
                    {
                        "vehicle_frames": 9,
                        "leaks": [
                            {"frame": f, "coverage": 0.0, "box": list(self.CAR)} for f in leaks
                        ],
                    }
                )
            )
        n, w, h = self.N, self.W, self.H

        class _Cap:  # no ``set``: a repair that seeks fails this test
            def __init__(self, path):
                self.i = -1

            def isOpened(self):
                return True

            def grab(self):
                if self.i + 1 >= n:
                    return False
                self.i += 1
                return True

            def retrieve(self):
                return True, np.full((h, w, 3), self.i % 256, dtype=np.uint8)

            def release(self):
                pass

        car = self.CAR
        seen = []

        class _Det:
            def detect(self, frame, conf):
                seen.append(int(frame[0, 0, 0]))
                return [car]

        import cv2

        monkeypatch.setattr(cv2, "VideoCapture", _Cap)
        monkeypatch.setattr("redactcam.detect.build_vehicle_detector", lambda *a, **k: _Det())
        monkeypatch.setattr(
            "gomoveshift.analysis.resolve_vehicle_model", lambda drive: tmp_path / "veh.onnx"
        )
        written = {}

        def _materialize(timeline, out_path, *a, **k):
            written["path"] = Path(out_path)
            Path(out_path).write_bytes(b"new-mask")
            return out_path

        monkeypatch.setattr("redactcam.materialize_mask_video", materialize or _materialize)
        cfg = Config()
        cfg.drive.blur_vehicle_cabin_frac = 0.5
        cfg.drive.blur_vehicle_conf = 0.5
        cfg.drive.blur_vehicle_min_height_frac = 0.0
        cfg.drive.blur_vehicle_ignore_bottom_frac = 0.0
        return raw, mask, sidecar, cfg, written, seen

    def test_adds_cabins_around_the_leak_and_nowhere_else(self, tmp_path, monkeypatch):
        from redactcam import timeline as br

        raw, mask, sidecar, cfg, written, seen = self._setup(tmp_path, monkeypatch, [50])
        verdict = blur_gate.cabin_verdict_path(mask)
        verdict.write_text('{"signature": {}, "error": "BLUR GATE (cabin): old"}')
        added = dme._repair_cabin_leaks(raw, cfg, mask, sidecar)
        assert added > 0
        assert not verdict.exists()  # the new mask never inherits the old verdict
        half = round(self.FPS * dme.CABIN_REPAIR_WINDOW_S)
        hold = dme.CABIN_REPAIR_HOLD_FRAMES
        # Only frames inside the window are run through the detector.
        assert seen == list(range(50 - half, 50 + half + 1))
        tl = br.load_sidecar(sidecar)
        boxed = {f for f, boxes in tl.frame_boxes.items() if self.CABIN in boxes}
        assert boxed == set(range(50 - half - hold, 50 + half + hold + 1))
        assert tl.frame_boxes[0] == [(1, 1, 4, 4)]  # what was there before stays
        # Written beside the mask, then swapped in whole.
        assert written["path"] != mask and written["path"].suffix == ".mkv"
        assert mask.read_bytes() == b"new-mask"
        assert not written["path"].exists()

    def test_the_gates_own_leaked_box_is_covered_even_when_detection_finds_nothing(
        self, tmp_path, monkeypatch
    ):
        from redactcam import timeline as br

        raw, mask, sidecar, cfg, written, seen = self._setup(tmp_path, monkeypatch, [50])

        class _Blind:
            def detect(self, frame, conf):
                return []

        monkeypatch.setattr("redactcam.detect.build_vehicle_detector", lambda *a, **k: _Blind())
        added = dme._repair_cabin_leaks(raw, cfg, mask, sidecar)
        assert added >= 1  # a repair given a leak can never add nothing
        tl = br.load_sidecar(sidecar)
        assert self.CABIN in tl.frame_boxes[50]
        assert mask.read_bytes() == b"new-mask"

    def test_the_leaked_box_is_covered_without_a_vehicle_model(self, tmp_path, monkeypatch):
        from redactcam import timeline as br

        raw, mask, sidecar, cfg, written, seen = self._setup(tmp_path, monkeypatch, [50])
        monkeypatch.setattr("gomoveshift.analysis.resolve_vehicle_model", lambda drive: None)
        assert dme._repair_cabin_leaks(raw, cfg, mask, sidecar) >= 1
        assert self.CABIN in br.load_sidecar(sidecar).frame_boxes[50]
        assert seen == []

    def test_no_leak_report_changes_nothing(self, tmp_path, monkeypatch):
        raw, mask, sidecar, cfg, written, seen = self._setup(tmp_path, monkeypatch, None)
        before = sidecar.read_text()
        assert dme._repair_cabin_leaks(raw, cfg, mask, sidecar) == 0
        assert seen == [] and written == {}
        assert mask.read_bytes() == b"old-mask" and sidecar.read_text() == before

    def test_a_failed_mask_write_keeps_the_old_mask_and_regions(self, tmp_path, monkeypatch):
        def _boom(timeline, out_path, *a, **k):
            Path(out_path).write_bytes(b"half-written")
            raise RuntimeError("mask ffmpeg failed")

        raw, mask, sidecar, cfg, _, _ = self._setup(tmp_path, monkeypatch, [50], _boom)
        before = sidecar.read_text()
        assert dme._repair_cabin_leaks(raw, cfg, mask, sidecar) == 0
        assert mask.read_bytes() == b"old-mask"
        assert sidecar.read_text() == before
        assert sorted(p.name for p in tmp_path.iterdir() if "repair" in p.name) == []
