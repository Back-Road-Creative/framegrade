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
            lambda raw, cfg, mask, sidecar, date: built.setdefault("mask", mask) or sentinel,
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


class TestCabinCoverageGateWiredIntoDriveMasterPath:
    """The drive day-master path must run the same occupant-coverage check as the
    legacy per-episode path (``blur_gate.check_cabin_coverage``) — after the day
    mask is built, before the master encode composites it."""

    def test_leak_aborts_before_encode(self, tmp_path, monkeypatch):
        cap = _mock_encode(monkeypatch)
        mask_path = tmp_path / "day_2026-02-24_blur_mask.mkv"
        monkeypatch.setattr(
            dme, "_build_day_blur_mask", lambda raw, cfg, mask, sidecar, date: mask_path
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
            dme, "_build_day_blur_mask", lambda raw, cfg, mask, sidecar, date: mask_path
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

    def _wire(self, tmp_path, monkeypatch, verdicts, added):
        cap = _mock_encode(monkeypatch)
        mask_path = tmp_path / "day_2026-02-24_blur_mask.mkv"
        monkeypatch.setattr(
            dme, "_build_day_blur_mask", lambda raw, cfg, mask, sidecar, date: mask_path
        )
        calls = {"check": 0, "repair": []}

        def _check(config, video, mask):
            calls["check"] += 1
            return verdicts[calls["check"] - 1]

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
            tmp_path, monkeypatch, ["BLUR GATE (cabin): EXPOSED", "BLUR GATE (cabin): STILL"], 4
        )
        with pytest.raises(blur_gate.BlurGateError, match="STILL"):
            dme.encode_day_master(_batch(tmp_path), tmp_path, config=Config())
        assert calls["check"] == 2
        assert cap == {}
        assert not (tmp_path / "day_2026-02-24_master.mp4").exists()

    def test_a_repair_that_adds_nothing_stops_without_a_second_check(self, tmp_path, monkeypatch):
        cap, _, calls = self._wire(tmp_path, monkeypatch, ["BLUR GATE (cabin): EXPOSED"], 0)
        with pytest.raises(blur_gate.BlurGateError, match="EXPOSED"):
            dme.encode_day_master(_batch(tmp_path), tmp_path, config=Config())
        assert calls["check"] == 1
        assert cap == {}


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
