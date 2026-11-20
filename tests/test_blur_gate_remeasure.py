"""The cabin gate's second pass re-measures the first pass's recorded sightings against the
repaired mask instead of re-decoding and re-detecting the source. It may only ever reach the
verdict a full pass on the same inputs reaches; anything it cannot prove runs in full."""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from gomoveshift import analysis
from gomoveshift.pipeline import blur_gate

W, H, N = 64, 48, 8
# (frame, box x,y,w,h): cabin = top 0.6 of the box. The last sighting has an empty region.
SIGHTINGS = [(2, (4, 4, 20, 20)), (2, (40, 4, 20, 20)), (5, (10, 60, 30, 20))]


def _write_video(path: Path, frames):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 10, (W, H), False)
    for f in frames:
        wr.write(f)
    wr.release()


def _mask(path: Path, white=(), frames=N):
    """A lossless mask; ``white`` = (frame, x0, y0, x1, y1) rectangles painted white."""
    imgs = [np.zeros((H, W), np.uint8) for _ in range(frames)]
    for f, x0, y0, x1, y1 in white:
        imgs[f][y0:y1, x0:x1] = 255
    _write_video(path, imgs)


def _cfg(tmp_path, conf=0.6):
    from types import SimpleNamespace

    drive = SimpleNamespace(
        blur_enabled=True,
        skip_blur=False,
        blur_vehicle_cabin_frac=0.6,
        blur_vehicle_conf=conf,
        blur_vehicle_min_height_frac=0.07,
        blur_vehicle_ignore_bottom_frac=0.12,
        blur_plate_max_area_frac=0.006,
        blur_cabin_gate_sample_fps=4.0,
        blur_cabin_gate_min_coverage=0.5,
    )
    video = tmp_path / "raw.mkv"
    _write_video(video, [np.zeros((H, W), np.uint8)] * N)
    mask = tmp_path / "mask.mkv"
    return SimpleNamespace(drive=drive), video, mask


class _FakeFull:
    """Stands in for ``verify_cabin_coverage``: a deterministic 'detector' whose sightings
    are fixed, measured against the real mask exactly as redactcam measures them."""

    def __init__(self):
        self.calls = 0

    def __call__(self, video, mask, veh, **kw):
        import redactcam
        from redactcam.coverage import cabin_region

        self.calls += 1
        cap = cv2.VideoCapture(str(mask))
        frames = []
        while cap.grab():
            frames.append(cap.retrieve()[1][:, :, 0])
        rep = redactcam.CoverageReport(plate_corroborated=True)
        for f, box in SIGHTINGS:
            rep.vehicle_frames += 1
            x0, y0, x1, y1 = cabin_region(box, kw["cabin_frac"], W, H)
            if x1 <= x0 or y1 <= y0:
                continue
            cov = float((frames[f][y0:y1, x0:x1] > 0).mean())
            if cov < kw["min_coverage"]:
                rep.leaks.append(redactcam.Leak(f, cov, box))
        return rep


@pytest.fixture
def fake(monkeypatch):
    f = _FakeFull()
    monkeypatch.setattr(analysis, "resolve_vehicle_model", lambda d: Path("v.onnx"))
    monkeypatch.setattr(analysis, "resolve_plate_model", lambda d: Path("p.onnx"))
    monkeypatch.setattr(analysis, "verify_cabin_coverage", f)
    return f


def _repaired(mask):
    # covers the first frame-2 cabin only: one leak less, the others remain
    _mask(mask, white=[(2, 0, 0, 30, 16)])


def _gate(config, video, mask):
    err = blur_gate.check_day_master_cabin_coverage(config, video, mask)
    rep = blur_gate.cabin_leak_report_path(mask)
    return err, (json.loads(rep.read_text()) if rep.exists() else None)


class TestRemeasure:
    def test_remeasured_pass_two_equals_a_full_pass_two(self, tmp_path, fake):
        config, video, mask = _cfg(tmp_path)
        _mask(mask)
        err1, _ = _gate(config, video, mask)
        assert err1 and fake.calls == 1
        assert blur_gate.cabin_sightings_path(mask).exists()
        _repaired(mask)
        err2, rep2 = _gate(config, video, mask)
        assert fake.calls == 1, "pass 2 must be re-measured, not a second full pass"

        # reference: a full pass on identical inputs (same names, other directory)
        ref = tmp_path / "ref"
        ref.mkdir()
        shutil.copy(video, ref / video.name)
        shutil.copy(mask, ref / mask.name)
        ref_config, _, _ = _cfg(ref)
        shutil.copy(video, ref / video.name)
        err_ref, rep_ref = _gate(ref_config, ref / video.name, ref / mask.name)
        assert fake.calls == 2
        assert err2 == err_ref and rep2 == rep_ref
        assert [lk["frame"] for lk in rep2["leaks"]] == [2]  # one of two leaks repaired

    def test_a_repair_that_fixes_everything_passes_like_a_full_pass(self, tmp_path, fake):
        config, video, mask = _cfg(tmp_path)
        _mask(mask)
        assert _gate(config, video, mask)[0]
        _mask(mask, white=[(2, 0, 0, W, H), (5, 0, 0, W, H)])
        err, rep = _gate(config, video, mask)
        assert err is None and rep is None and fake.calls == 1

    def test_no_source_decode_and_no_detector(self, tmp_path, fake, monkeypatch):
        config, video, mask = _cfg(tmp_path)
        _mask(mask)
        _gate(config, video, mask)
        _repaired(mask)
        opened = []
        real = cv2.VideoCapture

        def spy(path, *a, **k):
            opened.append(str(path))
            return real(path, *a, **k)

        def boom(*a, **k):
            raise AssertionError("detector must not run on a re-measured pass")

        monkeypatch.setattr(cv2, "VideoCapture", spy)
        monkeypatch.setattr(analysis, "verify_cabin_coverage", boom)
        err, _ = _gate(config, video, mask)
        assert err and "EXPOSED" in err
        assert opened and set(opened) == {str(mask)}, "only the mask may be read"

    @pytest.mark.parametrize(
        "spoil, reason",
        [
            ("no_record", "no usable sightings record"),
            ("garbage", "no usable sightings record"),
            ("partial", "partial"),
            ("schema", "partial or of another schema"),
            ("source", "signature differs"),
            ("config", "signature differs"),
            ("mask_frames", "decodes"),
        ],
    )
    def test_every_fallback_runs_the_full_pass_and_logs_why(
        self, tmp_path, fake, caplog, spoil, reason
    ):
        config, video, mask = _cfg(tmp_path)
        _mask(mask)
        _gate(config, video, mask)
        rec = blur_gate.cabin_sightings_path(mask)
        data = json.loads(rec.read_text())
        _repaired(mask)
        if spoil == "no_record":
            rec.unlink()
        elif spoil == "garbage":
            rec.write_text("{not json")
        elif spoil == "partial":
            data["sightings"] = data["sightings"][:-1]
            rec.write_text(json.dumps(data))
        elif spoil == "schema":
            data["schema"] = 1
            rec.write_text(json.dumps(data))
        elif spoil == "source":
            _write_video(video, [np.full((H, W), 9, np.uint8)] * (N + 1))
        elif spoil == "config":
            config.drive.blur_vehicle_conf = 0.7
        elif spoil == "mask_frames":
            _mask(mask, white=[(2, 0, 0, 30, 16)], frames=N + 2)
        caplog.set_level("INFO", logger=blur_gate.logger.name)
        err, _ = _gate(config, video, mask)
        assert fake.calls == 2, f"{spoil}: the full pass must run"
        assert err
        msgs = [r.getMessage() for r in caplog.records if "full pass" in r.getMessage()]
        assert msgs and reason in msgs[-1]

    def test_the_record_is_written_atomically_and_leaves_no_tmp(self, tmp_path, fake):
        config, video, mask = _cfg(tmp_path)
        _mask(mask)
        _gate(config, video, mask)
        assert not list(tmp_path.glob("*.tmp"))
        data = json.loads(blur_gate.cabin_sightings_path(mask).read_text())
        assert data["sighting_count"] == len(SIGHTINGS) - 1  # the empty-region one is not measured
        assert "mask_fp" not in data["signature"]


# --- contract tests against the REAL redactcam verifier (stub detector only) -------------

import redactcam.coverage as rcov  # noqa: E402

CW, CH, CN = 64, 48, 12
BOXES: dict = {}


class _Det:
    def detect(self, frame, conf):
        return list(BOXES.get(int(frame[0, 0, 0]), []))


@pytest.fixture
def real(tmp_path, monkeypatch):
    from types import SimpleNamespace

    BOXES.clear()
    BOXES.update(
        {
            0: [(4, 4, 20, 20)],
            4: [(4, 4, 20, 20), (40, 4, 20, 20)],
            8: [(30, 30, 30, 10)],
            11: [(50, 2, 20, 20)],
        }
    )
    monkeypatch.setattr(rcov, "build_vehicle_detector", lambda *a, **k: _Det())
    monkeypatch.setattr(rcov, "build_plate_detector", lambda *a, **k: None)
    monkeypatch.setattr(rcov, "filter_vehicles", lambda b, *a: b)
    monkeypatch.setattr(analysis, "resolve_vehicle_model", lambda d: Path("v.onnx"))
    monkeypatch.setattr(analysis, "resolve_plate_model", lambda d: None)
    calls = []
    real_fn = rcov.verify_cabin_coverage

    def spy(*a, **k):
        calls.append(k["min_coverage"])
        return real_fn(*a, **k)

    monkeypatch.setattr(analysis, "verify_cabin_coverage", spy)
    cfg, video, mask = _cfg(tmp_path)
    cfg.drive.blur_cabin_gate_sample_fps = 4.0
    return cfg, video, mask, calls, real_fn


def _src(video, n):
    _write_video(video, [np.full((CH, CW), min(i, 255), np.uint8) for i in range(n)])


def _cmask(path, white=(), n=CN):
    imgs = [np.zeros((CH, CW), np.uint8) for _ in range(n)]
    for f, x0, y0, x1, y1 in white:
        imgs[f][y0:y1, x0:x1] = 255
    _write_video(path, imgs)


class TestAgainstTheRealVerifier:
    def test_remeasure_equals_the_real_full_pass(self, real):
        cfg, video, mask, calls, real_fn = real
        _src(video, CN)
        _cmask(mask)
        assert blur_gate.check_day_master_cabin_coverage(cfg, video, mask)
        for white in (
            [(4, 0, 0, 30, 16)],
            [(f, 0, 0, CW, CH) for f in (0, 4, 8, 11)],
            [(4, 0, 0, CW, 6)],
        ):
            _cmask(mask, white)
            err = blur_gate.check_day_master_cabin_coverage(cfg, video, mask)
            assert calls == [2.0], "pass 2 must be re-measured"
            ref = real_fn(video, mask, "v", cabin_frac=0.6, min_coverage=0.5)
            rep = blur_gate.cabin_leak_report_path(mask)
            got = json.loads(rep.read_text()) if rep.exists() else None
            if ref.ok:
                assert err is None and got is None
            else:
                assert err and got["vehicle_frames"] == ref.vehicle_frames
                assert [(k["frame"], k["box"], k["coverage"]) for k in got["leaks"]] == [
                    (k.frame, list(k.box), round(k.coverage, 4)) for k in ref.leaks
                ]

    @pytest.mark.parametrize("boxes", ["sightings", "none"])
    def test_a_truncated_repaired_mask_with_an_intact_header_runs_in_full(self, real, boxes):
        cfg, video, mask, calls, _ = real
        if boxes == "none":
            BOXES.clear()
        _src(video, 60)
        _cmask(mask, n=60)
        blur_gate.check_day_master_cabin_coverage(cfg, video, mask)
        _cmask(mask, [(i, 0, 0, CW, CH) for i in range(60)], n=60)
        data = mask.read_bytes()
        mask.write_bytes(data[: len(data) // 2])
        cap = cv2.VideoCapture(str(mask))
        decoded = 0
        while cap.grab():
            decoded += 1
        assert decoded < 60 and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 60, "fixture"
        blur_gate.check_day_master_cabin_coverage(cfg, video, mask)
        assert calls == [2.0, 2.0], "a mask that does not decode in full must never be re-measured"

    def test_a_refusing_full_pass_after_truncation_is_a_refusal_not_a_pass(self, real, monkeypatch):
        cfg, video, mask, calls, _ = real
        _src(video, 60)
        _cmask(mask, n=60)
        blur_gate.check_day_master_cabin_coverage(cfg, video, mask)
        _cmask(mask, [(i, 0, 0, CW, CH) for i in range(60)], n=60)
        data = mask.read_bytes()
        mask.write_bytes(data[: len(data) // 2])

        def refuse(*a, **k):
            raise RuntimeError("mask ended early")

        monkeypatch.setattr(analysis, "verify_cabin_coverage", refuse)
        err = blur_gate.check_day_master_cabin_coverage(cfg, video, mask)
        assert err and "could not run" in err

    def test_the_record_carries_decoded_counts_and_the_decoded_frame_shape(self, real, monkeypatch):
        cfg, video, mask, calls, _ = real
        _src(video, CN)
        _cmask(mask)
        real_cap = cv2.VideoCapture

        class Lying:
            def __init__(self, *a, **k):
                self._c = real_cap(*a, **k)

            def get(self, prop):
                return (
                    999.0
                    if prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)
                    else self._c.get(prop)
                )

            def __getattr__(self, name):
                return getattr(self._c, name)

        monkeypatch.setattr(cv2, "VideoCapture", Lying)
        blur_gate.check_day_master_cabin_coverage(cfg, video, mask)
        rec = json.loads(blur_gate.cabin_sightings_path(mask).read_text())
        assert (rec["frame_w"], rec["frame_h"]) == (CW, CH)
        assert rec["mask_frames"] == CN and "source_frames" not in rec
        assert "complete" not in rec


class TestBuildIdentityIsInTheSignature:
    @pytest.mark.parametrize(
        "what",
        ["redactcam_build", "onnxruntime", "cv2", "providers"],
    )
    def test_any_changed_build_part_forces_a_full_pass(self, tmp_path, fake, monkeypatch, what):
        import onnxruntime

        config, video, mask = _cfg(tmp_path)
        _mask(mask)
        _gate(config, video, mask)
        _repaired(mask)
        if what == "redactcam_build":
            monkeypatch.setattr(blur_gate, "_redactcam_build", lambda: "other-commit")
        elif what == "onnxruntime":
            monkeypatch.setattr(onnxruntime, "__version__", "0.0.1")
        elif what == "cv2":
            monkeypatch.setattr(cv2, "__version__", "0.0.1")
        else:
            monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["Other"])
        _gate(config, video, mask)
        assert fake.calls == 2

    def test_the_redactcam_build_is_the_installed_commit(self):
        build = blur_gate._redactcam_build()
        assert isinstance(build, str) and build


class TestAFailureInsideTheOptimisationNeverDecides:
    def test_an_exception_in_the_remeasure_falls_back_to_a_full_pass(
        self, tmp_path, fake, monkeypatch
    ):
        config, video, mask = _cfg(tmp_path)
        _mask(mask)
        _gate(config, video, mask)
        _repaired(mask)
        monkeypatch.setattr(
            blur_gate, "_valid_sightings", lambda *a: (_ for _ in ()).throw(cv2.error("boom"))
        )
        err, rep = _gate(config, video, mask)
        assert fake.calls == 2 and err and [k["frame"] for k in rep["leaks"]] == [2]

    def test_a_failed_record_write_leaves_the_verdict_intact(self, tmp_path, fake, monkeypatch):
        config, video, mask = _cfg(tmp_path)
        _mask(mask)
        monkeypatch.setattr(
            blur_gate, "_decode_stats", lambda p: (_ for _ in ()).throw(cv2.error("boom"))
        )
        err, rep = _gate(config, video, mask)
        assert err and "EXPOSED" in err and len(rep["leaks"]) == 2
        assert not blur_gate.cabin_sightings_path(mask).exists()
        assert blur_gate.load_cached_json(blur_gate.cabin_verdict_path(mask))["error"] == err


class TestNoExtraSourceDecode:
    def _spy(self, monkeypatch):
        seen = []
        real_fn = blur_gate._decode_stats

        def spy(path):
            seen.append(Path(path).name)
            return real_fn(path)

        monkeypatch.setattr(blur_gate, "_decode_stats", spy)
        return seen

    def test_a_passing_full_pass_writes_no_record_and_decodes_nothing_extra(
        self, tmp_path, fake, monkeypatch
    ):
        config, video, mask = _cfg(tmp_path)
        _mask(mask, white=[(2, 0, 0, W, H), (5, 0, 0, W, H)])
        seen = self._spy(monkeypatch)
        err, _ = _gate(config, video, mask)
        assert err is None and fake.calls == 1
        assert seen == [] and not blur_gate.cabin_sightings_path(mask).exists()

    def test_a_leak_pass_takes_n_from_the_mask_and_never_decodes_the_source(
        self, tmp_path, fake, monkeypatch
    ):
        config, video, mask = _cfg(tmp_path)
        _mask(mask)
        seen = self._spy(monkeypatch)
        _gate(config, video, mask)
        assert seen == [mask.name], "only the mask may be fully decoded"
        rec = json.loads(blur_gate.cabin_sightings_path(mask).read_text())
        assert rec["mask_frames"] == N and (rec["frame_w"], rec["frame_h"]) == (W, H)
        assert rec["step"] >= 1

    def test_a_stale_record_is_removed_by_a_passing_full_pass(self, tmp_path, fake):
        config, video, mask = _cfg(tmp_path)
        _mask(mask)
        _gate(config, video, mask)
        assert blur_gate.cabin_sightings_path(mask).exists()
        _mask(mask, white=[(2, 0, 0, W, H), (5, 0, 0, W, H)], frames=N + 1)  # forces full pass
        err, _ = _gate(config, video, mask)
        assert err is None and not blur_gate.cabin_sightings_path(mask).exists()


class TestRealVerifierContract:
    def test_a_mask_shorter_than_its_source_is_refused_by_the_full_pass(self, real):
        """Design depends on this: a full pass that succeeds proves decoded(source) ==
        decoded(mask), so the mask's decoded count N IS the source's. Holds at the pinned
        redactcam (b0861b2 / 143b3be), which CI installs. The stale local venv (1937312)
        ends silently on a short mask, so this is EXPECTED to fail there."""
        from redactcam.frames import FrameSourceError

        cfg, video, mask, calls, real_fn = real
        _src(video, CN)
        _cmask(mask, n=CN // 2)
        with pytest.raises(FrameSourceError):
            real_fn(video, mask, "v", cabin_frac=0.6, min_coverage=0.5)


class TestEverySampledFrameMustRead:
    def test_an_unreadable_sampled_frame_without_a_sighting_runs_the_full_pass(
        self, real, monkeypatch
    ):
        cfg, video, mask, calls, _ = real
        _src(video, CN)
        _cmask(mask)
        assert blur_gate.check_day_master_cabin_coverage(cfg, video, mask)
        assert (
            json.loads(blur_gate.cabin_sightings_path(mask).read_text())["step"] == 2
        )  # 10 fps / 4 sample fps
        _cmask(mask, [(4, 0, 0, 30, 16)])
        real_cap = cv2.VideoCapture
        bad = {"on": True}

        class Flaky:
            def __init__(self, *a, **k):
                self._c = real_cap(*a, **k)
                self._n = 0

            def grab(self):
                ok = self._c.grab()
                self._n += ok
                return ok

            def retrieve(self):
                if bad["on"] and self._n - 1 == 6:  # sampled, no sighting there
                    return False, None
                return self._c.retrieve()

            def __getattr__(self, name):
                return getattr(self._c, name)

        monkeypatch.setattr(cv2, "VideoCapture", Flaky)
        blur_gate.check_day_master_cabin_coverage(cfg, video, mask)
        assert calls == [2.0, 2.0]


class TestBuildIdentityStrength:
    def test_the_source_hash_is_always_part_of_the_build(self, monkeypatch):
        a = blur_gate._redactcam_build()
        assert "commit:" in a and "sha256:" in a
        monkeypatch.setattr(blur_gate, "_redactcam_source_hash", lambda: "other")
        assert blur_gate._redactcam_build() != a

    def test_an_unidentifiable_build_never_matches_itself(self, monkeypatch):
        def boom():
            raise OSError("no source")

        monkeypatch.setattr(blur_gate, "_redactcam_source_hash", boom)
        assert blur_gate._redactcam_build() != blur_gate._redactcam_build()

    def test_a_changed_cpu_allowed_setting_forces_a_full_pass(self, tmp_path, fake, monkeypatch):
        config, video, mask = _cfg(tmp_path)
        monkeypatch.delenv("REDACTCAM_ALLOW_CPU", raising=False)
        _mask(mask)
        _gate(config, video, mask)
        _repaired(mask)
        monkeypatch.setenv("REDACTCAM_ALLOW_CPU", "1")
        _gate(config, video, mask)
        assert fake.calls == 2

    @pytest.mark.parametrize("mode", ["changed", "raises"])
    def test_the_providers_a_session_would_use_are_in_the_signature(
        self, tmp_path, fake, monkeypatch, mode
    ):
        import redactcam.detect as rd

        config, video, mask = _cfg(tmp_path)
        monkeypatch.setattr(rd, "check_inference_providers", lambda: ["CUDAExecutionProvider"])
        _mask(mask)
        _gate(config, video, mask)
        _repaired(mask)
        if mode == "changed":
            monkeypatch.setattr(rd, "check_inference_providers", lambda: ["CPUExecutionProvider"])
        else:
            monkeypatch.setattr(
                rd, "check_inference_providers", lambda: (_ for _ in ()).throw(RuntimeError("x"))
            )
        _gate(config, video, mask)
        assert fake.calls == 2
