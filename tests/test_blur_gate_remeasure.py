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
            ("incomplete_flag", "partial"),
            ("source", "signature differs"),
            ("config", "signature differs"),
            ("mask_frames", "frame count"),
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
        elif spoil == "incomplete_flag":
            data["complete"] = False
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
