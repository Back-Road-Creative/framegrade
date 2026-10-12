"""Measured day-master rate: record round-trip, staleness, calibration, ensure."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from gomoveshift.pipeline import calibration
from gomoveshift.pipeline.calibration import (
    CalibrationRecord,
    calibrate_master_chain,
    ensure_calibrated,
    load_calibration,
    save_calibration,
)
from gomoveshift.pipeline.day_master_encoder import MASTER_ENCODE_ALGO_VERSION


def _record(**over) -> CalibrationRecord:
    base = dict(
        measured_at="2026-10-07T12:00:00+00:00",
        chain_version=MASTER_ENCODE_ALGO_VERSION,
        encoder="hevc_nvenc",
        width=3840,
        height=2160,
        source_fps=30.0,
        slice_seconds=15.0,
        wall_seconds=22.5,
        blur=True,
        ffmpeg_version="ffmpeg version 6.1",
        cpu_count=16,
    )
    base.update(over)
    return CalibrationRecord(**base)


class TestRecord:
    def test_derived_rates(self):
        rec = _record(source_fps=30.0, slice_seconds=15.0, wall_seconds=22.5)
        assert rec.realtime_factor == pytest.approx(1.5)
        assert rec.chain_fps == pytest.approx(20.0)

    def test_round_trip(self, tmp_path):
        path = tmp_path / "sub" / "calibration.json"
        rec = _record()
        save_calibration(rec, path)
        assert load_calibration(path) == rec
        assert not list(path.parent.glob("*.tmp"))

    def test_missing_file_is_none(self, tmp_path):
        assert load_calibration(tmp_path / "nope.json") is None

    def test_wrong_shape_is_none_with_warning(self, tmp_path, caplog):
        path = tmp_path / "calibration.json"
        path.write_text('{"unexpected": 1}', encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            assert load_calibration(path) is None
        assert "calibration" in caplog.text

    def test_stale_chain_version_is_none(self, tmp_path, caplog):
        path = tmp_path / "calibration.json"
        save_calibration(_record(chain_version="v-old"), path)
        with caplog.at_level(logging.WARNING):
            assert load_calibration(path) is None
        assert "stale" in caplog.text
        assert "v-old" in caplog.text
        assert str(MASTER_ENCODE_ALGO_VERSION) in caplog.text

    def test_default_path_follows_intermediate_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GMS_INTERMEDIATE_DIR", str(tmp_path))
        assert calibration.calibration_path() == tmp_path / "calibration.json"


class TestCalibrateMasterChain:
    def _run(self, tmp_path, blur):
        calls = {}

        def cut_slice(source, slice_path, seconds):
            calls["cut"] = (source, slice_path, seconds)
            slice_path.write_bytes(b"s")

        def make_mask(mask_path, width, height, fps, seconds):
            calls["mask"] = (mask_path, width, height, fps, seconds)
            mask_path.write_bytes(b"m")

        def run_chain(slice_path, out_path, mask_path):
            calls["chain"] = (slice_path, out_path, mask_path)
            out_path.write_bytes(b"o")

        def probe(path):
            return {"width": 3840, "height": 2160, "fps": 29.97, "duration": 15.0}

        workdir = tmp_path / "work"
        rec = calibrate_master_chain(
            SimpleNamespace(),
            tmp_path / "clip.mp4",
            blur=blur,
            workdir=workdir,
            run_chain=run_chain,
            probe=probe,
            cut_slice=cut_slice,
            make_mask=make_mask,
        )
        return rec, calls, workdir

    def test_record_fields_and_cleanup(self, tmp_path):
        rec, calls, workdir = self._run(tmp_path, blur=True)
        assert rec.slice_seconds == 15.0
        assert rec.blur is True
        assert rec.chain_version == MASTER_ENCODE_ALGO_VERSION
        assert rec.wall_seconds > 0
        assert (rec.width, rec.height) == (3840, 2160)
        assert rec.source_fps == pytest.approx(29.97)
        assert calls["cut"][2] == 15.0
        assert calls["chain"][2] == calls["mask"][0]
        assert list(workdir.iterdir()) == []

    def test_no_blur_passes_no_mask(self, tmp_path):
        rec, calls, _ = self._run(tmp_path, blur=False)
        assert rec.blur is False
        assert "mask" not in calls
        assert calls["chain"][2] is None

    def test_cleanup_when_chain_raises(self, tmp_path):
        workdir = tmp_path / "work"

        def boom(slice_path, out_path, mask_path):
            out_path.write_bytes(b"partial")
            raise RuntimeError("encode died")

        with pytest.raises(RuntimeError, match="encode died"):
            calibrate_master_chain(
                SimpleNamespace(),
                tmp_path / "clip.mp4",
                blur=False,
                workdir=workdir,
                run_chain=boom,
                probe=lambda p: {"width": 1, "height": 1, "fps": 30.0, "duration": 15.0},
                cut_slice=lambda s, p, sec: p.write_bytes(b"s"),
                make_mask=lambda *a: None,
            )
        assert list(workdir.iterdir()) == []


class TestEnsureCalibrated:
    def test_valid_record_skips_calibration(self, tmp_path):
        rec = _record()

        def calibrate(*a, **k):
            raise AssertionError("must not calibrate")

        got = ensure_calibrated(
            SimpleNamespace(),
            [tmp_path / "a.mp4"],
            blur=True,
            workdir=tmp_path,
            calibrate=calibrate,
            load=lambda: rec,
            save=lambda r: None,
        )
        assert got is rec

    def test_absent_record_calibrates_once_and_saves(self, tmp_path):
        rec = _record()
        calls, saved = [], []

        def calibrate(config, source, **kw):
            calls.append((source, kw["blur"]))
            return rec

        got = ensure_calibrated(
            SimpleNamespace(),
            [tmp_path / "a.mp4", tmp_path / "b.mp4"],
            blur=True,
            workdir=tmp_path,
            calibrate=calibrate,
            load=lambda: None,
            save=saved.append,
        )
        assert got is rec
        assert calls == [(tmp_path / "a.mp4", True)]
        assert saved == [rec]

    def test_no_sources_returns_none_without_calibrating(self, tmp_path):
        got = ensure_calibrated(
            SimpleNamespace(),
            [],
            blur=False,
            workdir=tmp_path,
            calibrate=lambda *a, **k: pytest.fail("no sources"),
            load=lambda: None,
            save=lambda r: None,
        )
        assert got is None

    def test_failure_falls_back_with_warning(self, tmp_path, caplog):
        def calibrate(*a, **k):
            raise RuntimeError("no gpu")

        with caplog.at_level(logging.WARNING):
            got = ensure_calibrated(
                SimpleNamespace(),
                [tmp_path / "a.mp4"],
                blur=False,
                workdir=tmp_path,
                calibrate=calibrate,
                load=lambda: None,
                save=lambda r: None,
            )
        assert got is None
        assert "falls back" in caplog.text
        assert "no gpu" in caplog.text
