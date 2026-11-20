"""The cabin repair swaps the mask and the regions sidecar as a pair, and a round that
never had a detector is not a spent repair round.

Replacing two files cannot be one atomic step, so the swap is journaled: a pending
marker beside the sidecar is written before the first replace and removed after the
second. While it exists the mask and sidecar may disagree, so every encode gate
(``blur_gate.frame_pairing_error``) refuses and the mask cache will not reuse the
pair. A crash at any point leaves either the old pair (marker gone, nothing replaced),
the new pair (marker gone), or a marker that fails closed.
"""

import os
from pathlib import Path

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline import blur_gate
from gomoveshift.pipeline import day_master_encoder as dme
import tests.test_day_master_blur as _blur_tests

_helper = _blur_tests.TestRepairCabinLeaks()


def _fail_second_replace(monkeypatch, sidecar):
    real = os.replace

    def _replace(src, dst, *a, **k):
        if Path(dst) == sidecar:
            raise OSError("disk went away between the two replaces")
        return real(src, dst, *a, **k)

    monkeypatch.setattr(dme.os, "replace", _replace)


class TestPairSwapIsFailClosed:
    def test_a_failure_between_the_two_replaces_is_refused_downstream(self, tmp_path, monkeypatch):
        raw, mask, sidecar, cfg, _, _ = _helper._setup(tmp_path, monkeypatch, [50])
        old_sidecar = sidecar.read_text()
        _fail_second_replace(monkeypatch, sidecar)
        assert dme._repair_cabin_leaks(raw, cfg, mask, sidecar) == 0
        # The new mask is beside the OLD sidecar: the state the marker exists to flag.
        assert mask.read_bytes() == b"new-mask" and sidecar.read_text() == old_sidecar
        err = blur_gate.frame_pairing_error(raw, mask, sidecar)
        assert err and "repair" in err and "--fresh" in err
        # The pair is also refused by the repair itself and never repaired on top of.
        assert dme._repair_cabin_leaks(raw, cfg, mask, sidecar) == 0

    def test_a_clean_swap_leaves_no_marker(self, tmp_path, monkeypatch):
        raw, mask, sidecar, cfg, _, _ = _helper._setup(tmp_path, monkeypatch, [50])
        assert dme._repair_cabin_leaks(raw, cfg, mask, sidecar) > 0
        assert not blur_gate.repair_swap_pending_path(sidecar).exists()
        assert blur_gate.repair_swap_error(sidecar) is None

    def test_a_failure_before_any_replace_leaves_the_old_pair_and_no_marker(
        self, tmp_path, monkeypatch
    ):
        def _boom(timeline, out_path, *a, **k):
            raise RuntimeError("mask ffmpeg failed")

        raw, mask, sidecar, cfg, _, _ = _helper._setup(tmp_path, monkeypatch, [50], _boom)
        before = sidecar.read_text()
        assert dme._repair_cabin_leaks(raw, cfg, mask, sidecar) == 0
        assert mask.read_bytes() == b"old-mask" and sidecar.read_text() == before
        assert blur_gate.repair_swap_error(sidecar) is None

    def test_a_pending_marker_is_reported_as_a_swap_error_naming_the_sidecar(self, tmp_path):
        # The cache refusal and the marker's clearing are TestStaleMarkerRecovery's.
        sidecar = tmp_path / "day_2026-02-24_blur_regions.json"
        marker = blur_gate.repair_swap_pending_path(sidecar)
        assert marker.parent == sidecar.parent and marker != sidecar
        assert blur_gate.repair_swap_error(sidecar) is None
        marker.write_text("{}")
        err = blur_gate.repair_swap_error(sidecar)
        assert err and sidecar.name in err and "rebuilds the pair in place" in err


class TestDetectorlessRoundIsNotSpent:
    def _wire(self, tmp_path, monkeypatch, rounds=2):
        raw, mask, sidecar, cfg, _, _ = _helper._setup(tmp_path, monkeypatch, [50])
        cfg.drive.blur_cabin_repair_rounds = rounds
        monkeypatch.setattr("redactcam.detect.build_vehicle_detector", lambda *a, **k: None)
        calls = {"recorded": []}
        monkeypatch.setattr(
            blur_gate,
            "check_day_master_cabin_coverage",
            lambda config, video, m: "BLUR GATE (cabin): 1 of 9 EXPOSED",
        )
        monkeypatch.setattr(blur_gate, "cabin_repair_failure", lambda *a, **k: calls.get("failure"))
        monkeypatch.setattr(
            blur_gate,
            "record_cabin_repair_failure",
            lambda config, video, m, rounds, repair_key=None: calls["recorded"].append(rounds),
        )
        return raw, mask, sidecar, cfg, calls

    def test_it_fails_closed_names_the_detector_and_records_nothing(self, tmp_path, monkeypatch):
        raw, mask, sidecar, cfg, calls = self._wire(tmp_path, monkeypatch)
        err = dme._cabin_gate_with_repair(raw, cfg, mask, sidecar)
        assert err and "EXPOSED" in err and "detector unavailable" in err
        assert calls["recorded"] == []  # not exhausted: the detector never ran

    def test_a_missing_vehicle_model_is_the_same(self, tmp_path, monkeypatch):
        raw, mask, sidecar, cfg, calls = self._wire(tmp_path, monkeypatch)
        monkeypatch.setattr("gomoveshift.analysis.resolve_vehicle_model", lambda drive: None)
        err = dme._cabin_gate_with_repair(raw, cfg, mask, sidecar)
        assert err and "detector unavailable" in err and calls["recorded"] == []

    def test_a_retry_with_a_detector_gets_its_full_rounds(self, tmp_path, monkeypatch):
        raw, mask, sidecar, cfg, calls = self._wire(tmp_path, monkeypatch, rounds=2)
        assert "detector unavailable" in dme._cabin_gate_with_repair(raw, cfg, mask, sidecar)
        rounds_run = []

        def _repair(r, c, m, s):
            rounds_run.append(1)
            return 3

        monkeypatch.setattr(dme, "_repair_cabin_leaks", _repair)
        err = dme._cabin_gate_with_repair(raw, cfg, mask, sidecar)
        assert err and "detector unavailable" not in err
        assert len(rounds_run) == 2 and calls["recorded"] == [2]

    def test_a_pass_after_the_leaked_box_cover_still_passes(self, tmp_path, monkeypatch):
        raw, mask, sidecar, cfg, calls = self._wire(tmp_path, monkeypatch)
        verdicts = iter(["BLUR GATE (cabin): leak", None])
        monkeypatch.setattr(blur_gate, "check_day_master_cabin_coverage", lambda *a: next(verdicts))
        assert dme._cabin_gate_with_repair(raw, cfg, mask, sidecar) is None

    def test_one_round_with_no_detector_records_nothing(self, tmp_path, monkeypatch):
        raw, mask, sidecar, cfg, calls = self._wire(tmp_path, monkeypatch, rounds=1)
        err = dme._cabin_gate_with_repair(raw, cfg, mask, sidecar)
        assert err and "detector unavailable" in err and "leaked boxes covered" in err
        assert calls["recorded"] == []


class TestHalfSwappedPairNeverReachesTheEncode:
    def _wire_gate(self, tmp_path, monkeypatch):
        raw, mask, sidecar, cfg, _, _ = _helper._setup(tmp_path, monkeypatch, [50])
        cfg.drive.blur_cabin_repair_rounds = 2
        verdicts = iter(["BLUR GATE (cabin): 1 of 9 EXPOSED", None, None])
        monkeypatch.setattr(blur_gate, "check_day_master_cabin_coverage", lambda *a: next(verdicts))
        monkeypatch.setattr(blur_gate, "cabin_repair_failure", lambda *a, **k: None)
        monkeypatch.setattr(blur_gate, "record_cabin_repair_failure", lambda *a, **k: None)
        _fail_second_replace(monkeypatch, sidecar)
        return raw, mask, sidecar, cfg

    def test_the_gate_refuses_when_the_swap_is_interrupted_inside_the_loop(
        self, tmp_path, monkeypatch
    ):
        raw, mask, sidecar, cfg = self._wire_gate(tmp_path, monkeypatch)
        err = dme._cabin_gate_with_repair(raw, cfg, mask, sidecar)
        # The new mask would pass the gate on its own; the pair must still be refused.
        assert err and "repair swap" in err

    def test_the_encode_re_proves_the_pair_after_the_cabin_gate(self, tmp_path, monkeypatch):
        mask = tmp_path / "day_2026-02-24_blur_mask.mkv"
        cap = _blur_tests._mock_encode(monkeypatch)
        monkeypatch.setattr(dme, "_build_day_blur_mask", lambda *a, **k: mask)
        order = []

        def _cabin(*a):
            order.append("cabin")

        def _pairing(*a, **k):
            order.append("pairing")
            return "BLUR GATE (frame pairing): half-swapped" if order.count("pairing") > 1 else None

        monkeypatch.setattr(dme, "_cabin_gate_with_repair", _cabin)
        monkeypatch.setattr(blur_gate, "frame_pairing_error", _pairing)
        with pytest.raises(blur_gate.BlurGateError, match="half-swapped"):
            dme.encode_day_master(_blur_tests._batch(tmp_path), tmp_path, config=Config())
        assert order == ["pairing", "cabin", "pairing"] and cap == {}


class TestInterruptedSwapIsRebuiltInPlaceOnce:
    def _wire(self, tmp_path, monkeypatch, rebuild_clears_marker):
        raw, mask, sidecar, cfg, _, _ = _helper._setup(tmp_path, monkeypatch, [50])
        cap = _blur_tests._mock_encode(monkeypatch)
        marker = blur_gate.repair_swap_pending_path(sidecar)
        builds = []

        def _build(*a, **k):
            builds.append(1)
            if len(builds) > 1 and rebuild_clears_marker:
                marker.unlink(missing_ok=True)  # what a re-detect does
            return mask

        verdicts = iter(["BLUR GATE (cabin): 1 of 9 EXPOSED", None])
        monkeypatch.setattr(dme, "_build_day_blur_mask", _build)
        monkeypatch.setattr(blur_gate, "check_day_master_cabin_coverage", lambda *a: next(verdicts))
        monkeypatch.setattr(blur_gate, "cabin_repair_failure", lambda *a, **k: None)
        monkeypatch.setattr(blur_gate, "record_cabin_repair_failure", lambda *a, **k: None)
        monkeypatch.setattr(
            blur_gate, "frame_pairing_error", lambda v, m, s: blur_gate.repair_swap_error(s)
        )
        _fail_second_replace(monkeypatch, sidecar)
        return cfg, cap, builds

    def test_the_rebuild_runs_once_and_the_encode_proceeds_on_the_fresh_pair(
        self, tmp_path, monkeypatch
    ):
        cfg, cap, builds = self._wire(tmp_path, monkeypatch, rebuild_clears_marker=True)
        out = dme.encode_day_master(_blur_tests._batch(tmp_path), tmp_path, config=cfg)
        assert out == tmp_path / "day_2026-02-24_master.mp4"
        assert len(builds) == 2 and cap != {}

    def test_a_rebuild_that_leaves_the_marker_is_refused_without_a_third_build(
        self, tmp_path, monkeypatch
    ):
        cfg, cap, builds = self._wire(tmp_path, monkeypatch, rebuild_clears_marker=False)
        with pytest.raises(blur_gate.BlurGateError, match="repair swap"):
            dme.encode_day_master(_blur_tests._batch(tmp_path), tmp_path, config=cfg)
        assert len(builds) == 2 and cap == {}


class TestStaleMarkerRecovery:
    """A marker left by an interrupted swap makes ``_build_day_blur_mask`` re-detect
    instead of reusing the cached pair, and the fresh pair clears the marker."""

    def _build(self, tmp_path, monkeypatch):
        raw = tmp_path / "day_2026-02-24_raw.mp4"
        raw.write_bytes(b"raw")
        veh = tmp_path / "veh.onnx"
        veh.write_bytes(b"model")
        mask = tmp_path / "day_2026-02-24_blur_mask.mkv"
        sidecar = tmp_path / "day_2026-02-24_blur_regions.json"
        detected = []

        def _detect(path, **k):
            detected.append(1)
            return {0: [(1, 1, 4, 4)]}, 10, 0, 0

        def _materialize(timeline, out_path, *a, **k):
            Path(out_path).write_bytes(b"mask")
            return out_path

        monkeypatch.setattr("redactcam.detect_and_track", _detect)
        monkeypatch.setattr("redactcam.materialize_mask_video", _materialize)
        monkeypatch.setattr("redactcam.timeline.probe_video", lambda p: (10.0, 10, 64, 48))
        monkeypatch.setattr("gomoveshift.analysis.resolve_vehicle_model", lambda d: veh)
        monkeypatch.setattr("gomoveshift.analysis.resolve_face_model", lambda d: None)
        monkeypatch.setattr("gomoveshift.analysis.resolve_plate_model", lambda d: None)
        monkeypatch.setattr("gomoveshift.analysis.resolve_person_model", lambda d: None)
        monkeypatch.setattr(dme, "_counts_allow_reuse", lambda *a, **k: True)
        cfg = Config()

        def build():
            return dme._build_day_blur_mask(raw, cfg, mask, sidecar, "2026-02-24")

        return build, detected, sidecar

    def test_a_marker_forces_re_detection_and_is_cleared(self, tmp_path, monkeypatch):
        build, detected, sidecar = self._build(tmp_path, monkeypatch)
        build()
        assert len(detected) == 1
        build()
        assert len(detected) == 1  # control: an intact pair is reused
        marker = blur_gate.repair_swap_pending_path(sidecar)
        marker.write_text("{}")
        build()
        assert len(detected) == 2  # the cache refused the possibly mismatched pair
        assert not marker.exists()  # and the fresh pair cleared the marker
        build()
        assert len(detected) == 2  # the rebuilt pair is reused again
