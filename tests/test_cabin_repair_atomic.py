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

    def test_a_pending_marker_stops_the_mask_cache_reusing_the_pair(self, tmp_path, monkeypatch):
        sidecar = tmp_path / "day_2026-02-24_blur_regions.json"
        marker = blur_gate.repair_swap_pending_path(sidecar)
        assert marker.parent == sidecar.parent and marker != sidecar
        marker.write_text("{}")
        assert blur_gate.repair_swap_error(sidecar)


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
