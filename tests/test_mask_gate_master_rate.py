"""The publish gate never counts source frames on a master encoded at another rate.

The mask pairs 1:1 with the source's frames; the 29.97 fps CFR conversion runs after the
composite (``single_pass._build_blur_filter_complex``), and the encoder deletes the raw concat.
A 59.94 fps day's master therefore holds about half the mask's frames, and comparing the two
refused every such day as a mismatch (Galley Head 2026-11-27: sidecar 491848 at 59.94 fps,
master 245925 at 29.97 fps, though the encode's own pairing check had passed). With the raw
gone and the master at another rate, only the persisted source-frame proof can speak.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from gomoveshift.publishing import mask_frame_gate as gate

from tests.test_mask_frame_gate import DATE, make_day, persist, rows


@pytest.fixture
def incidents():
    """The conftest-isolated incidents dir (``GMS_HEALTH_INCIDENTS_DIR``)."""
    return Path(os.environ["GMS_HEALTH_INCIDENTS_DIR"])


SRC_FPS = 59.939831059779685


def day_at(stages, count, *, fps=SRC_FPS, master=True):
    """A day whose sidecar records ``count`` frames at ``fps``, the raw gone, a master kept."""
    day = make_day(stages, sidecar=count, raw=False)
    sidecar = day / f"day_{DATE}_blur_regions.json"
    head = {"schema": "redactcam.timeline/v1", "source_hash": "x", "source_fps": fps,
            "source_frame_count": count}  # fmt: skip
    sidecar.write_text(json.dumps(head, indent=2)[:-2] + ',\n  "frame_boxes": {}\n}')
    if master:
        (day / f"day_{DATE}_master.mp4").write_bytes(b"")
    return day


def probes(monkeypatch, frames, rate):
    monkeypatch.setattr(gate, "probe_frames", lambda path: frames)
    monkeypatch.setattr(gate, "probe_rate", lambda path: rate)


def test_a_master_at_half_the_source_rate_defers_to_the_persisted_proof(
    tmp_path, monkeypatch, incidents
):
    day = day_at(tmp_path / "st", 491848)
    persist(day, count=491848)
    (day / f"day_{DATE}_master.mp4").write_bytes(b"")  # persist() keeps the master
    probes(monkeypatch, 245925, 30000 / 1001)
    assert gate.mask_frame_refusal(tmp_path / "st", drive="galley") is None
    assert rows(incidents) == []


def test_a_master_at_another_rate_with_no_proof_is_unknown_naming_both_rates(tmp_path, monkeypatch):
    day_at(tmp_path / "st", 491848)
    probes(monkeypatch, 245925, 30000 / 1001)
    why = gate.mask_frame_refusal(tmp_path / "st", drive="galley", record=False)
    assert why and "mask_frame_count_unknown" in why
    assert "mismatch" not in why  # never "re-render on the fixed renderer" for a rate change
    assert "29.970" in why and "59.940" in why and "source_frames.json" in why


def test_a_proof_that_disagrees_with_the_sidecar_is_still_a_mismatch(tmp_path, monkeypatch):
    day = day_at(tmp_path / "st", 368837)
    persist(day, count=368832)  # Old Head: the mask ran 5 frames past its source
    (day / f"day_{DATE}_master.mp4").write_bytes(b"")
    probes(monkeypatch, 184421, 30000 / 1001)
    why = gate.mask_frame_refusal(tmp_path / "st", drive="old-head", record=False)
    assert why and "mask_frame_count_mismatch" in why and "+5" in why


def test_a_master_at_the_source_rate_still_wins_over_the_proof(tmp_path, monkeypatch):
    day = day_at(tmp_path / "st", 306377, fps=30000 / 1001)
    persist(day, count=306377)
    (day / f"day_{DATE}_master.mp4").write_bytes(b"")
    probes(monkeypatch, 306000, 30000 / 1001)  # same rate: the live count is the measure
    why = gate.mask_frame_refusal(tmp_path / "st", drive="kerry", record=False)
    assert why and "mask_frame_count_mismatch" in why and "306000" in why


def test_an_unreadable_master_rate_keeps_the_master_count(tmp_path, monkeypatch):
    day = day_at(tmp_path / "st", 491848)
    persist(day, count=491848)
    (day / f"day_{DATE}_master.mp4").write_bytes(b"")
    probes(monkeypatch, 245925, None)  # rate unknown: the proof is not consulted
    why = gate.mask_frame_refusal(tmp_path / "st", drive="galley", record=False)
    assert why and "mask_frame_count_mismatch" in why


def test_a_present_raw_is_counted_whatever_the_master_rate(tmp_path, monkeypatch):
    day = day_at(tmp_path / "st", 491848)
    (day / f"day_{DATE}_raw.mp4").write_bytes(b"")
    probes(monkeypatch, 491848, 30000 / 1001)
    assert gate.mask_frame_refusal(tmp_path / "st", drive="galley", record=False) is None


def test_probe_rate_reads_avg_frame_rate(monkeypatch):
    seen = {}

    class Done:
        stdout = "30000/1001\n"

    def run(argv, **kw):
        seen["argv"] = argv
        return Done()

    monkeypatch.setattr(gate.subprocess, "run", run)
    assert abs(gate.probe_rate("m.mp4") - 29.97003) < 1e-4
    assert "stream=avg_frame_rate" in seen["argv"]
    Done.stdout = "0/0\n"
    assert gate.probe_rate("m.mp4") is None
