"""A master at another frame rate never counts as the mask's source, and says so.

The mask pairs 1:1 with the source's frames; the 29.97 fps CFR conversion runs after the
composite, and the encoder deletes the raw concat. A 59.94 fps day's master therefore holds about
half the mask's frames (Galley Head 2026-11-27: sidecar 491,848 at 59.94 fps, master 245,925 at
29.97 fps). ``container_frame_count`` already sets such a master aside; when nothing else can
count the day, the ``mask_frame_count_unknown`` reason names both rates, so the operator knows the
master was set aside for its rate and not because it could not be read.
"""

from __future__ import annotations

import pytest

from gomoveshift.publishing import mask_frame_gate as gate

from tests.test_mask_frame_gate import DATE, finished_day, probe_by_name

HALF = 30000 / 1001


def test_a_master_at_another_rate_with_nothing_else_is_unknown_naming_both_rates(
    tmp_path, monkeypatch
):
    finished_day(tmp_path / "st", [], sidecar=491848, master=True, record=False)
    probe_by_name(monkeypatch, {f"{DATE}": 245925}, rate=HALF)
    why = gate.mask_frame_refusal(tmp_path / "st", drive="galley", record=False)
    assert why and "mask_frame_count_unknown" in why
    assert "mismatch" not in why  # never "re-render on the fixed renderer" for a rate change
    assert "master is 29.970 fps" in why and "59.939 fps" in why


def test_the_rate_note_rides_on_the_incident_row(tmp_path, monkeypatch):
    finished_day(tmp_path / "st", [], sidecar=491848, master=True, record=False)
    probe_by_name(monkeypatch, {f"{DATE}": 245925}, rate=HALF)
    (row,) = gate.day_findings(tmp_path / "st")
    assert row["reason"] == gate.REASON_UNKNOWN and "29.970" in row["rate_note"]


def test_an_unknown_day_with_no_master_carries_no_rate_note(tmp_path, monkeypatch):
    finished_day(tmp_path / "st", [], sidecar=491848, record=False)
    probe_by_name(monkeypatch, {}, rate=HALF)
    (row,) = gate.day_findings(tmp_path / "st")
    assert row["reason"] == gate.REASON_UNKNOWN and "rate_note" not in row


def test_an_unreadable_master_rate_refuses_unknown_without_a_rate_note(tmp_path, monkeypatch):
    finished_day(tmp_path / "st", [], sidecar=491848, master=True, record=False)
    probe_by_name(monkeypatch, {f"{DATE}": 245925}, rate=None)  # rate unknown: never counted
    (row,) = gate.day_findings(tmp_path / "st")
    assert row["reason"] == gate.REASON_UNKNOWN and "rate_note" not in row


def test_a_present_raw_is_counted_whatever_the_master_rate(tmp_path, monkeypatch):
    day = finished_day(tmp_path / "st", [], sidecar=491848, master=True, record=False)
    (day / f"day_{DATE}_raw.mp4").write_bytes(b"")
    monkeypatch.setattr(gate, "probe_frames", lambda path: 491848)
    monkeypatch.setattr(gate, "probe_rate", lambda path: HALF)
    assert gate.mask_frame_refusal(tmp_path / "st", drive="galley", record=False) is None


@pytest.mark.parametrize(
    "stdout, rate", [("30000/1001\n", 30000 / 1001), ("60\n", 60.0), ("0/0\n", None), ("", None)]
)
def test_probe_rate_reads_avg_frame_rate(monkeypatch, stdout, rate):
    seen = {}

    class Done:
        pass

    def run(argv, **kw):
        seen["argv"] = argv
        done = Done()
        done.stdout = stdout
        return done

    monkeypatch.setattr(gate.subprocess, "run", run)
    got = gate.probe_rate("m.mp4")
    assert (got is None) if rate is None else abs(got - rate) < 1e-6
    assert "stream=avg_frame_rate" in seen["argv"]
