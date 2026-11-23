"""Follow-up to the source-frames proof: nothing skipped is silent, a failed write leaves no
debris, a proof is written only for a real day, the date check is load-bearing, cleanup keeps a
master the gate cannot prove, and a day dir with no day artefacts is visible."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from unittest.mock import patch

from gomoveshift.pipeline import blur_gate, cleanup
from gomoveshift.publishing import mask_frame_gate as gate

from tests.test_mask_frame_gate import DATE, is_unknown, make_day, persist, real_persist

_OK = {"ok": True, "reason": "", "missing": [], "bad": [], "count_actual": 1, "count_expected": 1}


def _call(day, mask, sidecar):
    return blur_gate.persist_source_frames(
        day / f"day_{DATE}_raw.mp4", mask, sidecar, source_fingerprint="fp"
    )


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


# ── m1: every None return of the persist path is logged ──────────────────────


def test_unequal_counts_at_persist_are_logged_with_day_and_reason(tmp_path, monkeypatch, caplog):
    day = make_day(tmp_path / "st", sidecar=306377)
    mask = day / f"day_{DATE}_blur_mask.mkv"
    mask.write_bytes(b"m")
    monkeypatch.setattr(blur_gate, "container_frame_count", lambda p: 306377)
    monkeypatch.setattr(blur_gate, "sidecar_frame_count", lambda p: 306385)
    with caplog.at_level(logging.DEBUG):
        assert _call(day, mask, day / f"day_{DATE}_blur_regions.json") is None
    (msg,) = _warnings(caplog)
    assert DATE in msg and "306377" in msg and "306385" in msg


def test_unprobeable_container_at_persist_is_logged(tmp_path, monkeypatch, caplog):
    day = make_day(tmp_path / "st", sidecar=306377)
    mask = day / f"day_{DATE}_blur_mask.mkv"
    mask.write_bytes(b"m")
    monkeypatch.setattr(blur_gate, "container_frame_count", lambda p: -1)
    monkeypatch.setattr(blur_gate, "sidecar_frame_count", lambda p: -1)
    with caplog.at_level(logging.DEBUG):
        assert _call(day, mask, day / f"day_{DATE}_blur_regions.json") is None
    (msg,) = _warnings(caplog)
    assert DATE in msg


def test_missing_mask_or_sidecar_at_persist_is_logged(tmp_path, caplog):
    day = make_day(tmp_path / "st", sidecar=306377)
    sidecar = day / f"day_{DATE}_blur_regions.json"
    with caplog.at_level(logging.DEBUG):
        assert _call(day, None, sidecar) is None
    (msg,) = _warnings(caplog)
    assert DATE in msg and "mask" in msg


# ── m4: a failed replace leaves no .tmp behind ───────────────────────────────


def test_failed_replace_removes_its_tmp_and_is_logged(tmp_path, monkeypatch, caplog):
    day = make_day(tmp_path / "st", sidecar=306377)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(blur_gate.os, "replace", boom)
    with caplog.at_level(logging.WARNING):
        assert real_persist(day, monkeypatch) is None
    assert not list(day.glob("*.tmp"))
    assert any("disk full" in m for m in _warnings(caplog))


# ── m3: no proof for a call that is not a real day ───────────────────────────


def test_stage_path_writes_no_proof_and_says_why(tmp_path, monkeypatch, caplog):
    stages = tmp_path / "stages"
    stages.mkdir()
    (stages / "proj_blur_regions.json").write_text('{"source_frame_count": 5}')
    mask = stages / "proj_blur_mask.mkv"
    mask.write_bytes(b"m")
    monkeypatch.setattr(blur_gate, "container_frame_count", lambda p: 5)
    monkeypatch.setattr(blur_gate, "sidecar_frame_count", lambda p: 5)
    with caplog.at_level(logging.DEBUG, logger=blur_gate.logger.name):
        out = blur_gate.persist_source_frames(
            stages / "proj.mp4", mask, stages / "proj_blur_regions.json", source_fingerprint="fp"
        )
    assert out is None
    assert not list(stages.glob("*source_frames*"))
    assert any("proj" in r.getMessage() and r.levelno >= logging.DEBUG for r in caplog.records)


def test_a_real_day_still_gets_its_proof(tmp_path, monkeypatch):
    day = make_day(tmp_path / "st", sidecar=306377)
    assert real_persist(day, monkeypatch) is not None


# ── m2: the gate's date check is load-bearing ────────────────────────────────


def test_proof_whose_date_mismatches_alone_is_unknown(tmp_path):
    day = make_day(tmp_path / "st", sidecar=306377)
    proof = persist(day)
    doc = json.loads(proof.read_text())
    doc["date"] = "2025-09-17"  # day_dir, counts, sidecar hash and mask all still match
    proof.write_text(json.dumps(doc))
    assert is_unknown(tmp_path / "st")


# ── m5: cleanup keeps a master the gate cannot prove ─────────────────────────


def _day_with_finals(base: Path, with_sidecar=True) -> Path:
    day = make_day(base, sidecar=306377 if with_sidecar else None, raw=False)
    for sub in ("regular_lfs", "shorts"):
        (day / sub).mkdir()
    (day / "regular_lfs" / "lf_01.mp4").write_bytes(b"lf")
    (day / "shorts" / "short_01.mp4").write_bytes(b"s")
    (day / f"day_{DATE}_master.mp4").write_bytes(b"master")
    return day


def _drop(day):
    with patch("gomoveshift.pipeline.cleanup.verify_shorts_integrity", return_value=_OK):
        return cleanup.drop_day_master(day, date=DATE)


def test_master_kept_when_no_proof_validates(tmp_path, caplog):
    day = _day_with_finals(tmp_path / "st")
    (day / f"day_{DATE}_blur_mask.mkv").write_bytes(b"mask")
    with caplog.at_level(logging.WARNING):
        assert _drop(day)["dropped"] is False
    assert (day / f"day_{DATE}_master.mp4").exists()
    assert any(DATE in m for m in _warnings(caplog))


def test_master_kept_after_an_mtime_only_change_so_the_gate_can_still_probe_it(
    tmp_path, monkeypatch
):
    day = _day_with_finals(tmp_path / "st")
    persist(day)
    mask = day / f"day_{DATE}_blur_mask.mkv"
    st = mask.stat()
    os.utime(mask, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert _drop(day)["dropped"] is False
    monkeypatch.setattr(gate, "probe_frames", lambda p: 306377)
    assert gate.mask_frame_refusal(tmp_path / "st", drive="k", record=False) is None


def test_master_dropped_when_the_persisted_proof_validates(tmp_path):
    day = _day_with_finals(tmp_path / "st")
    persist(day)
    assert _drop(day)["dropped"] is True
    assert not (day / f"day_{DATE}_master.mp4").exists()
    assert not is_unknown(tmp_path / "st")


def test_master_dropped_when_the_raw_is_present_and_probes(tmp_path, monkeypatch):
    day = _day_with_finals(tmp_path / "st")
    (day / f"day_{DATE}_raw.mp4").write_bytes(b"raw")
    monkeypatch.setattr(gate, "probe_frames", lambda p: 306377)
    assert _drop(day)["dropped"] is True


def test_master_dropped_for_a_day_with_no_blur_artefacts(tmp_path):
    day = _day_with_finals(tmp_path / "st", with_sidecar=False)
    assert _drop(day)["dropped"] is True


# ── pre-existing: a day dir with no day_<date>_* file is visible ─────────────


def test_day_dir_with_finals_but_no_day_artefacts_is_unknown(tmp_path):
    day = make_day(tmp_path / "st", sidecar=306377)
    persist(day)
    (day / "regular_lfs").mkdir()
    (day / "regular_lfs" / "lf_01.mp4").write_bytes(b"x")
    day.rename(tmp_path / "st" / "day_2025-09-17")
    why = gate.mask_frame_refusal(tmp_path / "st", drive="k", record=False)
    assert why and "mask_frame_count_unknown" in why and "2025-09-17" in why


def test_empty_day_dir_is_skipped_with_a_warning(tmp_path, caplog):
    (tmp_path / "st" / "day_2025-09-18" / "work").mkdir(parents=True)
    with caplog.at_level(logging.WARNING):
        assert gate.day_findings(tmp_path / "st") == []
    assert any("2025-09-18" in m for m in _warnings(caplog))
