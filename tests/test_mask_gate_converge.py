"""A kept day master converges with no hand step: cleanup drops the master only once the
publish gate can measure the day without it (a probeable raw, else a validating proof), and
writes that proof from the master when none validates. Fail closed when it cannot."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from gomoveshift.pipeline import blur_gate, day_master_encoder
from gomoveshift.publishing import mask_frame_gate as gate

from tests.test_mask_frame_gate import DATE, is_unknown, make_day, persist
from tests.test_mask_gate_followup import _drop, _warnings

N = 306377


def _finals(day: Path, subs=("regular_lfs", "shorts")) -> None:
    for sub in subs:
        (day / sub).mkdir(exist_ok=True)
    (day / "regular_lfs" / "lf_01.mp4").write_bytes(b"lf")
    (day / "shorts" / "short_01.mp4").write_bytes(b"s")


def _day(tmp_path, *, raw=False) -> Path:
    day = make_day(tmp_path / "st", sidecar=N, raw=raw)
    (day / f"day_{DATE}_blur_mask.mkv").write_bytes(b"mask")
    _finals(day)
    (day / f"day_{DATE}_master.mp4").write_bytes(b"master")
    return day


def _master_probe(monkeypatch, n=N):
    """Answer by file name: the master (and its mask) count ``n``, anything else, the raw
    included, is unreadable. A proof measured from any path but the master is never written."""
    monkeypatch.setattr(
        gate, "probe_frames", lambda p: n if Path(p).name.endswith("_master.mp4") else None
    )
    monkeypatch.setattr(
        blur_gate,
        "container_frame_count",
        lambda p: n if Path(p).name.endswith(("_master.mp4", "_blur_mask.mkv")) else -1,
    )
    monkeypatch.setattr(blur_gate, "sidecar_frame_count", lambda p: n)


# ── MINOR-1: a present raw is not a measurable raw ───────────────────────────


def test_zero_byte_raw_is_cleared_and_the_master_drops(tmp_path, monkeypatch, caplog):
    """The 0-byte raw is the pipeline's own day concat intermediate (Step 3 meant to unlink it):
    once a proof written from the master validates, it goes and the master can drop."""
    day = make_day(tmp_path / "st", sidecar=N, raw=True)  # raw is 0 bytes
    (day / f"day_{DATE}_blur_mask.mkv").write_bytes(b"mask")
    _finals(day)
    master = day / f"day_{DATE}_master.mp4"
    master.write_bytes(b"master")
    _master_probe(monkeypatch)
    assert not is_unknown(tmp_path / "st")
    with caplog.at_level(logging.WARNING):
        assert _drop(day)["dropped"] is True
    assert not master.exists()
    assert not (day / f"day_{DATE}_raw.mp4").exists()
    assert gate.persisted_source_count(day, DATE) == N
    assert any("raw.mp4" in m for m in _warnings(caplog))
    assert not is_unknown(tmp_path / "st")
    assert gate.mask_frame_refusal(tmp_path / "st", drive="k", record=False) is None


def test_dangling_raw_symlink_is_cleared_and_the_master_drops(tmp_path, monkeypatch):
    day = _day(tmp_path)
    (day / f"day_{DATE}_raw.mp4").symlink_to(tmp_path / "offline" / "raw.mp4")
    _master_probe(monkeypatch)
    assert _drop(day)["dropped"] is True
    assert not os.path.lexists(day / f"day_{DATE}_raw.mp4")
    assert not (day / f"day_{DATE}_master.mp4").exists()
    assert not is_unknown(tmp_path / "st")


def test_directory_raw_is_never_unlinked_and_keeps_the_master(tmp_path, monkeypatch, caplog):
    day = _day(tmp_path)
    raw = day / f"day_{DATE}_raw.mp4"
    raw.mkdir()
    (raw / "keep.txt").write_text("x")
    _master_probe(monkeypatch)
    with caplog.at_level(logging.WARNING):
        assert _drop(day)["dropped"] is False
    assert (raw / "keep.txt").exists()
    assert (day / f"day_{DATE}_master.mp4").exists()
    assert any("raw.mp4" in m and "directory" in m for m in _warnings(caplog))
    assert not is_unknown(tmp_path / "st")


def test_mismatched_master_count_unlinks_nothing_and_keeps_the_master(
    tmp_path, monkeypatch, caplog
):
    day = _day(tmp_path)
    raw = day / f"day_{DATE}_raw.mp4"
    raw.write_bytes(b"")
    _master_probe(monkeypatch)
    monkeypatch.setattr(
        blur_gate,
        "container_frame_count",
        lambda p: N - 8 if Path(p).name.endswith("_master.mp4") else N,
    )
    with caplog.at_level(logging.WARNING):
        assert _drop(day)["dropped"] is False
    assert raw.exists()
    assert (day / f"day_{DATE}_master.mp4").exists()
    assert not (day / f"day_{DATE}_source_frames.json").exists()
    assert any(DATE in m and "master" in m for m in _warnings(caplog))


def test_raw_stays_when_the_proof_does_not_validate(tmp_path, monkeypatch, caplog):
    day = _day(tmp_path)
    raw = day / f"day_{DATE}_raw.mp4"
    raw.write_bytes(b"")
    _master_probe(monkeypatch)
    monkeypatch.setattr(blur_gate, "persist_source_frames", lambda *a, **k: None)
    with caplog.at_level(logging.WARNING):
        assert _drop(day)["dropped"] is False
    assert raw.exists() and (day / f"day_{DATE}_master.mp4").exists()
    assert any(DATE in m and "no source frame proof" in m for m in _warnings(caplog))


def test_unlink_failure_keeps_the_master_with_a_warning(tmp_path, monkeypatch, caplog):
    day = _day(tmp_path)
    raw = day / f"day_{DATE}_raw.mp4"
    raw.write_bytes(b"")
    _master_probe(monkeypatch)
    real_unlink = Path.unlink

    def deny(self, *a, **k):
        if self == raw:
            raise PermissionError("denied")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", deny)
    with caplog.at_level(logging.WARNING):
        assert _drop(day)["dropped"] is False
    assert raw.exists() and (day / f"day_{DATE}_master.mp4").exists()
    assert any("raw.mp4" in m and "denied" in m for m in _warnings(caplog))


def test_cleanup_writes_no_proof_while_a_repair_swap_is_pending(tmp_path, monkeypatch, caplog):
    """persist_source_frames may only follow a clean frame_pairing_error."""
    day = _day(tmp_path)
    _master_probe(monkeypatch)
    sidecar = day / f"day_{DATE}_blur_regions.json"
    blur_gate.repair_swap_pending_path(sidecar).write_text("{}")
    with caplog.at_level(logging.WARNING):
        assert _drop(day)["dropped"] is False
    assert (day / f"day_{DATE}_master.mp4").exists()
    assert not (day / f"day_{DATE}_source_frames.json").exists()
    assert any(DATE in m and "repair swap" in m for m in _warnings(caplog))


def test_the_proof_fingerprint_names_the_master(tmp_path, monkeypatch):
    day = _day(tmp_path)
    _master_probe(monkeypatch)
    master = day / f"day_{DATE}_master.mp4"
    expected = "master:" + day_master_encoder.master_fingerprint(master)
    assert _drop(day)["dropped"] is True
    doc = json.loads((day / f"day_{DATE}_source_frames.json").read_text())
    assert doc["source_fingerprint"] == expected


def test_unprobeable_raw_with_a_valid_proof_keeps_the_master(tmp_path, monkeypatch):
    """The gate never reads the proof while a raw exists, so the proof cannot vouch; with the
    master itself unmeasurable here nothing can be re-proved, so nothing is removed."""
    day = _day(tmp_path)
    persist(day)
    (day / f"day_{DATE}_raw.mp4").write_bytes(b"raw")
    monkeypatch.setattr(
        gate, "probe_frames", lambda p: N if Path(p).name.endswith("_master.mp4") else None
    )
    assert _drop(day)["dropped"] is False
    assert not is_unknown(tmp_path / "st")


def test_probeable_raw_lets_the_master_go_and_the_day_stays_measured(tmp_path, monkeypatch):
    day = _day(tmp_path)
    (day / f"day_{DATE}_raw.mp4").write_bytes(b"raw")
    monkeypatch.setattr(gate, "probe_frames", lambda p: N)
    assert _drop(day)["dropped"] is True
    assert not is_unknown(tmp_path / "st")


@pytest.mark.parametrize("raw_kind", ["dangling", "dir"])
def test_helper_refuses_a_present_raw_even_with_a_valid_proof(tmp_path, raw_kind):
    """The gate skips the proof fallback while any raw name exists, so the helper must too."""
    day = _day(tmp_path)
    persist(day)
    raw = day / f"day_{DATE}_raw.mp4"
    if raw_kind == "dir":
        raw.mkdir()
    else:
        raw.symlink_to(tmp_path / "offline" / "raw.mp4")
    assert gate.persisted_source_count(day, DATE) == N
    assert gate.can_measure_without_master(day, DATE) is False
    (day / f"day_{DATE}_master.mp4").unlink()
    assert is_unknown(tmp_path / "st")  # the gate agrees


def test_helper_is_the_single_rule(tmp_path, monkeypatch):
    day = _day(tmp_path)
    assert gate.can_measure_without_master(day, DATE) is False
    persist(day)
    assert gate.can_measure_without_master(day, DATE) is True


# ── MINOR-2: a kept master is never offered for drop again ───────────────────


def test_no_proof_writes_one_from_the_master_then_drops_and_the_day_is_measured(
    tmp_path, monkeypatch
):
    day = _day(tmp_path)
    _master_probe(monkeypatch)
    assert not (day / f"day_{DATE}_source_frames.json").exists()
    assert _drop(day)["dropped"] is True
    assert not (day / f"day_{DATE}_master.mp4").exists()
    assert (day / f"day_{DATE}_source_frames.json").is_file()
    assert gate.mask_frame_refusal(tmp_path / "st", drive="k", record=False) is None


def test_failed_proof_write_keeps_the_master_with_a_warning(tmp_path, monkeypatch, caplog):
    day = _day(tmp_path)
    _master_probe(monkeypatch)
    monkeypatch.setattr(blur_gate, "container_frame_count", lambda p: -1)
    with caplog.at_level(logging.WARNING):
        assert _drop(day)["dropped"] is False
    assert (day / f"day_{DATE}_master.mp4").exists()
    assert any(DATE in m and "master" in m for m in _warnings(caplog))


def test_proof_that_does_not_validate_keeps_the_master_with_a_warning(
    tmp_path, monkeypatch, caplog
):
    day = _day(tmp_path)
    _master_probe(monkeypatch)
    monkeypatch.setattr(blur_gate, "persist_source_frames", lambda *a, **k: day / "x.json")
    with caplog.at_level(logging.WARNING):
        assert _drop(day)["dropped"] is False
    assert (day / f"day_{DATE}_master.mp4").exists()
    assert any(DATE in m and "master" in m for m in _warnings(caplog))


# ── M10: every finals subdir counts, alone ───────────────────────────────────


def test_shorts_only_day_dir_with_no_day_files_is_unknown(tmp_path):
    day = tmp_path / "st" / f"day_{DATE}"
    (day / "shorts").mkdir(parents=True)
    (day / "shorts" / "short_01.mp4").write_bytes(b"x")
    assert is_unknown(tmp_path / "st")


def test_superlong_only_day_dir_with_no_day_files_is_unknown(tmp_path):
    day = tmp_path / "st" / f"day_{DATE}"
    (day / "superlong").mkdir(parents=True)
    (day / "superlong" / "sl_01.mp4").write_bytes(b"x")
    assert is_unknown(tmp_path / "st")


def test_regular_lfs_only_day_dir_with_no_day_files_is_unknown(tmp_path):
    day = tmp_path / "st" / f"day_{DATE}"
    (day / "regular_lfs").mkdir(parents=True)
    (day / "regular_lfs" / "lf_01.mp4").write_bytes(b"x")
    assert is_unknown(tmp_path / "st")


def test_intermediate_only_mp4_counts_as_finals_fail_closed(tmp_path):
    day = tmp_path / "st" / f"day_{DATE}"
    (day / "shorts").mkdir(parents=True)
    (day / "shorts" / "short_01_raw.mp4").write_bytes(b"x")
    assert is_unknown(tmp_path / "st")


# ── M15: a mask swapped mid-count is logged ──────────────────────────────────


def test_mask_swapped_while_counting_logs_a_warning(tmp_path, monkeypatch, caplog):
    day = make_day(tmp_path / "st", sidecar=N)
    mask = day / f"day_{DATE}_blur_mask.mkv"
    mask.write_bytes(b"maskbytes")

    def count(p):
        if Path(p) == mask:
            mask.write_bytes(b"a different mask of other length")
        return N

    monkeypatch.setattr(blur_gate, "container_frame_count", count)
    monkeypatch.setattr(blur_gate, "sidecar_frame_count", lambda p: N)
    with caplog.at_level(logging.WARNING):
        out = blur_gate.persist_source_frames(
            day / f"day_{DATE}_raw.mp4",
            mask,
            day / f"day_{DATE}_blur_regions.json",
            source_fingerprint="fp",
        )
    assert out is None
    (msg,) = _warnings(caplog)
    assert DATE in msg and "changed while" in msg


# ── regressions ──────────────────────────────────────────────────────────────


def test_no_sidecar_day_master_drops_as_before(tmp_path):
    day = make_day(tmp_path / "st", sidecar=None, raw=False)
    _finals(day)
    (day / f"day_{DATE}_master.mp4").write_bytes(b"master")
    assert _drop(day)["dropped"] is True
    assert is_unknown(tmp_path / "st")


def test_mtime_change_after_a_drop_is_unknown_fail_closed(tmp_path):
    day = make_day(tmp_path / "st", sidecar=N)
    persist(day)
    _finals(day)
    (day / f"day_{DATE}_master.mp4").write_bytes(b"master")
    assert _drop(day)["dropped"] is True
    mask = day / f"day_{DATE}_blur_mask.mkv"
    st = mask.stat()
    os.utime(mask, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert is_unknown(tmp_path / "st")
