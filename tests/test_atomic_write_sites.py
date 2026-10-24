"""An interrupted write leaves the old file intact and no stray temp file behind.

Covers the shared helper (``gomoveshift.pipeline._atomic_write``) and the call
sites that used to hand-roll a fixed-name ``.tmp`` write: the health incident
row, the pending-publish sidecar, the trail-opening marker and the video
routing table.
"""

from __future__ import annotations

import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from gomoveshift.health import store
from gomoveshift.pipeline import pending_journal, trail_opening
from gomoveshift.pipeline._atomic_write import atomic_write_text
from gomoveshift.pipeline.site_sync.journey_reconcile import merge_routing


def _interrupt_replace(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise OSError("simulated interruption before the swap")

    monkeypatch.setattr(os, "replace", boom)


def _names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


# --- the helper ---------------------------------------------------------------


def test_atomic_write_text_creates_and_replaces(tmp_path):
    target = tmp_path / "out.json"
    atomic_write_text(target, "one")
    atomic_write_text(target, "two")
    assert target.read_text(encoding="utf-8") == "two"
    assert _names(tmp_path) == ["out.json"]


def test_atomic_write_text_writes_utf8(tmp_path):
    target = tmp_path / "place.txt"
    atomic_write_text(target, "Zürich – Säntis")
    assert target.read_bytes() == "Zürich – Säntis".encode("utf-8")


def test_atomic_write_text_follows_the_umask_not_0600(tmp_path):
    previous = os.umask(0o022)
    try:
        target = tmp_path / "shared.json"
        atomic_write_text(target, "{}")
    finally:
        os.umask(previous)
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_interrupted_swap_keeps_old_file_and_leaves_no_temp(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    atomic_write_text(target, "old")
    _interrupt_replace(monkeypatch)
    with pytest.raises(OSError, match="simulated interruption"):
        atomic_write_text(target, "new")
    assert target.read_text() == "old"
    assert _names(tmp_path) == ["state.json"]


def test_interrupted_write_keeps_old_file_and_leaves_no_temp(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    atomic_write_text(target, "old")

    def boom(_fd):
        raise OSError("simulated disk full")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError, match="simulated disk full"):
        atomic_write_text(target, "new")
    assert target.read_text() == "old"
    assert _names(tmp_path) == ["state.json"]


def test_each_call_uses_a_fresh_temp_name(tmp_path, monkeypatch):
    seen: list[str] = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append(Path(src).name)
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    target = tmp_path / "state.json"
    atomic_write_text(target, "a")
    atomic_write_text(target, "b")
    assert len(set(seen)) == 2


def test_concurrent_writers_leave_one_complete_file_and_no_temp(tmp_path):
    target = tmp_path / "state.json"
    payloads = [str(i) * 5000 for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda text: atomic_write_text(target, text), payloads))
    assert target.read_text(encoding="utf-8") in payloads
    assert _names(tmp_path) == ["state.json"]


# --- the call sites -----------------------------------------------------------


def _row(file: str, note: str) -> dict:
    return {"schema": store.SCHEMA, "file": file, "note": note}


def _seed_incident(tmp_path):
    store.write_row(tmp_path, _row("a.json", "old"))
    return tmp_path / "a.json", lambda: store.write_row(tmp_path, _row("a.json", "new"))


def _entry(youtube_id: str) -> pending_journal.PendingEntry:
    return pending_journal.PendingEntry(
        youtube_id=youtube_id,
        publish_at="2030-01-01T00:00:00+00:00",
        title="t",
        duration="1:00",
        day_date="2025-09-14",
        journey_path="/site/x.md",
        lf_index=1,
    )


def _seed_pending(tmp_path):
    path = tmp_path / pending_journal.PENDING_FILENAME
    pending_journal.save_pending(path, [_entry("old")])
    return path, lambda: pending_journal.save_pending(path, [_entry("new")])


def _seed_marker(tmp_path):
    path = tmp_path / "lf_01.trail_opening.json"
    trail_opening._write_json(path, {"state": "old"})
    return path, lambda: trail_opening._write_json(path, {"state": "new"})


def _seed_routing(tmp_path):
    path = tmp_path / "video_routing.yaml"
    merge_routing(path, {"old": {"journey": "j", "date": "2025-01-01"}})
    return path, lambda: merge_routing(path, {"new": {"journey": "j", "date": "2025-01-02"}})


SITES = {
    "health-incident-row": _seed_incident,
    "pending-journal": _seed_pending,
    "trail-opening-marker": _seed_marker,
    "video-routing": _seed_routing,
}


@pytest.mark.parametrize("site", sorted(SITES))
def test_site_interrupted_swap_keeps_old_file_and_leaves_no_temp(site, tmp_path, monkeypatch):
    path, rewrite = SITES[site](tmp_path)
    before = path.read_bytes()
    _interrupt_replace(monkeypatch)
    try:
        rewrite()
    except OSError:
        pass  # some sites log and swallow; either way the old file must survive
    monkeypatch.undo()
    assert path.read_bytes() == before
    assert _names(tmp_path) == [path.name]


@pytest.mark.parametrize("site", sorted(SITES))
def test_site_still_writes_the_new_content(site, tmp_path):
    path, rewrite = SITES[site](tmp_path)
    before = path.read_bytes()
    rewrite()
    assert path.read_bytes() != before
    assert _names(tmp_path) == [path.name]


def test_incident_row_content_is_unchanged(tmp_path):
    store.write_row(tmp_path, _row("a.json", "n"))
    assert json.loads((tmp_path / "a.json").read_text()) == _row("a.json", "n")
