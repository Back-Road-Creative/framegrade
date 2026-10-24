"""An interrupted write leaves the old file intact and no stray temp file behind.

Covers the shared helper (``gomoveshift.atomic_io``) and the call sites that
used to hand-roll a fixed-name ``.tmp`` write: the health incident row and
resume recipe, the pending-publish sidecar, the trail-opening marker and the
video routing table.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from gomoveshift.atomic_io import write_text_atomic
from gomoveshift.health import resume, store
from gomoveshift.pipeline import pending_journal, trail_opening
from gomoveshift.pipeline.site_sync.journey_reconcile import merge_routing


def _interrupt_replace(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise OSError("simulated interruption before the swap")

    monkeypatch.setattr(os, "replace", boom)


def _names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


# --- the helper ---------------------------------------------------------------


def test_write_text_atomic_creates_and_replaces(tmp_path):
    target = tmp_path / "out.json"
    write_text_atomic(target, "one")
    write_text_atomic(target, "two")
    assert target.read_text(encoding="utf-8") == "two"
    assert _names(tmp_path) == ["out.json"]


def test_write_text_atomic_writes_utf8(tmp_path):
    target = tmp_path / "place.txt"
    write_text_atomic(target, "Zürich – Säntis")
    assert target.read_bytes() == "Zürich – Säntis".encode("utf-8")


def test_write_text_atomic_follows_the_umask_not_0600(tmp_path):
    previous = os.umask(0o022)
    try:
        target = tmp_path / "shared.json"
        write_text_atomic(target, "{}")
    finally:
        os.umask(previous)
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_interrupted_swap_keeps_old_file_and_leaves_no_temp(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    write_text_atomic(target, "old")
    _interrupt_replace(monkeypatch)
    with pytest.raises(OSError, match="simulated interruption"):
        write_text_atomic(target, "new")
    assert target.read_text() == "old"
    assert _names(tmp_path) == ["state.json"]


def test_interrupted_write_keeps_old_file_and_leaves_no_temp(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    write_text_atomic(target, "old")

    def boom(_fd):
        raise OSError("simulated disk full")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError, match="simulated disk full"):
        write_text_atomic(target, "new")
    assert target.read_text() == "old"
    assert _names(tmp_path) == ["state.json"]


def test_two_writers_never_share_a_temp_name(tmp_path, monkeypatch):
    seen: list[str] = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append(Path(src).name)
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    target = tmp_path / "state.json"
    write_text_atomic(target, "a")
    write_text_atomic(target, "b")
    assert len(set(seen)) == 2


# --- the call sites -----------------------------------------------------------


def _row(file: str, note: str) -> dict:
    return {"schema": store.SCHEMA, "file": file, "note": note}


def _seed_incident(tmp_path):
    store.write_row(tmp_path, _row("a.json", "old"))
    return tmp_path / "a.json", lambda: store.write_row(tmp_path, _row("a.json", "new"))


def _seed_recipe(tmp_path):
    kwargs = dict(cwd="/w", sha="abc", now_iso="t")
    resume.write_recipe(tmp_path, ["drive", "x"], **kwargs)
    return tmp_path / resume.RECIPE_NAME, lambda: resume.write_recipe(
        tmp_path, ["drive", "y"], **kwargs
    )


def _seed_phase(tmp_path):
    resume.write_recipe(tmp_path, ["drive", "x"], cwd="/w", sha="abc", now_iso="t")
    return tmp_path / resume.RECIPE_NAME, lambda: resume.mark_phase(tmp_path, "phase_d")


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
    "health-resume-recipe": _seed_recipe,
    "health-resume-phase": _seed_phase,
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
