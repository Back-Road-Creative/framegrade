"""Unit tests for the pending_journal sidecar module."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gomoveshift.pipeline import pending_journal as pending_journal_module
from gomoveshift.pipeline.pending_journal import (
    PENDING_FILENAME,
    PendingEntry,
    load_pending,
    merge_pending,
    save_pending,
    split_by_readiness,
)


def _make_entry(
    youtube_id: str, publish_at: str, journey_path: str = "/site/x.md", lf_index: int = 1
) -> PendingEntry:
    return PendingEntry(
        youtube_id=youtube_id,
        publish_at=publish_at,
        title=f"Title {youtube_id}",
        duration="10:00",
        day_date="2025-09-14",
        journey_path=journey_path,
        lf_index=lf_index,
    )


def test_load_pending_missing_file_returns_empty(tmp_path: Path):
    assert load_pending(tmp_path / "nope.json") == []


def test_save_then_load_roundtrip(tmp_path: Path):
    entries = [_make_entry("vid1", "2026-06-03T15:00:00Z")]
    path = tmp_path / PENDING_FILENAME
    save_pending(path, entries)
    assert load_pending(path) == entries


def test_save_empty_deletes_file(tmp_path: Path):
    path = tmp_path / PENDING_FILENAME
    save_pending(path, [_make_entry("vid1", "2026-06-03T15:00:00Z")])
    assert path.is_file()
    save_pending(path, [])
    assert not path.is_file()


def test_save_empty_when_no_file_is_noop(tmp_path: Path):
    path = tmp_path / PENDING_FILENAME
    save_pending(path, [])  # must not crash
    assert not path.is_file()


def test_merge_pending_appends_new(tmp_path: Path):
    path = tmp_path / PENDING_FILENAME
    save_pending(path, [_make_entry("v1", "2026-06-03T15:00:00Z")])
    merge_pending(path, [_make_entry("v2", "2026-06-10T15:00:00Z")])
    loaded = load_pending(path)
    ids = sorted(e.youtube_id for e in loaded)
    assert ids == ["v1", "v2"]


def test_merge_pending_updates_same_key(tmp_path: Path):
    path = tmp_path / PENDING_FILENAME
    save_pending(path, [_make_entry("v1", "2026-06-03T15:00:00Z")])
    merge_pending(path, [_make_entry("v1", "2026-06-04T15:00:00Z")])  # updated time
    loaded = load_pending(path)
    assert len(loaded) == 1
    assert loaded[0].publish_at == "2026-06-04T15:00:00Z"


def test_merge_pending_keys_by_journey_and_id(tmp_path: Path):
    path = tmp_path / PENDING_FILENAME
    e1 = _make_entry("v1", "2026-06-03T15:00:00Z", journey_path="/a.md")
    e2 = _make_entry("v1", "2026-06-03T15:00:00Z", journey_path="/b.md")
    merge_pending(path, [e1, e2])
    loaded = load_pending(path)
    journeys = sorted(e.journey_path for e in loaded)
    assert journeys == ["/a.md", "/b.md"]


def test_merge_pending_survives_concurrent_writer(tmp_path: Path, monkeypatch):
    """Two overlapping ``merge_pending`` read-modify-writes on the SAME sidecar
    (the hourly replay cron vs. a live drive run) must not lose an update. A
    slow reader that snapshots before a fast writer's save must not clobber
    the fast writer's entry when it finally saves its own — and the sidecar
    must still exist afterward (both entries are live, never empty)."""
    path = tmp_path / PENDING_FILENAME
    slow_started = threading.Event()
    real_load = pending_journal_module.load_pending

    def instrumented_load(p):
        result = real_load(p)
        if threading.current_thread().name == "slow-writer":
            slow_started.set()
            time.sleep(0.3)  # give the fast writer time to complete its RMW
        return result

    monkeypatch.setattr(pending_journal_module, "load_pending", instrumented_load)

    def slow_writer():
        merge_pending(path, [_make_entry("slow-video", "2026-06-03T15:00:00Z")])

    def fast_writer():
        assert slow_started.wait(timeout=2), "slow writer never started its read"
        merge_pending(path, [_make_entry("fast-video", "2026-06-03T15:00:00Z")])

    t_slow = threading.Thread(target=slow_writer, name="slow-writer")
    t_fast = threading.Thread(target=fast_writer, name="fast-writer")
    t_slow.start()
    t_fast.start()
    t_slow.join(timeout=5)
    t_fast.join(timeout=5)

    assert path.is_file(), "sidecar was deleted out from under a live entry"
    ids = {e.youtube_id for e in load_pending(path)}
    assert ids == {"slow-video", "fast-video"}, f"lost update: got {ids}"


def test_split_by_readiness_past_is_ready():
    now = datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc)
    past = _make_entry("v1", (now - timedelta(days=1)).isoformat().replace("+00:00", "Z"))
    future = _make_entry("v2", (now + timedelta(days=1)).isoformat().replace("+00:00", "Z"))
    ready, still = split_by_readiness([past, future], now)
    assert [e.youtube_id for e in ready] == ["v1"]
    assert [e.youtube_id for e in still] == ["v2"]


def test_split_by_readiness_exactly_now_is_ready():
    now = datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc)
    on_time = _make_entry("v1", now.isoformat().replace("+00:00", "Z"))
    ready, still = split_by_readiness([on_time], now)
    assert [e.youtube_id for e in ready] == ["v1"]
    assert still == []


def test_split_by_readiness_unparseable_stays_pending():
    now = datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc)
    junk = _make_entry("v1", "not-an-iso-date")
    ready, still = split_by_readiness([junk], now)
    assert ready == []
    assert [e.youtube_id for e in still] == ["v1"]


def test_load_pending_rejects_non_list(tmp_path: Path):
    path = tmp_path / PENDING_FILENAME
    path.write_text('{"not": "a list"}')
    with pytest.raises(ValueError, match="expected a JSON list"):
        load_pending(path)
