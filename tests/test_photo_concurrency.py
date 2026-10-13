"""Photo-run concurrency safety — two defects pinned here.

The photo ledger booking was an unguarded load→add→save: two concurrent photo
runs (or a photo run racing a drive run) both read the same snapshot and each
wrote back only its own addition, so the loser's booking vanished — and the
ledger is what stops two runs double-booking a publish slot. Separately, the
photo pipeline had no equivalent of the drive orchestrator's run lock, so two
concurrent ``gomoveshift photos`` runs could interleave over the same state.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline import content_ledger
from gomoveshift.pipeline.orchestrator_lock import (
    ConcurrentRunError,
    _acquire_run_lock,
    _release_run_lock,
)
from gomoveshift.pipeline.photos.orchestrator import PHOTO_RUN_LOCK_NAME, PhotoOrchestrator
from gomoveshift.publishing.photo_run_hooks import photo_run_hooks


def _book(tmp_path: Path, date: str, text: str) -> None:
    orch = PhotoOrchestrator(site_root=tmp_path, output_dir=tmp_path, **photo_run_hooks())
    orch._book_slots(
        [SimpleNamespace(date=date, time="12:00:00")],
        [SimpleNamespace(main_caption=text)],
        tmp_path / "photos.csv",
    )


def _orchestrator(tmp_path: Path) -> PhotoOrchestrator:
    """A publish-ready orchestrator whose real work is stubbed out."""
    orch = PhotoOrchestrator(site_root=tmp_path, output_dir=tmp_path)
    orch._batch_keys = MagicMock(return_value=[])
    orch._publish_batch = MagicMock(return_value=tmp_path / "photos.csv")
    orch._generate_batch = MagicMock(return_value=SimpleNamespace(captions=[]))
    return orch


class TestLedgerBookingIsSerialized:
    """The photo booking must go through the locked ledger writer."""

    def test_concurrent_photo_bookings_do_not_lose_each_other(self, tmp_path, monkeypatch):
        ledger_path = tmp_path / ".content_ledger.json"
        monkeypatch.setattr(content_ledger, "_default_ledger_path", lambda: ledger_path)

        # Widen the read-modify-write window so an unguarded booking reliably
        # loses an update; a lock-guarded one just serializes behind it.
        real_add = content_ledger.ContentLedger.add

        def slow_add(self, entry):
            time.sleep(0.3)
            return real_add(self, entry)

        monkeypatch.setattr(content_ledger.ContentLedger, "add", slow_add)

        threads = [
            threading.Thread(target=_book, args=(tmp_path, "2026-03-01", "Sunrise over the pass")),
            threading.Thread(target=_book, args=(tmp_path, "2026-03-02", "Low tide at harbour")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        entries = json.loads(ledger_path.read_text())["entries"]
        booked = sorted(e["date"] for e in entries)
        assert booked == ["2026-03-01", "2026-03-02"], (
            f"a concurrent photo booking was lost: {booked}"
        )
        assert all(e["content_type"] == "photo" for e in entries)

    def test_booking_failure_does_not_fail_a_shipped_batch(self, tmp_path, monkeypatch):
        """Photos are already on Drive and in the CSV by booking time, so a
        bookkeeping error must be logged, not raised back at the caller."""
        monkeypatch.setattr(content_ledger, "_default_ledger_path", lambda: tmp_path / "x.json")
        monkeypatch.setattr(
            content_ledger.ContentLedger, "save", MagicMock(side_effect=OSError("disk gone"))
        )
        _book(tmp_path, "2026-03-01", "Sunrise")


class TestPhotoRunLock:
    """A second concurrent photo run must be refused, not interleaved."""

    def test_publish_is_rejected_while_another_run_holds_the_lock(self, tmp_path):
        orch = _orchestrator(tmp_path)
        fd, lock_path = _acquire_run_lock(tmp_path, tmp_path, lock_name=PHOTO_RUN_LOCK_NAME)
        try:
            with pytest.raises(ConcurrentRunError):
                orch.publish_batch("captions.yaml")
        finally:
            _release_run_lock(fd, lock_path)
        orch._publish_batch.assert_not_called()

    def test_generate_is_rejected_while_another_run_holds_the_lock(self, tmp_path):
        orch = _orchestrator(tmp_path)
        fd, lock_path = _acquire_run_lock(tmp_path, tmp_path, lock_name=PHOTO_RUN_LOCK_NAME)
        try:
            with pytest.raises(ConcurrentRunError):
                orch.generate_batch(week_start="2026-03-01")
        finally:
            _release_run_lock(fd, lock_path)
        orch._generate_batch.assert_not_called()

    def test_lock_lives_under_the_photo_output_dir_and_is_released(self, tmp_path):
        orch = _orchestrator(tmp_path)
        orch.publish_batch("captions.yaml")
        orch._publish_batch.assert_called_once()
        # Released on exit, so a sequential second run is never blocked.
        assert not (tmp_path / PHOTO_RUN_LOCK_NAME).exists()
        orch.publish_batch("captions.yaml")
        assert orch._publish_batch.call_count == 2

    def test_photo_lock_is_independent_of_the_drive_lock(self, tmp_path):
        """A running drive must not block a photo batch — separate sidecars."""
        orch = _orchestrator(tmp_path)
        fd, lock_path = _acquire_run_lock(tmp_path, tmp_path)
        try:
            orch.publish_batch("captions.yaml")
        finally:
            _release_run_lock(fd, lock_path)
        orch._publish_batch.assert_called_once()
