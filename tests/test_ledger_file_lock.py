"""ContentLedger.save takes the file lock it advertises — no lost-update double-book.

Two concurrent drives (the documented two-USA-drives no-double-book mode) each
do a load→add→save against the shared ledger. Without an inter-process lock held
across that whole read-modify-write, both load the same snapshot and each saves
back only its own addition — last writer wins and one confirmed booking is lost
permanently (reconcile never re-adds it from YouTube → silent double-book).
"""

import contextlib
import json
import threading

from gomoveshift.pipeline import content_ledger as cl


def _entry(video_id, date, project):
    return cl.ContentEntry(
        date=date,
        time="16:00:00",
        content_type="longform",
        title=f"Drive {video_id}",
        project_name=project,
        status="scheduled",
        source="pipeline",
        video_id=video_id,
        region="usa",
    )


def test_concurrent_record_confirmed_upload_keeps_both_bookings(tmp_path, monkeypatch):
    """Two drives booking distinct LFs into the same ledger must both survive.

    A barrier planted just after each drive's ``load`` forces the interleave
    that used to drop a booking: both loaded the same snapshot, each saved its
    own single addition, last writer won. With the file lock held across
    load→add→save the second drive serializes behind the first and both
    bookings persist; the barrier simply times out.
    """
    ledger_path = tmp_path / ".content_ledger.json"
    barrier = threading.Barrier(2, timeout=3.0)
    real_load = cl.ContentLedger.load

    def slow_load(path=None):
        led = real_load(path)
        with contextlib.suppress(threading.BrokenBarrierError):
            barrier.wait()
        return led

    monkeypatch.setattr(cl.ContentLedger, "load", staticmethod(slow_load))

    def book(video_id, date, project):
        cl.record_confirmed_upload([_entry(video_id, date, project)], ledger_path)

    t_a = threading.Thread(target=book, args=("AAA", "2026-08-04", "usa_drive_a"))
    t_b = threading.Thread(target=book, args=("BBB", "2026-08-07", "usa_drive_b"))
    t_a.start()
    t_b.start()
    t_a.join(timeout=30)
    t_b.join(timeout=30)
    assert not t_a.is_alive() and not t_b.is_alive(), "booking thread deadlocked"

    data = json.loads(ledger_path.read_text())
    ids = {e.get("video_id") for e in data["entries"]}
    assert ids == {"AAA", "BBB"}, f"lost-update double-book: only {sorted(ids)} survived"


def test_record_confirmed_upload_single_call_is_not_reentrant_deadlocked(tmp_path):
    """A single booking (outer lock wrapping load→add→save, then save's own
    lock) must not self-deadlock: the ledger lock is reentrant within a thread.
    """
    ledger_path = tmp_path / ".content_ledger.json"
    done = threading.Event()

    def book():
        cl.record_confirmed_upload([_entry("SOLO", "2026-08-04", "usa_drive")], ledger_path)
        done.set()

    threading.Thread(target=book).start()
    assert done.wait(timeout=10), "single record_confirmed_upload deadlocked on its own lock"
    ids = {e.video_id for e in cl.ContentLedger.load(ledger_path).entries}
    assert ids == {"SOLO"}
