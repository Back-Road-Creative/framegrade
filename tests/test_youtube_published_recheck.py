"""A published YouTube row whose video the channel stops returning becomes
``missing_remote`` on its own, and the lost-response adopt never takes a video
another row already owns.

Before: ``confirm_scheduled_live`` left a row ``published`` when ``videos.list``
returned nothing, nothing ever looked at a published row again, and
``_find_existing_upload`` could adopt another row's video. Fakes only, synthetic ids."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
from freezegun import freeze_time

from gomoveshift.health.store import load_all
from gomoveshift.publishing import queue as q
from gomoveshift.publishing import youtube_adapter
from gomoveshift.publishing.models import YouTubePayload
from gomoveshift.publishing.youtube_adapter import (
    RECHECK_BATCH,
    YouTubeShortsAdapter,
    confirm_scheduled_live,
    recheck_published_videos,
)
from gomoveshift.publishing.youtube_uploads_listing import Upload

NOW = "2026-11-20T12:00:00+00:00"
SLOT = "2026-11-20T09:00:00+00:00"


@pytest.fixture(autouse=True)
def _no_pause(monkeypatch):
    monkeypatch.setattr(youtube_adapter, "_sleep", lambda _s: None)


@pytest.fixture
def incidents(tmp_path, monkeypatch):
    out = tmp_path / "incidents"
    monkeypatch.setenv("GMS_HEALTH_INCIDENTS_DIR", str(out))
    return out


@pytest.fixture
def conn(tmp_path):
    c = q.open_queue_db(tmp_path / "seo")
    yield c
    c.close()


class _Svc:
    """``videos().list(part, id).execute()`` returning only the ids in ``live``;
    ``raises`` makes every lookup raise."""

    def __init__(self, live=(), raises=False):
        self.live, self.raises, self.calls = set(live), raises, []

    def videos(self):
        return self

    def list(self, part, id):  # noqa: A002
        self._ids = id.split(",")
        return self

    def execute(self):
        self.calls.append(self._ids)
        if self.raises:
            raise OSError("network down")
        return {
            "items": [
                {"id": v, "status": {"privacyStatus": "private"}}
                for v in self._ids
                if v in self.live
            ]
        }


def _published(conn, index, video_id, *, scheduled=False):
    did = q.enqueue(conn, "proj", index, "youtube", f"h{index}", scheduled_at=SLOT)
    fields = {"published_at_source": q.PUBLISHED_AT_PLATFORM_CONFIRMED}
    if scheduled:
        fields = {
            "published_at_source": q.PUBLISHED_AT_PLATFORM_SCHEDULED,
            "platform_published_at": SLOT,
        }
    q.transition(conn, did, "queued", "published", platform_upload_id=video_id, **fields)
    return did


def _status(conn, did):
    return conn.execute("SELECT status FROM deliveries WHERE id=?", (did,)).fetchone()[0]


def test_scheduled_row_whose_video_is_absent_twice_is_missing_remote_with_incident(conn, incidents):
    gone = _published(conn, 0, "VIDGONE0001", scheduled=True)
    svc = _Svc()
    assert confirm_scheduled_live(conn, svc, now=NOW) == 0
    assert len(svc.calls) == 2
    assert _status(conn, gone) == q.STATUS_MISSING_REMOTE
    rows, _ = load_all(incidents)
    assert [r["kind"] for r in rows] == [youtube_adapter.MISSING_VIDEO_KIND]
    assert rows[0]["delivery_id"] == gone and rows[0]["video_id"] == "VIDGONE0001"


def test_published_rows_are_rechecked_in_one_batch_and_an_absent_one_goes_missing(conn, incidents):
    live = _published(conn, 0, "VIDLIVE0001")
    gone = _published(conn, 1, "VIDGONE0001")
    svc = _Svc(live={"VIDLIVE0001"})
    assert recheck_published_videos(conn, svc, now=NOW) == 1
    assert svc.calls == [["VIDLIVE0001", "VIDGONE0001"], ["VIDGONE0001"]]
    assert _status(conn, live) == "published"
    assert _status(conn, gone) == q.STATUS_MISSING_REMOTE
    assert len(load_all(incidents)[0]) == 1


def test_a_row_is_rechecked_at_most_once_a_day(conn, incidents):
    _published(conn, 0, "VIDLIVE0001")
    svc = _Svc(live={"VIDLIVE0001"})
    recheck_published_videos(conn, svc, now=NOW)
    recheck_published_videos(conn, svc, now="2026-11-21T11:00:00+00:00")
    assert len(svc.calls) == 1
    recheck_published_videos(conn, svc, now="2026-11-21T12:00:00+00:00")
    assert len(svc.calls) == 2


def test_a_pass_asks_about_at_most_one_batch(conn, incidents):
    ids = [f"VID{i:08d}" for i in range(RECHECK_BATCH + 5)]
    for i, vid in enumerate(ids):
        _published(conn, i, vid)
    svc = _Svc(live=set(ids))
    recheck_published_videos(conn, svc, now=NOW)
    assert [len(c) for c in svc.calls] == [RECHECK_BATCH]


@pytest.mark.parametrize("scheduled", [True, False])
def test_a_raising_lookup_leaves_the_row_published(conn, incidents, scheduled):
    did = _published(conn, 0, "VIDGONE0001", scheduled=scheduled)
    svc = _Svc(raises=True)
    if scheduled:
        assert confirm_scheduled_live(conn, svc, now=NOW) == 0
    else:
        assert recheck_published_videos(conn, svc, now=NOW) == 0
    assert _status(conn, did) == "published"
    assert load_all(incidents)[0] == []


def test_adopt_refuses_a_video_another_row_already_owns(conn):
    _published(conn, 0, "VIDOWNED001")
    mine = q.enqueue(conn, "proj", 1, "youtube", "h1")
    payload = YouTubePayload(title="Same title", description="d")
    upload = Upload("VIDOWNED001", "Same title", "2026-11-26T00:00:00Z", None, "private", 30.0)
    adapter = YouTubeShortsAdapter(SimpleNamespace(_build_service=object), SimpleNamespace())
    with (
        freeze_time("2026-11-26T00:30:00+00:00"),
        mock.patch.object(youtube_adapter, "list_uploads", return_value=[upload]),
    ):
        assert adapter._find_existing_upload(payload, conn=conn, delivery_id=mine) is None
        # The same video is adopted when no other row records it.
        conn.execute("UPDATE deliveries SET platform_upload_id=NULL WHERE id != ?", (mine,))
        assert adapter._find_existing_upload(payload, conn=conn, delivery_id=mine) == "VIDOWNED001"
