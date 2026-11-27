"""The same content never queues twice under two names.

The queue knew a row only by ``(project_name, short_index, platform)``. A day reel
is queued as ``day-reel-<drive file id>``, so it and the drive Short of the same
footage (or two reels of two Drive files) never collided, and a second Day 10
Short went out. Every row now stores its media sha256 and, on YouTube, its
normalised title; a record another live row already carries is refused at
enqueue. All data is synthetic."""

from __future__ import annotations

import dataclasses

import pytest
from test_cli_enqueue_plan import _media_for, _record, _sidecar, _titled

from gomoveshift.health.store import load_all
from gomoveshift.publishing.enqueue_from_plan import content_title_key, enqueue_from_plan_sidecar
from gomoveshift.publishing.models import PlatformPayloads, YouTubePayload
from gomoveshift.publishing.queue import cancel_delivery, open_queue_db, transition

REEL = "day-reel-SYNTHFILE01"


@pytest.fixture
def incidents(tmp_path, monkeypatch):
    out = tmp_path / "incidents"
    monkeypatch.setenv("GMS_HEALTH_INCIDENTS_DIR", str(out))
    return out


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    yield c
    c.close()


def _plan(tmp_path, name, *records):
    out = tmp_path / name
    out.mkdir()
    return _sidecar(out, *records)


def _short(index, title, media_index=None, project="proj"):
    record = _record(short_index=index, media=_media_for(media_index or index), payloads=_titled(0))
    payloads = PlatformPayloads(youtube=YouTubePayload(title, "d"))
    return dataclasses.replace(record, project_name=project, payloads=payloads)


def _enqueue(conn, sidecar):
    # The final-text review is not under test here; ``review=False`` skips it.
    return enqueue_from_plan_sidecar(conn, sidecar, review=False)


def _projects(conn):
    return sorted(r[0] for r in conn.execute("SELECT DISTINCT project_name FROM deliveries"))


def test_title_key_drops_case_spacing_and_closing_hashtags():
    assert content_title_key("Day 10:  Dingle  Loop #ireland #kerry") == "day 10: dingle loop"
    assert content_title_key("#ireland") is None


def test_a_day_reel_with_a_queued_shorts_title_is_refused(tmp_path, conn, incidents):
    _enqueue(conn, _plan(tmp_path, "drive", _short(1, "Day 10 Dingle loop")))
    reel = _short(0, "day 10   DINGLE loop #ireland", media_index=99, project=REEL)
    result = _enqueue(conn, _plan(tmp_path, "reel", reel))
    assert result.enqueued == ()
    assert [(r[0], r[1]) for r in result.refused] == [(REEL, 0)]
    assert "title" in result.refused[0][2]
    assert _projects(conn) == ["proj"]
    rows, _ = load_all(incidents)
    assert [(r["kind"], r["project"]) for r in rows] == [("duplicate_content_refused", REEL)]


def test_the_same_media_under_another_project_is_refused(tmp_path, conn, incidents):
    _enqueue(conn, _plan(tmp_path, "drive", _short(1, "Day 10 Dingle loop")))
    reel = _short(0, "Something else entirely", media_index=1, project=REEL)
    other = _short(1, "A different reel", media_index=7, project="day-reel-SYNTHFILE02")
    result = _enqueue(conn, _plan(tmp_path, "reel", reel, other))
    assert "media sha256" in result.refused[0][2]
    assert _projects(conn) == ["day-reel-SYNTHFILE02", "proj"]
    assert len(load_all(incidents)[0]) == 1


@pytest.mark.parametrize("ended", ["cancelled", "failed_permanent"])
def test_a_row_that_never_went_out_blocks_nothing(tmp_path, conn, incidents, ended):
    first = _enqueue(conn, _plan(tmp_path, "drive", _short(1, "Day 10")))
    delivery_id = first.enqueued[0][3]
    if ended == "cancelled":
        assert cancel_delivery(conn, delivery_id)
    else:
        transition(conn, delivery_id, "queued", "failed_permanent", error_class="x")
    reel = _short(0, "Day 10", media_index=1, project=REEL)
    result = _enqueue(conn, _plan(tmp_path, "reel", reel))
    assert result.refused == () and len(result.enqueued) == 1


def test_a_published_row_still_blocks(tmp_path, conn, incidents):
    first = _enqueue(conn, _plan(tmp_path, "drive", _short(1, "Day 10")))
    transition(
        conn,
        first.enqueued[0][3],
        "queued",
        "published",
        platform_upload_id="SYNTHVID001",
        published_at_source="platform_confirmed",
    )
    reel = _short(0, "Day 10", media_index=2, project=REEL)
    assert _enqueue(conn, _plan(tmp_path, "reel", reel)).refused


def test_re_enqueueing_the_same_short_returns_its_row(tmp_path, conn, incidents):
    sidecar = _plan(tmp_path, "drive", _short(1, "Day 10"))
    first = _enqueue(conn, sidecar)
    again = _enqueue(conn, sidecar)
    assert again.refused == ()
    assert again.already_present == first.enqueued
    assert load_all(incidents)[0] == []
