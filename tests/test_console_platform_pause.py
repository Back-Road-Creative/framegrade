"""The console can pause a platform, because the queue already could.

``pubqueue.html`` said per-platform pause was "Not available yet: neither
``dispatch-queue`` nor the queue itself accepts a per-platform filter". That
stopped being true: ``queue.pause_platform``/``resume_platform`` exist and
``dispatch._select_due`` excludes a paused platform in the SELECT itself, so
every drain path -- the CLI, the console's own drain button, a future worker --
is governed by one rule that cannot be routed around. The only thing missing
was a way to say so from the page, which is what these tests pin.

The refusal shape is the one the other console forms use: a bad platform name comes
back on the Publishing page with a sentence, never a separate error page.
"""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from gomoveshift.console import pubqueue
from gomoveshift.console.app import create_app
from gomoveshift.publishing.queue import enqueue, is_platform_paused, open_queue_db


def _client() -> TestClient:
    return TestClient(create_app())


def _config(tmp_path):
    return SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))


def _point_at(tmp_path, monkeypatch):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))


def _pause(client, **over):
    form = {"platform": "instagram", "action": "pause"}
    form.update(over)
    return client.post("/publishing/pause", data=form, follow_redirects=False)


def test_paused_platforms_missing_db_is_empty(tmp_path):
    """Reading the pause list must not create the queue file -- the same rule
    ``queue_rows`` already follows for viewing the tab."""
    assert pubqueue.paused_platforms(_config(tmp_path)) == []
    assert not (tmp_path / "social_queue.sqlite3").exists()


def test_paused_platforms_reads_what_the_queue_wrote(tmp_path):
    open_queue_db(tmp_path).close()
    pubqueue.set_platform_pause(pubqueue.queue_db_path(_config(tmp_path)), "tiktok", paused=True)

    assert pubqueue.paused_platforms(_config(tmp_path)) == ["tiktok"]


def test_set_platform_pause_refuses_an_unknown_platform(tmp_path):
    """The queue returns a bare ``False`` rather than creating a pause row
    nothing will ever match. A redirect to an unchanged page would be
    indistinguishable from success, so the console turns it into a sentence."""
    open_queue_db(tmp_path).close()
    db = pubqueue.queue_db_path(_config(tmp_path))

    try:
        pubqueue.set_platform_pause(db, "myspace", paused=True)
    except ValueError as exc:
        assert "myspace" in str(exc)
    else:  # pragma: no cover - the raise below reports the miss
        raise AssertionError("an unknown platform must not be pausable")


def test_resuming_a_platform_that_was_not_paused_is_refused_not_silent(tmp_path):
    open_queue_db(tmp_path).close()
    db = pubqueue.queue_db_path(_config(tmp_path))

    try:
        pubqueue.set_platform_pause(db, "youtube", paused=False)
    except ValueError as exc:
        assert "youtube" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("resuming a platform that is not paused must say so")


def test_the_page_no_longer_claims_pause_is_unavailable(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)

    body = _client().get("/publishing").text

    assert "Not available yet" not in body
    assert 'action="/publishing/pause"' in body


def test_the_page_offers_every_platform_the_queue_knows(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)

    body = _client().get("/publishing").text

    for platform in ("youtube", "instagram", "facebook", "tiktok"):
        assert f'value="{platform}"' in body


def test_pausing_from_the_page_stops_the_next_drain_selecting_that_platform(tmp_path, monkeypatch):
    """The point of the button. The pause lands in the queue DB both the
    console and ``dispatch-queue`` open, not in console settings."""
    _point_at(tmp_path, monkeypatch)
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "instagram", "h1")
    conn.close()

    response = _pause(_client())

    assert response.status_code == 303
    assert response.headers["location"] == "/publishing"
    conn = open_queue_db(tmp_path)
    try:
        assert is_platform_paused(conn, "instagram")
    finally:
        conn.close()


def test_the_page_lists_what_is_currently_paused(tmp_path, monkeypatch):
    """A pause with nothing on screen saying so is how a queue sits still for a
    week without anyone knowing why."""
    _point_at(tmp_path, monkeypatch)
    open_queue_db(tmp_path).close()
    _pause(_client(), platform="facebook")

    body = _client().get("/publishing").text

    # Anchored on the list's own id: every platform name also appears as an
    # option in the form below it, so splitting on the heading would pass
    # whether or not anything is actually paused.
    listed = body.split('id="paused-platforms"', 1)[1].split("</span>", 1)[0]
    assert "facebook" in listed


def test_resuming_from_the_page_clears_the_pause(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    open_queue_db(tmp_path).close()
    _pause(_client(), platform="tiktok")

    response = _pause(_client(), platform="tiktok", action="resume")

    assert response.status_code == 303
    conn = open_queue_db(tmp_path)
    try:
        assert not is_platform_paused(conn, "tiktok")
    finally:
        conn.close()


def test_an_unknown_platform_is_answered_on_the_publishing_page(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    open_queue_db(tmp_path).close()

    response = _pause(_client(), platform="myspace")

    assert response.status_code == 400
    assert "myspace" in response.text
    assert "Delivery queue" in response.text


def test_a_refused_pause_still_shows_the_queue_and_the_pause_list(tmp_path, monkeypatch):
    """It re-renders the whole tab, not a stripped-down form: the operator who
    lands back here still sees the deliveries they were looking at."""
    _point_at(tmp_path, monkeypatch)
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 7, "youtube", "h1")
    conn.close()
    _pause(_client(), platform="youtube")

    response = _pause(_client(), platform="myspace")

    assert "proj" in response.text
    listed = response.text.split('id="paused-platforms"', 1)[1].split("</span>", 1)[0]
    assert "youtube" in listed


def test_pausing_does_not_touch_rows_already_in_flight(tmp_path, monkeypatch):
    """A pause governs the NEXT pickup. Nothing here may mutate a delivery --
    the platform call a leased row is mid-flight on cannot be un-sent."""
    _point_at(tmp_path, monkeypatch)
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "instagram", "h1")
    before = [dict(r) for r in conn.execute("SELECT * FROM deliveries").fetchall()]
    conn.close()

    _pause(_client())

    conn = open_queue_db(tmp_path)
    try:
        after = [dict(r) for r in conn.execute("SELECT * FROM deliveries").fetchall()]
    finally:
        conn.close()
    assert after == before
