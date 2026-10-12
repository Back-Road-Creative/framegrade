"""A manual "override state" control on the Publishing tab -- the
sticking half of the bounded response. Same refusal shape as the other console forms and
``test_console_platform_pause.py``: a bad input comes back on
the Publishing page with a sentence, never a separate error page.
"""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from gomoveshift.console import pubqueue
from gomoveshift.console.app import create_app
from gomoveshift.publishing.queue import get_platform_daily_cap, open_queue_db


def _client() -> TestClient:
    return TestClient(create_app())


def _config(tmp_path):
    return SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))


def _point_at(tmp_path, monkeypatch):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))


def _override(client, **over):
    form = {"platform": "youtube", "state": "SUPPRESSED", "cap": ""}
    form.update(over)
    return client.post("/publishing/override-state", data=form, follow_redirects=False)


def test_platform_daily_caps_missing_db_is_empty(tmp_path):
    assert pubqueue.platform_daily_caps(_config(tmp_path)) == []
    assert not (tmp_path / "social_queue.sqlite3").exists()


def test_setting_an_override_writes_a_manual_reasoned_cap(tmp_path):
    open_queue_db(tmp_path).close()
    db = pubqueue.queue_db_path(_config(tmp_path))

    pubqueue.set_state_override(db, "youtube", "SUPPRESSED", cap=None)

    conn = open_queue_db(tmp_path)
    try:
        row = get_platform_daily_cap(conn, "youtube")
    finally:
        conn.close()
    assert row["state"] == "SUPPRESSED"
    assert row["cap"] == 1
    assert row["reason"] == "manual:SUPPRESSED"


def test_setting_an_override_honours_an_explicit_cap(tmp_path):
    open_queue_db(tmp_path).close()
    db = pubqueue.queue_db_path(_config(tmp_path))

    pubqueue.set_state_override(db, "instagram", "RECOVERING", cap=2)

    conn = open_queue_db(tmp_path)
    try:
        row = get_platform_daily_cap(conn, "instagram")
    finally:
        conn.close()
    assert row["cap"] == 2


def test_clear_removes_a_previously_set_override(tmp_path):
    open_queue_db(tmp_path).close()
    db = pubqueue.queue_db_path(_config(tmp_path))
    pubqueue.set_state_override(db, "tiktok", "SUPPRESSED", cap=None)

    pubqueue.set_state_override(db, "tiktok", "clear", cap=None)

    conn = open_queue_db(tmp_path)
    try:
        assert get_platform_daily_cap(conn, "tiktok") is None
    finally:
        conn.close()


def test_set_state_override_refuses_an_unknown_platform(tmp_path):
    open_queue_db(tmp_path).close()
    db = pubqueue.queue_db_path(_config(tmp_path))
    try:
        pubqueue.set_state_override(db, "myspace", "SUPPRESSED", cap=None)
    except ValueError as exc:
        assert "myspace" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an unknown platform must not be overridable")


def test_set_state_override_refuses_an_unknown_state(tmp_path):
    open_queue_db(tmp_path).close()
    db = pubqueue.queue_db_path(_config(tmp_path))
    try:
        pubqueue.set_state_override(db, "youtube", "ON_FIRE", cap=None)
    except ValueError as exc:
        assert "ON_FIRE" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an unrecognized state must be refused")


def test_the_route_sets_an_override_and_redirects(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    open_queue_db(tmp_path).close()

    response = _override(_client())

    assert response.status_code == 303
    assert response.headers["location"] == "/publishing"
    conn = open_queue_db(tmp_path)
    try:
        row = get_platform_daily_cap(conn, "youtube")
    finally:
        conn.close()
    assert row["reason"] == "manual:SUPPRESSED"


def test_an_unknown_platform_is_answered_on_the_publishing_page(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    open_queue_db(tmp_path).close()

    response = _override(_client(), platform="myspace")

    assert response.status_code == 400
    assert "myspace" in response.text
    assert "Delivery queue" in response.text


def test_a_bad_cap_is_answered_on_the_publishing_page(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    open_queue_db(tmp_path).close()

    response = _override(_client(), cap="not-a-number")

    assert response.status_code == 400
    assert "cap" in response.text.lower()


def test_a_negative_cap_is_refused(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    open_queue_db(tmp_path).close()

    response = _override(_client(), cap="-1")

    assert response.status_code == 400


def test_the_page_lists_the_current_override(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    open_queue_db(tmp_path).close()
    _override(_client())

    body = _client().get("/publishing").text

    assert "youtube" in body
    assert "manual:SUPPRESSED" in body
