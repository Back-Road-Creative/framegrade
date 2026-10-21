"""The Publishing tab names what a ``published`` row actually is.

A row sits in the ``published`` group once the queue is finished with it. That is not proof
the platform published: a platform-scheduled row goes live later, and an operator-confirmed or
legacy row has no platform confirmation. The Status cell must keep those apart, or a coverage
claim read off the tab overstates what is out.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from gomoveshift.console import pubqueue
from gomoveshift.console.app import create_app
from gomoveshift.publishing.queue import _now_iso, open_queue_db

_EXPECTED = {
    "platform_confirmed": "published",
    "platform_reconciled": "published",
    "platform_scheduled": "remotely scheduled",
    "operator_confirmed": "unverified",
    None: "unverified",
}


@pytest.mark.parametrize("source", list(_EXPECTED))
def test_delivery_state_of_a_published_row(source):
    row = {"status": "published", "published_at_source": source}
    assert pubqueue.delivery_state(row) == _EXPECTED[source]


@pytest.mark.parametrize("status", ["queued", "uploaded", "failed_permanent", "cancelled"])
def test_delivery_state_of_any_other_row_is_its_status(status):
    assert pubqueue.delivery_state({"status": status, "published_at_source": None}) == status


def test_rows_by_state_attaches_the_state():
    rows = [{"status": "published", "id": 1, "published_at_source": "platform_scheduled"}]
    (_status, grouped), *_ = pubqueue.rows_by_state(rows)
    assert grouped[0]["delivery_state"] == "remotely scheduled"


def test_the_page_status_cells_do_not_all_say_published(tmp_path, monkeypatch):
    conn = open_queue_db(tmp_path)
    now = _now_iso()
    for i, source in enumerate(["platform_confirmed", "platform_scheduled", "operator_confirmed"]):
        conn.execute(
            "INSERT INTO deliveries (project_name, short_index, platform, plan_hash, status, "
            "published_at_source, shadow, created_at, updated_at) "
            "VALUES (?, 1, 'youtube', 'h', 'published', ?, 0, ?, ?)",
            (f"proj{i}", source, now, now),
        )
    conn.commit()
    conn.close()
    cfg = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
    monkeypatch.setattr(pubqueue, "_config", lambda config=None: cfg)
    html = TestClient(create_app()).get("/publishing").text
    cells = re.findall(r'<td data-label="Status">([^<]*)</td>', html)
    assert sorted(c.strip() for c in cells) == ["published", "remotely scheduled", "unverified"]


def _seed(tmp_path, monkeypatch, source):
    conn = open_queue_db(tmp_path)
    now = _now_iso()
    cur = conn.execute(
        "INSERT INTO deliveries (project_name, short_index, platform, plan_hash, status, "
        "published_at_source, shadow, created_at, updated_at) "
        "VALUES ('proj', 1, 'youtube', 'h', 'published', ?, 0, ?, ?)",
        (source, now, now),
    )
    conn.commit()
    delivery_id = cur.lastrowid
    conn.close()
    cfg = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
    monkeypatch.setattr(pubqueue, "_config", lambda config=None: cfg)
    return delivery_id


@pytest.mark.parametrize("source", ["platform_scheduled", "operator_confirmed"])
def test_the_calendar_does_not_call_an_unproven_row_published(tmp_path, monkeypatch, source):
    _seed(tmp_path, monkeypatch, source)
    html = TestClient(create_app()).get("/publishing/calendar").text
    cells = re.findall(r"<td>(published|remotely scheduled|unverified)</td>", html)
    assert cells == [_EXPECTED[source]]


@pytest.mark.parametrize("source", ["platform_scheduled", "operator_confirmed"])
def test_the_detail_page_does_not_call_an_unproven_row_published(tmp_path, monkeypatch, source):
    delivery_id = _seed(tmp_path, monkeypatch, source)
    html = TestClient(create_app()).get(f"/publishing/{delivery_id}").text
    assert re.search(rf"<th>Status</th><td>{_EXPECTED[source]}</td>", html)
    assert "this delivery is complete" not in html


def test_the_detail_page_of_a_platform_confirmed_row_is_complete(tmp_path, monkeypatch):
    delivery_id = _seed(tmp_path, monkeypatch, "platform_confirmed")
    html = TestClient(create_app()).get(f"/publishing/{delivery_id}").text
    assert re.search(r"<th>Status</th><td>published</td>", html)
    assert "this delivery is complete" in html


@pytest.mark.parametrize("source", ["platform_scheduled", "operator_confirmed"])
def test_the_calendar_card_names_the_delivery_state_not_the_status(tmp_path, monkeypatch, source):
    _seed(tmp_path, monkeypatch, source)
    noon = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)
    conn = open_queue_db(tmp_path)
    conn.execute("UPDATE deliveries SET scheduled_at = ?", (noon.isoformat(),))
    conn.commit()
    conn.close()
    html = TestClient(create_app()).get(f"/publishing/calendar?week={noon.date()}").text
    assert f"youtube &mdash; {_EXPECTED[source]}<br>" in html
    assert "youtube &mdash; published" not in html


@pytest.mark.parametrize("source", ["platform_scheduled", "operator_confirmed"])
def test_the_detail_sentence_names_the_delivery_state_not_the_status(tmp_path, monkeypatch, source):
    delivery_id = _seed(tmp_path, monkeypatch, source)
    html = TestClient(create_app()).get(f"/publishing/{delivery_id}").text
    assert f"This delivery is {_EXPECTED[source]} --" in html
    assert "This delivery is published" not in html
