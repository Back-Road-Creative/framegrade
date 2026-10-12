"""The dashboard and Publishing use one queue vocabulary, and the
Publishing delivery table collapses to labelled cards on a phone."""

from __future__ import annotations

from fastapi.testclient import TestClient

from gomoveshift.console import overview, pubqueue
from gomoveshift.console.app import create_app
from gomoveshift.publishing.queue import enqueue, open_queue_db
from test_console_pubqueue import _config


def _section(rows):
    return {"rows": rows, "unreadable": []}


def test_queued_delivery_never_sits_beside_an_unqualified_nothing_queued(monkeypatch):
    monkeypatch.setattr(
        overview,
        "build_overview",
        lambda _store, _config=None, **_kw: {
            "needs_attention": _section([]),
            "in_progress": _section([]),
            "coming_up": _section(
                [{"href": "/publishing/7", "label": "proj#1 (youtube) — queued"}]
            ),
        },
    )

    text = TestClient(create_app()).get("/").text

    assert "proj#1 (youtube)" in text
    assert "Nothing is running or queued" not in text
    assert "No jobs are running or queued." in text
    assert "Jobs in progress" in text
    assert "Deliveries coming up" in text


def test_publishing_table_has_phone_card_hooks(tmp_path, monkeypatch):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "youtube", "h1")
    conn.close()

    text = TestClient(create_app()).get("/publishing").text

    assert 'class="delivery-table"' in text
    for label in ("Platform", "Scheduled", "Title", "Status"):
        assert f'data-label="{label}"' in text
    assert "@media (max-width" in text
