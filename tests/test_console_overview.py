"""The console's first screen names what needs an operator instead of
listing where things live. Unit-level coverage of the read model
(``gomoveshift.console.overview``); the index route is covered by the
existing ``test_console_app.py`` fixtures for the nav list itself.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from gomoveshift.console import jobs, overview


def _store(tmp_path) -> jobs.JobStore:
    return jobs.JobStore(tmp_path / "jobs.sqlite3")


def _empty_config(tmp_path) -> SimpleNamespace:
    """A ``Config`` whose delivery queue db simply doesn't exist yet --
    the queue's own definition of an empty queue (see ``pubqueue.queue_rows``)."""
    return SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path / "no-such-dir")))


def test_needs_attention_says_so_when_nothing_needs_you(tmp_path):
    result = overview.needs_attention(_store(tmp_path), _empty_config(tmp_path))
    assert result == {"rows": [], "unreadable": []}


def test_needs_attention_lists_failed_and_interrupted_jobs_only(tmp_path):
    store = _store(tmp_path)
    ok_id = store.create("dispatch-queue", ["x"], "")
    store.update(ok_id, status="done", rc=0)
    failed_id = store.create("dispatch-queue", ["x"], "")
    store.update(failed_id, status="failed", rc=1)
    running_id = store.create("dispatch-queue", ["x"], "")
    store.update(running_id, status="running")

    result = overview.needs_attention(store, _empty_config(tmp_path))

    hrefs = [item["href"] for item in result["rows"]]
    assert f"/jobs/{failed_id}" in hrefs
    assert f"/jobs/{running_id}" not in hrefs
    assert f"/jobs/{ok_id}" not in hrefs
    assert result["unreadable"] == []


def test_needs_attention_reports_unreadable_when_the_job_store_raises(tmp_path, monkeypatch):
    store = _store(tmp_path)
    monkeypatch.setattr(
        jobs,
        "list_jobs",
        lambda _store: (_ for _ in ()).throw(sqlite3.OperationalError("disk I/O error")),
    )

    result = overview.needs_attention(store, _empty_config(tmp_path))

    # Unreadable is reported as its own fact -- never silently folded into
    # an empty ``items`` list, which would read as "nothing needs you".
    assert result["unreadable"] == ["jobs"]
    assert result["rows"] == []


def test_in_progress_lists_queued_and_running_jobs(tmp_path):
    store = _store(tmp_path)
    running_id = store.create("dispatch-queue", ["x"], "")
    store.update(running_id, status="running")
    done_id = store.create("dispatch-queue", ["x"], "")
    store.update(done_id, status="done", rc=0)

    result = overview.in_progress(store)

    hrefs = [item["href"] for item in result["rows"]]
    assert f"/jobs/{running_id}" in hrefs
    assert f"/jobs/{done_id}" not in hrefs


def test_coming_up_is_empty_and_honest_with_no_deliveries(tmp_path):
    result = overview.coming_up(_empty_config(tmp_path))
    assert result == {"rows": [], "unreadable": []}


def test_build_overview_has_all_three_sections(tmp_path):
    result = overview.build_overview(_store(tmp_path), _empty_config(tmp_path))
    assert set(result) == {"needs_attention", "in_progress", "coming_up"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_coming_up_shows_a_delivery_the_publishing_tab_calls_scheduled(tmp_path, monkeypatch):
    """The overview must not be less truthful than the page it summarises.

    ``pubqueue`` treats the row's own ``scheduled_at`` column as authoritative
    but falls back to the sidecar's time when that column is NULL -- the
    Publishing tab shows such a delivery as scheduled (see
    ``pubqueue.enrich_rows_with_sidecar``'s ``elif sidecar_scheduled is not
    None`` branch). Reading the raw column here instead of the effective one
    would silently drop that delivery from "Coming up" while Publishing lists
    it, which is the overview quietly disagreeing with the truth it points at.
    """
    from test_console_pubqueue import _short_record, _write_sidecar

    sidecar_path = _write_sidecar(tmp_path, _short_record())
    monkeypatch.setattr(
        overview.pubqueue,
        "queue_rows",
        lambda _config=None: [
            {
                "id": 7,
                "project_name": "proj",
                "short_index": 1,
                "platform": "youtube",
                "status": "queued",
                "scheduled_at": None,
                "sidecar_path": str(sidecar_path),
            }
        ],
    )

    from datetime import datetime, timezone

    before_the_sidecar_schedule = datetime(2026, 9, 1, tzinfo=timezone.utc)
    result = overview.coming_up(_empty_config(tmp_path), now=before_the_sidecar_schedule)

    assert result["unreadable"] == []
    assert [item["href"] for item in result["rows"]] == ["/publishing/7"]


def _now():
    from datetime import datetime, timezone

    return datetime(2026, 9, 14, 0, 45, tzinfo=timezone.utc)


def test_an_overdue_queued_delivery_is_needs_attention_not_coming_up(tmp_path, monkeypatch):
    monkeypatch.setattr(
        overview.pubqueue,
        "queue_rows",
        lambda _config=None: [
            {
                "id": 24,
                "project_name": "proj",
                "short_index": 1,
                "platform": "tiktok",
                "status": "queued",
                "scheduled_at": "2026-09-09T12:00:00+00:00",
                "sidecar_path": None,
            }
        ],
    )
    monkeypatch.setattr(
        overview.pubqueue,
        "enrich_rows_with_sidecar",
        lambda rows, cache=None: [
            {**row, "scheduled_at_effective_local": "2026-09-09 12:00 UTC"} for row in rows
        ],
    )

    attention = overview.needs_attention(_store(tmp_path), _empty_config(tmp_path), now=_now())
    upcoming = overview.coming_up(_empty_config(tmp_path), now=_now())

    attention_hrefs = [item["href"] for item in attention["rows"]]
    assert "/publishing/24" in attention_hrefs
    attention_item = next(item for item in attention["rows"] if item["href"] == "/publishing/24")
    assert "overdue" in attention_item["label"]
    assert [item["href"] for item in upcoming["rows"]] == []


def test_a_future_queued_delivery_stays_in_coming_up_only(tmp_path, monkeypatch):
    monkeypatch.setattr(
        overview.pubqueue,
        "queue_rows",
        lambda _config=None: [
            {
                "id": 99,
                "project_name": "proj",
                "short_index": 2,
                "platform": "youtube",
                "status": "queued",
                "scheduled_at": "2026-09-20T12:00:00+00:00",
                "sidecar_path": None,
            }
        ],
    )
    monkeypatch.setattr(
        overview.pubqueue,
        "enrich_rows_with_sidecar",
        lambda rows, cache=None: [
            {**row, "scheduled_at_effective_local": "2026-09-20 12:00 UTC"} for row in rows
        ],
    )

    attention = overview.needs_attention(_store(tmp_path), _empty_config(tmp_path), now=_now())
    upcoming = overview.coming_up(_empty_config(tmp_path), now=_now())

    assert [item["href"] for item in upcoming["rows"]] == ["/publishing/99"]
    assert [item["href"] for item in attention["rows"]] == []


def test_a_delivery_with_no_parsable_instant_is_not_called_overdue(tmp_path, monkeypatch):
    monkeypatch.setattr(
        overview.pubqueue,
        "queue_rows",
        lambda _config=None: [
            {
                "id": 50,
                "project_name": "proj",
                "short_index": 3,
                "platform": "youtube",
                "status": "queued",
                "scheduled_at": None,
                "sidecar_path": None,
            }
        ],
    )
    monkeypatch.setattr(
        overview.pubqueue,
        "enrich_rows_with_sidecar",
        lambda rows, cache=None: [{**row, "scheduled_at_effective_local": None} for row in rows],
    )

    attention = overview.needs_attention(_store(tmp_path), _empty_config(tmp_path), now=_now())

    assert [item["href"] for item in attention["rows"]] == []


def test_delivery_labels_carry_project_name_and_short_index(tmp_path, monkeypatch):
    monkeypatch.setattr(
        overview.pubqueue,
        "queue_rows",
        lambda _config=None: [
            {
                "id": 20,
                "project_name": "trip-42",
                "short_index": 3,
                "platform": "tiktok",
                "status": "queued",
                "scheduled_at": "2026-09-09T16:00:00+00:00",
                "sidecar_path": None,
            }
        ],
    )
    monkeypatch.setattr(
        overview.pubqueue,
        "enrich_rows_with_sidecar",
        lambda rows, cache=None: [
            {**row, "scheduled_at_effective_local": "2026-09-09 16:00 UTC"} for row in rows
        ],
    )

    attention = overview.needs_attention(_store(tmp_path), _empty_config(tmp_path), now=_now())

    attention_item = next(item for item in attention["rows"] if item["href"] == "/publishing/20")
    assert "trip-42#3" in attention_item["label"]


def test_index_page_shows_media_library_link_exactly_once():
    from fastapi.testclient import TestClient

    from gomoveshift.console.app import create_app

    response = TestClient(create_app()).get("/")

    assert response.status_code == 200
    assert response.text.count("Media library") == 1
