"""The console can move a delivery's slot.

`queue.reschedule_delivery` has existed and been tested since the queue layer
was built, with every rule already decided there -- a terminal row is final, a
live foreign lease may already be mid-publish, and moving a slot FORWARD past a
stale backoff clears it while a future backoff is left alone. It had no caller
at all: not the CLI, not the console. An operator who needed a delivery to go
out later had to cancel it and re-enqueue the whole plan.

Two things this pins beyond "the button works":

* **A slot the queue cannot compare is refused by the QUEUE**, not by the page.
  `scheduled_at` is a TEXT column that `dispatch._select_due` compares as a
  string, so storing "tomorrow" makes a row never due, or always due, silently.
  `transition` already refuses an unparseable `platform_published_at` for the
  same reason; this is that rule, for the other timestamp. A route-level check
  would leave the CLI and any future worker able to write the bad value.
* **A refusal comes back on the page with the typed value intact** -- the shape
  the other console forms already use. A 409 error page throws away the timestamp the
  operator just worked out.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from gomoveshift.console import pubqueue
from gomoveshift.console.app import create_app
from gomoveshift.publishing.queue import (
    NEXT_ATTEMPT_RETRY_BACKOFF,
    PUBLISHED_AT_OPERATOR_CONFIRMED,
    acquire_lease,
    enqueue,
    open_queue_db,
    reschedule_delivery,
    transition,
)

T_NOW = "2026-09-12T12:00:00+00:00"
T_LATER = "2026-09-13T09:00:00+00:00"
T_LATEST = "2026-09-14T09:00:00+00:00"


def _config(tmp_path):
    return SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))


def _point_at(tmp_path, monkeypatch):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))


def _queued(tmp_path) -> int:
    conn = open_queue_db(tmp_path)
    delivery_id = enqueue(conn, "proj", 1, "instagram", "h1", scheduled_at=T_NOW)
    conn.close()
    return delivery_id


def _scheduled_at(tmp_path, delivery_id: int):
    conn = open_queue_db(tmp_path)
    try:
        return conn.execute(
            "SELECT scheduled_at FROM deliveries WHERE id=?", (delivery_id,)
        ).fetchone()["scheduled_at"]
    finally:
        conn.close()


def test_the_queue_refuses_a_slot_it_could_never_compare(tmp_path):
    """The robust tier, and the reason it belongs here rather than in the
    route: `scheduled_at` is compared as a string by the drain's own SELECT, so
    an unparseable value does not fail loudly -- it makes the row never due, or
    due forever. No caller should be able to write one."""
    conn = open_queue_db(tmp_path)
    delivery_id = enqueue(conn, "proj", 1, "instagram", "h1")
    try:
        with pytest.raises(ValueError, match="scheduled_at"):
            reschedule_delivery(conn, delivery_id, "tomorrow afternoon")
    finally:
        conn.close()
    assert _scheduled_at(tmp_path, delivery_id) is None


def test_the_queue_still_accepts_a_real_timestamp(tmp_path):
    conn = open_queue_db(tmp_path)
    delivery_id = enqueue(conn, "proj", 1, "instagram", "h1", scheduled_at=T_NOW)
    try:
        assert reschedule_delivery(conn, delivery_id, T_LATER) is True
    finally:
        conn.close()
    assert _scheduled_at(tmp_path, delivery_id) == T_LATER


def test_the_detail_page_offers_the_control(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    delivery_id = _queued(tmp_path)

    body = TestClient(create_app()).get(f"/publishing/{delivery_id}").text

    assert f'action="/publishing/{delivery_id}/reschedule"' in body
    assert 'name="scheduled_at"' in body


def test_a_finished_delivery_is_offered_no_control(tmp_path, monkeypatch):
    """Same rule the cancel/retry buttons follow: a control that can only
    refuse is worse than no control."""
    _point_at(tmp_path, monkeypatch)
    delivery_id = _queued(tmp_path)
    conn = open_queue_db(tmp_path)
    transition(
        conn,
        delivery_id,
        "queued",
        "published",
        published_at_source=PUBLISHED_AT_OPERATOR_CONFIRMED,
    )
    conn.close()

    body = TestClient(create_app()).get(f"/publishing/{delivery_id}").text

    assert f'action="/publishing/{delivery_id}/reschedule"' not in body


def test_rescheduling_moves_the_slot_and_returns_to_the_row(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    delivery_id = _queued(tmp_path)

    response = TestClient(create_app()).post(
        f"/publishing/{delivery_id}/reschedule",
        data={"scheduled_at": T_LATER},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == f"/publishing/{delivery_id}"
    assert _scheduled_at(tmp_path, delivery_id) == T_LATER


def test_the_reschedule_route_is_not_swallowed_by_the_action_route(tmp_path, monkeypatch):
    """`POST /publishing/{id}/{action}` already existed and 404s an unknown
    action. Registered after it, this path would never be reached."""
    _point_at(tmp_path, monkeypatch)
    delivery_id = _queued(tmp_path)

    response = TestClient(create_app()).post(
        f"/publishing/{delivery_id}/reschedule", data={"scheduled_at": T_LATER}
    )

    assert response.status_code == 200
    assert "Unknown action" not in response.text


def test_a_slot_the_queue_cannot_read_comes_back_on_the_page(tmp_path, monkeypatch):
    """The refusal is a sentence on this page, and the value the
    operator typed is still in the field."""
    _point_at(tmp_path, monkeypatch)
    delivery_id = _queued(tmp_path)

    response = TestClient(create_app()).post(
        f"/publishing/{delivery_id}/reschedule", data={"scheduled_at": "tomorrow afternoon"}
    )

    assert response.status_code == 400
    assert "tomorrow afternoon" in response.text
    assert _scheduled_at(tmp_path, delivery_id) == T_NOW


def test_an_empty_slot_is_refused_with_a_sentence(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    delivery_id = _queued(tmp_path)

    response = TestClient(create_app()).post(
        f"/publishing/{delivery_id}/reschedule", data={"scheduled_at": ""}
    )

    assert response.status_code == 400
    assert _scheduled_at(tmp_path, delivery_id) == T_NOW


def test_a_live_lease_refusal_names_the_worker_holding_it(tmp_path, monkeypatch):
    """The queue signals both refusals with a bare `False`, so the reason is
    read back off the row -- an operator told only "refused" would retry."""
    _point_at(tmp_path, monkeypatch)
    delivery_id = _queued(tmp_path)
    conn = open_queue_db(tmp_path)
    acquire_lease(conn, delivery_id, "worker-1", "2099-01-01T00:00:00+00:00", now=T_NOW)
    conn.close()

    response = TestClient(create_app()).post(
        f"/publishing/{delivery_id}/reschedule", data={"scheduled_at": T_LATER}
    )

    assert response.status_code == 409
    assert "worker-1" in response.text
    assert _scheduled_at(tmp_path, delivery_id) == T_NOW


def test_a_finished_delivery_posted_directly_is_refused(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    delivery_id = _queued(tmp_path)
    conn = open_queue_db(tmp_path)
    transition(conn, delivery_id, "queued", "cancelled")
    conn.close()

    response = TestClient(create_app()).post(
        f"/publishing/{delivery_id}/reschedule", data={"scheduled_at": T_LATER}
    )

    assert response.status_code == 409
    assert "cancelled" in response.text


def test_an_unknown_delivery_is_a_404(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    _queued(tmp_path)

    response = TestClient(create_app()).post(
        "/publishing/9999/reschedule", data={"scheduled_at": T_LATER}
    )

    assert response.status_code == 404


def test_moving_a_slot_forward_clears_a_stale_backoff_through_the_console(tmp_path, monkeypatch):
    """The queue's own rule, asserted through the console path so the surface
    cannot quietly bypass it: a row waiting on an already-past retry backoff
    must not become due the instant its new schedule arrives."""
    _point_at(tmp_path, monkeypatch)
    delivery_id = _queued(tmp_path)
    conn = open_queue_db(tmp_path)
    transition(
        conn,
        delivery_id,
        "queued",
        "queued",
        next_attempt_at=T_NOW,
        next_attempt_reason=NEXT_ATTEMPT_RETRY_BACKOFF,
    )
    conn.close()

    TestClient(create_app()).post(
        f"/publishing/{delivery_id}/reschedule", data={"scheduled_at": T_LATEST}
    )

    conn = open_queue_db(tmp_path)
    row = conn.execute(
        "SELECT scheduled_at, next_attempt_at FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    conn.close()
    assert row["scheduled_at"] == T_LATEST
    assert row["next_attempt_at"] is None
