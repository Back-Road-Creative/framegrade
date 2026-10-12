"""A delivery the platform has not published yet does not
read as published.

`PUBLISHED_AT_PLATFORM_SCHEDULED` landed in the queue and Facebook
started writing it: the delivery is ACCEPTED now and goes live at a
recorded later instant, `platform_published_at`. The console never learned the
word. `_PUBLISH_CONFIRMATION_LABELS` had no entry for it, so
`.get(..., "unknown (legacy)")` labelled the strongest, most precise evidence
this queue can hold as the weakest -- and the row sat in the `published` group
with nothing on the page saying the platform had not run it yet.

Both halves are the same mistake: the labels exist so a delayed confirmation
never reads as an immediate one, and a scheduled one is neither.
"""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from gomoveshift.console import pubqueue
from gomoveshift.console.app import create_app
from gomoveshift.publishing.queue import PUBLISHED_AT_PLATFORM_SCHEDULED, open_queue_db

#: When the queue handed the delivery to Facebook, and when Facebook publishes
#: it. They are different facts about different moments; the page used to show
#: only the first.
HANDED_OVER = "2026-09-05T19:00:00+00:00"
GOES_LIVE = "2026-09-06T13:30:00+00:00"


def _client() -> TestClient:
    return TestClient(create_app())


def _config(tmp_path):
    return SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))


def _insert(conn, **over):
    from gomoveshift.publishing.queue import _now_iso

    fields = {
        "project_name": "proj",
        "short_index": 1,
        "platform": "facebook",
        "plan_hash": "h1",
        "status": "published",
        "shadow": 0,
        "published_at": HANDED_OVER,
        "published_at_source": PUBLISHED_AT_PLATFORM_SCHEDULED,
        "platform_published_at": GOES_LIVE,
        **over,
    }
    now = _now_iso()
    cols = [*fields, "created_at", "updated_at"]
    vals = [*fields.values(), now, now]
    with conn:
        cursor = conn.execute(
            f"INSERT INTO deliveries ({','.join(cols)}) VALUES ({','.join('?' * len(vals))})",  # noqa: S608
            vals,
        )
        delivery_id = cursor.lastrowid
        cursor.close()
    return delivery_id


def _page(tmp_path, monkeypatch, path, **over):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    conn = open_queue_db(tmp_path)
    delivery_id = _insert(conn, **over)
    conn.close()
    return _client().get(path.format(id=delivery_id))


def test_a_scheduled_publication_has_its_own_label():
    grouped = dict(
        pubqueue.rows_by_state(
            [
                {
                    "status": "published",
                    "id": 1,
                    "published_at_source": PUBLISHED_AT_PLATFORM_SCHEDULED,
                }
            ]
        )
    )

    label = grouped["published"][0]["publish_confirmation"]
    assert label == "platform scheduled (publishes later)"
    # The point of the label set is that one source never wears another's
    # words. "unknown (legacy)" was the default this fell through to.
    assert "unknown" not in label


def test_the_existing_labels_are_unchanged():
    rows = [
        {"status": "published", "id": 1, "published_at_source": "operator_confirmed"},
        {"status": "published", "id": 2, "published_at_source": "platform_reconciled"},
        {"status": "published", "id": 3, "published_at_source": None},
        {"status": "published", "id": 4, "published_at_source": "platform_confirmed"},
    ]

    grouped = dict(pubqueue.rows_by_state(rows))

    assert {r["id"]: r["publish_confirmation"] for r in grouped["published"]} == {
        1: "operator confirmed",
        2: "platform reconciled (delayed)",
        3: "unknown (legacy)",
        4: "platform confirmed",
    }


def test_the_detail_page_names_both_moments(tmp_path, monkeypatch):
    response = _page(tmp_path, monkeypatch, "/publishing/{id}")

    assert response.status_code == 200
    assert "platform scheduled (publishes later)" in response.text
    # Handed over at one instant, live at another. Showing only the handover
    # invited reading it as the publication time.
    assert HANDED_OVER in response.text
    assert GOES_LIVE in response.text


def test_a_delivery_published_on_the_call_shows_one_moment(tmp_path, monkeypatch):
    response = _page(
        tmp_path,
        monkeypatch,
        "/publishing/{id}",
        published_at_source="platform_confirmed",
        platform_published_at=None,
    )

    assert "platform confirmed" in response.text
    assert GOES_LIVE not in response.text
    # No empty "goes live" line for a delivery that has no second moment.
    assert "the platform publishes it" not in response.text.lower()


def test_the_queue_list_says_a_scheduled_row_is_not_live_yet(tmp_path, monkeypatch):
    response = _page(tmp_path, monkeypatch, "/publishing")

    assert response.status_code == 200
    # The row sits in the `published` group because the queue is finished with
    # it. The list has to say the platform is not.
    assert GOES_LIVE in response.text


def test_the_queue_list_leaves_an_immediate_publication_alone(tmp_path, monkeypatch):
    response = _page(
        tmp_path,
        monkeypatch,
        "/publishing",
        published_at_source="platform_confirmed",
        platform_published_at=None,
    )

    assert response.status_code == 200
    assert GOES_LIVE not in response.text
