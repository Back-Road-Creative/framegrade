"""Calendar reading of the publishing queue (``console/pubqueue_calendar.py``).

Same queue rows as the list view (``pubqueue.rows_by_state``), bucketed by day
instead of grouped by status. Never invents a schedule: a row the list view
would show as "no schedule recorded" lands in the "not scheduled" group here,
never on a day -- see ``_bucket_key``.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from fastapi.testclient import TestClient

from gomoveshift.console import pubqueue, pubqueue_calendar
from gomoveshift.console.app import create_app
from gomoveshift.publishing.queue import enqueue, open_queue_db


def _client() -> TestClient:
    return TestClient(create_app())


def _config(tmp_path):
    return SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))


def test_week_calendar_places_a_scheduled_row_on_its_day(monkeypatch):
    monkeypatch.setattr(
        pubqueue,
        "queue_rows",
        lambda config=None: [
            {
                "project_name": "proj",
                "short_index": 1,
                "platform": "youtube",
                "status": "queued",
                "scheduled_at": "2026-09-08T19:00:00+00:00",
            }
        ],
    )

    result = pubqueue_calendar.week_calendar(anchor=date(2026, 9, 8))

    day = next(d for d in result["days"] if d["date"] == date(2026, 9, 8))
    assert [r["project_name"] for r in day["rows"]] == ["proj"]
    assert result["unscheduled"] == []


def test_week_calendar_never_places_an_unscheduled_row_on_a_day(monkeypatch):
    monkeypatch.setattr(
        pubqueue,
        "queue_rows",
        lambda config=None: [
            {
                "project_name": "proj",
                "short_index": 1,
                "platform": "youtube",
                "status": "queued",
                "scheduled_at": None,
            }
        ],
    )

    result = pubqueue_calendar.week_calendar(anchor=date(2026, 9, 8))

    assert all(d["rows"] == [] for d in result["days"])
    assert [r["project_name"] for r in result["unscheduled"]] == ["proj"]


def test_week_calendar_includes_every_day_of_the_week_even_when_empty(monkeypatch):
    monkeypatch.setattr(pubqueue, "queue_rows", lambda config=None: [])

    result = pubqueue_calendar.week_calendar(anchor=date(2026, 9, 10))

    assert len(result["days"]) == 7
    assert result["week_start"] == date(2026, 9, 7)  # Monday
    assert result["week_end"] == date(2026, 9, 13)  # Sunday


def test_week_calendar_orders_rows_within_a_day_by_scheduled_instant(monkeypatch):
    monkeypatch.setattr(
        pubqueue,
        "queue_rows",
        lambda config=None: [
            {
                "project_name": "later",
                "short_index": 1,
                "platform": "youtube",
                "status": "queued",
                "scheduled_at": "2026-09-08T19:00:00+00:00",
                "scheduled_at_effective_local": "2026-09-08 19:00 UTC",
            },
            {
                "project_name": "earlier",
                "short_index": 1,
                "platform": "youtube",
                "status": "published",
                "scheduled_at": "2026-09-08T08:00:00+00:00",
                "scheduled_at_effective_local": "2026-09-08 08:00 UTC",
            },
        ],
    )

    result = pubqueue_calendar.week_calendar(anchor=date(2026, 9, 8))

    day = next(d for d in result["days"] if d["date"] == date(2026, 9, 8))
    assert [r["project_name"] for r in day["rows"]] == ["earlier", "later"]


def test_week_calendar_gives_each_row_a_time_label(monkeypatch):
    # Goes through ``rows_by_state`` directly (bypassing sidecar enrichment)
    # so a malformed ``scheduled_at_effective_local`` can be exercised even
    # though ``enrich_rows_with_sidecar`` never produces one in practice.
    monkeypatch.setattr(
        pubqueue,
        "rows_by_state",
        lambda rows: [
            (
                "queued",
                [
                    {
                        "project_name": "proj",
                        "short_index": 1,
                        "platform": "youtube",
                        "status": "queued",
                        "scheduled_at": "2026-09-08T19:00:00+00:00",
                        "scheduled_at_effective_local": "2026-09-08 19:00 UTC",
                    },
                    {
                        "project_name": "notime",
                        "short_index": 2,
                        "platform": "youtube",
                        "status": "queued",
                        "scheduled_at": "2026-09-08T20:00:00+00:00",
                        "scheduled_at_effective_local": "2026-09-08",
                    },
                ],
            )
        ],
    )
    monkeypatch.setattr(pubqueue, "queue_rows", lambda config=None: [])

    result = pubqueue_calendar.week_calendar(anchor=date(2026, 9, 8))

    day = next(d for d in result["days"] if d["date"] == date(2026, 9, 8))
    by_project = {r["project_name"]: r["time_label"] for r in day["rows"]}
    assert by_project["proj"] == "19:00"
    assert by_project["notime"] == ""


def test_week_calendar_groups_rows_by_project_within_a_day(monkeypatch):
    monkeypatch.setattr(
        pubqueue,
        "queue_rows",
        lambda config=None: [
            {
                "project_name": "proj",
                "short_index": 3,
                "platform": "youtube",
                "status": "published",
                "scheduled_at": "2026-09-08T12:00:00+00:00",
                "scheduled_at_effective_local": "2026-09-08 12:00 UTC",
            },
            {
                "project_name": "other",
                "short_index": 1,
                "platform": "tiktok",
                "status": "queued",
                "scheduled_at": "2026-09-08T09:00:00+00:00",
                "scheduled_at_effective_local": "2026-09-08 09:00 UTC",
            },
            {
                "project_name": "proj",
                "short_index": 1,
                "platform": "youtube",
                "status": "queued",
                "scheduled_at": "2026-09-08T08:00:00+00:00",
                "scheduled_at_effective_local": "2026-09-08 08:00 UTC",
            },
            {
                "project_name": "proj",
                "short_index": 2,
                "platform": "youtube",
                "status": "awaiting_manual_publish",
                "scheduled_at": "2026-09-08T10:00:00+00:00",
                "scheduled_at_effective_local": "2026-09-08 10:00 UTC",
            },
        ],
    )

    result = pubqueue_calendar.week_calendar(anchor=date(2026, 9, 8))

    day = next(d for d in result["days"] if d["date"] == date(2026, 9, 8))
    assert [g["project_name"] for g in day["groups"]] == ["proj", "other"]
    proj_group = day["groups"][0]
    assert [r["short_index"] for r in proj_group["rows"]] == [1, 2, 3]


def test_publishing_calendar_route_shows_project_name_once_per_day(tmp_path, monkeypatch):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    conn = open_queue_db(tmp_path)
    enqueue(conn, "zzzuniqueproj", 1, "youtube", "h1")
    enqueue(conn, "zzzuniqueproj", 2, "youtube", "h2")
    enqueue(conn, "zzzuniqueproj", 3, "youtube", "h3")
    conn.close()

    def _fake_queue_rows(config=None):
        base = "2026-09-08T{h:02d}:00:00+00:00"
        return [
            {
                "project_name": "zzzuniqueproj",
                "short_index": i,
                "platform": "youtube",
                "status": "queued",
                "scheduled_at": base.format(h=8 + i),
            }
            for i in (1, 2, 3)
        ]

    monkeypatch.setattr(pubqueue, "queue_rows", _fake_queue_rows)

    # The rows are fixed on 2026-09-08, so the request pins that week: with no
    # anchor the route shows TODAY's week and the rows would fall off the grid.
    response = _client().get("/publishing/calendar?week=2026-09-08")

    assert response.status_code == 200
    assert response.text.count("zzzuniqueproj") == 1


def test_publishing_calendar_route_renders_week_grid(tmp_path, monkeypatch):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "youtube", "h1")
    conn.close()

    response = _client().get("/publishing/calendar")

    assert response.status_code == 200
    assert "not scheduled" in response.text.lower()


def test_publishing_list_links_to_the_calendar():
    response = _client().get("/publishing")

    assert response.status_code == 200
    assert "/publishing/calendar" in response.text


def test_publishing_calendar_shows_title_and_format_for_a_scheduled_row(tmp_path, monkeypatch):
    from test_console_pubqueue import _short_record, _write_sidecar

    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    conn = open_queue_db(tmp_path)
    sidecar_path = _write_sidecar(tmp_path, _short_record(format="short"))
    from gomoveshift.publishing.queue import _now_iso

    with conn:
        conn.execute(
            "INSERT INTO deliveries "
            "(project_name, short_index, platform, plan_hash, status, sidecar_path, "
            "scheduled_at, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "proj",
                1,
                "youtube",
                "h1",
                "queued",
                str(sidecar_path),
                "2026-09-08T19:00:00+00:00",
                _now_iso(),
                _now_iso(),
            ),
        )
    conn.close()

    response = _client().get("/publishing/calendar?week=2026-09-08")

    assert response.status_code == 200
    assert "The Title" in response.text
    assert "short" in response.text


def test_publishing_calendar_distinguishes_the_three_no_title_states(tmp_path, monkeypatch):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    conn = open_queue_db(tmp_path)
    from gomoveshift.publishing.queue import _now_iso

    bad_sidecar = tmp_path / "bad.json"
    bad_sidecar.write_text("not json", encoding="utf-8")

    with conn:
        # not_recorded: no sidecar_path at all.
        conn.execute(
            "INSERT INTO deliveries "
            "(project_name, short_index, platform, plan_hash, status, "
            "scheduled_at, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                "norecord",
                1,
                "youtube",
                "h1",
                "queued",
                "2026-09-08T08:00:00+00:00",
                _now_iso(),
                _now_iso(),
            ),
        )
        # unreadable: sidecar_path points at a malformed file.
        conn.execute(
            "INSERT INTO deliveries "
            "(project_name, short_index, platform, plan_hash, status, sidecar_path, "
            "scheduled_at, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "unreadable",
                1,
                "youtube",
                "h2",
                "queued",
                str(bad_sidecar),
                "2026-09-08T09:00:00+00:00",
                _now_iso(),
                _now_iso(),
            ),
        )
    conn.close()

    response = _client().get("/publishing/calendar?week=2026-09-08")

    assert response.status_code == 200
    assert "no title recorded" in response.text
    assert "could not read sidecar" in response.text


def test_publishing_calendar_truncates_a_long_title_and_shows_it_full_via_title_attribute(
    tmp_path, monkeypatch
):
    from test_console_pubqueue import _short_record, _write_sidecar
    from gomoveshift.publishing.models import PlatformPayloads, YouTubePayload

    long_title = "x" * 200
    sidecar_path = _write_sidecar(
        tmp_path,
        _short_record(payloads=PlatformPayloads(youtube=YouTubePayload(long_title, "desc"))),
    )
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    conn = open_queue_db(tmp_path)
    from gomoveshift.publishing.queue import _now_iso

    with conn:
        conn.execute(
            "INSERT INTO deliveries "
            "(project_name, short_index, platform, plan_hash, status, sidecar_path, "
            "scheduled_at, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "proj",
                1,
                "youtube",
                "h1",
                "queued",
                str(sidecar_path),
                "2026-09-08T19:00:00+00:00",
                _now_iso(),
                _now_iso(),
            ),
        )
    conn.close()

    response = _client().get("/publishing/calendar?week=2026-09-08")

    assert response.status_code == 200
    # The full title is expected in the `title=` attribute (constraint 4: the
    # full text stays reachable) -- what must NOT appear is the untruncated
    # text as visible cell content.
    assert f">{long_title}<" not in response.text
    assert f'title="{long_title}"' in response.text
    assert "…" in response.text


def test_publishing_calendar_says_when_nothing_is_scheduled_this_week(tmp_path, monkeypatch):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))

    response = _client().get("/publishing/calendar?week=2026-09-08")

    assert response.status_code == 200
    assert "nothing is scheduled this week" in response.text.lower()


def test_format_label_separates_unreadable_from_not_stated():
    """``format`` is ``None`` both when a sidecar stated no format and when the
    sidecar could not be read at all — the same collapse the Title column
    already refuses. The Format column must refuse it too, or an unreadable
    sidecar reads as a deliberate "not stated"."""
    unreadable = {"format": None, "sidecar_read_error": True}
    not_stated = {"format": None, "sidecar_read_error": False}
    stated = {"format": "carousel", "sidecar_read_error": False}

    assert pubqueue_calendar._format_label(unreadable) == "could not read sidecar"
    assert pubqueue_calendar._format_label(not_stated) == "not stated"
    assert pubqueue_calendar._format_label(stated) == "carousel"
    assert pubqueue_calendar._format_label(unreadable) != pubqueue_calendar._format_label(
        not_stated
    )


def test_format_label_never_reads_a_missing_format_as_short():
    """``PublishRecord.format`` is ``Optional[str]`` and ``models.py`` is
    explicit that ``None`` must never be read as FORMAT_SHORT by any consumer.
    The calendar is a consumer."""
    label = pubqueue_calendar._format_label({"format": None, "sidecar_read_error": False})

    assert "short" not in label.lower()
