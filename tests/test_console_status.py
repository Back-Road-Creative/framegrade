"""The console's live status page: GET /status and GET /status.json.

Both are rendered from ``status.snapshot.build_snapshot`` -- the same reader behind
``gomoveshift status`` -- and never write or launch anything.
"""

import re

import pytest
from fastapi.testclient import TestClient

import gomoveshift.console.app as console_app
from gomoveshift.console.app import NAV_ENTRIES, create_app

NOW = "2026-11-13T15:00:00+00:00"


def _snapshot(**over):
    snap = {
        "schema": "gms-status/1",
        "as_of": NOW,
        "jobs": {
            "as_of": NOW,
            "source": "/jobs",
            "running": [
                {
                    "name": "drive-1",
                    "run_id": "r1",
                    "state": "running",
                    "drive": "D1",
                    "phase": "encode",
                    "heartbeat_at": NOW,
                    "created_at": NOW,
                }
            ],
            "recent": [
                {
                    "name": "bad-job",
                    "state": "failed",
                    "exit_code": 2,
                    "ended_at": NOW,
                    "error": "boom",
                },
                {"name": "ok-job", "state": "succeeded", "exit_code": 0, "ended_at": NOW},
            ],
            "counts": {"running": 1, "failed": 1, "succeeded": 1},
            "unreadable": 0,
        },
        "incidents": {
            "as_of": NOW,
            "source": "/inc",
            "open": [{"kind": "drive", "key": "D9", "error": "disk gone"}],
            "resolved": 3,
            "unreadable": 0,
        },
        "queue": {
            "as_of": NOW,
            "source": "/q",
            "counts": {"youtube": {"scheduled": 4, "published": 9}},
            "overdue_manual": [
                {
                    "id": 77,
                    "platform": "tiktok",
                    "project_name": "proj-a",
                    "short_index": 2,
                    "status": "awaiting_manual_publish",
                    "scheduled_at": NOW,
                }
            ],
            "next": [
                {
                    "id": 78,
                    "platform": "youtube",
                    "project_name": "proj-b",
                    "short_index": 1,
                    "status": "scheduled",
                    "scheduled_at": NOW,
                }
            ],
        },
        "runway": {
            "as_of": NOW,
            "source": "gomoveshift schedule --json",
            "queue_ends": "2026-12-01",
            "days_of_runway": 18,
            "weekly_longform_capacity": 3,
            "scheduled_longforms": [
                {"date": "2026-11-24", "time": "12:00", "region": "usa", "title": "t1"},
                {"date": "2026-12-01", "time": "12:00", "region": "ireland", "title": "t2"},
            ],
            "scheduled_shorts_count": 12,
            "scheduled_shorts_by_region": {"usa": 9, "ireland": 3},
            "next_available": [
                {
                    "region": "usa",
                    "day": "tuesday",
                    "slot": "2026-12-08 12:00",
                    "earliest_for_region": True,
                },
                {
                    "region": "usa",
                    "day": "friday",
                    "slot": "2026-12-11 12:00",
                    "earliest_for_region": False,
                },
            ],
        },
        "backlog": {
            "as_of": NOW,
            "source": "/b",
            "rows": 5,
            "by_region": {"SC": 3, "PA": 2},
            "projects": [],
        },
        "machine": {
            "as_of": NOW,
            "source": "this host",
            "volumes": [
                {
                    "label": "data",
                    "path": "/d",
                    "free_bytes": 2 * 1024**3,
                    "total_bytes": 4 * 1024**3,
                }
            ],
            "gpu": {"source": "/gpu", "held": False, "owner": None, "pid": None, "acquired": None},
            "cron": [],
        },
    }
    snap.update(over)
    return snap


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.delenv("GMS_CONSOLE_TOKEN", raising=False)
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(console_app, "_status_snapshot", lambda: _snapshot())
    return TestClient(create_app())


def _html(client):
    resp = client.get("/status")
    assert resp.status_code == 200
    return resp.text


def test_status_json_is_the_snapshot_with_six_sections(client):
    resp = client.get("/status.json")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()
    assert body["schema"] == "gms-status/1"
    for key in ("jobs", "incidents", "queue", "runway", "backlog", "machine"):
        assert key in body


def test_status_page_has_all_six_blocks_in_order(client):
    text = _html(client)
    headings = [
        "Needs you",
        "Rendering now",
        "Runway",
        "Publishing by platform",
        "Backlog",
        "Machine",
    ]
    positions = [text.index(f">{h}<") for h in headings]
    assert positions == sorted(positions)


def test_needs_you_lists_incidents_failed_jobs_and_overdue_with_links(client):
    text = _html(client)
    assert "disk gone" in text
    assert "bad-job" in text
    assert "ok-job" not in text.split(">Rendering now<")[0]
    assert 'href="/publishing/77"' in text
    assert 'href="/schedule"' in text
    assert 'href="/jobs"' in text


def test_runway_shows_days_bookings_shorts_and_next_free_slot(client):
    text = _html(client).split(">Runway<")[1].split(">Publishing by platform<")[0]
    assert "2026-12-01" in text and "18 days" in text
    assert "2 long-forms booked" in text and "3 a week" in text
    assert "12 Shorts booked" in text and "usa 9" in text and "ireland 3" in text
    assert "2026-12-08 12:00" in text and "2026-12-11 12:00" not in text


def test_rendering_now_shows_running_job_and_heartbeat(client):
    text = _html(client).split(">Rendering now<")[1].split(">Runway<")[0]
    assert "drive-1" in text and "encode" in text and "heartbeat" in text.lower()


def test_unavailable_section_says_why_and_never_looks_healthy(monkeypatch, tmp_path):
    monkeypatch.delenv("GMS_CONSOLE_TOKEN", raising=False)
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    snap = _snapshot(
        queue={"as_of": NOW, "source": "/q", "unavailable": "FileNotFoundError: no file /q"},
        backlog={"as_of": NOW, "source": "/b", "unavailable": "no backlog file"},
    )
    monkeypatch.setattr(console_app, "_status_snapshot", lambda: snap)
    text = TestClient(create_app()).get("/status").text
    assert "unavailable: FileNotFoundError: no file /q" in text
    assert "unavailable: no backlog file" in text
    pub = text.split(">Publishing by platform<")[1].split(">Backlog<")[0]
    assert "0 " not in pub.replace("unavailable", "")


def test_no_meta_refresh_and_no_form(client):
    text = _html(client).lower()
    assert "http-equiv" not in text
    assert "<form" not in text


def test_pause_control_refresh_link_and_in_place_script(client):
    text = _html(client)
    assert re.search(r"<button[^>]*>\s*Pause updates\s*</button>", text)
    assert '<a href="/status">Refresh</a>' in text
    assert "/status.json" in text and "setInterval" in text
    assert "stale since" in text


def test_waypost_is_the_first_nav_entry(client):
    assert NAV_ENTRIES[0] == ("status", "Waypost")
    nav = re.search(r'<nav class="console-nav".*?</nav>', _html(client), re.S).group(0)
    assert re.findall(r'href="/([a-z-]+)"', nav)[0] == "status"


def test_both_routes_require_the_token_when_set(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_CONSOLE_TOKEN", "s3cret")
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(console_app, "_status_snapshot", lambda: _snapshot())
    client = TestClient(create_app())
    assert client.get("/status").status_code == 401
    assert client.get("/status.json").status_code == 401
    auth = {"Authorization": "Bearer s3cret"}
    assert client.get("/status", headers=auth).status_code == 200
    assert client.get("/status.json", headers=auth).status_code == 200
