"""C1: read-only cross-platform reconciliation of the delivery queue.

Every fixture is synthetic. The view keeps queued / uploaded / remotely scheduled / published /
awaiting-manual / unverified apart, shows per-platform-and-format coverage gaps up to a horizon,
and separates recovered failures from unresolved ones. It never writes and never republishes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import re
import pytest
from fastapi.testclient import TestClient

from gomoveshift.console import reconcile
from gomoveshift.console.app import create_app
from gomoveshift.publishing.queue import enqueue, open_queue_db

NOW = datetime(2026, 10, 17, 16, 0, tzinfo=timezone.utc)  # noon ET


def _row(id_, platform="youtube", status="queued", fmt="short", day=None, **extra):
    row = {
        "id": id_,
        "project_name": f"proj{id_}",
        "short_index": 1,
        "platform": platform,
        "status": status,
        "format": fmt,
        "scheduled_at": f"2026-10-{day:02d}T16:00:00+00:00" if day else None,
        "published_at_source": None,
        "recovers_delivery_id": None,
        "shadow": 0,
        "permalink": None,
    }
    row.update(extra)
    return row


def _buckets(report):
    return {name: [r["id"] for r in rows] for name, rows in report["states"].items()}


def test_states_are_kept_distinct():
    rows = [
        _row(1, status="queued"),
        _row(2, status="uploaded"),
        _row(3, status="published", published_at_source="platform_scheduled"),
        _row(4, status="published", published_at_source="platform_confirmed"),
        _row(5, status="awaiting_manual_publish", platform="tiktok"),
        _row(6, status="published", published_at_source="operator_confirmed"),
        _row(7, status="awaiting_ig_reconcile", platform="instagram"),
        _row(8, status="missing_remote"),
        _row(9, status="published", published_at_source=None),
    ]
    got = _buckets(reconcile.build(rows, NOW))
    assert got["queued"] == [1]
    assert got["uploaded"] == [2]
    assert got["remotely_scheduled"] == [3]
    assert got["published"] == [4]
    assert got["awaiting_manual"] == [5]
    assert sorted(got["unverified"]) == [6, 7, 8, 9]


def test_shadow_rows_are_not_counted_anywhere():
    report = reconcile.build([_row(1, shadow=1)], NOW)
    assert all(not rows for rows in report["states"].values())
    assert report["shadow_ignored"] == 1


def test_failures_split_into_recovered_in_flight_and_unresolved():
    rows = [
        _row(10, status="failed_permanent"),
        _row(
            11,
            status="published",
            published_at_source="platform_confirmed",
            recovers_delivery_id=10,
        ),
        _row(20, status="failed_permanent"),
        _row(21, status="queued", recovers_delivery_id=20),
        _row(30, status="failed_permanent"),
    ]
    failures = reconcile.build(rows, NOW)["failures"]
    assert [f["id"] for f in failures["recovered"]] == [10]
    assert failures["recovered"][0]["recovered_by"] == 11
    assert [f["id"] for f in failures["recovery_in_flight"]] == [20]
    assert [f["id"] for f in failures["unresolved"]] == [30]


def test_a_failed_instagram_row_reports_a_prior_publication_without_acting():
    rows = [
        _row(26, platform="instagram", status="failed_permanent", project_name="p", short_index=3),
        _row(
            5,
            platform="instagram",
            status="published",
            project_name="p",
            short_index=3,
            published_at_source="platform_confirmed",
            permalink="https://example.test/p/5",
        ),
        _row(77, platform="instagram", status="failed_permanent", project_name="q", short_index=1),
    ]
    by_id = {f["id"]: f for f in reconcile.build(rows, NOW)["failures"]["unresolved"]}
    assert by_id[26]["prior_publication"] == {"id": 5, "permalink": "https://example.test/p/5"}
    assert by_id[77]["prior_publication"] is None
    assert by_id[26]["action"] == "report only"


_NON_PUBLISHED_SIBLINGS = {
    "queued": {"status": "queued"},
    "uploaded": {"status": "uploaded"},
    "platform_scheduled": {"status": "published", "published_at_source": "platform_scheduled"},
    "awaiting_manual": {"status": "awaiting_manual_publish"},
    "unverified_operator": {"status": "published", "published_at_source": "operator_confirmed"},
    "unverified_ig_reconcile": {"status": "awaiting_ig_reconcile"},
    "unverified_missing_remote": {"status": "missing_remote"},
}


@pytest.mark.parametrize("sibling", list(_NON_PUBLISHED_SIBLINGS))
def test_only_a_published_sibling_counts_as_a_prior_publication(sibling):
    rows = [
        _row(26, platform="instagram", status="failed_permanent", project_name="p", short_index=3),
        _row(
            5,
            platform="instagram",
            project_name="p",
            short_index=3,
            **_NON_PUBLISHED_SIBLINGS[sibling],
        ),
    ]
    (failure,) = reconcile.build(rows, NOW)["failures"]["unresolved"]
    assert failure["prior_publication"] is None


@pytest.mark.parametrize(
    "sibling",
    [
        {"project_name": "other", "short_index": 3, "platform": "instagram"},
        {"project_name": "p", "short_index": 2, "platform": "instagram"},
        {"project_name": "p", "short_index": 3, "platform": "youtube"},
    ],
    ids=["other_project", "other_short", "other_platform"],
)
def test_a_published_row_for_another_project_short_or_platform_is_not_a_prior_publication(sibling):
    rows = [
        _row(26, platform="instagram", status="failed_permanent", project_name="p", short_index=3),
        _row(5, status="published", published_at_source="platform_confirmed", **sibling),
    ]
    (failure,) = reconcile.build(rows, NOW)["failures"]["unresolved"]
    assert failure["prior_publication"] is None


def test_gaps_per_platform_and_format_up_to_the_horizon():
    # NOW is 2026-10-17 ET. Horizon 3 days covers the 17th, 18th and 19th.
    rows = [
        _row(1, platform="youtube", fmt="short", day=17, status="queued"),
        _row(
            2,
            platform="youtube",
            fmt="short",
            day=19,
            status="published",
            published_at_source="platform_confirmed",
        ),
        _row(3, platform="instagram", fmt="short", day=18, status="failed_permanent"),
        _row(4, platform="youtube", fmt="long-form", day=18, status="queued"),
        _row(5, platform="youtube", fmt="short", day=25, status="queued"),  # past horizon
    ]
    gaps = reconcile.build(rows, NOW, horizon_days=3)["gaps"]
    assert gaps[("youtube", "short")] == ["2026-10-18"]
    # a failed row covers nothing; nothing is scheduled for instagram on any day
    assert gaps[("instagram", "short")] == ["2026-10-17", "2026-10-18", "2026-10-19"]
    assert gaps[("youtube", "long-form")] == ["2026-10-17", "2026-10-19"]


def test_a_remotely_scheduled_row_covers_its_day_but_is_not_a_publication():
    rows = [_row(1, status="published", published_at_source="platform_scheduled", day=18)]
    report = reconcile.build(rows, NOW, horizon_days=3)
    assert "2026-10-18" not in report["gaps"][("youtube", "short")]
    assert [r["id"] for r in report["states"]["published"]] == []


def test_the_page_renders_from_a_real_queue_and_writes_nothing(tmp_path, monkeypatch):
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "youtube", "h1", scheduled_at="2026-10-18T16:00:00+00:00")
    conn.close()
    before = sorted(p.name for p in tmp_path.iterdir())
    cfg = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
    monkeypatch.setattr(reconcile.pubqueue, "_config", lambda config=None: cfg)
    html = TestClient(create_app()).get("/publishing/reconcile").text
    for heading in (
        "Queued",
        "Remotely scheduled",
        "Awaiting manual",
        "Unverified",
        "Coverage gaps",
    ):
        assert heading in html
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_every_reconcile_table_stacks_on_a_phone_with_labelled_cells(tmp_path, monkeypatch):
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "youtube", "h1", scheduled_at="2026-10-18T16:00:00+00:00")
    failed = enqueue(conn, "proj", 2, "instagram", "h2", scheduled_at="2026-10-18T17:00:00+00:00")
    conn.execute(
        "UPDATE deliveries SET status='failed_permanent', error_class='boom' WHERE id=?", (failed,)
    )
    conn.commit()
    conn.close()
    cfg = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
    monkeypatch.setattr(reconcile.pubqueue, "_config", lambda config=None: cfg)
    html = TestClient(create_app()).get("/publishing/reconcile").text
    tables = re.findall(r"<table border[^>]*>", html)
    assert len(tables) >= 3  # queued state, coverage gaps, failures
    assert all('class="delivery-table"' in t for t in tables)
    cells = re.findall(r"<td[^>]*>", "".join(re.findall(r"<table border.*?</table>", html, re.S)))
    assert cells and all("data-label=" in c for c in cells)
