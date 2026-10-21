"""U-R1: a YouTube short the channel no longer has becomes ``missing_remote``.

``missing_remote`` is TERMINAL: nothing re-uploads, deletes or rewrites the
remote video in this wave. Two writers move a row there -- the adapter's
post-upload ``videos.list`` check and ``gomoveshift queue mark-missing-remote``
driven by ``catalog-fetch`` output -- and ``publishing-status --validate``
names every such row. Synthetic ids only."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import test_publishing_youtube_adapter as _yt
from test_publishing_youtube_adapter import _publish, _setup

from gomoveshift.catalog.fetch import fetch_catalog
from gomoveshift.cli.handlers.pub_status import validate_restored_state
from gomoveshift.cli.handlers.queue_recover import cmd_queue, missing_remote_from_catalog
from gomoveshift.cli.parser import _build_parser
from gomoveshift.publishing import youtube_adapter
from gomoveshift.publishing.dispatch import drain_queue
from gomoveshift.publishing.queue import (
    STATUS_MISSING_REMOTE,
    TERMINAL_STATUSES,
    enqueue,
    find_delivery,
    mark_missing_remote,
    open_queue_db,
    transition,
)


@pytest.fixture(autouse=True)
def _publication_gate_open(monkeypatch):
    # The publication gate holds every insert until the final text is bound;
    # these upload tests predate the gate, so they open it in-test only.
    monkeypatch.setattr(
        "gomoveshift.pipeline.drive_phases.youtube_uploader.FINAL_TEXT_BIND_ON", True
    )


NOW = "2026-10-15T12:00:00+00:00"
GONE = "SYNTHGONE01"
LIVE = "SYNTHLIVE01"

# The adapter suite's fixtures, shared rather than copied.
mock_service = _yt.mock_service
queue_conn = _yt.queue_conn


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def _published(conn, short_index, video_id, platform="youtube"):
    delivery_id = enqueue(conn, "proj", short_index, platform, f"hash-{short_index}")
    transition(
        conn,
        delivery_id,
        "queued",
        "published",
        platform_upload_id=video_id,
        published_at_source="platform_confirmed",
    )
    return delivery_id


def _row(conn, delivery_id):
    return conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone()


def _audit(conn):
    return conn.execute(
        "SELECT action, reason FROM response_log WHERE action='mark_missing_remote'"
    ).fetchall()


# --- the status ---------------------------------------------------------


def test_missing_remote_is_terminal_and_never_leaves(conn):
    assert STATUS_MISSING_REMOTE == "missing_remote"
    assert STATUS_MISSING_REMOTE in TERMINAL_STATUSES
    delivery_id = _published(conn, 0, GONE)

    assert mark_missing_remote(conn, delivery_id, evidence={"source": "test"}, now=NOW)

    row = _row(conn, delivery_id)
    assert row["status"] == "missing_remote"
    assert row["platform_upload_id"] == GONE  # the receipt is kept, never cleared
    assert row["error_class"] == "MissingRemote"
    assert GONE in row["error_detail"]
    assert not transition(conn, delivery_id, "missing_remote", "queued")
    [(_action, reason)] = _audit(conn)
    assert json.loads(reason)["delivery_id"] == delivery_id


def test_only_a_youtube_row_with_an_upload_id_can_be_marked(conn):
    queued = enqueue(conn, "proj", 1, "youtube", "hash-q")
    instagram = _published(conn, 2, "IGSYNTH", platform="instagram")

    assert not mark_missing_remote(conn, queued, evidence={}, now=NOW)
    assert not mark_missing_remote(conn, instagram, evidence={}, now=NOW)
    assert _row(conn, queued)["status"] == "queued"
    assert _row(conn, instagram)["status"] == "published"
    assert _audit(conn) == []


def test_a_missing_remote_short_is_never_re_enqueued_under_a_new_plan(conn):
    """No re-upload in this wave: a re-plan with a new hash must not slip a
    fresh ``queued`` row in beside the terminal one."""
    delivery_id = _published(conn, 0, GONE)
    mark_missing_remote(conn, delivery_id, evidence={}, now=NOW)

    assert enqueue(conn, "proj", 0, "youtube", "a-new-plan-hash") == delivery_id
    assert conn.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0] == 1


# --- post-upload verify in the adapter -----------------------------------


@pytest.fixture
def no_wait(monkeypatch):
    waits = []
    monkeypatch.setattr(youtube_adapter, "_sleep", waits.append)
    return waits


def test_a_fresh_upload_the_channel_does_not_return_is_missing_remote_not_published(
    mock_service, queue_conn, tmp_path, no_wait
):
    mock_service.videos.return_value.list.return_value.execute.return_value = {"items": []}
    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)

    result = _publish(adapter, record, queue_conn, delivery_id, video_path)

    row = find_delivery(queue_conn, record.project_name, record.short_index, "youtube")
    assert row["status"] == "missing_remote"
    assert row["platform_upload_id"] == "SHORTVID123"
    assert result.missing_remote is True
    assert mock_service.videos.return_value.insert.call_count == 1  # never re-uploaded
    mock_service.playlistItems.return_value.insert.assert_not_called()
    assert len(no_wait) == 1  # one re-check before deciding, never a retry loop


def test_a_second_look_that_finds_the_video_publishes_it(
    mock_service, queue_conn, tmp_path, no_wait
):
    mock_service.videos.return_value.list.return_value.execute.side_effect = [
        {"items": []},
        {"items": [{"id": "SHORTVID123"}]},
    ]
    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)

    result = _publish(adapter, record, queue_conn, delivery_id, video_path)

    row = find_delivery(queue_conn, record.project_name, record.short_index, "youtube")
    assert row["status"] == "published"
    assert result.missing_remote is False


def test_a_verify_call_that_errors_still_publishes(mock_service, queue_conn, tmp_path, no_wait):
    """Unknown is not missing: a failed lookup proves nothing, and the
    nightly catalog pass checks the row again."""
    mock_service.videos.return_value.list.return_value.execute.side_effect = RuntimeError("503")
    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)

    _publish(adapter, record, queue_conn, delivery_id, video_path)

    row = find_delivery(queue_conn, record.project_name, record.short_index, "youtube")
    assert row["status"] == "published"


def test_the_drain_never_counts_a_missing_remote_row_as_published(conn):
    delivery_id = enqueue(conn, "proj", 0, "youtube", "hash-a")

    def upload_then_vanish(row):
        transition(conn, row["id"], "queued", "submitting")
        transition(conn, row["id"], "submitting", "uploaded", platform_upload_id=GONE)
        mark_missing_remote(conn, row["id"], evidence={"source": "test"}, now=NOW)

    result = drain_queue(conn, {"youtube": upload_then_vanish}, owner="w", now=NOW)

    row = _row(conn, delivery_id)
    assert row["status"] == "missing_remote"
    assert row["lease_owner"] is None
    assert result.succeeded == 0
    assert result.failed_permanent == 1
    assert any("missing_remote" in e for e in result.errors)


# --- catalog-driven marking ------------------------------------------------


class _Videos:
    def __init__(self, live):
        self.live = live

    def list(self, *, part, id, maxResults):  # noqa: N803 -- mirrors the API kwarg
        items = [
            {
                "id": i,
                "snippet": {"title": i},
                "contentDetails": {"duration": "PT1M"},
                "status": {"privacyStatus": "public"},
            }
            for i in id.split(",")
            if i in self.live
        ]
        return SimpleNamespace(execute=lambda: {"items": items})


def test_fetch_catalog_records_every_id_it_asked_about():
    svc = SimpleNamespace(videos=lambda: _Videos({LIVE}))
    cat = fetch_catalog(svc, [LIVE, GONE], known_ids=["SYNTHKNOWN"])
    assert set(cat["videos"]) == {LIVE}
    assert cat["checked_ids"] == sorted([LIVE, GONE, "SYNTHKNOWN"])


def _catalog(*, videos, checked):
    return {"videos": {v: {"title": v} for v in videos}, "checked_ids": sorted(checked)}


def test_only_an_id_the_catalog_checked_and_did_not_find_is_marked(conn):
    gone = _published(conn, 0, GONE)
    live = _published(conn, 1, LIVE)
    unchecked = _published(conn, 2, "SYNTHNEVERASKED")
    cat = _catalog(videos=[LIVE], checked=[LIVE, GONE])

    dry = missing_remote_from_catalog(conn, cat, apply=False, now=NOW)
    assert [d for d, _ in dry.marked] == [gone]
    assert [d for d, _ in dry.unchecked] == [unchecked]
    assert _row(conn, gone)["status"] == "published"  # dry run writes nothing

    applied = missing_remote_from_catalog(conn, cat, apply=True, now=NOW)
    assert [d for d, _ in applied.marked] == [gone]
    assert _row(conn, gone)["status"] == "missing_remote"
    assert _row(conn, live)["status"] == "published"
    assert _row(conn, unchecked)["status"] == "published"


def test_a_catalog_without_checked_ids_is_refused(conn):
    _published(conn, 0, GONE)
    with pytest.raises(ValueError, match="checked_ids"):
        missing_remote_from_catalog(conn, {"videos": {}}, apply=True, now=NOW)
    assert conn.execute("SELECT status FROM deliveries").fetchone()[0] == "published"


def test_cli_mark_missing_remote_dry_runs_unless_applied(tmp_path, capsys):
    storage = tmp_path / "seo"
    conn = open_queue_db(storage)
    gone = _published(conn, 0, GONE)
    conn.close()
    cat_path = tmp_path / "live_catalog.json"
    cat_path.write_text(json.dumps(_catalog(videos=[], checked=[GONE])))
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(storage)))
    parser = _build_parser()

    args = parser.parse_args(["queue", "mark-missing-remote", "--live-catalog", str(cat_path)])
    assert cmd_queue(args, config) == 0
    assert "dry run" in capsys.readouterr().out

    args = parser.parse_args(
        ["queue", "mark-missing-remote", "--live-catalog", str(cat_path), "--apply"]
    )
    assert cmd_queue(args, config) == 0
    conn = open_queue_db(storage)
    try:
        assert _row(conn, gone)["status"] == "missing_remote"
    finally:
        conn.close()


def test_catalog_fetch_can_check_every_queue_upload_id(tmp_path, monkeypatch):
    from gomoveshift.cli.handlers import catalog_fetch as h

    storage = tmp_path / "seo"
    conn = open_queue_db(storage)
    _published(conn, 0, GONE)
    conn.close()
    seen = {}

    def fake_enum(service, *, known_ids=()):
        seen["known"] = sorted(known_ids)
        return {}

    def fake_fetch(service, ids, *, known_ids=()):
        return {"enumerated": 0, "listed": 0, "videos": {}, "checked_ids": sorted(known_ids)}

    monkeypatch.setattr(h, "_build_readonly_youtube_service", lambda: MagicMock())
    monkeypatch.setattr(h, "enumerate_uploads", fake_enum)
    monkeypatch.setattr(h, "fetch_catalog", fake_fetch)
    out = tmp_path / "cat.json"
    args = _build_parser().parse_args(
        ["catalog-fetch", "--out", str(out), "--known-ids-from-queue"]
    )
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(storage)))

    assert h.cmd_catalog_fetch(args, config) == 0
    assert seen["known"] == [GONE]
    assert json.loads(out.read_text())["checked_ids"] == [GONE]


# --- validate ----------------------------------------------------------------


def test_validate_names_every_missing_remote_row(conn, tmp_path):
    delivery_id = _published(conn, 0, GONE)
    mark_missing_remote(conn, delivery_id, evidence={}, now=NOW)

    problems = validate_restored_state(conn, str(tmp_path / "credentials"), now=NOW)

    assert [(p.delivery_id, p.kind) for p in problems] == [(delivery_id, "missing_remote")]
    assert GONE in problems[0].detail
