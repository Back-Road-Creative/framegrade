"""``gomoveshift queue recover-terminal``: every gate refuses and writes nothing;
a pass inserts exactly one linked row and leaves the terminal row untouched.
All offline -- the Graph client, sidecar loader, privacy loader and rendition
resolver are fakes."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gomoveshift.cli.handlers.queue_recover import (
    InstagramRecoveryGraph,
    RecoveryDeps,
    recover_terminal,
)
from gomoveshift.cli.parser import _build_parser as build_parser
from gomoveshift.publishing.dispatch import PRIVACY_REQUIRED_CLASSES
from gomoveshift.publishing.queue import (
    RecoveryRefusedError,
    enqueue,
    open_queue_db,
)

NOW = "2026-10-12T23:50:00+00:00"
CAPTION = "Rolling through the Smokies at dawn #roadtrip #gomoveshift extra tail text"


class FakeGraph:
    account_id = "17841400000000000"

    def __init__(self, state="EXPIRED", media=None):
        self.state = state
        self.media = [{"id": "o", "caption": "other"}] if media is None else media
        self.calls = []

    def container_state(self, container_id):
        self.calls.append(("container", container_id))
        return self.state

    def recent_media(self, limit):
        self.calls.append(("media", limit))
        return self.media


def _attested(state="verified"):
    return {"classes": {c: {"state": state} for c in PRIVACY_REQUIRED_CLASSES}}


def _record():
    return SimpleNamespace(
        project_name="proj",
        short_index=3,
        media=SimpleNamespace(path="/x/s3.mp4", sha256="a" * 64),
        payloads=SimpleNamespace(instagram=SimpleNamespace(caption=CAPTION)),
    )


def _deps(tmp_path, graph=None, *, privacy=None, record=None, rendition=None):
    media = tmp_path / "s3_ig.mp4"
    media.write_bytes(b"x")

    def _rendition(rec):
        return media

    def _missing(rec):
        raise FileNotFoundError("short not found")

    return RecoveryDeps(
        graph=graph or FakeGraph(),
        record_loader=(lambda project, idx: record or _record()),
        privacy_loader=(lambda row: privacy if privacy is not None else _attested()),
        rendition_path=rendition or _rendition,
    ), _missing


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "spool")
    yield c
    c.close()


def _terminal_row(conn, status="failed_permanent", upload_id="17900000000000001"):
    did = enqueue(conn, "proj", 3, "instagram", "hash-1", sidecar_path="/x/plan.json")
    conn.execute(
        "UPDATE deliveries SET status=?, attempt_count=8, error_class='PermanentDeliveryError',"
        " platform_upload_id=?, scheduled_at='2026-10-01T00:00:00+00:00' WHERE id=?",
        (status, upload_id, did),
    )
    conn.commit()
    return did


def _snapshot(conn):
    return (
        [tuple(r) for r in conn.execute("SELECT * FROM deliveries ORDER BY id")],
        [tuple(r) for r in conn.execute("SELECT * FROM response_log ORDER BY id")],
    )


def _refuses(conn, did, deps, match):
    before = _snapshot(conn)
    with pytest.raises(RecoveryRefusedError, match=match):
        recover_terminal(conn, did, "operator 2026-10-12", deps, now=NOW)
    assert _snapshot(conn) == before


def test_non_terminal_row_refused(conn, tmp_path):
    did = _terminal_row(conn, status="queued")
    _refuses(conn, did, _deps(tmp_path)[0], "failed_permanent")


def test_published_row_refused(conn, tmp_path):
    did = _terminal_row(conn, status="published")
    _refuses(conn, did, _deps(tmp_path)[0], "failed_permanent")


def test_unknown_row_refused(conn, tmp_path):
    _refuses(conn, 999, _deps(tmp_path)[0], "no delivery")


@pytest.mark.parametrize("state", ["FINISHED", "IN_PROGRESS", "PUBLISHED", "ERROR"])
def test_live_container_refused(conn, tmp_path, state):
    did = _terminal_row(conn)
    _refuses(conn, did, _deps(tmp_path, FakeGraph(state=state))[0], "container")


def test_not_found_container_passes_gate_b(conn, tmp_path):
    did = _terminal_row(conn)
    assert recover_terminal(
        conn, did, "operator", _deps(tmp_path, FakeGraph(state="NOT_FOUND"))[0], NOW
    )


def test_null_upload_id_treated_as_gone_without_graph_call(conn, tmp_path):
    did = _terminal_row(conn, upload_id=None)
    graph = FakeGraph(state="FINISHED")
    recover_terminal(conn, did, "operator", _deps(tmp_path, graph)[0], now=NOW)
    assert ("container", None) not in graph.calls
    assert not [c for c in graph.calls if c[0] == "container"]


def test_remote_caption_prefix_match_refused(conn, tmp_path):
    did = _terminal_row(conn)
    media = [{"id": "m1", "caption": CAPTION[:40] + " but different tail"}]
    _refuses(conn, did, _deps(tmp_path, FakeGraph(media=media))[0], "duplicate")


def test_remote_fingerprint_match_refused(conn, tmp_path):
    did = _terminal_row(conn)
    media = [{"id": "m1", "caption": "unrelated", "sha256": "a" * 64}]
    _refuses(conn, did, _deps(tmp_path, FakeGraph(media=media))[0], "duplicate")


def test_empty_scan_is_inconclusive_and_refused(conn, tmp_path):
    did = _terminal_row(conn)
    deps = _deps(tmp_path, FakeGraph(media=[]))[0]
    _refuses(conn, did, deps, "scan")


def test_missing_rendition_refused(conn, tmp_path):
    did = _terminal_row(conn)
    deps, missing = _deps(tmp_path, FakeGraph(media=[{"id": "o", "caption": "other"}]))
    deps = RecoveryDeps(deps.graph, deps.record_loader, deps.privacy_loader, missing)
    _refuses(conn, did, deps, "media")


def test_unreadable_sidecar_refused(conn, tmp_path):
    did = _terminal_row(conn)
    deps, _ = _deps(tmp_path, FakeGraph(media=[{"id": "o", "caption": "other"}]))

    def boom(project, idx):
        raise LookupError("no short_index 3")

    deps = RecoveryDeps(deps.graph, boom, deps.privacy_loader, deps.rendition_path)
    _refuses(conn, did, deps, "sidecar")


@pytest.mark.parametrize("attestation", [None, _attested("failed"), _attested("inconclusive")])
def test_privacy_unattested_or_unverified_refused(conn, tmp_path, attestation):
    did = _terminal_row(conn)
    graph = FakeGraph(media=[{"id": "o", "caption": "other"}])
    deps = _deps(tmp_path, graph)[0]
    deps = RecoveryDeps(
        deps.graph, deps.record_loader, (lambda row: attestation), deps.rendition_path
    )
    _refuses(conn, did, deps, "privacy")


def _passing(tmp_path):
    return _deps(tmp_path, FakeGraph(media=[{"id": "o", "caption": "other"}]))[0]


def test_pass_inserts_one_linked_row_and_leaves_old_row_identical(conn, tmp_path):
    did = _terminal_row(conn)
    old_before = tuple(conn.execute("SELECT * FROM deliveries WHERE id=?", (did,)).fetchone())
    new_id = recover_terminal(conn, did, "operator 2026-10-12 7:45 PM ET", _passing(tmp_path), NOW)
    assert tuple(conn.execute("SELECT * FROM deliveries WHERE id=?", (did,)).fetchone()) == (
        old_before
    )
    rows = conn.execute("SELECT * FROM deliveries ORDER BY id").fetchall()
    assert len(rows) == 2
    new = rows[1]
    assert new["id"] == new_id and new["recovers_delivery_id"] == did
    assert (new["project_name"], new["short_index"], new["platform"]) == ("proj", 3, "instagram")
    assert new["status"] == "queued" and new["attempt_count"] == 0
    assert new["platform_upload_id"] is None and new["platform_post_id"] is None
    assert new["next_attempt_at"] is None and new["sidecar_path"] == "/x/plan.json"
    assert new["plan_hash"] != "hash-1"
    assert new["scheduled_at"] == NOW
    assert new["recovery_authorized_by"] == "operator 2026-10-12 7:45 PM ET"
    ev = json.loads(new["recovery_evidence"])
    assert ev["container"] == {"id": "17900000000000001", "state": "EXPIRED"}
    assert ev["duplicate_scan"]["scanned"] == 1 and ev["duplicate_scan"]["matches"] == 0
    assert ev["rendition_path"].endswith("s3_ig.mp4")
    assert ev["privacy"] == "verified"
    log = conn.execute("SELECT * FROM response_log").fetchall()
    assert len(log) == 1 and log[0]["action"] == "recover_terminal"
    assert "operator 2026-10-12" in log[0]["reason"]


def test_second_recovery_of_same_row_refused(conn, tmp_path):
    did = _terminal_row(conn)
    recover_terminal(conn, did, "operator", _passing(tmp_path), NOW)
    _refuses(conn, did, _passing(tmp_path), "already recovered")


def test_blank_authorization_refused(conn, tmp_path):
    did = _terminal_row(conn)
    before = _snapshot(conn)
    with pytest.raises(RecoveryRefusedError, match="authorized-by"):
        recover_terminal(conn, did, "   ", _passing(tmp_path), now=NOW)
    assert _snapshot(conn) == before


def test_migration_adds_columns_to_old_db(tmp_path):
    import sqlite3

    spool = tmp_path / "old"
    spool.mkdir()
    raw = sqlite3.connect(spool / "social_queue.sqlite3")
    raw.execute(
        "CREATE TABLE deliveries (id INTEGER PRIMARY KEY AUTOINCREMENT, project_name TEXT NOT NULL,"
        " short_index INTEGER NOT NULL, platform TEXT NOT NULL, plan_hash TEXT NOT NULL,"
        " status TEXT NOT NULL, attempt_count INTEGER NOT NULL DEFAULT 0, next_attempt_at TEXT,"
        " platform_upload_id TEXT, platform_post_id TEXT, error_class TEXT, lease_owner TEXT,"
        " lease_expires_at TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,"
        " UNIQUE (project_name, short_index, platform, plan_hash))"
    )
    raw.commit()
    raw.close()
    c = open_queue_db(spool)
    cols = {r["name"] for r in c.execute("PRAGMA table_info(deliveries)")}
    assert {"recovers_delivery_id", "recovery_authorized_by", "recovery_evidence"} <= cols
    c.close()


def test_cli_requires_authorized_by():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["queue", "recover-terminal", "--id", "50"])
    with pytest.raises(SystemExit):
        parser.parse_args(["queue", "recover-terminal", "--authorized-by", "operator"])
    args = parser.parse_args(
        ["queue", "recover-terminal", "--id", "50", "--authorized-by", "operator"]
    )
    assert (args.command, args.queue_action, args.id) == ("queue", "recover-terminal", 50)


def test_real_graph_client_maps_not_found_and_states():
    from gomoveshift.publishing.instagram_adapter import InstagramPermanentError

    class Reels:
        account_id = "acct"

        def __init__(self, result):
            self.result = result

        def check_container_status(self, cid):
            if isinstance(self.result, Exception):
                raise self.result
            return {"status_code": self.result}

        def list_recent_media(self, limit):
            return [{"id": "1"}]

    assert InstagramRecoveryGraph(Reels("EXPIRED")).container_state("c") == "EXPIRED"
    gone = InstagramPermanentError(
        "Graph API permanently rejected: Object with ID 'c' does not exist"
    )
    assert InstagramRecoveryGraph(Reels(gone)).container_state("c") == "NOT_FOUND"
    other = InstagramPermanentError("Graph API permanently rejected: something else")
    with pytest.raises(InstagramPermanentError):
        InstagramRecoveryGraph(Reels(other)).container_state("c")
    assert InstagramRecoveryGraph(Reels("x")).recent_media(5) == [{"id": "1"}]


def _full_page(oldest_ts):
    """A full scan page (SCAN_LIMIT items, newest first) with no caption match."""
    from gomoveshift.cli.handlers.queue_recover import SCAN_LIMIT

    page = [
        {"id": str(i), "caption": f"other {i}", "timestamp": "2026-10-12T00:00:00+0000"}
        for i in range(SCAN_LIMIT - 1)
    ]
    page.append({"id": "last", "caption": "other last", "timestamp": oldest_ts})
    return page


def _created(conn, did, created_at):
    conn.execute("UPDATE deliveries SET created_at=? WHERE id=?", (created_at, did))
    conn.commit()


def test_full_scan_page_newer_than_the_row_is_refused(conn, tmp_path):
    # Row 50's "Media ID is not available" can mean the post went live. A full
    # page that stops short of the row's creation cannot prove no duplicate.
    did = _terminal_row(conn)
    _created(conn, did, "2026-09-01T00:00:00+00:00")
    deps = _deps(tmp_path, FakeGraph(media=_full_page("2026-09-20T00:00:00+0000")))[0]
    _refuses(conn, did, deps, "does not reach back")


def test_full_scan_page_without_timestamps_is_refused(conn, tmp_path):
    did = _terminal_row(conn)
    page = _full_page("2026-09-20T00:00:00+0000")
    page[-1].pop("timestamp")
    _refuses(conn, did, _deps(tmp_path, FakeGraph(media=page))[0], "timestamp")


def test_full_scan_page_reaching_past_the_row_passes(conn, tmp_path):
    did = _terminal_row(conn)
    _created(conn, did, "2026-10-01T00:00:00+00:00")
    deps = _deps(tmp_path, FakeGraph(media=_full_page("2026-09-20T00:00:00+0000")))[0]
    assert recover_terminal(conn, did, "operator", deps, now=NOW)
