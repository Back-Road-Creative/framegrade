"""``queue recover-terminal`` on a YouTube ``missing_remote`` row: the one
audited way to re-queue a short the channel no longer returns. Every gate
refuses and writes nothing; a pass inserts one linked row, and the
``missing_remote`` validate problem clears once a recovery row exists.
All offline -- the YouTube reader, loaders and media resolver are fakes."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from freezegun import freeze_time

from gomoveshift.cli.handlers.pub_status import validate_restored_state
from gomoveshift.cli.handlers.queue_recover import (
    RecoveryDeps,
    YouTubeRecoveryReader,
    recover_terminal,
)
from gomoveshift.publishing.dispatch import PRIVACY_REQUIRED_CLASSES
from gomoveshift.publishing.queue import (
    RecoveryRefusedError,
    enqueue,
    mark_missing_remote,
    open_queue_db,
)

NOW = "2026-10-15T16:00:00+00:00"
FUTURE_SLOT = "2026-10-20T14:00:00Z"
TITLE = "Smoky Mountains at dawn | Sevier County, TN"


class FakeYouTube:
    def __init__(self, exists=False, uploads=None, exists_error=None):
        self.exists = exists
        self.exists_error = exists_error
        self.uploads = [{"id": "other1", "title": "another"}] if uploads is None else uploads
        self.calls = []

    def video_exists(self, video_id):
        self.calls.append(("exists", video_id))
        if self.exists_error:
            raise self.exists_error
        return self.exists

    def recent_uploads(self, limit):
        self.calls.append(("uploads", limit))
        return self.uploads


def _record(slot=FUTURE_SLOT, title=TITLE):
    return SimpleNamespace(
        project_name="proj",
        short_index=4,
        media=SimpleNamespace(path="/x/s4.mp4", sha256="b" * 64),
        schedule=SimpleNamespace(scheduled_at_utc=slot),
        payloads=SimpleNamespace(youtube=SimpleNamespace(title=title)),
    )


def _attested(state="verified"):
    return {"classes": {c: {"state": state} for c in PRIVACY_REQUIRED_CLASSES}}


def _deps(tmp_path, reader=None, *, record=None, privacy=None, media_ok=True):
    media = tmp_path / "s4.mp4"
    if media_ok:
        media.write_bytes(b"x")
    return RecoveryDeps(
        graph=reader or FakeYouTube(),
        record_loader=lambda project, idx: record or _record(),
        privacy_loader=lambda row: privacy if privacy is not None else _attested(),
        rendition_path=lambda rec: media,
    )


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "spool")
    yield c
    c.close()


def _gone_row(conn, video_id="GONEvid0001"):
    did = enqueue(conn, "proj", 4, "youtube", "hash-yt", sidecar_path="/x/plan.json")
    conn.execute(
        "UPDATE deliveries SET status='published', platform_upload_id=? WHERE id=?",
        (video_id, did),
    )
    conn.commit()
    assert mark_missing_remote(conn, did, evidence={"source": "test"}, now=NOW)
    return did


def _missing_remote_problems(conn):
    # The recovery row's fake sidecar is unreadable here; only missing_remote matters.
    return [p for p in validate_restored_state(conn, None, now=NOW) if p.kind == "missing_remote"]


def _snapshot(conn):
    return (
        [tuple(r) for r in conn.execute("SELECT * FROM deliveries ORDER BY id")],
        [tuple(r) for r in conn.execute("SELECT * FROM response_log ORDER BY id")],
    )


def _refuses(conn, did, deps, match):
    before = _snapshot(conn)
    with pytest.raises(RecoveryRefusedError, match=match):
        recover_terminal(conn, did, "operator 2026-10-15", deps, now=NOW)
    assert _snapshot(conn) == before


def test_published_youtube_row_refused(conn, tmp_path):
    did = enqueue(conn, "proj", 4, "youtube", "h", sidecar_path="/x/plan.json")
    conn.execute(
        "UPDATE deliveries SET status='published', platform_upload_id='v' WHERE id=?", (did,)
    )
    conn.commit()
    _refuses(conn, did, _deps(tmp_path), "only failed_permanent or missing_remote")


def test_failed_permanent_youtube_row_still_refused(conn, tmp_path):
    did = enqueue(conn, "proj", 4, "youtube", "h", sidecar_path="/x/plan.json")
    conn.execute("UPDATE deliveries SET status='failed_permanent' WHERE id=?", (did,))
    conn.commit()
    _refuses(conn, did, _deps(tmp_path), "youtube recovers missing_remote only")


def test_gate_reads_the_old_video_id_fresh(conn, tmp_path):
    did = _gone_row(conn)
    reader = FakeYouTube()
    recover_terminal(conn, did, "operator", _deps(tmp_path, reader), now=NOW)
    assert ("exists", "GONEvid0001") in reader.calls


def test_video_back_on_channel_refused(conn, tmp_path):
    did = _gone_row(conn)
    _refuses(conn, did, _deps(tmp_path, FakeYouTube(exists=True)), "returns GONEvid0001 again")


def test_lookup_error_refused_not_assumed_gone(conn, tmp_path):
    did = _gone_row(conn)
    reader = FakeYouTube(exists_error=RuntimeError("quota"))
    _refuses(conn, did, _deps(tmp_path, reader), "could not confirm")


def test_passed_slot_refused(conn, tmp_path):
    did = _gone_row(conn)
    deps = _deps(tmp_path, record=_record(slot="2026-10-10T14:00:00Z"))
    _refuses(conn, did, deps, "slot 2026-10-10T14:00:00Z has passed")


def test_missing_media_refused(conn, tmp_path):
    did = _gone_row(conn)
    _refuses(conn, did, _deps(tmp_path, media_ok=False), "not deliverable")


def test_privacy_unverified_refused(conn, tmp_path):
    did = _gone_row(conn)
    _refuses(conn, did, _deps(tmp_path, privacy=_attested("pending")), "privacy")


def test_same_title_on_channel_refused(conn, tmp_path):
    did = _gone_row(conn)
    reader = FakeYouTube(uploads=[{"id": "dupe1", "title": TITLE}])
    _refuses(conn, did, _deps(tmp_path, reader), "dupe1")


def test_empty_uploads_scan_refused(conn, tmp_path):
    did = _gone_row(conn)
    _refuses(conn, did, _deps(tmp_path, FakeYouTube(uploads=[])), "empty")


def test_pass_inserts_linked_row_and_clears_validate_problem(conn, tmp_path):
    did = _gone_row(conn)
    assert len(_missing_remote_problems(conn)) == 1
    old = tuple(conn.execute("SELECT * FROM deliveries WHERE id=?", (did,)).fetchone())
    new_id = recover_terminal(conn, did, "operator 2026-10-15 noon ET", _deps(tmp_path), now=NOW)
    assert tuple(conn.execute("SELECT * FROM deliveries WHERE id=?", (did,)).fetchone()) == old
    new = conn.execute("SELECT * FROM deliveries WHERE id=?", (new_id,)).fetchone()
    assert (new["platform"], new["status"], new["recovers_delivery_id"]) == (
        "youtube",
        "queued",
        did,
    )
    assert new["platform_upload_id"] is None
    ev = json.loads(new["recovery_evidence"])
    assert ev["status"] == "missing_remote"
    assert ev["old_video"] == {"id": "GONEvid0001", "returned": False}
    assert ev["slot"] == FUTURE_SLOT
    assert ev["duplicate_scan"] == {"scanned": 1, "matches": 0}
    assert ev["privacy"] == "verified"
    assert _missing_remote_problems(conn) == []


def test_second_recovery_refused(conn, tmp_path):
    did = _gone_row(conn)
    recover_terminal(conn, did, "operator", _deps(tmp_path), now=NOW)
    _refuses(conn, did, _deps(tmp_path), "already recovered")


def test_unrecovered_problem_names_the_recovery_command(conn):
    did = _gone_row(conn)
    (problem,) = _missing_remote_problems(conn)
    assert f"queue recover-terminal --id {did}" in problem.detail


def test_real_reader_maps_items_and_titles():
    class _Req:
        def __init__(self, body):
            self._body = body

        def execute(self):
            return self._body

    class _Videos:
        def list(self, **kw):
            return _Req({"items": [{"id": kw["id"]}]} if kw["id"] == "here" else {"items": []})

    service = SimpleNamespace(videos=lambda: _Videos())
    uploads = [SimpleNamespace(video_id="a1", title="t1")]
    reader = YouTubeRecoveryReader(service, lister=lambda svc, limit: uploads)
    assert reader.video_exists("here") is True
    assert reader.video_exists("gone") is False
    assert reader.recent_uploads(5) == [{"id": "a1", "title": "t1"}]


# --- dry run by default -----------


def test_dry_run_checks_every_gate_and_writes_nothing(conn, tmp_path):
    did = _gone_row(conn)
    reader = FakeYouTube()
    before = _snapshot(conn)
    assert (
        recover_terminal(conn, did, "operator", _deps(tmp_path, reader), now=NOW, apply=False)
        is None
    )
    assert _snapshot(conn) == before
    assert ("exists", "GONEvid0001") in reader.calls and ("uploads", 200) in reader.calls


def test_dry_run_still_refuses_a_failing_gate(conn, tmp_path):
    did = _gone_row(conn)
    with pytest.raises(RecoveryRefusedError, match="not missing"):
        recover_terminal(
            conn, did, "operator", _deps(tmp_path, FakeYouTube(exists=True)), now=NOW, apply=False
        )


def _cli(tmp_path, monkeypatch, conn, did, *extra):
    from gomoveshift.cli.handlers import queue_recover as qr
    from gomoveshift.cli.parser import _build_parser

    monkeypatch.setattr(qr, "_build_real_deps", lambda *a, **k: _deps(tmp_path))
    args = _build_parser().parse_args(
        ["queue", "recover-terminal", "--id", str(did), "--authorized-by", "operator", *extra]
    )
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path / "spool")))
    # The CLI stamps the gates with the real clock; freeze it at NOW so
    # FUTURE_SLOT stays in the future on any run date.
    with freeze_time(NOW):
        return qr.cmd_queue(args, config)


def test_cli_without_apply_is_a_dry_run(conn, tmp_path, monkeypatch, capsys):
    did = _gone_row(conn)
    before = _snapshot(conn)
    assert _cli(tmp_path, monkeypatch, conn, did) == 0
    assert _snapshot(conn) == before
    out = capsys.readouterr().out
    assert f"dry run: delivery {did} passes every recovery gate" in out
    assert "--apply" in out


def test_cli_with_apply_inserts_the_recovery_row(conn, tmp_path, monkeypatch, capsys):
    did = _gone_row(conn)
    assert _cli(tmp_path, monkeypatch, conn, did, "--apply") == 0
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM deliveries WHERE recovers_delivery_id=?", (did,)
        ).fetchone()[0]
        == 1
    )
    assert f"recovered delivery {did} as new delivery" in capsys.readouterr().out
