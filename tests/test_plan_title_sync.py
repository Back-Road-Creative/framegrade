"""U-R3: a retitled short's publish-plan sidecar follows the live title.

``tools/retitle_apply.py`` changed four shorts' live titles on 2026-10-06 and
recorded APPLIED in its audit log, but never touched the publish-plan sidecar.
``publishing-status --validate --live-catalog`` then reported every one of
them as ``published_title_drift`` -- a defect manufactured by our own tool.

These tests pin the three halves of the fix:

* robust: an APPLIED retitle writes the new title into the sidecar in the
  same step (``plan_sync``);
* backstop: validate tells "plan stale, retitled by us" apart from genuine
  drift using the retitle audit log;
* history: ``gomoveshift queue sync-plan-titles --audit`` replays an audit log
  into the sidecars (local file write only, dry run unless ``--apply``).

All data is synthetic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_cli_enqueue_plan import _record, _sidecar

from gomoveshift.cli.handlers.pub_status import validate_against_live_catalog
from gomoveshift.cli.handlers.queue_recover import cmd_queue
from gomoveshift.publishing.plan_io import load_publish_plan
from gomoveshift.publishing.plan_title_sync import applied_retitles, sync_plan_title
from gomoveshift.publishing.queue import enqueue, open_queue_db

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_retitle_apply import FakeService, _row, _write_worklist  # noqa: E402
from tools import retitle_apply as ra  # noqa: E402

VID = "SYNTHvid01"
PLAN_TITLE = "t"  # _record()'s YouTube title
LIVE_TITLE = "Clonmel to Clonakilty, County Cork"


@pytest.fixture
def queue(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    sidecar = _sidecar(out, _record())
    conn = open_queue_db(tmp_path)
    delivery_id = enqueue(conn, "proj", 3, "youtube", "HASH1", sidecar_path=str(sidecar))
    conn.execute(
        "UPDATE deliveries SET status='published', platform_upload_id=? WHERE id=?",
        (VID, delivery_id),
    )
    conn.commit()
    yield SimpleNamespace(conn=conn, sidecar=sidecar, delivery_id=delivery_id, dir=tmp_path)
    conn.close()


def _plan_title(sidecar):
    return load_publish_plan(sidecar).shorts[0].payloads.youtube.title


def _audit(path, *records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _applied(new, ts="2026-10-06T21:27:00+00:00", result="APPLIED", vid=VID):
    return {"video_id": vid, "old": PLAN_TITLE, "new": new, "result": result, "timestamp": ts}


# --- the shared sync ---------------------------------------------------------


def test_dry_run_reports_the_change_and_writes_nothing(queue):
    before = queue.sidecar.read_bytes()

    changes = sync_plan_title(queue.conn, VID, LIVE_TITLE, apply=False)

    assert [(c.delivery_id, c.old, c.new, c.written) for c in changes] == [
        (queue.delivery_id, PLAN_TITLE, LIVE_TITLE, False)
    ]
    assert queue.sidecar.read_bytes() == before


def test_apply_rewrites_only_the_title_and_logs_it(queue):
    before = load_publish_plan(queue.sidecar).shorts[0]

    changes = sync_plan_title(queue.conn, VID, LIVE_TITLE, apply=True)

    after = load_publish_plan(queue.sidecar).shorts[0]
    assert [c.written for c in changes] == [True]
    assert after.payloads.youtube.title == LIVE_TITLE
    assert after.payloads.youtube.description == before.payloads.youtube.description
    assert after.media == before.media
    assert queue.sidecar.read_text(encoding="utf-8").endswith("}\n")
    logged = queue.conn.execute(
        "SELECT platform, action, reason FROM response_log WHERE action='sync_plan_title'"
    ).fetchall()
    assert len(logged) == 1
    reason = json.loads(logged[0]["reason"])
    assert (reason["video_id"], reason["old"], reason["new"]) == (VID, PLAN_TITLE, LIVE_TITLE)


def test_nothing_to_do_when_the_plan_already_matches_or_the_video_is_unknown(queue):
    assert sync_plan_title(queue.conn, VID, PLAN_TITLE, apply=True) == []
    assert sync_plan_title(queue.conn, "SYNTHother", LIVE_TITLE, apply=True) == []


def test_applied_retitles_keeps_the_latest_applied_row_per_video(tmp_path):
    path = _audit(
        tmp_path / "a.audit.jsonl",
        _applied("first", ts="2026-10-06T10:00:00+00:00"),
        _applied("snap", result="SNAPSHOT"),
        _applied("second", ts="2026-10-06T11:00:00+00:00"),
        _applied("stale", result="SKIPPED_STALE"),
    )

    got = applied_retitles([path])

    assert got[VID].title == "second"
    assert got[VID].at == "2026-10-06T11:00:00+00:00"
    assert got[VID].audit_path == path


# --- validate backstop -------------------------------------------------------


def _catalog(title=LIVE_TITLE):
    return {"videos": {VID: {"title": title}}}


def test_a_title_we_retitled_is_plan_stale_not_drift(queue, tmp_path):
    retitles = applied_retitles([_audit(tmp_path / "a.audit.jsonl", _applied(LIVE_TITLE))])

    problems = validate_against_live_catalog(queue.conn, _catalog(), retitles=retitles)

    assert [p.kind for p in problems] == ["plan_stale_retitled"]
    assert "2026-10-06T21:27" in problems[0].detail
    assert "queue sync-plan-titles" in problems[0].detail


def test_without_a_matching_retitle_it_is_still_drift(queue, tmp_path):
    other = applied_retitles([_audit(tmp_path / "a.audit.jsonl", _applied("something else"))])

    assert [p.kind for p in validate_against_live_catalog(queue.conn, _catalog())] == [
        "published_title_drift"
    ]
    assert [
        p.kind for p in validate_against_live_catalog(queue.conn, _catalog(), retitles=other)
    ] == ["published_title_drift"]


# --- retitle_apply: the robust half -------------------------------------------


def test_retitle_apply_syncs_the_plan_in_the_same_step(queue):
    worklist = _write_worklist(queue.dir, [_row(video_id=VID, current_title=PLAN_TITLE)])
    service = FakeService({VID: {"title": PLAN_TITLE, "categoryId": "19"}})

    rc = ra.main(
        ["--worklist", str(worklist), "--apply", "--yes-live", "--queue-dir", str(queue.dir)],
        service=service,
        sleep=lambda _s: None,
    )

    assert rc == 0
    assert service.videos().update_calls, "the live write happened"
    assert _plan_title(queue.sidecar) == LIVE_TITLE


def test_a_live_run_with_no_resolvable_queue_is_refused_before_any_write(tmp_path, monkeypatch):
    worklist = _write_worklist(tmp_path, [_row()])

    def _no_config(*_a, **_k):
        raise FileNotFoundError("no config.yaml")

    monkeypatch.setattr("gomoveshift.config.get_config", _no_config)
    err = []
    rc = ra.main(
        ["--worklist", str(worklist), "--apply", "--yes-live"],
        err=SimpleNamespace(write=err.append, flush=lambda: None),
    )

    assert rc == 2
    assert "--queue-dir" in "".join(err)


# --- queue sync-plan-titles: the history half ---------------------------------


def _queue_args(audit, apply):
    return SimpleNamespace(queue_action="sync-plan-titles", audit=[str(audit)], apply=apply)


def test_sync_plan_titles_cli_is_a_dry_run_until_apply(queue, tmp_path, capsys):
    audit = _audit(tmp_path / "a.audit.jsonl", _applied(LIVE_TITLE))
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(queue.dir)))

    assert cmd_queue(_queue_args(audit, apply=False), config) == 0
    assert _plan_title(queue.sidecar) == PLAN_TITLE
    assert "dry run" in capsys.readouterr().out

    assert cmd_queue(_queue_args(audit, apply=True), config) == 0
    assert _plan_title(queue.sidecar) == LIVE_TITLE
    assert f"delivery {queue.delivery_id}" in capsys.readouterr().out


def test_publishing_status_reports_plan_stale_without_failing(queue, tmp_path, capsys):
    from gomoveshift.cli.handlers.pub_status import cmd_publishing_status

    catalog = tmp_path / "live_catalog.json"
    catalog.write_text(json.dumps(_catalog()), encoding="utf-8")
    audit = _audit(tmp_path / "a.audit.jsonl", _applied(LIVE_TITLE))
    config = SimpleNamespace(
        seo=SimpleNamespace(storage_path=str(queue.dir), credentials_dir=str(tmp_path / "c" / "x"))
    )

    def run(audits):
        args = SimpleNamespace(validate=True, live_catalog=str(catalog), retitle_audit=audits)
        return cmd_publishing_status(args, config)

    assert run([str(audit)]) == 0
    assert "plan_stale_retitled" in capsys.readouterr().out
    assert run(None) == 1
    assert "published_title_drift" in capsys.readouterr().out
