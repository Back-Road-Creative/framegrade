"""U-R2: `publishing-status --validate` problems gate the publishing path.

A delivery the validator flags (a short the channel never got, an Instagram
post parked past its bound, a lost sidecar) used to be a report nobody had to
read: enqueue-plan kept spooling more shorts for the same project and
platform, and dispatch-queue kept draining them. These tests pin the gate:

* scope is per platform AND per project -- a problem on one pair never blocks
  another pair (no blanket refusal);
* the refusal names each offending row (delivery id, kind, platform, project)
  and the exact validate command;
* the escape is ``--ignore-validate`` with a required ``--reason``, written to
  the ``response_log`` audit table;
* the drive's automatic enqueue goes through the same gate.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

import test_cli_enqueue_plan as _ep

from gomoveshift.cli.handlers.dispatch_queue import cmd_dispatch_queue
from gomoveshift.cli.handlers.enqueue_plan import cmd_enqueue_plan
from gomoveshift.cli.handlers.pub_status import gating_problems
from gomoveshift.publishing.dispatch import DrainResult, drain_queue
from gomoveshift.publishing.queue import enqueue, open_queue_db

NOW = "2026-10-15T12:00:00+00:00"
VALIDATE_CMD = "gomoveshift publishing-status --validate"


def _missing_remote(conn, project="proj", short_index=7):
    delivery_id = enqueue(conn, project, short_index, "youtube", "HASH0")
    conn.execute(
        "UPDATE deliveries SET status='missing_remote', platform_upload_id='GONE1' WHERE id=?",
        (delivery_id,),
    )
    conn.commit()
    return delivery_id


def _row_count(tmp_path):
    conn = open_queue_db(tmp_path)
    try:
        return conn.execute("SELECT COUNT(*) AS n FROM deliveries").fetchone()["n"]
    finally:
        conn.close()


def _enqueue_run(tmp_path, capsys, **arg_over):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    _ep._sidecar(out, _ep._record(selected=("youtube",)))
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")
    with mock.patch(
        "gomoveshift.cli.handlers.enqueue_plan._load_state",
        return_value=_ep._approved_state(out),
    ):
        rc = cmd_enqueue_plan(_ep._args(state_path, apply=True, **arg_over), _ep._config(tmp_path))
    return rc, capsys.readouterr().out


# --- the shared helper ------------------------------------------------------


def test_gating_problems_carry_the_project_and_skip_stale_leases(tmp_path):
    conn = open_queue_db(tmp_path)
    gone = _missing_remote(conn)
    leased = enqueue(conn, "proj", 9, "instagram", "HASH0", sidecar_path=None)
    conn.execute(
        "UPDATE deliveries SET lease_owner='dead', lease_expires_at='2099-01-01T00:00:00+00:00' "
        "WHERE id=?",
        (leased,),
    )
    conn.commit()

    problems = gating_problems(conn, None, now=NOW)
    conn.close()

    kinds = {(p.delivery_id, p.kind, p.platform, p.project) for p in problems}
    assert (gone, "missing_remote", "youtube", "proj") in kinds
    # A live lease is work in flight on a healthy box, not a reason to stop
    # publishing; only the restore validator reports it.
    assert all(p.kind != "stale_lease" for p in problems)


# --- enqueue-plan --apply ---------------------------------------------------


def test_enqueue_refuses_a_plan_whose_project_and_platform_have_a_problem(tmp_path, capsys):
    conn = open_queue_db(tmp_path)
    gone = _missing_remote(conn)
    conn.close()
    before = _row_count(tmp_path)

    rc, out = _enqueue_run(tmp_path, capsys)

    assert rc == 1
    assert _row_count(tmp_path) == before, "a refusal must write nothing"
    assert "nothing was written to the queue" in out
    assert f"delivery={gone} kind=missing_remote platform=youtube project=proj" in out
    assert VALIDATE_CMD in out


def test_enqueue_is_not_blocked_by_another_projects_problem(tmp_path, capsys):
    conn = open_queue_db(tmp_path)
    _missing_remote(conn, project="other")
    conn.close()

    rc, out = _enqueue_run(tmp_path, capsys)

    assert rc == 0, out
    assert "enqueued=1" in out


def test_enqueue_is_not_blocked_by_another_platforms_problem(tmp_path, capsys):
    conn = open_queue_db(tmp_path)
    parked = enqueue(conn, "proj", 7, "instagram", "HASH0")
    old = "2020-01-01T00:00:00+00:00"  # parked long before any real now
    conn.execute(
        "UPDATE deliveries SET status='awaiting_ig_reconcile', updated_at=? WHERE id=?",
        (old, parked),
    )
    conn.commit()
    conn.close()

    rc, out = _enqueue_run(tmp_path, capsys)

    assert rc == 0, out
    assert "enqueued=1" in out


def test_ignore_validate_without_a_reason_is_refused(tmp_path, capsys):
    conn = open_queue_db(tmp_path)
    _missing_remote(conn)
    conn.close()
    before = _row_count(tmp_path)

    rc, out = _enqueue_run(tmp_path, capsys, ignore_validate=True, reason=None)

    assert rc == 1
    assert "--reason" in out
    assert _row_count(tmp_path) == before


def test_ignore_validate_with_a_reason_proceeds_and_is_audited(tmp_path, capsys):
    conn = open_queue_db(tmp_path)
    gone = _missing_remote(conn)
    conn.close()

    rc, out = _enqueue_run(
        tmp_path, capsys, ignore_validate=True, reason="re-upload decided by the operator"
    )

    assert rc == 0, out
    assert "enqueued=1" in out
    conn = open_queue_db(tmp_path)
    rows = conn.execute(
        "SELECT platform, action, mode, reason FROM response_log WHERE action='ignore_validate'"
    ).fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0]["platform"] == "youtube"
    reason = json.loads(rows[0]["reason"])
    assert reason["reason"] == "re-upload decided by the operator"
    assert reason["command"] == "enqueue-plan"
    assert reason["problems"][0]["delivery_id"] == gone


def test_the_drives_automatic_enqueue_goes_through_the_gate(tmp_path, capsys):
    """auto_enqueue builds its own args with no escape flag; the gate must
    still refuse, so a drive cannot pile more shorts onto a broken pair."""
    conn = open_queue_db(tmp_path)
    _missing_remote(conn)
    conn.close()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _ep._sidecar(out_dir, _ep._record(selected=("youtube",)))
    state_path = out_dir / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")
    args = SimpleNamespace(
        state_path=str(state_path), apply=True, output_dir=None, max_shorts=None, shadow=False
    )
    with mock.patch(
        "gomoveshift.cli.handlers.enqueue_plan._load_state",
        return_value=_ep._approved_state(out_dir),
    ):
        rc = cmd_enqueue_plan(args, _ep._config(tmp_path))

    assert rc == 1
    assert "kind=missing_remote" in capsys.readouterr().out


# --- dispatch-queue --apply -------------------------------------------------


def test_drain_holds_only_the_gated_project_and_platform(tmp_path):
    conn = open_queue_db(tmp_path)
    held = enqueue(conn, "proj", 3, "youtube", "HASH1")
    other = enqueue(conn, "other", 3, "youtube", "HASH1")
    called = []

    result = drain_queue(
        conn,
        {"youtube": lambda row: called.append(row["id"])},
        now=NOW,
        held_scopes={("youtube", "proj")},
    )
    status = {
        r["id"]: (r["status"], r["attempt_count"], r["lease_owner"])
        for r in conn.execute("SELECT id, status, attempt_count, lease_owner FROM deliveries")
    }
    conn.close()

    assert called == [other]
    assert result.held_validate == 1
    assert status[held] == ("queued", 0, None), "a hold never transitions or leases"
    assert any(f"youtube#{held}" in e and "validate hold" in e for e in result.errors)


def _drain_ok():
    return DrainResult(
        attempted=0,
        succeeded=0,
        retried=0,
        failed_permanent=0,
        skipped_leased=0,
        skipped_no_handler=0,
        throttled=0,
        errors=[],
    )


def _dispatch(tmp_path, **arg_over):
    import test_cli_dispatch_queue as _dq

    _dq._sidecar(tmp_path, _dq._record())
    with (
        mock.patch(
            "gomoveshift.cli.handlers.dispatch_queue._build_routed_handlers",
            return_value={},
        ),
        mock.patch("gomoveshift.cli.handlers.dispatch_queue.drain_queue") as drain,
    ):
        drain.return_value = _drain_ok()
        rc = cmd_dispatch_queue(
            _dq._args(apply=True, **arg_over), _dq._config(tmp_path, youtube="first_party")
        )
    return rc, drain


def test_dispatch_passes_the_gated_scopes_to_the_drain(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "_console"))
    monkeypatch.setenv("GMS_TIKTOK_PACK_DIR", str(tmp_path / "pack"))
    conn = open_queue_db(tmp_path)
    gone = _missing_remote(conn)
    enqueue(conn, "proj", 3, "youtube", "HASH1")
    conn.close()

    rc, drain = _dispatch(tmp_path)
    out = capsys.readouterr().out

    assert rc == 0
    assert drain.call_args.kwargs["held_scopes"] == {("youtube", "proj")}
    assert f"delivery={gone} kind=missing_remote platform=youtube project=proj" in out
    assert VALIDATE_CMD in out


def test_dispatch_ignore_validate_needs_a_reason_and_audits_it(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "_console"))
    monkeypatch.setenv("GMS_TIKTOK_PACK_DIR", str(tmp_path / "pack"))
    conn = open_queue_db(tmp_path)
    _missing_remote(conn)
    conn.close()

    rc, drain = _dispatch(tmp_path, ignore_validate=True, reason=None)
    assert rc == 1
    assert "--reason" in capsys.readouterr().out
    drain.assert_not_called()

    rc, drain = _dispatch(tmp_path, ignore_validate=True, reason="checked by hand")
    assert rc == 0
    assert drain.call_args.kwargs["held_scopes"] == set()
    conn = open_queue_db(tmp_path)
    n = conn.execute(
        "SELECT COUNT(*) AS n FROM response_log WHERE action='ignore_validate'"
    ).fetchone()["n"]
    conn.close()
    assert n == 1
