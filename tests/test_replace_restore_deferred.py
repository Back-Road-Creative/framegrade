"""A put-off restore is not a success; the launcher's own run_id is the one pinned (fixup 11)."""

import inspect
import json
import subprocess
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from gomoveshift.health import heal, launch
from test_health_backlog_launch import (  # noqa: F401
    NOW,
    _replace_rig,
    _three_lfs,
    _marker,
    _tick_at,
    live_worker,
    job,
)
from test_replace_restore_set_aside import _crashed, _old_run_is_back
from test_replace_restore_bound import (
    RID,
    _die_between_the_renames,
    _marker_file,
    _set_aside,
    _tick,
)


def _deferred_incidents():
    from gomoveshift.health.store import default_data_root, default_incidents_dir, load_all

    rows, _bad = load_all(default_incidents_dir(default_data_root()))
    return [r for r in rows if r.get("code") == "restore_deferred" and not r.get("resolved_at")]


def _restore_failed_incidents():
    from gomoveshift.health.store import default_data_root, default_incidents_dir, load_all

    rows, _bad = load_all(default_incidents_dir(default_data_root()))
    return [r for r in rows if r.get("code") == "restore_failed"]


def _running_since_the_attempt(rig, *, live=True, run_id="stale"):
    at = launch._read_retry_marker(_marker_file(rig))["at"]
    job(rig, run_id, "day-a", "running", datetime.fromisoformat(at).timestamp() + 5, live=live)


def _end_the_job(rig, run_id="stale", state="succeeded"):
    path = rig.tmp / "jobs" / run_id / "job.json"
    rec = json.loads(path.read_text())
    rec["result"] = {"state": state, "exit_code": 0}
    path.write_text(json.dumps(rec))


# ---- 1. a put-off restore is not a success ----


def test_a_permanent_refusal_while_the_restore_is_put_off_keeps_the_marker(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    _crashed(rig, monkeypatch)
    _running_since_the_attempt(rig)
    rig.slots = [(NOW + timedelta(days=41), "fri")]
    res = _tick(rig, NOW + timedelta(days=40), jobs_dir=rig.tmp / "jobs")
    assert res["replace_retry"]["outcome"] == "bumped", res["replace_retry"]
    assert _marker_file(rig).exists() and len(_set_aside(rig)) == 1
    assert not _old_run_is_back(rig)
    assert not _restore_failed_incidents()
    _end_the_job(rig)  # the block lifts: the next tick restores the old run
    rig.slots = [(NOW + timedelta(days=61), "fri")]
    res = _tick(rig, NOW + timedelta(days=60), jobs_dir=rig.tmp / "jobs")
    assert _old_run_is_back(rig), res["replace_retry"]
    assert not _restore_failed_incidents()


def test_an_unclaimed_give_up_while_the_restore_is_put_off_keeps_the_marker(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    _crashed(rig, monkeypatch)
    held = json.loads(_marker_file(rig).read_text())
    held.pop("region", None)
    _marker_file(rig).write_text(json.dumps(held))
    rig.rows = [r for r in rig.rows if r.project != "day-a"]
    _running_since_the_attempt(rig)
    res = _tick(rig, NOW + timedelta(days=2), jobs_dir=rig.tmp / "jobs")
    assert "day-a" not in (res.get("given_up_replace_retry") or [])
    assert _marker_file(rig).exists() and len(_set_aside(rig)) == 1
    assert not _old_run_is_back(rig) and not _restore_failed_incidents()
    _end_the_job(rig)
    res = _tick(rig, NOW + timedelta(days=20), jobs_dir=rig.tmp / "jobs")
    assert "day-a" in res["given_up_replace_retry"] and _old_run_is_back(rig)
    assert not _marker_file(rig).exists()


def test_a_put_off_restore_is_a_distinguishable_result(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    _crashed(rig, monkeypatch)
    _running_since_the_attempt(rig)
    got = launch.restore_set_aside(rig.out, "day-a", NOW, rig.stages, rig.tmp / "jobs", "")
    assert got.startswith("deferred") and "stale" in got, got
    assert launch.restore_set_aside(rig.out, "day-a", NOW, rig.stages, None, "a drive is running").startswith("deferred")  # fmt: skip


def test_a_restore_put_off_for_many_ticks_opens_a_visible_incident_and_keeps_the_marker(tmp_path, monkeypatch):  # fmt: skip
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    _crashed(rig, monkeypatch)
    _running_since_the_attempt(rig)
    rig.preflight_error = RuntimeError("blocked")
    n = heal.HEAL_PREFLIGHT_RETRIES
    for i in range(1, n):
        _tick(rig, NOW + timedelta(days=3 * i), jobs_dir=rig.tmp / "jobs")
        assert not _deferred_incidents(), f"opened after only {i} ticks"
    _tick(rig, NOW + timedelta(days=3 * n), jobs_dir=rig.tmp / "jobs")
    rows = _deferred_incidents()
    assert len(rows) == 1 and "day-a" in json.dumps(rows[0]) and "stale" in json.dumps(rows[0]), rows  # fmt: skip
    assert _marker_file(rig).exists() and len(_set_aside(rig)) == 1 and not _old_run_is_back(rig)
    _end_the_job(rig)
    _tick(rig, NOW + timedelta(days=3 * n + 3), jobs_dir=rig.tmp / "jobs")
    assert _old_run_is_back(rig) and not _deferred_incidents()


def test_a_tick_that_is_not_put_off_resets_the_streak(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    _crashed(rig, monkeypatch)
    _running_since_the_attempt(rig)
    rig.preflight_error = RuntimeError("blocked")
    for i in range(1, 4):
        _tick(rig, NOW + timedelta(days=3 * i), jobs_dir=rig.tmp / "jobs")
    assert launch._read_retry_marker(_marker_file(rig)).get("deferred_ticks") == 3
    _end_the_job(rig)
    _tick(rig, NOW + timedelta(days=30), jobs_dir=rig.tmp / "jobs")
    assert _old_run_is_back(rig)
    assert not launch._read_retry_marker(_marker_file(rig)).get("deferred_ticks")


# ---- 2. a dead "running" record does not put the restore off ----


def test_a_dead_running_record_does_not_put_the_restore_off(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    _crashed(rig, monkeypatch)
    _running_since_the_attempt(rig, live=False)
    rig.preflight_error = RuntimeError("blocked")
    res = _tick(rig, NOW + timedelta(days=2), jobs_dir=rig.tmp / "jobs")
    assert res["replace_retry"]["outcome"] == "bumped"
    assert _old_run_is_back(rig), "a dead record held the restore back"


def test_a_live_running_record_puts_the_restore_off(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    _crashed(rig, monkeypatch)
    _running_since_the_attempt(rig, live=True)
    assert launch._job_running_since(rig.tmp / "jobs", "day-a", 0.0)
    rig.preflight_error = RuntimeError("blocked")
    _tick(rig, NOW + timedelta(days=2), jobs_dir=rig.tmp / "jobs")
    assert not _old_run_is_back(rig)


# ---- 3. stages_root is a required keyword ----


@pytest.mark.parametrize("fn", [launch.run_tick, launch._run_tick, launch.give_up_unclaimed_replace_retries, launch.restore_set_aside])  # fmt: skip
def test_stages_root_has_no_default(fn):
    param = inspect.signature(fn).parameters["stages_root"]
    assert param.default is inspect.Parameter.empty


def test_q1b_transient_through_the_shared_helper_keeps_the_live_stage(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    _die_between_the_renames(rig, monkeypatch)
    rig.preflight_error = RuntimeError("blocked")
    for d in (2, 5, 12):
        res = _tick_at(rig, NOW + timedelta(days=d))
    assert (rig.stages / "day-a" / "src" / "_stages" / "resume.json").exists()
    assert (rig.out / "day-a" / "youtube_uploads.json").exists()
    assert res["replace_retry"]["outcome"] == "bumped"


def test_q1b2_permanent_through_the_shared_helper_never_buries_the_stage(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    _die_between_the_renames(rig, monkeypatch)
    rig.slots = [(NOW + timedelta(days=41), "fri")]
    res = _tick_at(rig, NOW + timedelta(days=40))
    assert (res.get("replace_retry") or {}).get("outcome") == "bumped" or (
        rig.stages / "day-a" / "src" / "_stages" / "resume.json"
    ).exists()
    assert (rig.stages / "day-a" / "src" / "_stages" / "resume.json").exists()


# ---- 4. the launcher's own stdout run_id wins ----


def test_a_stderr_uuid_never_becomes_the_pin(monkeypatch):
    other = "ffffffff-1111-2222-3333-444455556666"
    done = SimpleNamespace(returncode=0, stdout=f"jobrun: detached nanny 1 run_id {RID} record /x\n", stderr=f"warn: old run_id {other} still listed")  # fmt: skip
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: done)
    ok, detail = launch.default_launcher(["x"], "/", {})
    assert ok and launch._LAUNCH_RUN_ID.search(detail).group(1) == RID


def test_a_stdout_run_id_survives_a_long_stderr(monkeypatch):
    done = SimpleNamespace(returncode=0, stdout=f"jobrun: detached nanny 1 run_id {RID} record /x\n", stderr="e" * 500)  # fmt: skip
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: done)
    ok, detail = launch.default_launcher(["x"], "/", {})
    assert ok and launch._LAUNCH_RUN_ID.search(detail).group(1) == RID


# ---- 5. a job that ended does not put it off; a live one created after the first attempt does ----


@pytest.mark.parametrize("state", ["succeeded", "failed", "failed-no-output"])
def test_a_job_that_ended_since_the_attempt_does_not_put_the_restore_off(tmp_path, state):
    rig = _replace_rig(tmp_path, _three_lfs())
    job(rig, "done", "day-a", state, 1000.0, live=True)
    assert not launch._job_running_since(rig.tmp / "jobs", "day-a", 0.0)


def test_a_live_job_between_the_first_and_the_latest_attempt_puts_the_restore_off(tmp_path, monkeypatch):  # fmt: skip
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    _crashed(rig, monkeypatch)
    first = launch._read_retry_marker(_marker_file(rig))["at"]
    later = (datetime.fromisoformat(first) + timedelta(days=2)).isoformat()
    launch.mark_replace_retry(rig.out, "day-a", "again", NOW, region="usa", at=later)
    held = launch._read_retry_marker(_marker_file(rig))
    assert held["earlier_at"] == [first] and held["at"] == later
    job(rig, "mid", "day-a", "running", datetime.fromisoformat(first).timestamp() + 5, live=True)
    got = launch.restore_set_aside(rig.out, "day-a", NOW, rig.stages, rig.tmp / "jobs", "")
    assert got.startswith("deferred"), got


# ---- 4. fixup 12: a put-off only when something is really set aside; counter and streak gaps ----


def _unclaimed(rig, monkeypatch):
    """A crashed replace whose marker names no region and whose manifest row is gone."""
    _crashed(rig, monkeypatch)
    held = json.loads(_marker_file(rig).read_text())
    held.pop("region", None)
    _marker_file(rig).write_text(json.dumps(held))
    rig.rows = [r for r in rig.rows if r.project != "day-a"]


def _due(rig):
    m = launch._read_retry_marker(_marker_file(rig))
    wait = heal._retry_wait_s(int(m["attempts"]))
    return datetime.fromtimestamp(float(m["last_at"]) + wait, NOW.tzinfo) + timedelta(seconds=1)


def test_a_busy_box_with_nothing_set_aside_never_claims_a_set_aside_run(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    rig.preflight_error = RuntimeError("blocked")
    _tick(rig, NOW + timedelta(days=2))  # sets aside, the preflight refuses, refuse() puts it back
    assert launch._read_retry_marker(_marker_file(rig)).get("at") and not _set_aside(rig)
    rig.preflight_error, rig.running = None, True
    for _ in range(heal.HEAL_PREFLIGHT_RETRIES + 1):
        _tick(rig, _due(rig))
    assert not _deferred_incidents(), "no run is set aside, so none is waiting to be put back"
    assert not launch._read_retry_marker(_marker_file(rig)).get("deferred_ticks")
    assert (rig.out / "day-a" / "youtube_uploads.json").exists()


def test_nothing_set_aside_is_nothing_to_restore_not_a_put_off(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    rig.preflight_error = RuntimeError("blocked")
    _tick(
        rig, NOW + timedelta(days=2)
    )  # the marker now names an attempt; refuse() put the run back
    assert launch._read_retry_marker(_marker_file(rig)).get("at") and not _set_aside(rig)
    assert launch.restore_set_aside(rig.out, "day-a", NOW, rig.stages, None, "a drive is running") == ""  # fmt: skip


def test_d1b_the_unclaimed_give_up_put_off_bumps_the_attempts_by_one(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    _unclaimed(rig, monkeypatch)
    _running_since_the_attempt(rig, run_id="live")
    before = launch._read_retry_marker(_marker_file(rig))["attempts"]
    _tick(rig, NOW + timedelta(days=2), jobs_dir=rig.tmp / "jobs")
    after = launch._read_retry_marker(_marker_file(rig))
    assert after["attempts"] == before + 1 and after.get("deferred_ticks") == 1, after


def test_d1c_the_unclaimed_give_up_put_off_counts_and_opens_the_incident(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    _unclaimed(rig, monkeypatch)
    _running_since_the_attempt(rig, run_id="live")
    n = heal.HEAL_PREFLIGHT_RETRIES
    for i in range(1, n):
        _tick(rig, NOW + timedelta(days=2, hours=i), jobs_dir=rig.tmp / "jobs")
        assert launch._read_retry_marker(_marker_file(rig)).get("deferred_ticks") == i
        assert not _deferred_incidents(), f"opened after only {i} ticks"
    _tick(rig, NOW + timedelta(days=2, hours=n), jobs_dir=rig.tmp / "jobs")
    assert launch._read_retry_marker(_marker_file(rig)).get("deferred_ticks") == n
    rows = _deferred_incidents()
    assert len(rows) == 1 and "day-a" in json.dumps(rows[0]), rows
    assert _marker_file(rig).exists() and len(_set_aside(rig)) == 1


def test_l2_a_live_nanny_with_the_worker_pid_still_none_counts_as_running(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    me = live_worker()
    d = rig.tmp / "jobs" / "starting"
    d.mkdir(parents=True)
    worker = {"pid": None, "pgid": None, "start_ticks": "", "boot_id": me["boot_id"]}
    rec = {"schema": "job/1", "run_id": "starting", "name": "gms-backlog-day-a", "created_at": 1000.0,
           "worker": worker, "nanny": me, "result": {"state": "running"}}  # fmt: skip
    (d / "job.json").write_text(json.dumps(rec))
    got = launch._job_running_since(rig.tmp / "jobs", "day-a", 0.0)
    assert got and got["run_id"] == "starting"


def test_r1_a_give_up_streak_then_a_successful_restore_leaves_no_open_incident(tmp_path, monkeypatch):  # fmt: skip
    rig = _replace_rig(tmp_path, _three_lfs())
    _unclaimed(rig, monkeypatch)
    _running_since_the_attempt(rig, run_id="live")
    n = heal.HEAL_PREFLIGHT_RETRIES
    for i in range(1, n + 1):
        _tick(rig, NOW + timedelta(days=2, hours=i), jobs_dir=rig.tmp / "jobs")
    assert len(_deferred_incidents()) == 1
    _end_the_job(rig, run_id="live")
    res = _tick(rig, NOW + timedelta(days=30), jobs_dir=rig.tmp / "jobs")
    assert "day-a" in res["given_up_replace_retry"] and _old_run_is_back(rig)
    assert not _deferred_incidents(), "the restore succeeded; its incident must not stay open"


def test_p2_a_streak_on_the_marker_tick_then_a_launch_leaves_no_open_incident(tmp_path, monkeypatch):  # fmt: skip
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    _crashed(rig, monkeypatch)
    _running_since_the_attempt(rig, run_id="live")
    rig.preflight_error = RuntimeError("blocked")
    for _ in range(heal.HEAL_PREFLIGHT_RETRIES):
        _tick(rig, _due(rig), jobs_dir=rig.tmp / "jobs")
    assert len(_deferred_incidents()) == 1
    _end_the_job(rig, run_id="live")
    rig.preflight_error = None
    _tick(rig, _due(rig), jobs_dir=rig.tmp / "jobs")
    assert rig.launches, "the replace launches once nothing blocks it"
    assert not _deferred_incidents(), (
        "the replace launched; the put-off incident must not stay open"
    )
