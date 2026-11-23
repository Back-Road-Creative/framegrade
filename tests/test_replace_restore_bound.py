"""An ended launch is never replaced again; a restore is bound to its own attempt (fixup 10)."""

import json
import subprocess
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.health import launch
from test_health_backlog_launch import (  # noqa: F401
    NOW,
    Rig,
    _replace_rig,
    _three_lfs,
    _marker,
    _retry_tick,
    _tick_at,
    _named,
    _crash_after_the_renames,
    _crash_on_rename,
    job,
)
from test_replace_restore_set_aside import (  # noqa: F401
    _crashed,
    _old_run_is_back,
    _nothing_set_aside,
    _restore_incidents,
)

RID = "0a1b2c3d-1111-2222-3333-444455556666"


def _tick(rig, when, jobs_dir=None):
    def run(sup, rep):
        rig.supersede, rig.replace_scheduled = sup, rep
        return launch.run_backlog(**{**rig.kwargs(), "now": when})

    return launch.run_tick(output_root=rig.out, now=when, region="usa", regions={}, dry_run=False, supersede=None, replace=False, run=run, jobs_dir=jobs_dir, stages_root=rig.stages)  # fmt: skip


def _marker_file(rig):
    return launch._retry_marker(rig.out, "day-a")


def _set_aside(rig):
    return sorted((rig.out / launch.SUPERSEDED_DIR).glob("day-a-*"))


# ---- B1: a launched job that has since ended still counts as launched ----


def _launch_and_die(rig, monkeypatch, *, pin, how="death"):
    """The launcher returns ok (the real jobrun line, or nothing) and its job runs; the tick then
    dies (or raises) before the marker is cleared: the stamp branch's reason to exist."""

    def launcher(argv, cwd, env):
        rig.launches.append((argv, cwd, env))
        job(rig, RID, "day-a", "running", NOW.timestamp() + 3)
        return True, (f"jobrun: detached nanny 4242 run_id {RID} record /x/job.json" if pin else "")

    rig.launcher = launcher
    _marker(rig)
    name, boom = ("clear_replace_retry", KeyboardInterrupt()) if how == "death" else ("_record_margin_kept", RuntimeError("store down"))  # fmt: skip
    real = getattr(launch, name)
    monkeypatch.setattr(launch, name, lambda *a, **k: (_ for _ in ()).throw(boom))
    if how == "death":
        with pytest.raises(KeyboardInterrupt):
            _retry_tick(rig)
    else:
        assert _retry_tick(rig)["action"] == "launched"
    monkeypatch.setattr(launch, name, real)
    held = launch._read_retry_marker(_marker_file(rig))
    assert held.get("launching_at") and (held.get("launch_run_id") == RID) == pin, held


def _ended(rig, state, exit_code):
    p = rig.tmp / "jobs" / RID / "job.json"
    rec = json.loads(p.read_text())
    rec["result"] = {"state": state, "exit_code": exit_code, "ended_at": NOW.timestamp() + 7200}
    p.write_text(json.dumps(rec))


@pytest.mark.parametrize("pin", [True, False])
@pytest.mark.parametrize("ended", [("failed", 5), ("failed", 1), ("failed-no-output", 0)])
def test_a_launched_job_that_has_since_ended_is_never_replaced_again(
    tmp_path, monkeypatch, pin, ended
):  # noqa: E501
    rig = _replace_rig(tmp_path, _three_lfs())
    _launch_and_die(rig, monkeypatch, pin=pin)
    _ended(rig, *ended)
    res = _tick_at(rig, NOW + timedelta(days=1), jobs_dir=rig.tmp / "jobs")
    assert len(_named(rig)) == 1, "the replace ran a second time"
    assert res["replace_retry"]["outcome"] == "cleared", res["replace_retry"]


def test_after_a_raise_rather_than_a_death_an_ended_launch_is_the_same(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    _launch_and_die(rig, monkeypatch, pin=True, how="raise")
    _ended(rig, "failed", 5)
    _tick_at(rig, NOW + timedelta(days=1), jobs_dir=rig.tmp / "jobs")
    assert len(_named(rig)) == 1, "the replace ran a second time"


def test_a_running_launched_job_is_cleared(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    _launch_and_die(rig, monkeypatch, pin=True)
    res = _tick_at(rig, NOW + timedelta(days=1), jobs_dir=rig.tmp / "jobs")
    assert len(_named(rig)) == 1 and res["replace_retry"]["outcome"] == "cleared"


@pytest.mark.parametrize("what", ["another-run", "before-the-stamp", "another-project"])
def test_a_job_that_is_not_this_launch_never_counts(tmp_path, what):
    rig = _replace_rig(tmp_path, _three_lfs())
    marker = {"launching_at": 999.0, **({"launch_run_id": "mine"} if what == "another-run" else {})}  # fmt: skip
    job(rig, "r1", "day-b" if what == "another-project" else "day-a", "succeeded", 998.0 if what == "before-the-stamp" else 1000.0, exit_code=0)  # fmt: skip
    assert not launch._replace_launch_started(marker, rig.tmp / "jobs", "day-a")


# ---- B2: a restore is bound to the attempt the marker belongs to ----


def test_the_marker_records_the_attempt_that_set_a_run_aside_and_a_plan_only_retry_keeps_it(
    tmp_path, monkeypatch
):  # noqa: E501
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    _crashed(rig, monkeypatch)
    (aside,) = _set_aside(rig)
    at = json.loads((aside / launch.SUPERSEDE_RECORD).read_text())["at"]
    assert launch._read_retry_marker(_marker_file(rig)).get("at") == at
    rig.preflight_error = RuntimeError("blocked")
    _tick(rig, NOW + timedelta(days=2))  # plan-only: sets nothing aside
    assert launch._read_retry_marker(_marker_file(rig)).get("at") == at


def test_a_fresh_run_is_never_swapped_back_for_an_earlier_replaces_set_aside(tmp_path):
    """R0 was set aside by a COMPLETED replace; its fresh run has only the plan so far."""
    rig = _replace_rig(tmp_path, _three_lfs())
    assert rig.run()["action"] == "launched"
    assert not _marker_file(rig).exists()
    _marker(rig)  # a later attempt's marker, which has set nothing aside
    rig.running = True
    _tick(rig, NOW + timedelta(days=2))
    assert not (rig.out / "day-a" / "youtube_uploads.json").exists(), "R0 swapped back"
    assert len(_set_aside(rig)) == 1 and (_set_aside(rig)[0] / "youtube_uploads.json").exists()


def test_a_marker_with_no_attempt_restores_nothing(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    _crashed(rig, monkeypatch)
    held = json.loads(_marker_file(rig).read_text())
    held.pop("at"), held.pop("earlier_at", None)
    _marker_file(rig).write_text(json.dumps(held))
    rig.preflight_error = RuntimeError("blocked")
    res = _tick(rig, NOW + timedelta(days=2))
    assert res["replace_retry"]["outcome"] == "bumped", res
    assert not _old_run_is_back(rig) and len(_set_aside(rig)) == 1


def test_a_running_job_since_the_attempt_defers_the_restore(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    _crashed(rig, monkeypatch)
    job(rig, "fresh", "day-a", "running", NOW.timestamp() + 60, live=True)
    rig.preflight_error = RuntimeError("blocked")
    res = _tick(rig, NOW + timedelta(days=2), jobs_dir=rig.tmp / "jobs")
    assert res["replace_retry"]["outcome"] == "bumped", res
    assert not _old_run_is_back(rig) and len(_set_aside(rig)) == 1 and _marker_file(rig).exists()


def test_a_refusal_that_a_drive_is_running_defers_the_restore(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    _crashed(rig, monkeypatch)
    rig.running = True
    res = _tick(rig, NOW + timedelta(days=2))
    assert res["replace_retry"]["outcome"] == "bumped", res
    assert not _old_run_is_back(rig) and len(_set_aside(rig)) == 1


def test_a_later_attempts_marker_still_restores_an_earlier_attempts_set_aside(
    tmp_path, monkeypatch
):  # noqa: E501
    """Death between the stage rename and the out rename, then a retry that sets the out dir aside
    (its own ``at``) and is refused: the stage dir of the FIRST attempt is still put back."""
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    _die_between_the_renames(rig, monkeypatch)
    rig.preflight_error = RuntimeError("blocked")
    _tick(rig, NOW + timedelta(days=2))
    assert (rig.stages / "day-a" / "src" / "_stages" / "resume.json").exists()
    assert (rig.out / "day-a" / "youtube_uploads.json").exists()


# ---- M1: a death between the stage rename and the out rename never leaves the stage buried ----


def _die_between_the_renames(rig, monkeypatch):
    real, _aside = _crash_on_rename(monkeypatch, rig, after=False)
    with pytest.raises(KeyboardInterrupt):
        rig.run()
    monkeypatch.setattr(Path, "rename", real)
    assert (rig.out / "day-a" / "youtube_uploads.json").exists()
    assert not (rig.stages / "day-a").exists(), "precondition: the stage pair moved"


def test_a_transient_refusal_after_a_death_between_the_renames_puts_the_stage_back(
    tmp_path, monkeypatch
):  # noqa: E501
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    _die_between_the_renames(rig, monkeypatch)
    rig.preflight_error = RuntimeError("blocked")
    for d in (2, 5, 12):
        res = _tick(rig, NOW + timedelta(days=d))
    assert res["replace_retry"]["outcome"] == "bumped", res
    assert (rig.stages / "day-a" / "src" / "_stages" / "resume.json").exists(), "stage buried"
    assert not (rig.stages / launch.SUPERSEDED_DIR).exists()


def test_a_permanent_refusal_after_a_death_between_the_renames_puts_the_stage_back(
    tmp_path, monkeypatch
):  # noqa: E501
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    _die_between_the_renames(rig, monkeypatch)
    rig.slots = [(NOW + timedelta(days=41), "fri")]
    res = _tick(rig, NOW + timedelta(days=40))
    assert res["replace_retry"]["outcome"] == "cleared", res["replace_retry"]
    assert (rig.stages / "day-a" / "src" / "_stages" / "resume.json").exists(), "stage buried"
    assert not (rig.stages / "day-a" / launch.SUPERSEDE_RECORD).exists()


# ---- M2: the unclaimed give-up restores first ----


def _crashed_unclaimed(rig, monkeypatch):
    _crashed(rig, monkeypatch)
    held = json.loads(_marker_file(rig).read_text())
    held.pop("region", None)  # a legacy marker no region claims
    _marker_file(rig).write_text(json.dumps(held))
    rig.rows = [r for r in rig.rows if r.project != "day-a"]


def test_the_unclaimed_give_up_puts_the_set_aside_run_back_first(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    _crashed_unclaimed(rig, monkeypatch)
    res = _tick(rig, NOW + timedelta(days=2))
    assert res["given_up_replace_retry"] == ["day-a"], res
    assert _old_run_is_back(rig) and _nothing_set_aside(rig) and not _marker_file(rig).exists()


def test_the_unclaimed_give_up_keeps_the_marker_when_the_restore_fails(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    _crashed_unclaimed(rig, monkeypatch)
    (rig.out / "day-a" / "shorts").mkdir(parents=True)
    (rig.out / "day-a" / "shorts" / "fresh.mp4").write_text("fresh")
    res = _tick(rig, NOW + timedelta(days=2))
    assert "given_up_replace_retry" not in res and _marker_file(rig).exists(), res
    assert len(_set_aside(rig)) == 1 and _restore_incidents()


# ---- M3: pins, newest-first, the uploads guard, a failed restore on a permanent refusal ----


def test_a_new_stamp_and_an_unstamp_drop_the_old_pin(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    launch.stamp_replace_launch(rig.out, "day-a", 5.0)
    launch.pin_replace_launch_run_id(rig.out, "day-a", RID)
    assert launch._read_retry_marker(_marker_file(rig))["launch_run_id"] == RID
    launch.stamp_replace_launch(rig.out, "day-a", 6.0)  # a new launch: its id is not known yet
    held = launch._read_retry_marker(_marker_file(rig))
    assert held["launching_at"] == 6.0 and "launch_run_id" not in held
    launch.pin_replace_launch_run_id(rig.out, "day-a", RID)
    launch.stamp_replace_launch(rig.out, "day-a", None)
    assert not {"launching_at", "launch_run_id"} & set(launch._read_retry_marker(_marker_file(rig)))  # fmt: skip


def test_a_pin_needs_a_stamp(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    launch.pin_replace_launch_run_id(rig.out, "day-a", RID)  # no marker at all
    assert not _marker_file(rig).exists()
    _marker(rig)
    launch.pin_replace_launch_run_id(rig.out, "day-a", RID)  # a marker, but no stamp
    assert "launch_run_id" not in launch._read_retry_marker(_marker_file(rig))


def test_a_failed_launch_is_never_pinned(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    rig.launcher = lambda argv, cwd, env: (False, f"jobrun: could not launch run_id {RID} record /x")  # fmt: skip
    _retry_tick(rig)
    assert "launch_run_id" not in launch._read_retry_marker(_marker_file(rig))


def test_an_ordinary_launch_never_pins_a_run_id(tmp_path):
    rig = Rig(tmp_path)
    rig.launcher = lambda argv, cwd, env: (True, f"jobrun: detached nanny 1 run_id {RID} record /x")
    for p in ("day-a", "day-b"):
        launch.mark_replace_retry(rig.out, p, "why", NOW, region="usa")
        launch.stamp_replace_launch(rig.out, p, 1.0)
    res = rig.run()
    assert res["action"] == "launched", res
    held = launch._read_retry_marker(launch._retry_marker(rig.out, res["project"]))
    assert held and "launch_run_id" not in held


def test_the_newest_set_aside_run_is_the_one_a_retry_plans_from(tmp_path):
    out = tmp_path / "out" / "day-a"
    for stamp, vid in (("20260101T000000Z", "OLD"), ("20260201T000000Z", "NEW")):
        d = out.parent / launch.SUPERSEDED_DIR / f"day-a-{stamp}"
        d.mkdir(parents=True)
        (d / "youtube_uploads.json").write_text(json.dumps({"1": {"video_id": vid}}))
        (d / launch.SUPERSEDE_RECORD).write_text(json.dumps({"schema": "backlog-supersede/2", "project": "day-a", "at": stamp, "moves": []}))  # fmt: skip
    assert launch._superseded_uploads("day-a", out).parent.name == "day-a-20260201T000000Z"


def test_a_finished_run_holding_the_name_is_never_restored_over(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    _crashed(rig, monkeypatch)
    (rig.out / "day-a").mkdir()
    (rig.out / "day-a" / "youtube_uploads.json").write_text('{"2": {"video_id": "NEW"}}')
    assert launch.restore_set_aside(rig.out, "day-a", NOW, rig.stages) == ""
    assert len(_set_aside(rig)) == 1


def test_a_permanent_refusal_whose_restore_failed_keeps_the_marker(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    _crashed(rig, monkeypatch)
    (rig.out / "day-a" / "shorts").mkdir(parents=True)
    (rig.out / "day-a" / "shorts" / "fresh.mp4").write_text("fresh")
    rig.slots = [(NOW + timedelta(days=41), "fri")]  # permanent: nothing to replace
    res = _tick(rig, NOW + timedelta(days=40))
    assert res["replace_retry"]["outcome"] == "bumped", res["replace_retry"]
    assert _marker_file(rig).exists() and _restore_incidents()


def _moved_record(tmp_path, **over):
    live, dst = tmp_path / "out" / "day-a", tmp_path / "out" / "_superseded" / "day-a-x"
    move = {"from": str(live), "to": str(dst)}
    dst.mkdir(parents=True)
    rec = {"at": "A", "project": "day-a", "moves": [move], **over}
    (dst / launch.SUPERSEDE_RECORD).write_text(json.dumps(rec))
    return move


def test_moved_by_accepts_the_attempts_own_record(tmp_path):
    assert launch._moved_by(_moved_record(tmp_path), "A", "day-a")


@pytest.mark.parametrize("what", ["at", "project", "move"])
def test_moved_by_rejects_a_record_of_another_attempt_project_or_move(tmp_path, what):
    move = _moved_record(tmp_path, **({"at": "B"} if what == "at" else {"project": "day-b"} if what == "project" else {"moves": [{"from": "x", "to": "y"}]}))  # fmt: skip
    assert not launch._moved_by(move, "A", "day-a")


@pytest.mark.parametrize("line", [f"my_run_id {RID}", f"prerun_id {RID}", "run_id abcdef123456", "run_id 12345", "run_id"])  # fmt: skip
def test_the_run_id_pattern_rejects_a_longer_word_and_a_non_uuid_token(line):
    assert launch._LAUNCH_RUN_ID.search(line) is None


def test_the_run_id_pattern_reads_the_real_jobrun_line():
    assert launch._LAUNCH_RUN_ID.search(f"jobrun: detached nanny 9 run_id {RID} record /x").group(1) == RID  # fmt: skip


# ---- m2: text on stderr never drops the pin ----


@pytest.mark.parametrize("stderr", ["", "warning: something on stderr"])
def test_the_launcher_keeps_the_run_id_whatever_stderr_holds(monkeypatch, stderr):
    line = f"jobrun: detached nanny 4242 run_id {RID} record {'/x' * 100}/{RID}/job.json"
    done = SimpleNamespace(returncode=0, stdout=line + "\n", stderr=stderr)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: done)
    ok, detail = launch.default_launcher(["jobrun"], "/", {})
    assert ok and launch._LAUNCH_RUN_ID.search(detail).group(1) == RID, detail


# ---- m4: a restore never raises out of the tick ----


def test_a_restore_that_hits_an_oserror_is_a_failure_the_tick_survives(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    _crashed(rig, monkeypatch)
    monkeypatch.setattr(launch, "_unsupersede", lambda *a, **k: (_ for _ in ()).throw(OSError("EIO")))  # fmt: skip
    assert "EIO" in launch.restore_set_aside(rig.out, "day-a", NOW, rig.stages)
    rig.preflight_error = RuntimeError("blocked")
    res = _tick(rig, NOW + timedelta(days=2))
    assert res["replace_retry"]["outcome"] == "bumped" and _marker_file(rig).exists()
    assert _restore_incidents()


def test_a_restore_over_a_malformed_record_is_a_failure_the_tick_survives(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    _crashed(rig, monkeypatch)
    rec = _set_aside(rig)[0] / launch.SUPERSEDE_RECORD
    held = json.loads(rec.read_text())
    held["moves"] = [{"from": ["x"], "to": 5}, *held["moves"]]
    rec.write_text(json.dumps(held))
    got = launch.restore_set_aside(rig.out, "day-a", NOW, rig.stages)
    assert got and not got.startswith(launch.DEFERRED), got
    rig.slots = [(NOW + timedelta(days=41), "fri")]
    res = _tick(rig, NOW + timedelta(days=40))
    assert res["replace_retry"]["outcome"] == "bumped" and _marker_file(rig).exists()
    assert _restore_incidents()
