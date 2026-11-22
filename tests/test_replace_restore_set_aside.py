"""A set-aside old run is always put back, and only this attempt's job or move counts (fixup 9)."""

import json
import pathlib
from datetime import timedelta

import pytest

from gomoveshift.health import launch
from test_health_backlog_launch import (  # noqa: F401
    NOW,
    STAMP,
    _replace_rig,
    _three_lfs,
    _marker,
    _retry_tick,
    _attempts,
    FAR,
    job,
    _tick_at,
    _named,
    _crash_after_the_renames,
)


def _crashed(rig, monkeypatch):
    """A death after both renames and before the plan write: the old run sits under _superseded."""
    real = launch.replace_scheduled_mod.write_plan
    _crash_after_the_renames(rig, monkeypatch, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        rig.run()
    monkeypatch.setattr(launch.replace_scheduled_mod, "write_plan", real)
    assert not (rig.out / "day-a").exists()


def _old_run_is_back(rig):
    live = rig.out / "day-a"
    return (
        (live / "youtube_uploads.json").exists()
        and (live / "shorts" / "s01.mp4").exists()
        and (rig.stages / "day-a" / "src" / "_stages" / "resume.json").exists()
    )


def _nothing_set_aside(rig):
    return not any((root / launch.SUPERSEDED_DIR).exists() for root in (rig.out, rig.stages))


def _restore_incidents():
    from gomoveshift.health.store import default_data_root, default_incidents_dir, load_all

    rows, _bad = load_all(default_incidents_dir(default_data_root()))
    return [r for r in rows if r.get("code") == "restore_failed"]


# ---- 1. a crash-set-aside run is restored when the plan-only retry is refused ----


def test_a_transient_refusal_on_the_plan_only_path_puts_the_old_run_back(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    _crashed(rig, monkeypatch)
    rig.preflight_error = RuntimeError("disk is on fire")
    for d in (2, 4, 6):
        res = _tick_at(rig, NOW + timedelta(days=d))
        assert res["replace_retry"]["outcome"] == "bumped", res
        assert _old_run_is_back(rig), f"day {d}: the old run is still set aside"
        assert launch._retry_marker(rig.out, "day-a").exists()
    assert _nothing_set_aside(rig) or not list((rig.out / launch.SUPERSEDED_DIR).glob("day-a*"))
    rig.preflight_error = None
    rig.slots = [(NOW + timedelta(days=8, hours=12), "fri")]
    res = _tick_at(rig, NOW + timedelta(days=8))
    assert (res["action"], res["project"]) == ("launched", "day-a"), res.get("replace_retry")
    assert len(_named(rig)) == 1


def test_a_permanent_refusal_on_the_plan_only_path_restores_before_the_marker_clears(
    tmp_path, monkeypatch
):  # noqa: E501
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    _crashed(rig, monkeypatch)
    rig.slots = [(NOW + timedelta(days=41), "fri")]
    res = _tick_at(
        rig, NOW + timedelta(days=40)
    )  # every booked slot has passed: nothing to replace
    assert res["replace_retry"]["outcome"] == "cleared", res["replace_retry"]
    assert _old_run_is_back(rig) and _nothing_set_aside(rig)
    assert not (rig.out / "day-a" / launch.SUPERSEDE_RECORD).exists()


def test_a_refused_plan_only_retry_over_a_live_run_keeps_the_marker_and_both_runs(
    tmp_path, monkeypatch
):  # noqa: E501
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    _crashed(rig, monkeypatch)
    (rig.out / "day-a" / "shorts").mkdir(parents=True)
    (rig.out / "day-a" / "shorts" / "fresh.mp4").write_text("fresh")  # a real run, not the plan
    rig.preflight_error = RuntimeError("blocked")
    res = _tick_at(rig, NOW + timedelta(days=2))
    assert res["replace_retry"]["outcome"] == "bumped", res
    assert launch._retry_marker(rig.out, "day-a").exists()
    assert (rig.out / "day-a" / "shorts" / "fresh.mp4").read_text() == "fresh"
    (aside,) = (rig.out / launch.SUPERSEDED_DIR).glob("day-a-*")
    assert (aside / "youtube_uploads.json").exists()
    assert _restore_incidents(), "no restore_failed incident"


# ---- 2. only a running or succeeded job of THIS launch counts ----


def test_p4_a_failed_job_after_the_stamp_is_not_counted_as_launched(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    n = []

    def launcher(argv, cwd, env):
        rig.launches.append((argv, cwd, env))
        n.append(1)
        job(rig, f"r{len(n)}", "day-a", "failed", NOW.timestamp() + 3, worker_pid=None)
        return False, "jobrun: could not launch: ENOENT"

    rig.launcher = launcher
    _marker(rig)
    res = _retry_tick(rig)
    assert launch._retry_marker(rig.out, "day-a").exists(), res
    assert _old_run_is_back(rig)


def test_p5_a_stale_job_is_not_the_launch_whose_run_id_the_marker_pinned(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig, age=timedelta(days=30))
    t2 = (NOW + timedelta(days=1)).timestamp()
    launch.stamp_replace_launch(rig.out, "day-a", t2)
    launch.pin_replace_launch_run_id(rig.out, "day-a", "this-run")
    job(rig, "attempt1-late", "day-a", "running", t2 + 1)  # NOT this attempt's
    rig.launch_ok = False  # a real replace would otherwise launch and clear the marker itself
    res = _tick_at(rig, NOW + timedelta(days=5), jobs_dir=rig.tmp / "jobs")
    assert res["replace_retry"]["outcome"] == "bumped", res
    launch.stamp_replace_launch(
        rig.out, "day-a", t2
    )  # the bump dropped the stamp: this launch again
    launch.pin_replace_launch_run_id(rig.out, "day-a", "this-run")
    job(rig, "this-run", "day-a", "succeeded", t2 + 2, exit_code=0)
    res = _tick_at(rig, NOW + timedelta(days=9), jobs_dir=rig.tmp / "jobs")
    assert res["replace_retry"]["outcome"] == "cleared", res["replace_retry"]


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("state", ["running", "succeeded", "failed", "failed-no-output"])
def test_a_started_job_counts_in_any_state_jobrun_writes(tmp_path, state, pinned):
    """A launched drive that has since ended (exit 5 at the OST gate is ``failed``) was launched."""
    rig = _replace_rig(tmp_path, _three_lfs())
    job(rig, "r1", "day-a", state, 1000.0, exit_code=0)
    marker = {"launching_at": 999.0, **({"launch_run_id": "r1"} if pinned else {})}
    assert launch._replace_launch_started(marker, rig.tmp / "jobs", "day-a")


def test_a_launch_failure_job_does_not_count_as_started(tmp_path):
    """jobrun's own ``could not launch`` record: ``failed`` and the worker pid never set."""
    rig = _replace_rig(tmp_path, _three_lfs())
    job(rig, "r1", "day-a", "failed", 1000.0, worker_pid=None)
    assert not launch._replace_launch_started({"launching_at": 999.0}, rig.tmp / "jobs", "day-a")


def test_the_launchers_run_id_is_pinned_into_the_marker_as_soon_as_it_returns(
    tmp_path, monkeypatch
):  # noqa: E501
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    seen = []

    def launcher(argv, cwd, env):
        rig.launches.append((argv, cwd, env))
        return (
            True,
            "jobrun: detached 12345 run_id 0a1b2c3d-1111-2222-3333-444455556666 record /x/job.json",
        )  # noqa: E501

    rig.launcher = launcher
    real = launch.clear_replace_retry

    def spy(output_root, project):
        seen.append(launch._read_retry_marker(launch._retry_marker(output_root, project)))
        return real(output_root, project)

    monkeypatch.setattr(launch, "clear_replace_retry", spy)
    res = _retry_tick(rig)
    assert res["action"] == "launched", res
    assert seen and seen[0].get("launch_run_id") == "0a1b2c3d-1111-2222-3333-444455556666", seen


# ---- 3. _unsupersede counts a pair as moved only if this attempt moved it ----


def _stranger(path):
    path.mkdir(parents=True)
    (path / "youtube_uploads.json").write_text('{"2": {"video_id": "OLDER"}}')


def test_p2_an_unrelated_dir_at_dst_never_buries_the_live_run(tmp_path):
    live = tmp_path / "out" / "day-a"
    (live / "shorts").mkdir(parents=True)
    (live / "youtube_uploads.json").write_text('{"2": {"video_id": "LIVE"}}')
    (live / launch.SUPERSEDE_RECORD).write_text("{}")  # this attempt's record, never moved
    older = tmp_path / "out" / launch.SUPERSEDED_DIR / f"day-a-{STAMP}"
    _stranger(older)
    failed = launch._unsupersede([{"from": str(live), "to": str(older)}], live, NOW)
    assert not failed, failed
    assert json.loads((live / "youtube_uploads.json").read_text())["2"]["video_id"] == "LIVE"
    assert json.loads((older / "youtube_uploads.json").read_text())["2"]["video_id"] == "OLDER"
    assert not (live / launch.SUPERSEDE_RECORD).exists()


def test_p2_the_same_holds_with_no_record_in_src(tmp_path):
    live = tmp_path / "out" / "day-a"
    (live / "shorts").mkdir(parents=True)
    (live / "youtube_uploads.json").write_text('{"2": {"video_id": "LIVE"}}')
    older = tmp_path / "out" / launch.SUPERSEDED_DIR / f"day-a-{STAMP}"
    _stranger(older)
    launch._unsupersede([{"from": str(live), "to": str(older)}], live, NOW)
    assert json.loads((live / "youtube_uploads.json").read_text())["2"]["video_id"] == "LIVE"


def test_p2b_a_same_second_dst_made_after_the_plan_never_buries_the_live_run(tmp_path, monkeypatch):  # noqa: E501
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    real_plan = launch._supersede_plan

    def plan_then_collide(*a, **k):
        moves, why = real_plan(*a, **k)
        for m in moves:
            if m["from"] == str(rig.out / "day-a"):
                d = pathlib.Path(m["to"])
                d.mkdir(parents=True)
                (d / "youtube_uploads.json").write_text('{"9": {"video_id": "STRANGER"}}')
        return moves, why

    monkeypatch.setattr(launch, "_supersede_plan", plan_then_collide)
    _retry_tick(rig)
    live = rig.out / "day-a" / "youtube_uploads.json"
    assert live.exists() and "STRANGER" not in live.read_text()
    assert not (rig.out / "day-a" / launch.SUPERSEDE_RECORD).exists()


def test_a_pair_this_attempt_did_move_is_still_put_back(tmp_path):
    """The record in dst names this attempt (``at``, project, moves): that pair did move."""
    old = tmp_path / "out" / "day-a"
    (old / "shorts").mkdir(parents=True)
    (old / "youtube_uploads.json").write_text('{"1": {"video_id": "OLD"}}')
    aside = tmp_path / "out" / launch.SUPERSEDED_DIR / f"day-a-{STAMP}"
    moves = [{"from": str(old), "to": str(aside)}]
    launch._supersede("day-a", "why", moves, NOW)
    old.mkdir()  # the fresh dir the failed attempt left
    assert not launch._unsupersede(moves, old, NOW)
    assert (old / "youtube_uploads.json").exists() and not aside.exists()
    assert not (old / launch.SUPERSEDE_RECORD).exists()


# ---- 4. a stray src record is removed on give-up and on permanent refusal ----


def _stray(rig):
    from gomoveshift.pipeline.cleanup import _superseded_by_record

    live = rig.out / "day-a"
    record = {"schema": "backlog-supersede/2", "project": "day-a", "moves": [], "at": "x"}
    (live / launch.SUPERSEDE_RECORD).write_text(json.dumps(record))
    assert _superseded_by_record(live)
    return live, _superseded_by_record


def test_a_stray_record_is_removed_when_a_marker_is_given_up_as_unclaimed(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    live, by_record = _stray(rig)
    launch.mark_replace_retry(rig.out, "day-a", "why", NOW - timedelta(days=3))
    res = _tick_at(rig, NOW)
    assert res["given_up_replace_retry"] == ["day-a"], res
    assert not by_record(live) and (live / "youtube_uploads.json").exists()


def test_a_stray_record_is_removed_on_a_permanent_refusal(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    live, by_record = _stray(rig)
    rig.slots = [(NOW + timedelta(days=41), "fri")]
    res = _tick_at(
        rig, NOW + timedelta(days=40)
    )  # every booked slot has passed: nothing to replace
    assert res["replace_retry"]["outcome"] == "cleared", res["replace_retry"]
    assert not by_record(live) and (live / "youtube_uploads.json").exists()


# ---- 5. test gaps ----


def test_the_not_moved_branch_removes_the_empty_superseded_dir(tmp_path):
    live = tmp_path / "out" / "day-a"
    live.mkdir(parents=True)
    (live / launch.SUPERSEDE_RECORD).write_text("{}")
    dst = tmp_path / "out" / launch.SUPERSEDED_DIR / f"day-a-{STAMP}"
    dst.parent.mkdir()  # made by this attempt, the rename never happened
    assert not launch._unsupersede([{"from": str(live), "to": str(dst)}], live, NOW)
    assert (
        not dst.parent.exists() and live.is_dir() and not (live / launch.SUPERSEDE_RECORD).exists()
    )  # noqa: E501
