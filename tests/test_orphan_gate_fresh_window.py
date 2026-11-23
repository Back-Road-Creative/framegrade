"""The orphan gate's fresh window (rr45): it starts when the dead publish job ENDED, it counts every
dead state the backlog retries, a hand ``health publish`` converges like the tick, and a title
another drive derives too is never a silent pass."""

from __future__ import annotations

import json
import os
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gomoveshift.config import get_config
from gomoveshift.health import launch, publish
from gomoveshift.health.heal import _STALL_S
from gomoveshift.publishing.live_catalog import live_catalog_snapshot_path
from tests.test_orphan_upload_incident import (
    ORPHAN,
    catalog,
    iso,
    rig_with_state,
    rows,
    run_with,
    state,
)
from tests.test_publish_command import CAD, NOW

T0 = 1_800_000_000.0


def _inc():
    return Path(os.environ["GMS_HEALTH_INCIDENTS_DIR"])


def job(rig, *, created_at, state, exit_code=None, ended_at=None, name="gms-publish-day-a"):
    path = rig.jobs / f"{name}-{int(created_at)}" / "job.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "job/1", "name": name, "created_at": created_at,
                                "result": {"state": state, "exit_code": exit_code, "ended_at": ended_at}}))  # fmt: skip


def refresher(calls):
    def refresh():
        calls.append(1)
        catalog(
            {"vidKerry": ORPHAN}, fetched_at=iso(1_900_000_000.0)
        )  # YouTube DOES hold the orphan

    return refresh


def failed_drive(tmp_path):
    rig = rig_with_state(tmp_path)
    rig.store.set_status("day-a", "failed")
    return rig


def refused_after_refresh(rig, calls):
    res = run_with(rig, refresher(calls))
    assert calls == [1] and res["action"] == "refused" and "vidKerry" in res["reason"], res


# ── MAJOR A: every dead state is a failed publish, newest wins ───────────────


@pytest.mark.parametrize("dead", ["cancelled", "stalled", "killed", "failed"])
def test_every_dead_job_state_needs_a_snapshot_fetched_after_it(tmp_path, dead):
    rig = failed_drive(tmp_path)
    job(rig, created_at=T0, state=dead, ended_at=T0 + 60)
    catalog({"v": {"title": "Cork in 4K"}}, fetched_at=iso(T0 - 100))  # predates the dead job
    refused_after_refresh(rig, [])


def test_the_newest_dead_job_sets_the_window_not_an_older_failure(tmp_path):
    rig = failed_drive(tmp_path)
    job(rig, created_at=T0, state="failed", exit_code=1, ended_at=T0 + 10)
    job(rig, created_at=T0 + 5e7, state="stalled", ended_at=T0 + 5e7 + 10)  # newest, dead
    catalog({"v": {"title": "Cork in 4K"}}, fetched_at=iso(T0 + 2e7))  # between the two
    refused_after_refresh(rig, [])


def test_the_gate_and_the_backlog_share_one_dead_publish_rule():
    for res in ({"state": "cancelled"}, {"state": "failed", "exit_code": 1}, {"state": "killed"}):
        assert launch.dead_publish_result(res)
    for res in ({"state": "succeeded", "exit_code": 0}, {"state": "running"}, {}, None):
        assert not launch.dead_publish_result(res)


# ── MAJOR B: the window starts when the job ended ────────────────────────────


def test_a_snapshot_fetched_mid_job_does_not_count_as_fresh(tmp_path):
    rig = failed_drive(tmp_path)
    job(rig, created_at=T0, state="failed", exit_code=1, ended_at=T0 + 3600)
    catalog({"v": {"title": "Cork in 4K"}}, fetched_at=iso(T0 + 300))  # 5 min in, pre-insert
    refused_after_refresh(rig, [])


def test_a_snapshot_fetched_after_the_job_ended_needs_no_refresh(tmp_path):
    rig = failed_drive(tmp_path)
    job(rig, created_at=T0, state="failed", exit_code=1, ended_at=T0 + 3600)
    catalog({"v": {"title": "Cork in 4K"}}, fetched_at=iso(T0 + 3700))
    calls: list = []
    assert run_with(rig, refresher(calls))["action"] == "launched" and calls == []


def test_a_dead_job_with_no_end_stamp_forces_a_refresh_now(tmp_path):
    rig = failed_drive(tmp_path)
    job(rig, created_at=T0, state="cancelled", ended_at=None)
    catalog({"v": {"title": "Cork in 4K"}}, fetched_at=iso(NOW.timestamp() - 3600))
    refused_after_refresh(rig, [])


def test_a_failed_drive_with_no_job_record_refuses_an_ancient_snapshot(tmp_path):
    rig = failed_drive(tmp_path)
    catalog({"v": {"title": "Cork in 4K"}}, fetched_at="2001-01-01T00:00:00+00:00")
    refused_after_refresh(rig, [])


def test_a_failed_drive_with_no_job_record_and_an_unstamped_snapshot_refreshes(tmp_path):
    rig = failed_drive(tmp_path)
    catalog({"v": {"title": "Cork in 4K"}})  # no fetched_at
    refused_after_refresh(rig, [])


# ── wiring: a hand ``health publish`` and the tick both refresh ──────────────


def test_health_publish_passes_a_catalog_refresher(tmp_path, monkeypatch):
    from gomoveshift.cli.handlers import health as handler

    sentinel = object()
    seen: dict = {}
    monkeypatch.setattr(publish, "open_schedule_store", lambda config: (object(), object()))
    monkeypatch.setattr(publish, "catalog_refresher", lambda config: sentinel)
    monkeypatch.setattr(publish, "run_publish", lambda **kw: seen.update(kw) or {"action": "noop", "reason": "x", "steps": []})  # fmt: skip
    args = Namespace(project="day-a", region="usa", stages_root=tmp_path, code_root=None,
                     dry_run=False, lead_hours=None, expect_sha=None, json=False)  # fmt: skip
    handler._cmd_publish(args, get_config(), tmp_path, tmp_path)
    assert seen["refresh_catalog"] is sentinel


def _two_failed_drives(rig):
    rig.audit("day-b", status="auditing")
    state(rig, "day-b", label="Cork", date="2025-09-20")
    rig.store.set_status("day-b", "failed")
    for name in ("gms-publish-day-a", "gms-publish-day-b"):
        job(rig, created_at=T0, state="failed", exit_code=1, ended_at=T0 + 60, name=name)
    catalog({"v": {"title": "Elsewhere in 4K"}}, fetched_at=iso(T0 - 100))


def _tick_publisher(rig, monkeypatch, refresh):
    monkeypatch.setattr(publish, "open_schedule_store", lambda config: (rig.store, CAD))
    monkeypatch.setattr(publish, "catalog_refresher", lambda config: refresh)
    return publish.backlog_publisher(
        get_config(), region="usa", stages_root=rig.stages, code_root=rig.tmp, jobs_dir=rig.jobs,
        now=NOW, drive_running=lambda jobs: False,
    )  # fmt: skip


def test_the_backlog_publisher_passes_a_refresher(tmp_path, monkeypatch):
    rig = failed_drive(tmp_path)
    job(rig, created_at=T0, state="failed", exit_code=1, ended_at=T0 + 60)
    catalog({"v": {"title": "Cork in 4K"}}, fetched_at=iso(T0 - 100))
    calls: list = []
    res = _tick_publisher(rig, monkeypatch, refresher(calls))("day-a", False)
    assert calls == [1] and res["action"] == "refused" and "vidKerry" in res["reason"], res


def test_a_failing_refresh_is_tried_once_per_tick_not_once_per_failed_drive(tmp_path, monkeypatch):
    rig = failed_drive(tmp_path)
    _two_failed_drives(rig)
    calls: list = []

    def boom():
        calls.append(1)
        raise RuntimeError("quotaExceeded")

    run = _tick_publisher(rig, monkeypatch, boom)
    first, second = run("day-a", False), run("day-b", False)
    assert len(calls) == 1
    for res in (first, second):  # both still wait, with the failure visible
        assert res["action"] == "refused" and "quotaExceeded" in res["reason"], res


def test_the_next_tick_tries_the_refresh_again(tmp_path, monkeypatch):
    rig = failed_drive(tmp_path)
    _two_failed_drives(rig)
    calls: list = []

    def boom():
        calls.append(1)
        raise RuntimeError("quotaExceeded")

    for _tick in range(2):
        _tick_publisher(rig, monkeypatch, boom)("day-a", False)
    assert len(calls) == 2


# ── MINOR 1: the refresh writes nothing to stdout ────────────────────────────


def test_the_refresher_keeps_stdout_clean_for_backlog_json(tmp_path, monkeypatch, capsys):
    from gomoveshift.cli.handlers import catalog_fetch as cf

    monkeypatch.setattr(cf, "_build_readonly_youtube_service", lambda: object())
    monkeypatch.setattr(cf, "enumerate_uploads", lambda svc, known_ids: ["vidKerry"])
    monkeypatch.setattr(cf, "fetch_catalog", lambda svc, ids, known_ids: {"videos": {"vidKerry": ORPHAN}, "enumerated": 1, "listed": 1})  # fmt: skip
    monkeypatch.setattr(cf, "queue_youtube_upload_ids", lambda storage: [])
    publish.catalog_refresher(get_config())()
    print(json.dumps({"attempts": []}))  # what `health backlog --json` prints next
    out = capsys.readouterr().out
    assert json.loads(out) == {"attempts": []}, out
    assert "vidKerry" in live_catalog_snapshot_path(get_config().seo.storage_path).read_text()


# ── MINOR 3: a missing ledger refuses AND records the incident ───────────────


def _drop_ledger():
    (Path(get_config().seo.storage_path) / ".content_ledger.json").unlink()


def test_a_missing_ledger_refusal_says_so_and_opens_one_deduped_incident(tmp_path):
    rig = rig_with_state(tmp_path)
    _drop_ledger()
    catalog({"vidKerry": ORPHAN})
    res = rig.run()
    assert res["action"] == "refused" and "missing or unreadable" in res["reason"], res
    assert "vidKerry" in res["reason"]
    (row,) = rows(_inc())
    assert row["resolved_at"] is None and row["lf"] == 1
    rig.run()
    assert len(rows(_inc())) == 1


def test_a_missing_ledger_with_no_candidate_launches(tmp_path):
    rig = rig_with_state(tmp_path)
    _drop_ledger()
    catalog({"v": {"title": "Cork in 4K"}})
    assert rig.run()["action"] == "launched"


def test_a_missing_ledger_dry_run_records_nothing(tmp_path):
    rig = rig_with_state(tmp_path)
    _drop_ledger()
    catalog({"vidKerry": ORPHAN})
    assert rig.run(dry_run=True)["action"] == "refused" and rows(_inc()) == []


# ── MINOR 4: a title another drive derives too is never a silent pass ────────


def _other_ireland(rig, date="2025-09-20"):
    rig.audit("day-b", status="auditing")
    state(rig, "day-b", label="Ireland", date=date)


def test_a_bare_title_orphan_with_a_phase_d_sidecar_is_caught_despite_another_drive(tmp_path):
    rig = rig_with_state(tmp_path)
    _other_ireland(rig)
    out = rig.out / "day-a"
    out.mkdir(parents=True, exist_ok=True)
    (out / "day-a_lf01_youtube.json").write_text(json.dumps({"title": "Ireland in 4K"}))
    catalog({"vidKerry": {**ORPHAN, "title": "Ireland in 4K"}})
    res = rig.run()
    assert res["action"] == "refused" and "vidKerry" in res["reason"], res


def test_the_dated_title_is_caught_when_another_drive_shares_the_bare_label(tmp_path):
    rig = rig_with_state(tmp_path)
    _other_ireland(rig)
    catalog({"vidKerry": ORPHAN})
    res = rig.run()
    assert res["action"] == "refused" and "vidKerry" in res["reason"], res


def test_a_shared_bare_title_orphan_refuses_naming_both_drives(tmp_path):
    rig = rig_with_state(tmp_path)
    _other_ireland(rig)
    catalog({"vidKerry": {**ORPHAN, "title": "Ireland in 4K"}})
    res = rig.run()
    assert res["action"] == "refused" and not rig.launches, res
    assert "vidKerry" in res["reason"] and "day-a" in res["reason"] and "day-b" in res["reason"]
    assert len(rows(_inc())) == 1


def test_every_form_shared_with_another_drive_still_gates_this_one(tmp_path):
    rig = rig_with_state(tmp_path)
    _other_ireland(rig, date="2025-09-16")  # bare AND dated forms are the other drive's too
    catalog({"vidKerry": ORPHAN})
    res = rig.run()
    assert res["action"] == "refused" and not rig.launches, res
    assert "vidKerry" in res["reason"] and "day-a" in res["reason"] and "day-b" in res["reason"]


def test_a_shared_title_video_a_receipt_names_is_not_an_orphan(tmp_path):
    from gomoveshift.pipeline import longform_metadata as lm

    rig = rig_with_state(tmp_path)
    catalog({"vidKerry": ORPHAN})
    stages = rig.stages / "day-a" / "day-a" / "_stages"
    other = {"ireland in 4k": "day-b", "ireland (sep 16, 2025) in 4k": "day-b"}
    assert lm.orphan_upload_refusal("day-a", stages, receipt_ids={"vidKerry"}, other_forms=other) is None  # fmt: skip
    assert lm.orphan_upload_refusal("day-a", stages, receipt_ids=set(), other_forms=other)


def test_a_shared_title_video_uploaded_before_this_drives_first_attempt_is_not_its_orphan(tmp_path):
    rig = rig_with_state(tmp_path)
    _other_ireland(rig)
    job(rig, created_at=T0, state="succeeded", exit_code=0, ended_at=T0 + 10)
    old = {**ORPHAN, "title": "Ireland in 4K", "published_at": iso(T0 - 5000)}
    catalog({"vidOld": old}, fetched_at=iso(1_900_000_000.0))
    assert rig.run()["action"] == "launched"


# ── MINOR 5: a dropped LF is judged before shared forms are removed ──────────


def test_an_open_orphan_row_stays_open_when_another_drive_starts_sharing_the_title(tmp_path):
    rig = rig_with_state(tmp_path)
    catalog({"vidKerry": ORPHAN})
    assert rig.run()["action"] == "refused"
    (row,) = rows(_inc())
    assert row["resolved_at"] is None
    _other_ireland(rig, date="2025-09-16")
    res = rig.run()
    (row,) = rows(_inc())
    assert res["action"] == "refused" and "vidKerry" in res["reason"], res
    assert row["resolved_at"] is None
    assert "no longer has this LF" not in json.dumps(row)


def test_an_open_orphan_row_is_never_rewritten_when_another_drive_starts_sharing(
    tmp_path, monkeypatch
):
    """A close-then-reopen in one call ends in the same shape; every write stamps the clock, so a
    stamp that moves after the first run means the row was rewritten."""
    from gomoveshift.health import store as health_store

    rig = rig_with_state(tmp_path)
    catalog({"vidKerry": ORPHAN})
    assert rig.run()["action"] == "refused"
    (first,) = rows(_inc())
    ticks = iter(f"2030-01-01T00:00:{n:02d}+00:00" for n in range(60))
    monkeypatch.setattr(health_store, "utc_iso", lambda: next(ticks))
    _other_ireland(rig, date="2025-09-16")
    assert rig.run()["action"] == "refused"
    (row,) = rows(_inc())
    assert row == first, (first, row)
    assert next(ticks) == "2030-01-01T00:00:00+00:00"  # the second run stamped nothing


# ── MINOR 1 (rr49): a shared-title orphan blocks only the drive it is proven to belong to ──────

UP = datetime(2026, 11, 19, 17, 32, 49, tzinfo=timezone.utc).timestamp()  # ORPHAN published_at
SHARED = {**ORPHAN, "title": "Ireland in 4K"}


def _shared_pair(rig):
    _other_ireland(rig)
    catalog({"vidB": SHARED}, fetched_at=iso(UP + 7200))


def _publish_as(rig, project):
    return rig.run(project)


def test_the_drive_whose_job_window_covers_the_upload_is_the_only_one_refused(tmp_path):
    rig = rig_with_state(tmp_path)  # day-a: no job, no sidecar
    _shared_pair(rig)
    job(rig, created_at=UP - 600, state="failed", exit_code=1, ended_at=UP + 600, name="gms-publish-day-b")  # fmt: skip
    refused = _publish_as(rig, "day-b")
    assert refused["action"] == "refused" and "vidB" in refused["reason"], refused
    assert "day-b" in refused["reason"]
    assert _publish_as(rig, "day-a")["action"] == "launched"


def test_both_windows_covering_the_upload_refuses_both_drives(tmp_path):
    rig = rig_with_state(tmp_path)
    _shared_pair(rig)
    for name in ("gms-publish-day-a", "gms-publish-day-b"):
        job(rig, created_at=UP - 600, state="failed", exit_code=1, ended_at=UP + 600, name=name)
    for project in ("day-a", "day-b"):
        res = _publish_as(rig, project)
        assert res["action"] == "refused" and "vidB" in res["reason"], (project, res)


def test_an_upload_no_window_and_no_sidecar_accounts_for_refuses_both_drives(tmp_path):
    rig = rig_with_state(tmp_path)
    _shared_pair(rig)
    for project in ("day-a", "day-b"):
        res = _publish_as(rig, project)
        assert res["action"] == "refused" and "vidB" in res["reason"], (project, res)


def test_an_open_ended_other_window_attributes_nothing(tmp_path):
    rig = rig_with_state(tmp_path)
    _shared_pair(rig)
    job(rig, created_at=UP - 600, state="running", name="gms-publish-day-b")  # no end: unmeasured
    for project in ("day-a",):
        res = _publish_as(rig, project)
        assert res["action"] == "refused" and "vidB" in res["reason"], (project, res)


def test_the_other_drives_sidecar_naming_the_title_attributes_it_to_that_drive(tmp_path):
    rig = rig_with_state(tmp_path)
    _shared_pair(rig)
    out = rig.out / "day-b"
    out.mkdir(parents=True, exist_ok=True)
    (out / "day-b_lf01_youtube.json").write_text(json.dumps({"title": "Ireland in 4K"}))
    refused = _publish_as(rig, "day-b")
    assert refused["action"] == "refused" and "vidB" in refused["reason"], refused
    assert _publish_as(rig, "day-a")["action"] == "launched"


# ── MINOR 2 (rr49): a dead job with no end stamp anchors on a stable time, never on now ──────


def _ticks(rig, hours=3):
    calls: list = []
    results = []
    for hour in range(hours):
        now = NOW + timedelta(hours=hour)

        def refresh(now=now):
            calls.append(1)
            catalog({"vidKerry": ORPHAN}, fetched_at=(now + timedelta(seconds=5)).isoformat())

        results.append(
            publish.run_publish(
                project="day-a", region="usa", store=rig.store, cadence=CAD, stages_root=rig.stages,
                code_root=rig.tmp, jobs_dir=rig.jobs, now=now, dry_run=False,
                drive_running=lambda jobs: False, launcher=rig.launcher, refresh_catalog=refresh,
            )["action"]
        )  # fmt: skip
    return calls, results


def test_a_dead_job_with_no_end_stamp_fetches_once_across_ticks(tmp_path):
    rig = failed_drive(tmp_path)
    job(rig, created_at=NOW.timestamp() - 3 * 86400, state="cancelled")  # hand/legacy: no ended_at
    calls, results = _ticks(rig)
    assert results == ["refused"] * 3 and len(calls) == 1, (calls, results)


def test_a_dead_job_with_no_end_stamp_ends_when_its_record_was_last_written(tmp_path):
    rig = failed_drive(tmp_path)
    job(rig, created_at=UP - 3600, state="cancelled")
    (path,) = rig.jobs.glob("*/job.json")
    body = json.loads(path.read_text())
    path.write_text(json.dumps({**body, "run_id": path.parent.name}))
    os.utime(path, (UP + 90, UP + 90))
    _first, ended, _how = publish.publish_job_times(rig.jobs, "day-a")
    assert ended == datetime.fromtimestamp(UP + 90, timezone.utc)


def test_a_dead_job_whose_record_cannot_be_stat_ed_ends_at_creation_plus_the_stall_window(tmp_path):
    rig = failed_drive(tmp_path)
    job(rig, created_at=UP - 3600, state="cancelled")  # no run_id: no path to stat
    _first, ended, _how = publish.publish_job_times(rig.jobs, "day-a")
    assert ended == datetime.fromtimestamp(UP - 3600 + float(_STALL_S), timezone.utc)


# ── MINOR 3 (rr49): one refresher serves the whole backlog tick ──────────────────────────────


def test_a_replace_retry_tick_fetches_the_catalog_at_most_once(tmp_path, monkeypatch):
    from gomoveshift.cli.handlers import health as handler

    fetches: list = []
    monkeypatch.setattr(publish, "catalog_refresher", lambda config: lambda: fetches.append(1))
    monkeypatch.setattr(publish, "open_schedule_store", lambda config: (object(), object()))
    monkeypatch.setattr(publish, "run_publish", lambda **kw: kw["refresh_catalog"]() or {"action": "refused", "reason": "x", "steps": []})  # fmt: skip
    monkeypatch.setattr(launch, "load_inputs", lambda *a: ("usa", [], [], [], set(), None))
    monkeypatch.setattr(launch, "lease_fns", lambda root: (None, None))
    monkeypatch.setattr(launch, "run_backlog", lambda **kw: kw["publish"]("day-a", False) or {"action": "noop"})  # fmt: skip
    monkeypatch.setattr(launch, "run_tick", lambda **kw: (kw["run"](None, True), kw["run"](None, False))[1])  # fmt: skip
    args = Namespace(
        supersede=False, reason=None, replace_scheduled=False, region="usa", ledger_path=None,
        code_root=str(tmp_path), stages_root=tmp_path, output_root=tmp_path, sources_root=None,
        staging_sha="abc1234", dry_run=True, json=True,
    )  # fmt: skip
    handler._cmd_backlog(args, get_config(), tmp_path, tmp_path / "jobs", tmp_path / "inc")
    assert len(fetches) == 1, fetches


# ── MINOR 4 (rr49): the refusal names the anchor it really used ─────────────────────────────


def _stale_reason(rig):
    catalog({"v": {"title": "Cork in 4K"}}, fetched_at="2001-01-01T00:00:00+00:00")
    return run_with(rig, None)["reason"]


def test_the_refusal_says_ended_only_for_a_stamped_end(tmp_path):
    rig = failed_drive(tmp_path)
    job(rig, created_at=T0, state="failed", exit_code=1, ended_at=T0 + 60)
    assert "(ended " in _stale_reason(rig)


def test_the_refusal_names_the_record_write_when_the_job_has_no_end_stamp(tmp_path):
    rig = failed_drive(tmp_path)
    job(rig, created_at=T0, state="cancelled")
    (path,) = rig.jobs.glob("*/job.json")
    path.write_text(json.dumps({**json.loads(path.read_text()), "run_id": path.parent.name}))
    reason = _stale_reason(rig)
    assert "no end stamp; its job record was last written" in reason and "(ended " not in reason


def test_the_refusal_names_the_stall_window_when_nothing_else_anchors_the_job(tmp_path):
    rig = failed_drive(tmp_path)
    job(rig, created_at=T0, state="cancelled")
    assert "its creation plus the stall window" in _stale_reason(rig)


def test_the_refusal_names_a_reopen_with_no_job_record(tmp_path):
    reason = _stale_reason(failed_drive(tmp_path))
    assert "reopened from failed with no publish job record" in reason and "(ended " not in reason
