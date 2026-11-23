"""A scheduled upload with no ledger row is an orphan incident that blocks its own drive.

A publish job that dies after ``videos().insert`` leaves a private, scheduled video with no
receipt, no ledger row and no store ``complete`` (Kerry, 2026-11-19: ``Ireland (Sep 16, 2025) in
4K``). The only key that ties it to a drive without a receipt is the drive's expected title, so a
live-catalog video with that title and no ledger row opens one incident and refuses THAT drive's
publish. Nothing is deleted, unlisted or adopted, and no other drive is touched. No network: the
catalog snapshot, ledger and incidents dir are the conftest's per-test temp copies."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from gomoveshift.config import get_config
from gomoveshift.pipeline import longform_metadata as lm
from gomoveshift.pipeline.content_ledger import ContentEntry, ContentLedger
from gomoveshift.publishing.live_catalog import live_catalog_snapshot_path

from gomoveshift.health import publish
from tests.test_publish_command import CAD, NOW, Rig

TITLE = "Ireland (Sep 16, 2025) in 4K"
ORPHAN = {
    "title": TITLE,
    "privacy": "private",
    "publish_at": "2026-11-29T19:00:00Z",
    "published_at": "2026-11-19T17:32:49Z",
    "duration_s": 0,
}


@pytest.fixture
def incidents():
    return Path(os.environ["GMS_HEALTH_INCIDENTS_DIR"])


def rows(incidents):
    return [json.loads(p.read_text()) for p in sorted(incidents.glob("*orphan_upload*.json"))]


def catalog(videos, fetched_at=None):
    path = live_catalog_snapshot_path(lm.DEFAULT_SEO_STORAGE)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"videos": videos, **({"fetched_at": fetched_at} if fetched_at else {})}
    path.write_text(json.dumps(body), encoding="utf-8")


def state(rig, project="day-a", label="Ireland", date="2025-09-16"):
    stages = rig.stages / project / project / "_stages"
    stages.mkdir(parents=True, exist_ok=True)
    body = {"per_lf_ost_text": {"1": label}, "per_lf_recording_date": {"1": date}}
    (stages / "pipeline_state.json").write_text(json.dumps(body))


def ledger_row(video_id, project="somebody"):
    ledger = ContentLedger.load()
    ledger.add(
        ContentEntry(
            date="2026-11-29",
            time="14:00:00",
            content_type="longform",
            title=TITLE,
            project_name=project,
            status="scheduled",
            source="pipeline",
            video_id=video_id,
        )  # fmt: skip
    )
    ledger.save()


def rig_with_state(tmp_path, project="day-a"):
    rig = Rig(tmp_path)
    rig.audit(project)
    state(rig, project)
    ContentLedger.load().save()  # a ledger that exists and holds no rows (a missing one refuses)
    return rig


# ── the tie: a title match, no ledger row ────────────────────────────────────


def test_scheduled_catalog_video_without_ledger_row_is_an_orphan():
    expected = lm.expected_lf_titles({"per_lf_ost_text": {"1": "Ireland"},
                                      "per_lf_recording_date": {"1": "2025-09-16"}})  # fmt: skip
    found = lm.orphan_uploads(expected, {"vidKerry": ORPHAN, "other": {"title": "Elsewhere in 4K"}}, set())  # fmt: skip
    assert found == [{"lf": 1, "video_id": "vidKerry", "title": TITLE, "privacy": "private",
                      "publish_at": "2026-11-29T19:00:00Z"}]  # fmt: skip


def test_the_bare_and_the_dated_title_both_tie_a_video_to_the_drive():
    expected = lm.expected_lf_titles({"per_lf_ost_text": {"1": "Ireland"},
                                      "per_lf_recording_date": {"1": "2025-09-16"}})  # fmt: skip
    assert {"ireland in 4k", TITLE.lower()} <= expected[1]


def test_a_video_with_a_ledger_row_is_not_an_orphan():
    expected = {1: {TITLE.lower()}}
    assert lm.orphan_uploads(expected, {"vidKerry": ORPHAN}, {"vidKerry"}) == []


def test_no_matching_title_is_no_orphan():
    assert lm.orphan_uploads({1: {TITLE.lower()}}, {"v": {"title": "Cork in 4K"}}, set()) == []


# ── the block ────────────────────────────────────────────────────────────────


def test_orphan_blocks_publish_of_its_drive(tmp_path, incidents):
    rig = rig_with_state(tmp_path)
    catalog({"vidKerry": ORPHAN})
    res = rig.run()
    assert res["action"] == "refused" and not rig.launches
    for part in ("vidKerry", TITLE, "private", "2026-11-29T19:00:00Z"):
        assert part in res["reason"]
    (row,) = rows(incidents)
    assert row["drive"] == "day-a" and row["code"] == "orphan_upload" and row["lf"] == 1
    assert "vidKerry" in row["detail"] and row["resolved_at"] is None


def test_a_block_never_counts_a_publish_attempt_or_burns_the_drive(tmp_path, incidents):
    rig = rig_with_state(tmp_path)
    catalog({"vidKerry": ORPHAN})
    before = rig.snapshot()
    for _ in range(3):
        assert rig.run()["action"] == "refused"
    assert rig.snapshot() == before and rig.store.drive("day-a").status == "auditing"
    assert not rig.launches
    names = sorted(p.name for p in incidents.glob("*.json"))
    assert len(names) == 1 and "orphan_upload" in names[0]  # no relaunch/attempt row beside it
    assert "attempts" not in json.loads((incidents / names[0]).read_text())


def test_the_incident_is_not_duplicated_on_a_rerun(tmp_path, incidents):
    rig = rig_with_state(tmp_path)
    catalog({"vidKerry": ORPHAN})
    rig.run()
    first = rows(incidents)
    rig.run()
    rig.run()
    assert rows(incidents) == first and len(first) == 1


def test_a_dry_run_blocks_but_records_nothing(tmp_path, incidents):
    rig = rig_with_state(tmp_path)
    catalog({"vidKerry": ORPHAN})
    res = rig.run(dry_run=True)
    assert res["action"] == "refused" and "vidKerry" in res["reason"]
    assert rows(incidents) == []


def test_other_drives_are_not_blocked(tmp_path):
    rig = rig_with_state(tmp_path, "day-a")
    rig.audit("day-b", status="auditing")  # a second, unrelated drive with a different title
    state(rig, "day-b", label="Galway", date="2025-09-20")
    catalog({"vidKerry": ORPHAN})
    before = rig.snapshot()
    assert rig.run("day-a")["action"] == "refused"
    assert rig.snapshot() == before
    res = rig.run("day-b")
    assert res["action"] == "launched", res
    assert rig.store.drive("day-b").status == "claimed"


def test_a_catalog_with_no_match_changes_nothing(tmp_path, incidents):
    rig = rig_with_state(tmp_path)
    catalog({"v": {"title": "Cork in 4K", "privacy": "public"}})
    assert rig.run()["action"] == "launched"
    assert rows(incidents) == []


def test_a_ledger_row_clears_the_block(tmp_path, incidents):
    rig = rig_with_state(tmp_path)
    catalog({"vidKerry": ORPHAN})
    ledger_row("vidKerry")
    assert rig.run()["action"] == "launched"
    assert rows(incidents) == []


def test_another_drives_receipt_is_not_an_orphan(tmp_path):
    rig = rig_with_state(tmp_path, "day-a")
    other = rig.audit("day-b", recipe=False)
    other.mkdir(parents=True, exist_ok=True)
    receipt = {"lf1": {"video_id": "vidKerry", "publish_at": "2026-11-29T19:00:00+00:00"}}
    (other / "youtube_uploads.json").write_text(json.dumps(receipt))
    catalog({"vidKerry": ORPHAN})
    assert rig.run("day-a")["action"] == "launched"


def test_an_unreadable_ledger_blocks_fail_closed(tmp_path, incidents):
    rig = rig_with_state(tmp_path)
    catalog({"vidKerry": ORPHAN})
    path = Path(get_config().seo.storage_path) / ".content_ledger.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    res = rig.run()
    assert res["action"] == "refused" and "ledger" in res["reason"] and not rig.launches


def test_a_deleted_video_resolves_the_incident_and_an_empty_snapshot_does_not(tmp_path, incidents):
    rig = rig_with_state(tmp_path)
    stages = rig.stages / "day-a" / "day-a" / "_stages"
    catalog({"vidKerry": ORPHAN})
    assert lm.orphan_upload_refusal("day-a", stages)
    assert rows(incidents)[0]["resolved_at"] is None
    catalog({})  # an empty snapshot measures nothing: the incident stays open
    assert lm.orphan_upload_refusal("day-a", stages) is None
    assert rows(incidents)[0]["resolved_at"] is None
    catalog({"v": {"title": "Cork in 4K"}})  # JP deleted it in Studio; the next snapshot drops it
    assert lm.orphan_upload_refusal("day-a", stages) is None
    assert rows(incidents)[0]["resolved_at"]
    assert not rig.launches


def test_no_state_file_means_no_expected_title_and_no_block(tmp_path):
    rig = Rig(tmp_path)
    rig.audit()
    catalog({"vidKerry": ORPHAN})
    assert rig.run()["action"] == "launched"


# ── a replace's own upload is known (BLOCKER 1) ──────────────────────────────


def _replace_rig(tmp_path, *, receipt=True, state_name="uploaded"):
    rig = rig_with_state(tmp_path)
    out = rig.out / "day-a"
    out.mkdir(parents=True, exist_ok=True)
    old = {"video_id": "vidOld", "publish_at": "2026-12-20T19:00:00+00:00", "title": TITLE}
    plan = {"schema": "replace-scheduled/1", "kept": {}, "replace": {"1": old},
            "progress": {"1": {"old_id": "vidOld", "new_id": "vidNew", "state": state_name}}}  # fmt: skip
    (out / "replace_plan.json").write_text(json.dumps(plan))
    if receipt:  # the sidecar save_state wrote at insert: the replacement went up unscheduled
        (out / "youtube_uploads.json").write_text(json.dumps({"1": {"video_id": "vidNew", "publish_at": None}}))  # fmt: skip
    ledger_row("vidOld", project="day-a")  # the ledger names the OLD id until the swap
    catalog({"vidOld": {**ORPHAN, "publish_at": "2026-12-20T19:00:00Z"},
             "vidNew": {"title": TITLE, "privacy": "private", "publish_at": None}})  # fmt: skip
    return rig


@pytest.mark.parametrize("state_name", ["uploaded", "old_deleted"])
def test_a_half_finished_replace_does_not_call_its_own_upload_an_orphan(
    tmp_path, incidents, state_name
):
    rig = _replace_rig(tmp_path, state_name=state_name)
    res = rig.run()
    assert res["action"] == "launched", res
    assert rows(incidents) == []


def test_the_replace_plan_alone_names_the_new_upload(tmp_path):
    rig = _replace_rig(tmp_path, receipt=False)  # the receipt is not written yet: the plan is
    assert rig.run()["action"] == "launched"


def test_a_receipt_with_no_publish_at_is_a_known_id(tmp_path):
    rig = rig_with_state(tmp_path)
    other = rig.audit("day-b", recipe=False)  # another drive's receipt, saved with no publish_at
    other.mkdir(parents=True, exist_ok=True)
    (other / "youtube_uploads.json").write_text(json.dumps({"1": {"video_id": "vidKerry", "publish_at": None}}))  # fmt: skip
    catalog({"vidKerry": ORPHAN})
    assert rig.run()["action"] == "launched"


def test_a_superseded_runs_receipt_is_a_known_id(tmp_path):
    rig = rig_with_state(tmp_path)
    old = rig.out / "_superseded" / "day-a-20261101T000000Z"
    old.mkdir(parents=True)
    (old / "youtube_uploads.json").write_text(json.dumps({"1": {"video_id": "vidKerry", "publish_at": None}}))  # fmt: skip
    catalog({"vidKerry": ORPHAN})
    assert rig.run()["action"] == "launched"


# ── Phase D's exact title, then every derived form (MAJOR 2) ─────────────────


def test_the_exact_title_phase_d_wrote_is_what_ties_a_video_to_the_lf(tmp_path):
    rig = rig_with_state(tmp_path)
    (rig.out / "day-a").mkdir(parents=True, exist_ok=True)
    (rig.out / "day-a" / "day-a_lf01_youtube.json").write_text(json.dumps({"title": "Kerry Coast Run in 4K"}))  # fmt: skip
    catalog({"vidX": {**ORPHAN, "title": "Kerry Coast Run in 4K"}})
    res = rig.run()
    assert res["action"] == "refused" and "vidX" in res["reason"], res


def test_a_derived_title_is_only_a_fallback_when_phase_d_wrote_one(tmp_path):
    rig = rig_with_state(tmp_path)
    (rig.out / "day-a").mkdir(parents=True, exist_ok=True)
    (rig.out / "day-a" / "day-a_lf01_youtube.json").write_text(json.dumps({"title": "Kerry Coast Run in 4K"}))  # fmt: skip
    catalog({"vidX": {**ORPHAN, "title": "Ireland in 4K"}})
    assert rig.run()["action"] == "launched"


@pytest.mark.parametrize(
    "title",
    [
        "Ireland in 4K — Part 1 of 2",
        "Ireland (Sep 16, 2025) in 4K — Part 2 of 3",
        "Ireland in 4K (Part 1 of 2)",
        "Ireland in 4K (Part 2)",
    ],
)
def test_a_part_suffixed_title_ties_to_the_drive(tmp_path, title):
    rig = rig_with_state(tmp_path)
    catalog({"vidP": {**ORPHAN, "title": title}})
    res = rig.run()
    assert res["action"] == "refused" and "vidP" in res["reason"], res


def test_the_assembled_in_batch_part_title_ties_to_the_drive(tmp_path):
    rig = rig_with_state(tmp_path)
    catalog({"vidP": {**ORPHAN, "title": lm.assemble_final_title("Ireland (Part 1 of 2)")}})
    assert rig.run()["action"] == "refused"


def test_the_pre_render_decision_label_ties_to_the_drive(tmp_path):
    rig = rig_with_state(tmp_path)
    stages = rig.stages / "day-a" / "day-a" / "_stages"
    body = json.loads((stages / "pipeline_state.json").read_text())
    body["per_lf_pre_render_title_decisions"] = {"1": {"input_label": "Ring of Kerry"}}
    (stages / "pipeline_state.json").write_text(json.dumps(body))
    catalog({"vidR": {**ORPHAN, "title": lm.assemble_final_title("Ring of Kerry")}})
    res = rig.run()
    assert res["action"] == "refused" and "vidR" in res["reason"], res


# ── only a video this drive could have uploaded (MAJOR 4) ────────────────────


def test_a_public_historic_video_with_the_title_does_not_block(tmp_path, incidents):
    rig = rig_with_state(tmp_path)
    catalog({"vidOld": {"title": "Ireland in 4K", "privacy": "public", "publish_at": None,
                        "published_at": "2024-05-01T12:00:00Z"}})  # fmt: skip
    assert rig.run()["action"] == "launched"
    assert rows(incidents) == []


def test_the_refusal_says_verify_the_drive_and_never_adopt(tmp_path):
    rig = rig_with_state(tmp_path)
    catalog({"vidKerry": ORPHAN})
    reason = rig.run()["reason"]
    assert "verify which drive uploaded it" in reason and "YouTube Studio" in reason
    assert "book it into the ledger" in reason and "adopt" not in reason


@pytest.mark.parametrize(
    ("published_at", "action"), [(1_799_999_000.0, "launched"), (1_800_000_100.0, "refused")]
)
def test_a_public_video_blocks_only_if_published_after_the_first_attempt(
    tmp_path, published_at, action
):
    rig = rig_with_state(tmp_path)
    failed_job(rig, created_at=1_800_000_000.0)
    public = {"title": TITLE, "privacy": "public", "publish_at": None, "published_at": iso(published_at)}  # fmt: skip
    catalog({"vidPub": public}, fetched_at=iso(1_900_000_000.0))
    res = rig.run()
    assert res["action"] == action, res
    assert action == "launched" or "vidPub" in res["reason"]


def test_another_drives_same_title_upload_is_not_attributed_to_this_drive(tmp_path):
    rig = rig_with_state(tmp_path, "day-a")
    rig.audit("day-b", status="auditing")
    state(rig, "day-b", label="Ireland", date="2025-09-20")
    catalog({"vidB": {**ORPHAN, "title": "Ireland in 4K"}})  # the bare title both drives expect
    res = rig.run("day-a")
    # unbooked and matching only a shared title: not silently this drive's, not a pass either; the
    # refusal names both drives (a receipt naming the video clears it: test_orphan_gate_fresh_window)
    assert res["action"] == "refused" and "day-a" in res["reason"] and "day-b" in res["reason"], res
    catalog(
        {"vidB": {**ORPHAN, "title": "Ireland (Sep 16, 2025) in 4K"}}
    )  # day-a's own dated title
    assert rig.run("day-a")["action"] == "refused"


# ── a failed publish needs a catalog fetched after it (MAJOR 3) ──────────────


def iso(epoch):
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def failed_job(rig, *, created_at, state="failed", name="gms-publish-day-a"):
    path = rig.jobs / f"{name}-{int(created_at)}" / "job.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "job/1", "name": name, "created_at": created_at,
                                "result": {"state": state, "exit_code": 1}}))  # fmt: skip


def run_with(rig, refresh):
    return publish.run_publish(
        project="day-a", region="usa", store=rig.store, cadence=CAD, stages_root=rig.stages,
        code_root=rig.tmp, jobs_dir=rig.jobs, now=NOW, dry_run=False,
        drive_running=lambda jobs: False, launcher=rig.launcher, refresh_catalog=refresh,
    )  # fmt: skip


def test_a_failed_publish_with_a_stale_snapshot_refreshes_before_deciding(tmp_path):
    rig = rig_with_state(tmp_path)
    failed_job(rig, created_at=1_800_000_000.0)
    catalog({"v": {"title": "Cork in 4K"}}, fetched_at=iso(1_700_000_000.0))
    calls = []

    def refresh():
        calls.append(1)
        catalog({"vidKerry": ORPHAN}, fetched_at=iso(1_900_000_000.0))

    res = run_with(rig, refresh)
    assert calls == [1] and res["action"] == "refused" and "vidKerry" in res["reason"], res
    assert not rig.launches


def test_a_failed_publish_launches_once_a_refreshed_catalog_is_clean(tmp_path):
    rig = rig_with_state(tmp_path)
    failed_job(rig, created_at=1_800_000_000.0)
    rig.store.set_status("day-a", "failed")
    res = run_with(rig, lambda: catalog({"v": {"title": "Cork in 4K"}}, fetched_at=iso(1_900_000_000.0)))  # fmt: skip
    assert res["action"] == "launched", res


@pytest.mark.parametrize("refresh", [None, "raises", "no_write"])
def test_a_failed_publish_that_cannot_get_a_fresh_snapshot_refuses_until_the_next_tick(
    tmp_path, refresh
):
    rig = rig_with_state(tmp_path)
    failed_job(rig, created_at=1_800_000_000.0)

    # no snapshot at all, as after a restore: nothing proves the dead upload is absent
    def boom():
        raise RuntimeError("quota")

    res = run_with(rig, {"raises": boom, "no_write": lambda: None}.get(refresh))
    assert res["action"] == "refused" and "next tick" in res["reason"], res
    assert not rig.launches and rig.store.drive("day-a").status == "auditing"


def test_a_fresh_snapshot_after_the_failed_publish_needs_no_refresh(tmp_path):
    rig = rig_with_state(tmp_path)
    failed_job(rig, created_at=1_800_000_000.0)
    catalog({"v": {"title": "Cork in 4K"}}, fetched_at=iso(1_900_000_000.0))

    def refresh():
        raise AssertionError("a fresh snapshot is never refetched")

    assert run_with(rig, refresh)["action"] == "launched"


def test_a_clean_first_publish_needs_no_fresh_snapshot(tmp_path):
    rig = rig_with_state(tmp_path)
    catalog({"v": {"title": "Cork in 4K"}})  # no fetched_at, no failed job: stays open
    assert rig.run()["action"] == "launched"


def test_a_dry_run_never_refreshes_the_catalog(tmp_path):
    rig = rig_with_state(tmp_path)
    failed_job(rig, created_at=1_800_000_000.0)
    catalog({"v": {"title": "Cork in 4K"}}, fetched_at=iso(1_700_000_000.0))

    def refresh():
        raise AssertionError("a dry run changes nothing")

    res = publish.run_publish(
        project="day-a", region="usa", store=rig.store, cadence=CAD, stages_root=rig.stages,
        code_root=rig.tmp, jobs_dir=rig.jobs, now=NOW, dry_run=True,
        drive_running=lambda jobs: False, launcher=rig.launcher, refresh_catalog=refresh,
    )  # fmt: skip
    assert res["action"] == "refused" and "next tick" in res["reason"]


# ── incident bookkeeping (MINOR 5, 6, 7) ─────────────────────────────────────


def test_the_replace_flows_lf_done_resolve_leaves_the_orphan_row_open(tmp_path, incidents):
    from gomoveshift.pipeline import replace_scheduled

    rig = rig_with_state(tmp_path)
    catalog({"vidKerry": ORPHAN})
    assert rig.run()["action"] == "refused"
    replace_scheduled.resolve_incidents("day-a", lf=1)  # what advance() calls when an LF is DONE
    replace_scheduled.resolve_incidents("day-a", lf=1, keep=replace_scheduled._KEEP_OPEN_ON_ABANDON)  # fmt: skip
    (row,) = rows(incidents)
    assert row["resolved_at"] is None


def test_a_missing_ledger_file_blocks_fail_closed(tmp_path):
    rig = rig_with_state(tmp_path)
    (Path(get_config().seo.storage_path) / ".content_ledger.json").unlink()
    catalog({"vidKerry": ORPHAN})
    res = rig.run()
    assert res["action"] == "refused" and "ledger is missing or unreadable" in res["reason"], res
    assert not rig.launches


def test_a_row_for_an_lf_the_drive_no_longer_has_is_resolved(tmp_path, incidents):
    rig = rig_with_state(tmp_path)
    stages = rig.stages / "day-a" / "day-a" / "_stages"
    catalog({"vidKerry": ORPHAN})
    assert lm.orphan_upload_refusal("day-a", stages)
    from gomoveshift.pipeline.replace_scheduled import record_incident

    record_incident("day-a", "orphan_upload", "LF 2 orphan", lf=2)
    assert len(rows(incidents)) == 2
    assert lm.orphan_upload_refusal("day-a", stages)  # LF 2 is not in the checkpoint's labels
    by_lf = {r["lf"]: r["resolved_at"] for r in rows(incidents)}
    assert by_lf[1] is None and by_lf[2]
