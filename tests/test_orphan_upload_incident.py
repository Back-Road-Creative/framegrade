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

from tests.test_publish_command import Rig

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


def catalog(videos):
    path = live_catalog_snapshot_path(lm.DEFAULT_SEO_STORAGE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"videos": videos}), encoding="utf-8")


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
    assert rig.run("day-a")["action"] == "refused"
    assert rig.run("day-b")["action"] != "refused" or "vidKerry" not in rig.run("day-b")["reason"]


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
