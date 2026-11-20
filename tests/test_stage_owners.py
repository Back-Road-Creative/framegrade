"""B6: a stage root with no manifest row has an owner record with an expiry; an expired record is
listed for release through the host-cleanup release list, whose file name is one constant."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from gomoveshift.health import stage_owners as so

TODAY = date(2026, 11, 20)


def _owner(name="mizen-remeasure", expires="2026-12-18", **kw):
    rec = {
        "name": name, "owner": "JP", "reason": "evidence for the Mizen re-measure",
        "issue": "framegrade#398", "expires": expires,
    }  # fmt: skip
    rec.update(kw)
    return rec


def _write(tmp_path, records):
    path = tmp_path / "stage-owners.json"
    path.write_text(json.dumps({"schema": so.SCHEMA, "roots": records}))
    return path


def test_the_release_list_name_is_the_one_host_cleanup_reads():
    # Reader: workspaces _ops/host-cleanup.sh (GMS_STAGES_RELEASE, rule 7c). Pinned literally:
    # renaming either side alone fails here instead of silently releasing nothing.
    assert so.STAGES_RELEASE_LIST_NAME == "gms-stages-release.list"
    assert so.release_list_path(Path("/d")) == Path("/d/host-cleanup/gms-stages-release.list")


def test_the_tracked_owner_file_holds_the_mizen_remeasure_record():
    owners = so.load_owners(so.DEFAULT_OWNERS_PATH)
    mizen = {o.name: o for o in owners}["mizen-remeasure"]
    assert mizen.issue == "framegrade#398" and mizen.owner and mizen.reason and mizen.expires


def test_a_record_needs_owner_reason_issue_and_a_real_expiry(tmp_path):
    for bad in (_owner(owner=""), _owner(reason=""), _owner(issue=""), _owner(expires="soon"),
                _owner(expires=""), _owner(name="../x"), _owner(name="Has Caps")):  # fmt: skip
        with pytest.raises(so.OwnerRecordError):
            so.load_owners(_write(tmp_path, [bad]))


def test_a_duplicate_name_is_refused(tmp_path):
    with pytest.raises(so.OwnerRecordError):
        so.load_owners(_write(tmp_path, [_owner(), _owner()]))


def test_a_missing_file_is_no_owners_and_unreadable_json_raises(tmp_path):
    assert so.load_owners(tmp_path / "nope.json") == []
    path = tmp_path / "bad.json"
    path.write_text("{")
    with pytest.raises(so.OwnerRecordError):
        so.load_owners(path)


def test_expiry_is_inclusive_of_the_day_after_only(tmp_path):
    owners = so.load_owners(_write(tmp_path, [_owner(expires="2026-11-20")]))
    assert so.expired(owners, TODAY) == []  # expires today: still held
    assert [o.name for o in so.expired(owners, date(2026, 11, 21))] == ["mizen-remeasure"]


def test_unowned_roots_are_dirs_with_no_row_and_no_owner(tmp_path):
    stages = tmp_path / "stages"
    for name in ("day-a", "mizen-remeasure", "stray", so.SUPERSEDED_DIR):
        (stages / name).mkdir(parents=True)
    (stages / "a-file").write_text("x")
    owners = so.load_owners(_write(tmp_path, [_owner()]))
    assert so.unowned_roots(stages, {"day-a"}, owners) == ["stray"]


def test_release_candidates_are_expired_roots_that_exist(tmp_path):
    stages = tmp_path / "stages"
    (stages / "mizen-remeasure").mkdir(parents=True)
    owners = so.load_owners(_write(tmp_path, [_owner(), _owner(name="gone", expires="2026-01-01")]))
    assert so.release_candidates(owners, stages, TODAY) == []
    assert so.release_candidates(owners, stages, date(2027, 1, 1)) == ["mizen-remeasure"]


def test_listing_appends_once_keeps_existing_lines_and_names_why(tmp_path):
    lst = tmp_path / "host-cleanup" / so.STAGES_RELEASE_LIST_NAME
    lst.parent.mkdir()
    lst.write_text("# kept\nold-root\n")
    owner = so.load_owners(_write(tmp_path, [_owner(expires="2026-11-01")]))[0]
    assert so.list_for_release(lst, [owner], TODAY) == ["mizen-remeasure"]
    assert so.list_for_release(lst, [owner], TODAY) == []  # idempotent
    text = lst.read_text()
    assert text.startswith("# kept\nold-root\n") and text.count("mizen-remeasure\n") == 1
    assert "framegrade#398" in text and "owner record expired 2026-11-01" in text


def test_listing_creates_the_file_and_never_lists_a_non_name(tmp_path):
    lst = tmp_path / "new" / so.STAGES_RELEASE_LIST_NAME
    owner = so.Owner("mizen-remeasure", "JP", "r", "framegrade#398", date(2026, 1, 1))
    assert so.list_for_release(lst, [owner], TODAY) == ["mizen-remeasure"]
    assert lst.read_text().splitlines()[-1] == "mizen-remeasure"


def test_run_backlog_lists_an_expired_root_and_reports_unowned(tmp_path):
    from tests.test_health_backlog_launch import Rig  # noqa: PLC0415
    from gomoveshift.health import launch  # noqa: PLC0415

    rig = Rig(tmp_path)
    (rig.stages / "mizen-remeasure").mkdir()
    (rig.stages / "stray").mkdir()
    owners = so.load_owners(_write(tmp_path, [_owner(expires="2026-10-01")]))
    lst = tmp_path / "rel" / so.STAGES_RELEASE_LIST_NAME
    kw = {**rig.kwargs(), "stage_owners": owners, "release_list": lst}
    res = launch.run_backlog(**kw)
    assert res["stage_roots"]["listed"] == ["mizen-remeasure"]
    assert res["stage_roots"]["unowned"] == ["stray"]
    assert "mizen-remeasure" in lst.read_text()
    dry = tmp_path / "dry" / so.STAGES_RELEASE_LIST_NAME
    res = launch.run_backlog(
        **{**rig.kwargs(dry_run=True), "stage_owners": owners, "release_list": dry}
    )
    assert res["stage_roots"]["listed"] == ["mizen-remeasure"] and not dry.exists()
