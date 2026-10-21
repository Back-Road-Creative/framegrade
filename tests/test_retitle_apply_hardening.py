"""U7 hardening of ``tools/retitle_apply.py``: idempotent re-run, snapshot before
write, read-back after write, reversal worklist and ``--revert`` (synthetic data)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_retitle_apply import (
    NEW_DESC,
    OLD_DESC,
    FakeService,
    _desc_row,
    _req,
    _row,
    _write_worklist,
)
from tools import retitle_apply as ra

OLD_TITLE = "To Clonakilty in 4K"
NEW_TITLE = "Clonmel to Clonakilty, County Cork"
APPLY = ["--apply", "--yes-live"]


def _run(path, service, *extra):
    sleeps: list[float] = []
    rc = ra.main(["--worklist", str(path), *extra], service=service, sleep=sleeps.append)
    return rc, sleeps


def _all_audit(path):
    return [json.loads(x) for x in path.with_suffix(".audit.jsonl").read_text().splitlines()]


def _outcomes(path):
    return [r for r in _all_audit(path) if r["result"] != ra.SNAPSHOT]


# --- 1. idempotent re-run ----------------------------------------------------
def test_live_title_already_new_is_already_applied(tmp_path):
    path = _write_worklist(tmp_path, [_row()])
    service = FakeService({"vid1": {"title": NEW_TITLE, "categoryId": "19"}})
    rc, _ = _run(path, service, *APPLY)
    assert rc == 0
    assert service.videos().update_calls == []
    assert [r["result"] for r in _outcomes(path)] == [ra.ALREADY_APPLIED]


def test_description_row_needs_both_fields_new_for_already_applied(tmp_path):
    path = _write_worklist(tmp_path, [_desc_row()])
    service = FakeService({"vid1": {"title": NEW_TITLE, "description": OLD_DESC}})
    _run(path, service, *APPLY)
    # title new but description old: not a completed apply -> stale, never a write
    assert service.videos().update_calls == []
    assert [r["result"] for r in _outcomes(path)] == [ra.SKIPPED_STALE]
    service = FakeService({"vid1": {"title": NEW_TITLE, "description": NEW_DESC}})
    _run(path, service, *APPLY)
    assert service.videos().update_calls == []
    assert _outcomes(path)[-1]["result"] == ra.ALREADY_APPLIED


def test_second_run_after_full_apply_makes_no_update_calls(tmp_path):
    path = _write_worklist(tmp_path, [_row(), _row(video_id="vid2")])
    service = FakeService(
        {"vid1": {"title": OLD_TITLE, "categoryId": "19"}, "vid2": {"title": OLD_TITLE}}
    )
    _run(path, service, *APPLY)
    assert len(service.videos().update_calls) == 2
    _run(path, service, *APPLY)
    assert len(service.videos().update_calls) == 2
    assert [r["result"] for r in _outcomes(path)][2:] == [ra.ALREADY_APPLIED] * 2


# --- 2. snapshot before write --------------------------------------------------
def test_snapshot_is_durable_before_the_update_call(tmp_path):
    path = _write_worklist(tmp_path, [_desc_row()])
    seen: list[list[dict]] = []
    service = FakeService({"vid1": {"title": OLD_TITLE, "description": OLD_DESC}})
    real_update = service.videos().update

    def update(part, body):
        seen.append(_all_audit(path))
        return real_update(part, body)

    service.videos().update = update
    _run(path, service, *APPLY)
    [snap] = seen[0]
    assert snap["result"] == ra.SNAPSHOT
    assert (snap["snapshot_title"], snap["snapshot_description"]) == (OLD_TITLE, OLD_DESC)


def test_crash_mid_run_still_leaves_first_rows_records(tmp_path):
    path = _write_worklist(tmp_path, [_row(), _row(video_id="vid2")])
    service = FakeService({"vid1": {"title": OLD_TITLE}, "vid2": {"title": OLD_TITLE}})
    real_list = service.videos().list

    def list_(part, id):  # noqa: A002
        if id == "vid2":
            raise KeyboardInterrupt
        return real_list(part, id)

    service.videos().list = list_
    with pytest.raises(KeyboardInterrupt):
        _run(path, service, *APPLY)
    assert [r["result"] for r in _all_audit(path)] == [ra.SNAPSHOT, ra.APPLIED]


def test_audit_log_has_one_snapshot_and_one_outcome_per_pushed_row(tmp_path):
    path = _write_worklist(tmp_path, [_row(), _row(video_id="vid2")])
    service = FakeService({"vid1": {"title": OLD_TITLE}, "vid2": {"title": OLD_TITLE}})
    _run(path, service, *APPLY)
    assert [(r["video_id"], r["result"]) for r in _all_audit(path)] == [
        ("vid1", ra.SNAPSHOT),
        ("vid1", ra.APPLIED),
        ("vid2", ra.SNAPSHOT),
        ("vid2", ra.APPLIED),
    ]


def test_apply_updates_returns_outcome_records_only(tmp_path):
    results = [{"row": _row(), "verdict": ra.APPROVED}]
    service = FakeService({"vid1": {"title": OLD_TITLE}})
    records = ra.apply_updates(service, results, audit_path=tmp_path / "a.jsonl")
    assert [r["result"] for r in records] == [ra.APPLIED]
    assert len((tmp_path / "a.jsonl").read_text().splitlines()) == 2


# --- 3. read-back after write ----------------------------------------------------
class LaggyVideos:
    """Applies the update, but serves the OLD snippet for ``lag`` reads after it."""

    def __init__(self, snippet, lag):
        self.snippet, self.lag, self.stale_reads_left = dict(snippet), lag, 0
        self.update_calls: list[dict] = []

    def list(self, part, id):  # noqa: A002
        snippet = self.snippet
        if self.stale_reads_left > 0:
            self.stale_reads_left -= 1
            snippet = self.old
        return _req({"items": [{"id": id, "snippet": snippet}]})

    def update(self, part, body):
        self.update_calls.append(body)
        self.old, self.snippet = self.snippet, body["snippet"]
        self.stale_reads_left = self.lag
        return _req({})


class LaggyService:
    def __init__(self, snippet, lag):
        self._v = LaggyVideos(snippet, lag)

    def videos(self):
        return self._v


def test_read_back_retries_with_injectable_wait_then_applies(tmp_path):
    path = _write_worklist(tmp_path, [_row()])
    service = LaggyService({"title": OLD_TITLE}, lag=2)
    rc, sleeps = _run(path, service, *APPLY)
    assert rc == 0
    assert len(sleeps) == 2
    [rec] = _outcomes(path)
    assert rec["result"] == ra.APPLIED
    assert rec["readback_title"] == NEW_TITLE


def test_read_back_mismatch_is_verify_failed_and_stops_run(tmp_path):
    path = _write_worklist(tmp_path, [_row(), _row(video_id="vid2")])
    service = LaggyService({"title": OLD_TITLE}, lag=99)
    rc, sleeps = _run(path, service, *APPLY)
    assert rc == 0
    assert len(sleeps) == 2  # 3 reads after the write, 2 waits between
    assert len(service.videos().update_calls) == 1  # stopped, vid2 never touched
    [rec] = _outcomes(path)
    assert rec["result"] == ra.VERIFY_FAILED
    assert rec["readback_title"] == OLD_TITLE
    assert not path.with_suffix(".revert.csv").exists()


def test_read_back_checks_description_when_changed(tmp_path):
    path = _write_worklist(tmp_path, [_desc_row()])
    svc = FakeService({"vid1": {"title": OLD_TITLE, "description": OLD_DESC}})
    real_update = svc.videos().update

    def update(part, body):  # title lands, description silently does not
        body = {"id": body["id"], "snippet": {**body["snippet"], "description": OLD_DESC}}
        return real_update(part, body)

    svc.videos().update = update
    _run(path, svc, *APPLY)
    assert _outcomes(path)[0]["result"] == ra.VERIFY_FAILED


def test_read_back_ignores_surrounding_whitespace(tmp_path):
    path = _write_worklist(tmp_path, [_row()])
    svc = FakeService({"vid1": {"title": OLD_TITLE}})
    real_update = svc.videos().update

    def update(part, body):
        body = {"id": "vid1", "snippet": {"title": f" {NEW_TITLE}\n"}}
        return real_update(part, body)

    svc.videos().update = update
    _run(path, svc, *APPLY)
    assert _outcomes(path)[0]["result"] == ra.APPLIED


# --- 4. reversal worklist --------------------------------------------------------
def _revert_rows(p):
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def test_applied_rows_append_to_reversal_csv(tmp_path):
    path = _write_worklist(
        tmp_path,
        [
            _row(current_description="", suggested_description=""),
            _desc_row(video_id="vid2"),
        ],
    )
    service = FakeService(
        {"vid1": {"title": OLD_TITLE}, "vid2": {"title": OLD_TITLE, "description": OLD_DESC}}
    )
    _run(path, service, *APPLY)
    rows = _revert_rows(tmp_path / "worklist.revert.csv")
    assert rows[0]["video_id"] == "vid1"
    assert (rows[0]["current_title"], rows[0]["restore_title"]) == (NEW_TITLE, OLD_TITLE)
    assert rows[0]["current_description"] == rows[0]["restore_description"] == ""
    assert (rows[1]["current_description"], rows[1]["restore_description"]) == (NEW_DESC, OLD_DESC)


def test_revert_out_overrides_default_path(tmp_path):
    path = _write_worklist(tmp_path, [_row()])
    out = tmp_path / "custom.csv"
    _run(path, FakeService({"vid1": {"title": OLD_TITLE}}), *APPLY, "--revert-out", str(out))
    assert out.exists() and not (tmp_path / "worklist.revert.csv").exists()


def _forward_then_revert_file(tmp_path, with_desc=False):
    path = _write_worklist(tmp_path, [_desc_row() if with_desc else _row()])
    snippet = {"title": OLD_TITLE, **({"description": OLD_DESC} if with_desc else {})}
    service = FakeService({"vid1": snippet})
    _run(path, service, *APPLY)
    return tmp_path / "worklist.revert.csv", service


def _revert(rev, service, *extra):
    return ra.main(["--revert", str(rev), *extra], service=service, sleep=lambda s: None)


def _revert_audit(rev):
    return [json.loads(x) for x in rev.with_suffix(".audit.jsonl").read_text().splitlines()]


def test_revert_dry_run_makes_no_calls(tmp_path):
    rev, service = _forward_then_revert_file(tmp_path)
    before = len(service.videos().update_calls)
    assert _revert(rev, service) == 0
    assert len(service.videos().update_calls) == before


def test_revert_restores_prior_text_with_snapshot_and_readback(tmp_path):
    rev, service = _forward_then_revert_file(tmp_path, with_desc=True)
    assert _revert(rev, service, *APPLY) == 0
    snippet = service.videos().snippets["vid1"]
    assert (snippet["title"], snippet["description"]) == (OLD_TITLE, OLD_DESC)
    recs = _revert_audit(rev)
    assert [r["result"] for r in recs] == [ra.SNAPSHOT, ra.APPLIED]
    assert recs[0]["snapshot_title"] == NEW_TITLE


def test_revert_second_run_is_already_applied(tmp_path):
    rev, service = _forward_then_revert_file(tmp_path)
    _revert(rev, service, *APPLY)
    n = len(service.videos().update_calls)
    _revert(rev, service, *APPLY)
    assert len(service.videos().update_calls) == n
    assert _revert_audit(rev)[-1]["result"] == ra.ALREADY_APPLIED


def test_revert_skips_stale_when_live_differs(tmp_path):
    rev, service = _forward_then_revert_file(tmp_path)
    service.videos().snippets["vid1"]["title"] = "Edited in Studio"
    n = len(service.videos().update_calls)
    _revert(rev, service, *APPLY)
    assert len(service.videos().update_calls) == n
    assert _revert_audit(rev)[-1]["result"] == ra.SKIPPED_STALE


def test_revert_refuses_a_normal_worklist(tmp_path):
    path = _write_worklist(tmp_path, [_row()])
    service = FakeService({"vid1": {"title": NEW_TITLE}})
    assert _revert(path, service, *APPLY) == 2
    assert service.videos().update_calls == []


def test_revert_needs_both_flags(tmp_path):
    rev, service = _forward_then_revert_file(tmp_path)
    assert _revert(rev, service, "--apply") == 2


def test_revert_does_not_run_place_gates(tmp_path):
    rev = tmp_path / "r.revert.csv"
    rev.write_text(
        "video_id,current_title,restore_title,current_description,restore_description\n"
        "vid1,New Title,To to Somewhere in 4K,,\n"  # restore text would fail grammar gates
    )
    service = FakeService({"vid1": {"title": "New Title"}})
    _revert(rev, service, *APPLY)
    assert service.videos().snippets["vid1"]["title"] == "To to Somewhere in 4K"
