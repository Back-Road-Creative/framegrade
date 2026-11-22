"""Re-render a drive and replace its still-scheduled uploads (``--replace-scheduled``).

The classifier fails closed; Phase D keeps public LFs, uploads a replaced LF with the old
``publishAt`` and title, and deletes the old video only after the new upload succeeded."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gomoveshift.pipeline import replace_scheduled as rs
from gomoveshift.pipeline.content_ledger import ContentEntry, ContentLedger
from gomoveshift.pipeline.drive_phases import external_systems_coordinator as esc

from _state_helpers import make_state

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


def _rec(vid, when, **kw):
    return {"video_id": vid, "url": f"https://youtu.be/{vid}", "publish_at": when.isoformat(),
            "title": f"T {vid}", "uploaded_at": "2026-11-01T00:00:00+00:00",
            "playlist_id": "PL1", **kw}  # fmt: skip


PUBLIC = _rec("pub", NOW - timedelta(days=2))
SOON = NOW + timedelta(days=5)
LATER = NOW + timedelta(days=12)


def _live(states):
    return lambda ids: {i: states[i] for i in ids if i in states}


def test_all_scheduled_is_accepted():
    uploads = {"1": _rec("a", SOON), "2": _rec("b", LATER)}
    plan, why = rs.classify_uploads(uploads, NOW, _live({"a": "scheduled", "b": "scheduled"}))
    assert why == "" and plan["kept"] == {} and set(plan["replace"]) == {"1", "2"}
    assert plan["replace"]["1"]["video_id"] == "a" and plan["replace"]["1"]["playlist_id"] == "PL1"


def test_public_and_scheduled_mix_keeps_the_public_lf():
    uploads = {"1": PUBLIC, "2": _rec("b", SOON), "3": _rec("c", LATER)}
    live = _live({"b": "scheduled", "c": "scheduled"})
    plan, why = rs.classify_uploads(uploads, NOW, live)
    assert why == "" and set(plan["kept"]) == {"1"} and set(plan["replace"]) == {"2", "3"}


def test_a_record_inside_the_margin_refuses():
    near = _rec("n", NOW + timedelta(hours=rs.REPLACE_MARGIN_H - 1))
    plan, why = rs.classify_uploads({"1": near}, NOW, _live({"n": "scheduled"}))
    assert plan is None and "margin" in why


def test_a_live_check_that_says_public_refuses():
    plan, why = rs.classify_uploads({"1": _rec("a", SOON)}, NOW, _live({"a": "public"}))
    assert plan is None and "public" in why


def test_a_video_missing_on_youtube_refuses():
    plan, why = rs.classify_uploads({"1": _rec("a", SOON)}, NOW, _live({}))
    assert plan is None and "not found" in why


def test_a_failing_live_lookup_refuses():
    def boom(ids):
        raise RuntimeError("quota")

    plan, why = rs.classify_uploads({"1": _rec("a", SOON)}, NOW, boom)
    assert plan is None and "live YouTube check failed" in why


def test_unreadable_records_refuse():
    for bad in ("bareVideoId", {"video_id": "a"}, {"video_id": "a", "publish_at": "soon"}):
        plan, why = rs.classify_uploads({"1": bad}, NOW, _live({"a": "scheduled"}))
        assert plan is None and "not decidable" in why


def test_no_live_client_refuses():
    plan, why = rs.classify_uploads({"1": _rec("a", SOON)}, NOW, None)
    assert plan is None and "live" in why


# --- Phase D --------------------------------------------------------------------------------

OLD_AT = NOW + timedelta(days=9)


def _plan(tmp_path, *, replace=True):
    plan = {"schema": rs.REPLACE_PLAN_SCHEMA, "kept": {"1": PUBLIC},
            "replace": {"2": _rec("old2", OLD_AT, title="Old Two")}}  # fmt: skip
    rs.write_plan(tmp_path, plan)


def _run(tmp_path, live_state="scheduled"):
    calls = []
    yt = MagicMock()
    yt.video_states.side_effect = lambda ids: {i: live_state for i in ids}
    yt.delete_video.side_effect = lambda vid: calls.append(("delete", vid))

    def insert(**kw):
        calls.append(("upload", kw["lf"].index, kw["publish_at"], kw["title"]))
        rec = {"video_id": f"new{kw['lf'].index}", "publish_at": kw["publish_at"].isoformat(),
               "title": kw["title"]}  # fmt: skip
        kw["state"].youtube_uploads[kw["lf"].index] = rec
        return rec

    config = MagicMock()
    config.seo.storage_path = tmp_path / "seo"
    sched = SimpleNamespace(lf_slots=[SimpleNamespace(
        datetime_est=(NOW + timedelta(days=d)).isoformat(), date="x") for d in (30, 31, 32)])  # fmt: skip
    state = make_state()
    with (
        patch.object(esc, "_resolve_youtube_uploader_class", return_value=lambda: yt),
        patch.object(esc, "_upload_lf_to_youtube", side_effect=insert),
        patch.object(
            esc, "_book_lf_upload_in_ledger", side_effect=lambda **kw: calls.append(("book",))
        ),  # fmt: skip
    ):
        esc._upload_lfs_to_youtube(
            drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=i) for i in (1, 2, 3)]),
            regular_lf_paths=["/v/1", "/v/2", "/v/3"],
            metadata_per_lf={i: {"title": f"New {i}"} for i in (1, 2, 3)},
            drive_schedule=sched, youtube_uploads={}, state=state, output_base=tmp_path,
            config=config, save_state=lambda s: None, skip_youtube_upload=False,
            youtube_upload_enabled=True, clock=lambda: NOW,
        )  # fmt: skip
    return calls, state


def test_a_kept_lf_is_never_reuploaded(tmp_path):
    _plan(tmp_path)
    calls, state = _run(tmp_path)
    assert 1 not in [c[1] for c in calls if c[0] == "upload"]
    assert state.youtube_uploads[1]["video_id"] == "pub"


def test_replace_uploads_new_with_the_old_slot_and_title_then_deletes_old(tmp_path):
    _plan(tmp_path)
    calls, _ = _run(tmp_path)
    up = [c for c in calls if c[0] == "upload" and c[1] == 2][0]
    assert up[2] == OLD_AT and up[3] == "Old Two"
    assert calls.index(up) < calls.index(("delete", "old2"))
    assert json.loads((tmp_path / rs.REPLACE_PLAN_FILENAME).read_text())["deleted"] == ["old2"]


def test_an_old_video_that_went_public_is_kept_and_not_replaced(tmp_path):
    _plan(tmp_path)
    calls, state = _run(tmp_path, live_state="public")
    assert not [c for c in calls if c[0] == "upload" and c[1] == 2]
    assert ("delete", "old2") not in calls
    assert state.youtube_uploads[2]["video_id"] == "old2"
    assert "2" in json.loads((tmp_path / rs.REPLACE_PLAN_FILENAME).read_text())["kept_live"]


def test_no_plan_leaves_phase_d_unchanged(tmp_path):
    calls, _ = _run(tmp_path)
    assert [c[1] for c in calls if c[0] == "upload"] == [1, 2, 3]
    assert not [c for c in calls if c[0] == "delete"]


def test_ledger_entry_follows_the_new_video_id(tmp_path):
    path = tmp_path / ".content_ledger.json"
    ledger = ContentLedger()
    ledger.add(ContentEntry(date="2026-12-01", time="15:00:00", content_type="longform",
                            title="t", video_id="old2"))  # fmt: skip
    ledger.save(path)
    assert rs.swap_ledger_video_id(path, "old2", "new2")
    assert [e.video_id for e in ContentLedger.load(path).entries] == ["new2"]
