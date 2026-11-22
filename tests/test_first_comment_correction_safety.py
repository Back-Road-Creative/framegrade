"""A correction never deletes the intended comment or a stranger's, follows a second replace and
never goes silent at its cap (YouTube + Meta). No network: fake clients."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

import pytest

from gomoveshift.publishing import first_comment, meta_comments, youtube_comments
from gomoveshift.publishing.first_comment_stale import (
    CAPPED_KIND,
    KIND,
    raise_stale_incident,
    run_corrections,
)
from gomoveshift.publishing.queue import (
    FIRST_COMMENT_MAX_ATTEMPTS as CAP,
    first_comment_corrections,
    open_queue_db,
    pending_first_comment_corrections,
    record_stale_live_comment,
)
from gomoveshift.publishing.replace_relink import relink_replaced_lf
from tests.test_first_comment_meta_correction import FakeGraph, _adapter
from tests.test_first_comment_meta_correction import _drain as _meta_drain
from tests.test_first_comment_meta_correction import _relinked as _meta_relinked
from tests.test_first_comment_stale import (
    NEW_TEXT,
    OLD_TEXT,
    OWN,
    _drain,
    _incidents,
    _published,
    _rewrite,
    _row,
    _Service,
    _thread,
)
from tests.test_youtube_first_comment import _make_http_error

_NOW = "2026-08-23T00:06:00+00:00"


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "q")
    try:
        yield c
    finally:
        c.close()


@pytest.fixture
def inc(tmp_path):
    return tmp_path / "incidents"


def _upd(conn, sets, *args):
    with conn:
        conn.execute(f"UPDATE deliveries SET {sets}", args)  # noqa: S608


def _rows(row):
    return [(i["live_id"], i["action"]) for i in first_comment_corrections(row)]


def test_a_concurrent_race_keeps_the_new_comment_and_deletes_the_old_one(conn, inc):
    did = _published(conn)
    _rewrite(conn, did)
    r = youtube_comments.FirstCommentResult()
    youtube_comments.record_found_comments(conn, did, OLD_TEXT, "X", OLD_TEXT, (), r, inc)
    youtube_comments.record_found_comments(conn, did, NEW_TEXT, "Y", NEW_TEXT, (), r, inc)
    row = _row(conn, did)
    assert _rows(row) == [("X", "delete")] and row["first_comment_text"] == NEW_TEXT
    svc = _Service(fail=[_make_http_error(403, "forbidden")])
    _drain(conn, svc, inc)
    assert svc.comments_api.deletes == ["X"] and svc.comments_api.updates == []  # Y never touched
    ok = _Service()
    _drain(conn, ok, inc)
    assert ok.comments_api.deletes == ["X"] and not first_comment_corrections(_row(conn, did))


def test_a_second_mark_of_the_same_text_never_queues_a_delete_of_the_intended_comment(conn, inc):
    did = _published(conn)
    r = youtube_comments.FirstCommentResult()
    youtube_comments.mark_posted(conn, did, OLD_TEXT, r, "Y", inc)
    youtube_comments.mark_posted(conn, did, OLD_TEXT, r, "Y", inc)  # the same comment, again
    youtube_comments.record_live_comment(conn, did, OLD_TEXT, "Y", r, inc)
    assert not first_comment_corrections(_row(conn, did)) and not _incidents(inc)
    assert record_stale_live_comment(conn, did, OLD_TEXT, _NOW, "Y") is None
    # a second copy beside the recorded one gets the delete; the recorded one never does
    youtube_comments.mark_posted(conn, did, OLD_TEXT, r, "Y2", inc)
    assert _rows(_row(conn, did)) == [("Y2", "delete")]


def test_a_failed_item_stops_the_rows_other_items_this_pass(conn, inc):
    did = _published(conn)
    _rewrite(conn, did)
    assert record_stale_live_comment(conn, did, OLD_TEXT, _NOW, "X")
    assert record_stale_live_comment(conn, did, "someone else's text", _NOW, "Z")
    assert _rows(_row(conn, did)) == [("X", "edit"), ("Z", "delete")]
    svc = _Service(fail=[_make_http_error(400, "invalidValue")])
    _drain(conn, svc, inc)
    assert svc.comments_api.updates == [("X", NEW_TEXT)] and svc.comments_api.deletes == []


def test_a_delete_of_the_intended_text_is_never_run(conn, inc):
    did = _published(conn)
    gone = json.dumps([{"live_id": "Y", "live_text": OLD_TEXT, "action": "delete"}])
    _upd(conn, "first_comment_posted_at=?, first_comment_correction=? WHERE id=?", _NOW, gone, did)
    svc = _Service()
    _drain(conn, svc, inc)
    assert svc.comments_api.deletes == [] and not first_comment_corrections(_row(conn, did))


def test_both_live_returns_the_current_text_and_dupes_the_prior(conn):
    svc = _Service(pages=[{"items": [_thread(OLD_TEXT, "t-old"), _thread(NEW_TEXT, "t-new")]}])
    live_id, said, dupes = youtube_comments.post_first_comment_reporting(
        svc, "vid", NEW_TEXT, also_accept=(OLD_TEXT,)
    )
    assert (live_id, said, dupes) == ("t-new", NEW_TEXT, (("t-old", OLD_TEXT),))


def test_a_viewers_comment_with_the_old_text_is_never_edited_and_ours_is(conn, inc):
    did = _published(conn)
    _rewrite(conn, did)
    page = {"items": [_thread(OLD_TEXT, "viewer", author="UC-other"), _thread(OLD_TEXT, "mine")]}
    _drain(conn, _Service(pages=[page]), inc)
    assert _rows(_row(conn, did)) == [("mine", "edit")]
    svc = _Service()
    _drain(conn, svc, inc)
    assert svc.comments_api.updates == [("mine", NEW_TEXT)] and svc.comments_api.deletes == []


def test_only_a_viewers_old_comment_means_the_new_text_is_posted(conn, inc):
    did = _published(conn)
    _rewrite(conn, did)
    svc = _Service(pages=[{"items": [_thread(OLD_TEXT, "viewer", author="UC-other")]}])
    result = _drain(conn, svc, inc)
    assert svc.inserted == [NEW_TEXT] and result.posted == 1 and not _incidents(inc)


def test_a_thread_with_no_author_is_not_ours(conn, inc):
    did = _published(conn)
    _rewrite(conn, did)
    bare = {"id": "t", "snippet": {"topLevelComment": {"snippet": {"textOriginal": OLD_TEXT}}}}
    svc = _Service(pages=[{"items": [bare]}])
    _drain(conn, svc, inc)
    assert svc.inserted == [NEW_TEXT]


@pytest.mark.parametrize("answer", [{"items": []}, RuntimeError("boom")])
def test_an_unreadable_own_channel_fails_closed_and_says_why(conn, inc, answer):
    did = _published(conn)
    _rewrite(conn, did)

    class NoChannel(_Service):
        def channels(self):
            def go(**kw):
                if isinstance(answer, Exception):
                    raise answer
                return SimpleNamespace(execute=lambda: answer)

            return SimpleNamespace(list=go)

    svc = NoChannel(pages=[{"items": [_thread(OLD_TEXT, "t")]}])
    result = _drain(conn, svc, inc)
    assert svc.inserted == [] and svc.comments_api.updates == [] and result.failed == 1
    assert "whose comment" in _row(conn, did)["first_comment_error"]
    assert "whose comment" in result.errors[0]


def test_forbidden_is_a_refusal_not_a_scope_gap_and_counts_to_the_cap(conn, inc):
    did = _published(conn)
    _drain(conn, _Service(on_insert=lambda: _rewrite(conn, did)), inc)
    for n in range(1, CAP + 1):
        svc = _Service(fail=[_make_http_error(403, "forbidden")])
        result = _drain(conn, svc, inc)
        assert svc.comments_api.updates and not result.needs_reauth
        assert _row(conn, did)["first_comment_correction_attempts"] == n
    after = _Service()
    _drain(conn, after, inc)
    assert after.comments_api.updates == [] and after.comments_api.deletes == []


def test_only_insufficient_permissions_is_a_scope_gap_for_a_correction():
    assert youtube_comments._is_scope_error(_make_http_error(403, "insufficientPermissions"))
    forbidden = _make_http_error(403, "forbidden")
    assert youtube_comments._refused_for_a_correction(forbidden)
    assert not youtube_comments._refused_for_a_correction(
        _make_http_error(403, "insufficientPermissions")
    )


@pytest.mark.parametrize("platform", ["facebook", "instagram"])
def test_a_strangers_old_comment_on_meta_is_left_alone_and_ours_is_corrected(
    conn, inc, tmp_path, platform
):
    from tests.test_first_comment_meta_correction import OLD, NEW

    did = _meta_relinked(conn, platform)
    graph = FakeGraph(platform, [("v1", OLD, "stranger"), ("c1", OLD)])
    _meta_drain(conn, platform, graph, inc)
    assert _rows(_row(conn, did)) == [("c1", "edit")]
    _meta_drain(conn, platform, graph, inc)
    assert graph.comments["v1"] == OLD and "c1" not in graph.comments or graph.comments["c1"] == NEW
    assert not [e for e in graph.log if e == "delete:v1" or e == "edit:v1"]


@pytest.mark.parametrize("platform", ["facebook", "instagram"])
def test_a_viewers_old_comment_beside_our_current_one_is_never_touched(conn, inc, platform):
    from tests.test_first_comment_meta_correction import NEW, OLD

    did = _meta_relinked(conn, platform)
    graph = FakeGraph(platform, [("v1", OLD, "stranger"), ("c1", NEW)])
    for _ in range(2):
        assert _meta_drain(conn, platform, graph, inc).stale == 0
    assert graph.comments == {"v1": OLD, "c1": NEW} and graph.log == ["list"]
    assert not first_comment_corrections(_row(conn, did))


def test_a_viewers_old_comment_beside_our_current_one_on_youtube_is_left_alone(conn, inc):
    did = _published(conn)
    _rewrite(conn, did)
    page = {"items": [_thread(OLD_TEXT, "viewer", author="UC-other"), _thread(NEW_TEXT, "mine")]}
    for _ in range(2):
        svc = _Service(pages=[page])
        _drain(conn, svc, inc)
        assert (svc.inserted, svc.comments_api.updates, svc.comments_api.deletes) == ([], [], [])
    assert not first_comment_corrections(_row(conn, did)) and not _incidents(inc)


def test_the_meta_check_needs_our_id_only_when_a_correction_could_follow():
    graph = FakeGraph("facebook", [("a", "hello", "someone"), ("b", "old text")])
    kw = dict(text_field="message", permission="p", raise_for_error=lambda s, b: None)
    first_comment.post_graph_comment(graph, "https://g", "obj", "new", "t", **kw)
    assert graph.log[-1].startswith("post:")  # no own id, no replaced text: posted
    with pytest.raises(first_comment.CommentCheckError, match="own account id"):
        first_comment.post_graph_comment(
            graph, "https://g", "obj", "new", "t", also_accept=("old text",), **kw
        )
    assert not [e for e in graph.log if e.startswith(("delete:", "edit:"))]


def test_a_meta_comment_without_from_is_not_ours():
    graph = FakeGraph("facebook", [("a", "x")])
    graph.authors["a"] = ""
    comments, _ = first_comment._read_graph_comments(
        graph, "https://g", "obj", "t", text_field="message", permission="p",
        raise_for_error=lambda s, b: None, own_id="p",
    )  # fmt: skip
    assert comments == [("a", "x", False)]


def test_a_meta_replace_fails_closed_without_our_own_id():
    graph = FakeGraph("instagram", [("a", "x")])
    with pytest.raises(first_comment.CommentCheckError, match="own account id"):
        first_comment.replace_graph_comment(
            graph, "https://g", "obj", "a", "new", "t", text_field="text", permission="p",
            raise_for_error=lambda s, b: None,
        )  # fmt: skip
    assert graph.log == []


_PARENT_AT = "2026-11-01T00:00:00+00:00"
_SLOT = "2026-11-10T00:00:00+00:00"


def _comment(vid):
    return f"\U0001f3ac Watch the full drive: https://youtu.be/{vid}"


def test_a_second_replace_rewrites_the_pending_edit_and_reports_the_row(conn, inc, tmp_path):
    a, b, c = "aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"
    did = _published(conn)
    _upd(conn, "first_comment_text=?, scheduled_at=? WHERE id=?", _comment(a), _SLOT, did)
    now = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
    relink_replaced_lf("proj", tmp_path, {a: (b, _PARENT_AT)}, conn=conn, now=now)
    page = {"items": [_thread(_comment(a), "X")]}
    _drain(conn, _Service(pages=[page]), inc)
    assert first_comment_corrections(_row(conn, did))[0]["to"] == _comment(b)
    dry = relink_replaced_lf(
        "proj", tmp_path, {b: (c, _PARENT_AT)}, conn=conn, now=now, dry_run=True
    )
    assert [r.delivery_id for r in dry.queue_rewritten] == [did]
    assert first_comment_corrections(_row(conn, did))[0]["to"] == _comment(b)  # dry run: no write
    report = relink_replaced_lf("proj", tmp_path, {b: (c, _PARENT_AT)}, conn=conn, now=now)
    assert [(r.delivery_id, r.written) for r in report.queue_rewritten] == [(did, True)]
    svc = _Service()
    _drain(conn, svc, inc)
    assert svc.comments_api.updates == [("X", _comment(c))]
    assert _row(conn, did)["first_comment_text"] == _comment(c)


def test_a_404_on_a_delete_counts_as_resolved(conn, inc):
    did = _published(conn)
    _rewrite(conn, did)
    r = youtube_comments.FirstCommentResult()
    youtube_comments.record_found_comments(conn, did, OLD_TEXT, "X", OLD_TEXT, (), r, inc)
    youtube_comments.record_found_comments(conn, did, NEW_TEXT, "Y", NEW_TEXT, (), r, inc)
    result = _drain(conn, _Service(fail=[_make_http_error(404, "commentNotFound")]), inc)
    row = _row(conn, did)
    assert result.corrected == 1 and result.stale == 0
    assert not first_comment_corrections(row) and row["first_comment_error"] is None
    assert row["first_comment_correction_attempts"] == 0
    assert _incidents(inc)[0]["resolved_at"]


def test_a_404_on_an_edit_unposts_the_row_so_the_new_text_is_posted_again(conn, inc):
    did = _published(conn)
    _drain(conn, _Service(on_insert=lambda: _rewrite(conn, did)), inc)
    svc = _Service(fail=[_make_http_error(404, "commentNotFound")])
    result = _drain(conn, svc, inc)  # the same pass unposts, then the normal drain reposts
    row = _row(conn, did)
    assert svc.inserted == [NEW_TEXT] and result.posted == 1
    assert row["first_comment_text"] == NEW_TEXT and row["first_comment_posted_at"]
    assert not first_comment_corrections(row) and row["first_comment_error"] is None
    assert row["first_comment_correction_attempts"] == 0
    assert _incidents(inc)[0]["resolved_at"]


def _cap_the_correction(conn, inc, did):
    _drain(conn, _Service(on_insert=lambda: _rewrite(conn, did)), inc)
    for _ in range(CAP):
        _drain(conn, _Service(fail=[_make_http_error(400, "invalidValue")]), inc)


def test_the_two_counters_are_separate(conn, inc):
    did = _published(conn)
    _cap_the_correction(conn, inc, did)
    row = _row(conn, did)
    assert row["first_comment_correction_attempts"] == CAP and row["first_comment_attempts"] == 0


def test_a_capped_row_stays_reported_and_gets_a_distinct_capped_incident(conn, inc):
    did = _published(conn)
    _cap_the_correction(conn, inc, did)
    for _ in range(2):  # every later pass keeps saying so
        svc = _Service()
        result = _drain(conn, svc, inc)
        assert result.correction_capped == 1 and svc.comments_api.updates == []
    by_kind = {i["kind"]: i for i in _incidents(inc)}
    assert by_kind[KIND]["resolved_at"] and not by_kind[CAPPED_KIND].get("resolved_at")
    assert "by hand" in by_kind[CAPPED_KIND]["detail"]
    _upd(conn, "first_comment_correction=NULL WHERE id=?", did)  # a human cleared it
    assert _drain(conn, _Service(), inc).correction_capped == 0
    assert {i["kind"]: i for i in _incidents(inc)}[CAPPED_KIND]["resolved_at"]


def test_a_row_at_the_cap_is_not_pending(conn, inc):
    did = _published(conn)
    _drain(conn, _Service(on_insert=lambda: _rewrite(conn, did)), inc)
    assert [r["id"] for r in pending_first_comment_corrections(conn, ("youtube",))] == [did]
    _upd(conn, "first_comment_correction_attempts=? WHERE id=?", CAP, did)
    assert pending_first_comment_corrections(conn, ("youtube",)) == []
    _upd(
        conn,
        "first_comment_correction_attempts=?, first_comment_attempts=0 WHERE id=?",
        CAP - 1,
        did,
    )
    assert len(pending_first_comment_corrections(conn, ("youtube",))) == 1


def test_the_summary_last_line_keeps_the_capped_count(tmp_path, capsys):
    from gomoveshift.cli.handlers.dispatch_queue import cmd_dispatch_queue
    from gomoveshift.publishing.dispatch import DrainResult
    from gomoveshift.publishing.queue import enqueue, open_queue_db
    from tests.test_cli_dispatch_queue import _args, _config, _record, _sidecar

    c = open_queue_db(tmp_path)
    enqueue(c, "proj", 3, "tiktok", "HASH1")
    c.close()
    _sidecar(tmp_path, _record())
    fake = youtube_comments.FirstCommentResult(correction_capped=1)
    with (
        mock.patch("gomoveshift.cli.handlers.dispatch_queue._build_routed_handlers"),
        mock.patch(
            "gomoveshift.cli.handlers.dispatch_queue.drain_queue",
            return_value=DrainResult(attempted=1, succeeded=1),
        ),
        mock.patch(
            "gomoveshift.cli.handlers.dispatch_queue.drain_first_comments", return_value=fake
        ),
    ):
        rc = cmd_dispatch_queue(_args(apply=True), _config(tmp_path, youtube="first_party"))
    assert rc == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert "first_comment_correction_capped=1" in lines[-1]


def _relink_text(conn, tmp_path, did):
    now = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
    with conn:  # every row's slot is after the replacement's publish time: the gate passes
        conn.execute("UPDATE deliveries SET scheduled_at=? WHERE scheduled_at IS NULL", (_SLOT,))
    report = relink_replaced_lf(
        "proj", tmp_path, {"oldvideo_01": ("newvideo_02", _PARENT_AT)}, conn=conn, now=now
    )
    return [i for i in report.operator_items if f"delivery {did} " in i], report


def _posted_stale(conn, platform, items, attempts=0, post_id="media1"):
    from gomoveshift.publishing.queue import enqueue

    did = enqueue(conn, "proj", 5, platform, "h5", first_comment_text=_comment("oldvideo_01"))
    with conn:
        conn.execute(
            "UPDATE deliveries SET status='published', platform_post_id=?, platform_upload_id=?, "
            "first_comment_posted_at=?, first_comment_error=?, first_comment_correction=?, "
            "first_comment_correction_attempts=? WHERE id=?",
            (post_id, post_id if platform == "youtube" else None, _NOW,
             "stale comment live: links old", json.dumps(items) if items else None, attempts, did),
        )  # fmt: skip
    return did


@pytest.mark.parametrize("platform", ["facebook", "instagram", "youtube"])
def test_the_drain_corrects_wording_only_while_pending_and_uncapped(conn, tmp_path, platform):
    item = {"live_id": "c1", "live_text": "t", "action": "edit", "to": _comment("newvideo_02")}
    did = _posted_stale(conn, platform, [item])
    (note,), _ = _relink_text(conn, tmp_path, did)
    assert "the drain corrects it" in note and "by hand" not in note


@pytest.mark.parametrize("platform", ["facebook", "instagram"])
@pytest.mark.parametrize("case", ["capped", "id_unknown", "no_correction"])
def test_a_meta_row_the_drain_cannot_correct_says_fix_it_by_hand(conn, tmp_path, platform, case):
    item = {"live_id": "" if case == "id_unknown" else "c1", "live_text": "t", "action": "edit",
            "to": _comment("newvideo_02")}  # fmt: skip
    items = [] if case == "no_correction" else [item]
    did = _posted_stale(conn, platform, items, attempts=CAP if case == "capped" else 0)
    (note,), _ = _relink_text(conn, tmp_path, did)
    assert "fix it by hand" in note and "the drain corrects it" not in note


def test_an_instagram_edit_with_no_media_id_is_a_hand_step(conn, tmp_path):
    item = {"live_id": "c1", "live_text": "t", "action": "edit", "to": _comment("newvideo_02")}
    did = _posted_stale(conn, "instagram", [item], post_id=None)
    (note,), _ = _relink_text(conn, tmp_path, did)
    assert "fix it by hand" in note


@pytest.mark.parametrize("platform", ["facebook", "instagram"])
def test_drain_will_verify_lists_a_meta_row_only_if_the_drain_reads_its_post(
    conn, tmp_path, platform
):
    from gomoveshift.publishing.queue import enqueue

    def make(idx, post_id):
        did = enqueue(
            conn, "proj", idx, platform, f"h{idx}", first_comment_text=_comment("oldvideo_01")
        )
        _upd(conn, "status='published', platform_post_id=? WHERE id=?", post_id, did)
        return did

    ok, blind = make(1, "post1"), make(2, None)
    _, report = _relink_text(conn, tmp_path, ok)
    notes = {n for n in report.drain_will_verify}
    (checked,) = [n for n in notes if f"delivery {ok} " in n]
    (unchecked,) = [n for n in notes if f"delivery {blind} " in n]
    assert "checks whether the old text already went live" in checked
    assert "never checks" in unchecked and "by hand" in unchecked


def test_the_incident_refresh_also_refreshes_live_comments(conn, inc):
    did = _published(conn)
    _rewrite(conn, did)
    assert record_stale_live_comment(conn, did, OLD_TEXT, _NOW, "X")
    raise_stale_incident(conn, did, inc)
    assert record_stale_live_comment(conn, did, "someone else's text", _NOW, "Z")
    raise_stale_incident(conn, did, inc)
    (incident,) = _incidents(inc)
    assert [c["live_id"] for c in incident["live_comments"]] == ["X", "Z"]


def test_a_youtube_drain_never_resolves_a_meta_incident_and_the_reverse(conn, inc):
    yt = _published(conn)
    _drain(conn, _Service(on_insert=lambda: _rewrite(conn, yt)), inc)
    fb = _meta_relinked(conn, "facebook")
    graph = FakeGraph("facebook", [("c1", "Watch the full drive: https://youtu.be/old")])
    graph.comments["c1"] = "Watch the full drive: https://youtu.be/old"
    _meta_drain(conn, "facebook", graph, inc)  # a Facebook stale incident opens
    kinds = {(i["platform"], i["delivery_id"]) for i in _incidents(inc)}
    assert kinds == {("youtube", yt), ("facebook", fb)}
    _upd(conn, "first_comment_correction=NULL")  # both rows lose it: neither drain has them
    _drain(conn, _Service(), inc)  # a YouTube drain: the Facebook incident stays open
    open_now = {i["platform"] for i in _incidents(inc) if not i.get("resolved_at")}
    assert open_now == {"facebook"}
    _meta_drain(conn, "facebook", graph, inc)  # a Meta drain closes its own
    assert not {i["platform"] for i in _incidents(inc) if not i.get("resolved_at")}


def test_run_corrections_resolves_nothing_for_a_platform_it_was_not_given(conn, inc):
    did = _published(conn)
    _drain(conn, _Service(on_insert=lambda: _rewrite(conn, did)), inc)
    _upd(conn, "first_comment_correction=NULL")
    run_corrections(
        conn, platforms=("facebook",), client_for=lambda p: None, refused=lambda e: False,
        result=youtube_comments.FirstCommentResult(), incidents_dir=inc,
    )  # fmt: skip
    assert not _incidents(inc)[0].get("resolved_at")


# --- the parent-public gate holds at correction time; the incident never promises a stopped fix


class _Recorder:
    def __init__(self):
        self.calls = []

    def read_comment(self, comment_id):
        return "some stale text"

    def edit_comment(self, comment_id, text, object_id=None):
        self.calls.append(("edit", comment_id, text))

    def delete_comment(self, comment_id):
        self.calls.append(("delete", comment_id))


def _posted_with_edit(conn, parent_at, slot):
    did = _published(conn)
    _rewrite(conn, did)
    item = {"live_id": "c1", "live_text": OLD_TEXT, "action": "edit", "to": NEW_TEXT}
    if parent_at:
        item["parent_at"] = parent_at
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_posted_at=?, first_comment_text=?, "
            "first_comment_correction=?, first_comment_error='stale comment live: links old', "
            "scheduled_at=? WHERE id=?",
            (_NOW, OLD_TEXT, json.dumps([item]), slot, did),
        )
    return did


def _run(conn, inc, client):
    result = youtube_comments.FirstCommentResult()
    run_corrections(
        conn, platforms=("youtube",), client_for=lambda p: client, refused=lambda e: False,
        result=result, incidents_dir=inc,
    )  # fmt: skip
    return result


def test_an_edit_whose_parent_is_still_private_is_never_run_it_becomes_a_delete(conn, inc):
    did = _posted_with_edit(conn, "2026-08-24T00:00:00+00:00", "2026-08-23T00:00:00+00:00")
    client = _Recorder()
    result = _run(conn, inc, client)
    assert not [c for c in client.calls if c[0] == "edit"]  # never links the private video
    assert any("not public" in e for e in result.errors)
    row = _row(conn, did)
    (fix,) = first_comment_corrections(row)
    assert fix["action"] == "delete" and fix["live_id"] == "c1" and not row["first_comment_text"]
    again = _run(conn, inc, client)  # the next pass runs the delete
    assert client.calls == [("delete", "c1")] and again.corrected == 1
    assert not first_comment_corrections(_row(conn, did))


def test_an_edit_whose_parent_is_public_by_the_slot_still_runs(conn, inc):
    did = _posted_with_edit(conn, "2026-08-23T00:00:00+00:00", "2026-08-23T00:00:00+00:00")
    client = _Recorder()
    assert _run(conn, inc, client).corrected == 1
    assert client.calls == [("edit", "c1", NEW_TEXT)]
    assert not first_comment_corrections(_row(conn, did))


def test_a_capped_incident_never_says_the_drain_will_edit(conn, inc):
    did = _posted_with_edit(conn, None, "2026-08-23T00:00:00+00:00")
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_correction_attempts=? WHERE id=?", (CAP, did)
        )
    raise_stale_incident(conn, did, inc)  # the pass whose refusal reached the cap
    (incident,) = _incidents(inc)
    assert "the drain will" not in incident["detail"]
    assert "stopped trying" in incident["detail"] and "edit comment c1" in incident["detail"]
