"""An empty relink, a trusted "gone" answer or a private relink never blanks, doubles or hides a
first comment. Each test asserts the GOOD outcome. No network: the repo's own fakes."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from gomoveshift.publishing import queue
from gomoveshift.publishing.facebook_adapter import FacebookReelsAdapter
from gomoveshift.publishing.youtube_comments import FirstCommentResult
from gomoveshift.publishing.queue import (
    first_comment_corrections,
    open_queue_db,
    rewrite_unposted_first_comment,
)
from tests.test_first_comment_correction_safety import _posted_with_edit
from tests.test_first_comment_meta_correction import NEW, OLD, FakeGraph, Resp, _adapter
from tests.test_first_comment_meta_correction import _drain as _meta_drain
from tests.test_first_comment_meta_correction import _open_incidents
from tests.test_first_comment_meta_correction import _relinked as _meta_relinked
from tests.test_first_comment_meta_correction import _row as _meta_row
from tests.test_first_comment_stale import (
    _NOW,
    OLD_TEXT,
    _drain,
    _published,
    _row,
    _Service,
)

SLOT = "2026-11-10T00:00:00+00:00"
LATE = "2026-11-20T00:00:00+00:00"  # the replacement goes public after the slot
RELINK_NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


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


def _ids(row):
    return [(i["live_id"], i["action"]) for i in first_comment_corrections(row)]


# --- F4: an empty current text is a delete, never an edit to "" --------------------------------


def test_a_private_relink_during_the_post_queues_a_delete_never_an_edit_to_empty(conn, inc):
    did = _published(conn)
    with conn:
        conn.execute("UPDATE deliveries SET scheduled_at=? WHERE id=?", (SLOT, did))

    def relink_private():
        assert rewrite_unposted_first_comment(conn, did, OLD_TEXT, None, _NOW)

    _drain(conn, _Service(on_insert=relink_private), inc)
    items = first_comment_corrections(_row(conn, did))
    assert [(i["live_id"], i["action"]) for i in items] == [("new-thread", "delete")]
    assert all(not i.get("to") for i in items)
    nxt = _Service()
    _drain(conn, nxt, inc)
    assert nxt.comments_api.updates == []  # never blanked
    assert nxt.comments_api.deletes == ["new-thread"]  # the dead-link comment goes
    assert not first_comment_corrections(_row(conn, did))


# --- F6: a 100/33 "gone" answer for a comment that still reads is not trusted ------------------


class _GoneLies(FakeGraph):
    """Graph answers 100/33 on a DELETE of a comment that is still there (the pair's generic
    message also covers 'cannot be loaded due to missing permissions' -- UNVERIFIED)."""

    def get(self, url, params=None, **kw):
        if not url.endswith("/comments"):  # a GET of the one comment lies the same way
            self.log.append(f"read-100-33:{url.rsplit('/', 1)[1]}")
            return Resp({"error": {"code": 100, "error_subcode": 33, "message": "no such"}}, 400)
        return super().get(url, params=params, **kw)

    def delete(self, url, **kw):
        cid = url.rsplit("/", 1)[1]
        self.log.append(f"delete-100-33:{cid}")
        return Resp({"error": {"code": 100, "error_subcode": 33, "message": "no such"}}, 400)


def test_a_100_33_on_a_listed_own_comment_never_lets_the_replacement_post_beside_it(conn, inc):
    did = _meta_relinked(conn, "instagram")
    graph = _GoneLies("instagram", [("c1", OLD)])
    _meta_drain(conn, "instagram", graph, inc)
    assert _ids(_meta_row(conn, did)) == [("c1", "edit")]
    res = _meta_drain(conn, "instagram", graph, inc)
    assert graph.comments == {"c1": OLD}  # nothing posted beside the live dead link
    assert res.corrected == 0
    assert _ids(_meta_row(conn, did)) == [("c1", "edit")]  # still pending, never resolved
    assert not [e for e in graph.log if e.startswith("post:")]


@pytest.mark.parametrize("platform", ["facebook", "instagram"])
def test_a_plain_delete_answered_100_33_for_a_readable_comment_is_not_done(platform):
    graph = _GoneLies(platform, [("c1", OLD)])
    adapter = _adapter(platform, graph)
    with pytest.raises(Exception):  # noqa: B017 -- any failure: the delete is not confirmed
        adapter.delete_comment("c1", object_id="post")
    assert graph.comments == {"c1": OLD}


@pytest.mark.parametrize("platform", ["facebook", "instagram"])
def test_x5_a_gone_answer_with_no_post_to_list_is_never_trusted(platform):
    graph = FakeGraph(platform, [])
    with pytest.raises(Exception):  # noqa: B017 -- absence cannot be confirmed without a listing
        _adapter(platform, graph).delete_comment("c1")


@pytest.mark.parametrize("platform", ["facebook", "instagram"])
def test_x5_a_gone_answer_when_the_listing_fails_leaves_the_delete_pending(platform):
    class _ListFails(FakeGraph):
        def get(self, url, params=None, **kw):
            if url.endswith("/comments"):
                raise ConnectionError("connection reset")
            return super().get(url, params=params, **kw)

    with pytest.raises(Exception):  # noqa: B017 -- never done on a listing that did not answer
        _adapter(platform, _ListFails(platform, [])).delete_comment("c1", object_id="post")


@pytest.mark.parametrize("platform", ["facebook", "instagram"])
def test_a_plain_delete_answered_100_33_for_an_absent_comment_is_done(platform):
    graph = FakeGraph(platform, [])
    # gone, and the listing does not show it: the goal holds
    _adapter(platform, graph).delete_comment("c1", object_id="post")
    assert "list" in graph.log


class _EditGone(FakeGraph):
    def post(self, url, data=None, **kw):
        if url.endswith("/comments"):
            return super().post(url, data=data, **kw)
        self.log.append("edit-100-33")
        return Resp({"error": {"code": 100, "error_subcode": 33, "message": "no such"}}, 400)


def test_an_edit_answered_100_33_is_not_a_success():
    graph = _EditGone("facebook", [("c1", OLD)])
    adapter = FacebookReelsAdapter({"page_id": "p", "page_access_token": "t"}, session=graph)
    with pytest.raises(Exception):  # noqa: B017 -- the edit did not land
        adapter.edit_comment("c1", NEW)
    assert graph.comments == {"c1": OLD}


# --- x3: an Instagram replace records the id of the comment it posted -------------------------------


def test_x3_an_instagram_replace_updates_the_recorded_live_id_so_a_later_scan_keeps_it(conn, inc):
    did = _meta_relinked(conn, "instagram")
    graph = FakeGraph("instagram", [("c1", OLD)])
    _meta_drain(conn, "instagram", graph, inc)  # finds the stale comment, queues the replace
    _meta_drain(conn, "instagram", graph, inc)  # deletes c1, posts new1
    assert graph.comments == {"new1": NEW}
    assert _meta_row(conn, did)["first_comment_live_id"] == "new1"
    for _ in range(3):  # a later drain that finds new1 never queues, or runs, a delete of it
        _meta_drain(conn, "instagram", graph, inc)
    assert graph.comments == {"new1": NEW} and not [e for e in graph.log if "delete:new1" in e]
    assert not first_comment_corrections(_meta_row(conn, did))


# --- x4: a deleted comment with nothing to replace it stays an open incident --------------------


def _open(inc, kind):
    rows = [json.loads(p.read_text()) for p in sorted(inc.glob("*.json"))] if inc.is_dir() else []
    return [r for r in rows if r.get("kind") == kind and not r.get("resolved_at")]


def test_x4_a_private_relink_delete_leaves_an_open_incident_until_the_comment_is_back(conn, inc):
    did = _published(conn)
    with conn:
        conn.execute("UPDATE deliveries SET scheduled_at=? WHERE id=?", (SLOT, did))
    _drain(
        conn,
        _Service(on_insert=lambda: rewrite_unposted_first_comment(conn, did, OLD_TEXT, None, _NOW)),
        inc,
    )
    nxt = _Service()
    result = _drain(conn, nxt, inc)
    assert nxt.comments_api.deletes == ["new-thread"]
    row = _row(conn, did)
    assert row["first_comment_posted_at"] and not row["first_comment_text"]  # posted_at kept
    (incident,) = _open(inc, "first_comment_removed")
    assert incident["delivery_id"] == did
    assert any("by hand" in e for e in result.errors)
    _drain(conn, _Service(), inc)  # another pass: still no comment, still open
    assert len(_open(inc, "first_comment_removed")) == 1
    with conn:  # the row has its comment text again
        conn.execute("UPDATE deliveries SET first_comment_text=? WHERE id=?", (OLD_TEXT, did))
    _drain(conn, _Service(), inc)
    assert not _open(inc, "first_comment_removed")


# --- F5 / F7: a private relink keeps its comment findable and tells the truth ------------------


def _relink(conn, tmp_path, did_map):
    from gomoveshift.publishing.replace_relink import relink_replaced_lf

    return relink_replaced_lf("proj", tmp_path, did_map, conn=conn, now=RELINK_NOW)


def test_a_private_relink_of_a_published_row_reports_a_possibly_live_old_comment(conn, tmp_path):
    did = _published(conn)
    with conn:
        conn.execute("UPDATE deliveries SET scheduled_at=? WHERE id=?", (SLOT, did))

    def lost():
        raise TimeoutError("read timed out")

    _drain(conn, _Service(on_insert=lost), tmp_path / "i")
    assert _row(conn, did)["first_comment_posted_at"] is None
    rep = _relink(conn, tmp_path, {"old": ("newvideo123", LATE)})
    assert _row(conn, did)["first_comment_text"] is None
    mine = [i for i in rep.operator_items + rep.drain_will_verify if f"delivery {did}" in i]
    assert any("by hand" in i and "OLD" in i for i in mine), mine  # reported, never silent


def test_a_posted_row_gets_the_hand_note_never_the_backfill_wording(conn, tmp_path):
    from tests.test_replace_relink import MAP as _MAP, _comment, _row as _rrow

    did = _rrow(conn, 1, posted="2026-11-22T10:00:00+00:00", slot="2026-11-22T11:00:00+00:00")
    from gomoveshift.publishing.replace_relink import relink_replaced_lf
    from tests.test_replace_relink import NOW, PROJECT

    rep = relink_replaced_lf(PROJECT, tmp_path, _MAP, conn=conn, now=NOW)
    (note,) = [i for i in rep.operator_items if f"delivery {did}" in i]
    assert "re-add the comment by hand once the video is public" in note
    assert "backfill" not in note
    assert (
        _comment("oLdVideo_01")
        in (
            conn.execute("SELECT first_comment_text FROM deliveries WHERE id=?", (did,)).fetchone()[
                0
            ]
        )
    )


def test_a_gated_edit_to_delete_note_says_re_add_by_hand(conn, tmp_path):
    from gomoveshift.publishing.replace_relink import relink_replaced_lf
    from tests.test_replace_relink import EARLY, MAP as _MAP, NOW, OLD as OLDV, PROJECT, _comment
    from tests.test_replace_relink import _row as _rrow

    did = _rrow(conn, 1, posted="2026-11-22T10:00:00+00:00", slot=EARLY)
    conn.execute(
        "UPDATE deliveries SET first_comment_error=?, first_comment_correction=? WHERE id=?",
        (
            "stale comment live: links zeroVid_00",
            json.dumps([{"live_id": "c1", "live_text": _comment("zeroVid_00"), "action": "edit",
                         "to": _comment(OLDV)}]),
            did,
        ),
    )  # fmt: skip
    conn.commit()
    rep = relink_replaced_lf(PROJECT, tmp_path, _MAP, conn=conn, now=NOW)
    (note,) = [i for i in rep.operator_items if f"delivery {did}" in i and "became a delete" in i]
    assert "re-add the comment by hand once the video is public" in note


# --- F8: the id map's new id and time are strings ----------------------------------------------


def test_a_non_str_new_id_or_time_raises_before_any_write(conn, tmp_path):
    from gomoveshift.publishing.replace_relink import _split, relink_replaced_lf

    did = _published(conn)
    with conn:
        conn.execute(
            "UPDATE deliveries SET scheduled_at=? WHERE id=?", ("2026-11-25T00:00:00+00:00", did)
        )
    with pytest.raises(TypeError):
        relink_replaced_lf("proj", tmp_path, {"old": (123, LATE)}, conn=conn, now=RELINK_NOW)
    assert _row(conn, did)["first_comment_text"] == OLD_TEXT
    aware = datetime(2026, 11, 20, tzinfo=timezone.utc)
    with pytest.raises(TypeError):
        _split({"a": ("b", aware)})
    assert _split({"a": ("b", LATE)}) == ({"a": "b"}, {"a": LATE})


# --- F9: drop_private_edit is a compare-and-set with a retry -----------------------------------


def test_drop_private_edit_keeps_a_delete_a_concurrent_drain_appended(conn, tmp_path, monkeypatch):
    did = _posted_with_edit(conn, "2026-08-24T00:00:00+00:00", "2026-08-23T00:00:00+00:00")
    other = open_queue_db(tmp_path / "q")
    real = queue.first_comment_corrections
    fired = []

    def racing(row):
        out = real(row)
        if not fired:
            fired.append(1)
            items = out + [{"live_id": "Z", "live_text": "dead", "action": "delete"}]
            with other:
                other.execute(
                    "UPDATE deliveries SET first_comment_correction=? WHERE id=?",
                    (json.dumps(items), did),
                )
        return out

    item = real(_row(conn, did))[0]
    monkeypatch.setattr(queue, "first_comment_corrections", racing)
    queue.drop_private_edit(conn, did, item, _NOW)
    monkeypatch.setattr(queue, "first_comment_corrections", real)
    other.close()
    assert sorted(_ids(_row(conn, did))) == [("Z", "delete"), ("c1", "delete")]
    assert _row(conn, did)["first_comment_text"] is None


# --- F10: a failing Meta (and YouTube) correction pass is visible on the CLI -------------------


def test_a_failing_meta_correction_pass_is_printed(tmp_path, capsys):
    from tests.test_first_comment_links import _dispatch

    boom = FirstCommentResult(errors=["stale-comment corrections failed: database is locked"])
    lines, _, _ = _dispatch(tmp_path, capsys, meta_result=boom, instagram="first_party")
    assert "  error: stale-comment corrections failed: database is locked" in lines


def test_a_failing_youtube_correction_pass_is_printed(tmp_path, capsys):
    from tests.test_first_comment_links import _dispatch

    boom = FirstCommentResult(errors=["stale-comment corrections failed: quota"])
    lines, _, _ = _dispatch(
        tmp_path, capsys, yt_result=boom, meta_result=FirstCommentResult(), youtube="first_party"
    )
    assert "  error: stale-comment corrections failed: quota" in lines


# --- F11: zero comments live after delete-then-post is never silent ----------------------------


def test_an_instagram_zero_comment_window_keeps_an_open_incident_and_counts_attempts(conn, inc):
    did = _meta_relinked(conn, "instagram")
    graph = FakeGraph("instagram", [("c1", OLD)])
    _meta_drain(conn, "instagram", graph, inc)
    graph.fail_post = "boom"
    failed = _meta_drain(conn, "instagram", graph, inc)
    assert graph.comments == {} and failed.failed == 1
    row = _meta_row(conn, did)
    assert row["first_comment_posted_at"] is None
    assert row["first_comment_attempts"] >= 1  # a transient failure still counts toward the cap
    (open_one,) = _open_incidents(inc)
    assert open_one["delivery_id"] == did and "repost" in open_one["kind"]
    graph.fail_post = None
    ok = _meta_drain(conn, "instagram", graph, inc)
    assert ok.posted == 1 and list(graph.comments.values()) == [NEW]
    _meta_drain(conn, "instagram", graph, inc)  # the next pass closes it, like the stale one
    assert not _open_incidents(inc)  # resolved once the comment is back


# --- N11: the plan sidecar's parent publish time follows the new id ----------------------------


def test_walk_plan_rewrites_parent_lf_publish_at_beside_the_id():
    from gomoveshift.publishing.replace_relink import _url_pattern, _walk_plan

    ids, times = {"old": "new"}, {"old": "2031-01-01T00:00:00+00:00"}
    plan = {
        "shorts": [
            {"parent_lf_video_id": "old", "parent_lf_publish_at": "2020-01-01T00:00:00+00:00"},
            {"parent_lf_video_id": "other", "parent_lf_publish_at": "2020-01-01T00:00:00+00:00"},
        ]
    }
    out, count = _walk_plan(plan, _url_pattern(ids), ids, "", times)
    assert count == 1
    assert out["shorts"][0] == {"parent_lf_video_id": "new", "parent_lf_publish_at": times["old"]}
    assert out["shorts"][1]["parent_lf_publish_at"] == "2020-01-01T00:00:00+00:00"
