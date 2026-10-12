"""``reconcile-playlists --prune / --undo / --audit``: the pure removal
planner, the ledgered delete/undo glue and the end-to-end ``reconcile`` wiring,
all against a fake service -- no live API calls."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from gomoveshift.pipeline import playlist_reconciler as pr

IRISH = "Just Driving — County Kerry, Ireland"
AMERICAN = "Colleton County — 4K 🚗"
USA_SHORTS, IRE_SHORTS, UNSORTED = pr.PL_USA_SHORTS, pr.PL_IRELAND_SHORTS, pr.PL_UNSORTED


def _rows(title, *vids):
    return [
        {"item_id": f"it:{title}:{v}", "video_id": v, "position": i} for i, v in enumerate(vids)
    ]


def _prune(items, desired=None, status=None, inserts=(), **kw):
    items = {t: _rows(t, *vids) for t, vids in items.items()}
    return pr.compute_prune_plan(items, desired or {}, status or {}, inserts=inserts, **kw)


NOW = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)


def _removed(plan):
    return sorted((d["playlist_title"], d["video_id"], d["reason"]) for d in plan["deletes"])


def _insert(video_id, title):
    return {"video_id": video_id, "playlist_title": title}


# --- compute_prune_plan: one test per class and per safety rule -----------------


def test_unsorted_entries_are_removed_when_the_video_is_filed_elsewhere():
    plan = _prune(
        {UNSORTED: ["a"], USA_SHORTS: ["a"]},
        {"a": [USA_SHORTS]},
        {"a": "public"},
    )
    assert _removed(plan) == [(UNSORTED, "a", "unsorted")]


def test_unsorted_public_video_in_no_other_playlist_is_kept():
    plan = _prune({UNSORTED: ["a"]}, {}, {"a": "public"})
    assert plan["deletes"] == []
    assert [(s["video_id"], s["why"]) for s in plan["skipped"]] == [("a", "last_managed_playlist")]


def test_deleted_video_is_removed_everywhere_it_is_listed():
    """A video absent from videos.list is deleted; it needs no routing to be pruned."""
    plan = _prune({USA_SHORTS: ["gone", "ok"], UNSORTED: ["gone"]}, {}, {"ok": "public"})
    assert _removed(plan) == [(USA_SHORTS, "gone", "deleted"), (UNSORTED, "gone", "deleted")]


def test_a_video_missing_from_the_manifest_is_not_deleted():
    """The uploads-derived manifest omits real videos: absence from ``desired`` is
    never a reason to remove an entry, only absence from videos.list is."""
    plan = _prune({USA_SHORTS: ["real"], IRE_SHORTS: ["real"]}, {}, {"real": "public"})
    assert plan["deletes"] == []


def test_nonpublic_entries_need_the_explicit_flag():
    items = {USA_SHORTS: ["p", "u", "ok"]}
    status = {"p": "private", "u": "unlisted", "ok": "public"}
    assert _prune(items, {}, status)["deletes"] == []
    plan = _prune(items, {}, status, include_nonpublic=True)
    assert _removed(plan) == [(USA_SHORTS, "p", "nonpublic"), (USA_SHORTS, "u", "nonpublic")]


def test_misfiled_entry_goes_once_every_desired_membership_is_live():
    plan = _prune({USA_SHORTS: ["a"], IRE_SHORTS: ["a"]}, {"a": [IRE_SHORTS]}, {"a": "public"})
    assert _removed(plan) == [(USA_SHORTS, "a", "misfiled")]


def test_misfiled_move_waits_for_the_insert_it_depends_on():
    items, desired, status = {USA_SHORTS: ["a"]}, {"a": [IRE_SHORTS]}, {"a": "public"}
    waiting = _prune(items, desired, status)
    assert waiting["deletes"] == []
    assert [s["why"] for s in waiting["skipped"]] == ["pending_insert"]
    planned = _prune(items, desired, status, inserts=[_insert("a", IRE_SHORTS)])
    assert _removed(planned) == [(USA_SHORTS, "a", "misfiled")]


def test_usa_video_keeps_its_live_state_playlist():
    desired = {"a": [pr.PL_USA_DRIVES]}
    plan = _prune({pr.PL_USA_DRIVES: ["a"], "Tennessee Drives": ["a"]}, desired, {"a": "public"})
    assert plan["deletes"] == []


def test_usa_superlong_video_keeps_its_live_state_playlist():
    desired = {"a": [pr.PL_USA_SUPERLONG]}
    plan = _prune({pr.PL_USA_SUPERLONG: ["a"], "Tennessee Drives": ["a"]}, desired, {"a": "public"})
    assert plan["deletes"] == []


def test_superlong_membership_is_never_removed():
    desired = {"a": [pr.PL_IRELAND_DRIVES]}
    plan = _prune(
        {pr.PL_IRELAND_DRIVES: ["a"], pr.PL_IRELAND_SUPERLONG: ["a"]}, desired, {"a": "public"}
    )
    assert plan["deletes"] == []


def test_highlight_in_region_drives_is_removed_when_routed_to_highlights():
    desired = {"h": [pr.PL_USA_HIGHLIGHTS]}
    plan = _prune({pr.PL_USA_DRIVES: ["h"], pr.PL_USA_HIGHLIGHTS: ["h"]}, desired, {"h": "public"})
    assert _removed(plan) == [(pr.PL_USA_DRIVES, "h", "misfiled")]


def test_a_public_video_keeps_a_managed_playlist_through_all_of_its_removals():
    plan = _prune(
        {UNSORTED: ["a"], USA_SHORTS: ["a"], IRE_SHORTS: ["a"]},
        {"a": [IRE_SHORTS]},
        {"a": "public"},
    )
    assert _removed(plan) == [(USA_SHORTS, "a", "misfiled"), (UNSORTED, "a", "unsorted")]
    # b's desired playlist is neither live nor planned: only the Unsorted entry may go,
    # the USA Shorts entry is the one place it is still filed.
    waiting = _prune({UNSORTED: ["b"], USA_SHORTS: ["b"]}, {"b": [IRE_SHORTS]}, {"b": "public"})
    assert _removed(waiting) == [(UNSORTED, "b", "unsorted")]
    assert [s["why"] for s in waiting["skipped"]] == ["pending_insert"]


# --- compute_audit -------------------------------------------------------------


def test_compute_audit_counts_every_check():
    items = {
        UNSORTED: _rows(UNSORTED, "u1"),
        USA_SHORTS: _rows(USA_SHORTS, "gone", "priv", "mis", "dual"),
        IRE_SHORTS: _rows(IRE_SHORTS, "mis", "dual"),
        pr.PL_USA_DRIVES: _rows(pr.PL_USA_DRIVES, "hl"),
        pr.PL_USA_HIGHLIGHTS: _rows(pr.PL_USA_HIGHLIGHTS, "hl"),
    }
    status = {k: "public" for k in ("u1", "mis", "dual", "hl")} | {"priv": "private"}
    desired = {"mis": [IRE_SHORTS], "dual": [IRE_SHORTS], "hl": [pr.PL_USA_HIGHLIGHTS]}
    inserts = [_insert("u1", IRE_SHORTS)]
    prune = pr.compute_prune_plan(items, desired, status, inserts=inserts)
    manifest = [
        {"id": "orphan", "privacy_status": "public"},
        {"id": "mis", "privacy_status": "public"},
        {"id": "priv", "privacy_status": "private"},
    ]
    audit = pr.compute_audit(items, desired, status, [{"video_id": "x"}], manifest, inserts, prune)
    assert audit == {
        "unsorted_items": 1,
        "deleted_entries": 1,
        "nonpublic_entries": 1,
        "misfiled_entries": 3,  # mis + dual in USA Shorts, hl in USA Drives
        "dual_shorts": 2,
        "highlights_in_drives": 1,
        "unrouted": 1,
        "public_orphans": 1,
        "planned_inserts": 1,
        "planned_deletes": len(prune["deletes"]),
    }


# --- fake service --------------------------------------------------------------


class _Resp:
    def __init__(self, result=None, exc=None):
        self._result, self._exc = result, exc

    def execute(self):
        if self._exc:
            raise self._exc
        return self._result or {}


class _Svc:
    """``members``: playlist title -> [videoId]; ``privacy``: videoId -> status
    (absent = deleted). Records every write in ``calls`` in order."""

    def __init__(self, members, privacy, fail_delete=(), ledger_check=None):
        self.members, self.privacy = members, privacy
        self.fail_delete, self.ledger_check = set(fail_delete), ledger_check
        self.calls, self.video_list_calls = [], []

    def playlists(self):
        titles = sorted(pr.MANAGED_PLAYLISTS)
        page = {"items": [{"id": f"PL:{t}", "snippet": {"title": t}} for t in titles]}
        return SimpleNamespace(list=lambda **kw: _Resp(page))

    def playlistItems(self):
        def rows(**kw):
            title = kw["playlistId"].removeprefix("PL:")
            return _Resp(
                {
                    "items": [
                        {
                            "id": f"it:{title}:{v}",
                            "snippet": {"position": i, "resourceId": {"videoId": v}},
                        }
                        for i, v in enumerate(self.members.get(title, []))
                    ]
                }
            )

        def insert(part, body):
            sn = body["snippet"]
            vid, pos = sn["resourceId"]["videoId"], sn.get("position")
            self.calls.append(("insert", sn["playlistId"], vid, pos))
            return _Resp()

        def delete(id):
            if self.ledger_check:
                self.ledger_check(id)
            self.calls.append(("delete", id))
            return _Resp(exc=RuntimeError("boom") if id in self.fail_delete else None)

        return SimpleNamespace(list=rows, insert=insert, delete=delete)

    def videos(self):
        def list_(id, **kw):
            ids = id.split(",")
            assert len(ids) <= 50
            self.video_list_calls.append(ids)
            found = [
                {"id": i, "status": {"privacyStatus": self.privacy[i]}}
                for i in ids
                if i in self.privacy
            ]
            return _Resp({"items": found})

        return SimpleNamespace(list=list_)


def _cfg(**kw):
    base = dict(
        max_inserts_per_run=40,
        newest_first=True,
        backfill_since="2020-01-01",
        overrides_path="",
        prune_include_nonpublic=False,
    )
    return SimpleNamespace(**(base | kw))


def _manifest_video(vid, title, kind, privacy="public"):
    return {
        "id": vid,
        "title": title,
        "category": kind,
        "published_at": "2026-06-01T00:00:00Z",
        "privacy_status": privacy,
    }


def _patch_manifest(monkeypatch, *videos):
    monkeypatch.setattr(pr, "build_channel_manifest", lambda service, *, now: {"videos": videos})


# --- fetch helpers ---------------------------------------------------------------


def test_fetch_live_items_returns_item_ids_and_positions():
    svc = _Svc({USA_SHORTS: ["a", "b"]}, {})
    rows, unscannable = pr.fetch_live_items(svc, [f"PL:{USA_SHORTS}"])
    assert rows == {
        f"PL:{USA_SHORTS}": [
            {"item_id": f"it:{USA_SHORTS}:a", "video_id": "a", "position": 0},
            {"item_id": f"it:{USA_SHORTS}:b", "video_id": "b", "position": 1},
        ]
    }
    assert unscannable == set()
    members, _ = pr.fetch_live_membership(svc, [f"PL:{USA_SHORTS}"])
    assert members == {f"PL:{USA_SHORTS}": {"a", "b"}}


def test_fetch_video_privacy_batches_by_fifty_and_omits_deleted():
    ids = [f"v{i}" for i in range(120)]
    svc = _Svc({}, {i: "public" for i in ids if i != "v7"})
    status = pr.fetch_video_privacy(svc, ids)
    assert [len(c) for c in svc.video_list_calls] == [50, 50, 20]
    assert "v7" not in status and status["v8"] == "public" and len(status) == 119


def test_fetch_video_privacy_refuses_when_nothing_resolves():
    """An empty answer for a non-empty id list is a broken lookup, not 100 deleted videos."""
    with pytest.raises(RuntimeError, match="refusing"):
        pr.fetch_video_privacy(_Svc({}, {}), ["a", "b"])


# --- apply_deletes / undo_deletes ---------------------------------------------------


def _delete_row(n, title=USA_SHORTS, reason="misfiled"):
    ids = {"playlist_id": f"PL:{title}", "item_id": f"it:{n}", "video_id": f"v{n}"}
    return {**ids, "playlist_title": title, "position": n, "reason": reason}


def test_apply_deletes_writes_the_ledger_line_before_each_call(tmp_path):
    ledger = tmp_path / "sub" / "prune.jsonl"

    def check(item_id):
        lines = ledger.read_text().splitlines()
        assert json.loads(lines[-1])["item_id"] == item_id

    svc = _Svc({}, {}, ledger_check=check)
    result = pr.apply_deletes(svc, [_delete_row(1), _delete_row(2)], ledger)
    assert result == {"applied": 2, "errors": 0}
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [r["video_id"] for r in rows] == ["v1", "v2"]
    assert {"playlist_id", "playlist_title", "item_id", "video_id", "position", "reason"} <= set(
        rows[0]
    )


def test_apply_deletes_logs_a_failure_and_continues(tmp_path, caplog):
    svc = _Svc({}, {}, fail_delete={"it:1"})
    result = pr.apply_deletes(svc, [_delete_row(1), _delete_row(2)], tmp_path / "l.jsonl")
    assert result == {"applied": 1, "errors": 1}
    assert [c for c in svc.calls if c[0] == "delete"] == [("delete", "it:1"), ("delete", "it:2")]
    assert "it:1" in caplog.text


def test_undo_reinserts_at_the_recorded_position_and_skips_live_entries(tmp_path):
    ledger = tmp_path / "l.jsonl"
    rows = [_delete_row(5), _delete_row(2), _delete_row(9)]
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    svc = _Svc({USA_SHORTS: ["v9"]}, {})  # v9 is already back
    result = pr.undo_deletes(svc, ledger)
    assert result == {"restored": 2, "skipped": 1, "errors": 0}
    assert svc.calls == [
        ("insert", f"PL:{USA_SHORTS}", "v2", 2),
        ("insert", f"PL:{USA_SHORTS}", "v5", 5),
    ]


# --- reconcile end to end ------------------------------------------------------------


def _scenario(monkeypatch):
    """USA Shorts holds an Irish short (also live in Ireland Shorts), a deleted
    video and a private one; Unsorted holds a filed American short."""
    _patch_manifest(
        monkeypatch,
        _manifest_video("irish", IRISH, "short"),
        _manifest_video("amer", AMERICAN, "short"),
        _manifest_video("priv", "Colleton County — Dashcam", "short", "private"),
        _manifest_video("new", AMERICAN, "short"),  # live nowhere: one planned insert
    )
    return _Svc(
        {
            USA_SHORTS: ["irish", "gone", "priv", "amer"],
            IRE_SHORTS: ["irish"],
            UNSORTED: ["amer"],
        },
        {"irish": "public", "amer": "public", "priv": "private"},
    )


def test_prune_dry_run_plans_but_never_writes(monkeypatch, tmp_path):
    svc = _scenario(monkeypatch)
    summary = pr.reconcile(svc, _cfg(), apply=False, prune=True, max_deletes=150)
    assert svc.calls == []
    prune = summary["prune"]
    assert prune["planned"] == 3 and prune["applied"] == 0 and prune["errors"] == 0
    assert sorted(r["reason"] for r in prune["rows"]) == ["deleted", "misfiled", "unsorted"]


def test_prune_apply_orders_inserts_before_deletes_and_caps_deletes(monkeypatch, tmp_path):
    svc = _scenario(monkeypatch)
    summary = pr.reconcile(
        svc, _cfg(), apply=True, prune=True, max_deletes=2, ledger=tmp_path / "l.jsonl"
    )
    assert [c[0] for c in svc.calls] == ["insert", "delete", "delete"]
    prune = summary["prune"]
    assert (prune["planned"], prune["applied"], prune["deferred"]) == (2, 2, 1)
    assert len((tmp_path / "l.jsonl").read_text().splitlines()) == 2


def test_negative_max_deletes_plans_no_deletes(monkeypatch):
    svc = _scenario(monkeypatch)
    summary = pr.reconcile(svc, _cfg(), apply=False, prune=True, max_deletes=-1)
    assert summary["prune"]["planned"] == 0
    assert summary["prune"]["deferred"] == 3


def test_cap_is_filled_in_class_precedence_nonpublic_last(monkeypatch):
    """The scenario has 3 deleted/unsorted/misfiled entries and 1 nonpublic one: a cap
    that is smaller than or equal to 3 plans zero nonpublic deletes; only a leftover
    slot goes to the private video."""
    for cap, reasons in (
        (1, ["deleted"]),
        (2, ["deleted", "unsorted"]),
        (3, ["deleted", "unsorted", "misfiled"]),
        (4, ["deleted", "unsorted", "misfiled", "nonpublic"]),
    ):
        svc = _scenario(monkeypatch)
        summary = pr.reconcile(
            svc, _cfg(), apply=False, prune=True, max_deletes=cap, include_nonpublic=True
        )
        assert [r["reason"] for r in summary["prune"]["rows"]] == reasons, cap
        assert summary["prune"]["deferred"] == 4 - cap


def test_plan_orders_classes_before_playlist_and_position():
    """A nonpublic entry that sorts first by playlist title and position still comes last."""
    items = {
        "Colorado Drives": ["priv", "gone2"],
        USA_SHORTS: ["gone", "priv2"],
        UNSORTED: ["uns"],
    }
    status = {"priv": "private", "priv2": "unlisted", "uns": "public"}
    plan = _prune(
        items,
        {"uns": [USA_SHORTS]},
        status,
        inserts=[_insert("uns", USA_SHORTS)],
        include_nonpublic=True,
    )
    assert [(d["reason"], d["playlist_title"], d["video_id"]) for d in plan["deletes"]] == [
        ("deleted", "Colorado Drives", "gone2"),
        ("deleted", USA_SHORTS, "gone"),
        ("unsorted", UNSORTED, "uns"),
        ("nonpublic", "Colorado Drives", "priv"),
        ("nonpublic", USA_SHORTS, "priv2"),
    ]


def test_nightly_apply_without_prune_is_insert_only(monkeypatch):
    """The cron runs ``reconcile-playlists --apply``: no delete and no videos.list."""
    svc = _scenario(monkeypatch)
    pr.reconcile(svc, _cfg(), apply=True)
    assert [c for c in svc.calls if c[0] == "delete"] == []
    assert svc.video_list_calls == []


def test_prune_apply_requires_a_ledger(monkeypatch):
    svc = _scenario(monkeypatch)
    with pytest.raises(ValueError, match="ledger"):
        pr.reconcile(svc, _cfg(), apply=True, prune=True, max_deletes=5)
    assert svc.calls == []


def test_audit_is_read_only_and_reports_counts(monkeypatch):
    svc = _scenario(monkeypatch)
    summary = pr.reconcile(svc, _cfg(), apply=True, audit=True)
    assert svc.calls == []
    audit = summary["audit"]
    assert audit["unsorted_items"] == 1
    assert audit["deleted_entries"] == 1
    assert audit["nonpublic_entries"] == 1
    assert audit["misfiled_entries"] == 1
    assert audit["planned_deletes"] == 3


# --- scheduled uploads are exempt from pruning ---------------------------------------


def _video_svc(items):
    """A bare ``videos.list`` stub: ``items`` are the raw API items."""
    return SimpleNamespace(
        videos=lambda: SimpleNamespace(
            list=lambda **kw: SimpleNamespace(execute=lambda: {"items": items})
        )
    )


def test_fetch_video_privacy_reports_a_future_publish_at_as_scheduled():
    items = [
        {"id": "s", "status": {"privacyStatus": "private", "publishAt": "2026-10-06T08:00:00Z"}},
        {"id": "past", "status": {"privacyStatus": "private", "publishAt": "2026-10-01T08:00:00Z"}},
        {"id": "p", "status": {"privacyStatus": "private"}},
        {"id": "pub", "status": {"privacyStatus": "public", "publishAt": "2026-10-06T08:00:00Z"}},
    ]
    status = pr.fetch_video_privacy(_video_svc(items), ["s", "past", "p", "pub"], now=NOW)
    assert status == {"s": "scheduled", "past": "private", "p": "private", "pub": "public"}


def test_scheduled_video_entries_are_never_pruned_even_with_nonpublic_on():
    items = {USA_SHORTS: ["s"], UNSORTED: ["s"], pr.PL_IRELAND_SUPERLONG: ["s"]}
    plan = _prune(items, {"s": [IRE_SHORTS]}, {"s": "scheduled"}, include_nonpublic=True)
    assert plan["deletes"] == []
    assert {d["why"] for d in plan["skipped"]} == {"scheduled"}
    assert {d["playlist_title"] for d in plan["skipped"]} == {UNSORTED}  # only a class match skips


def test_a_scheduled_video_is_not_counted_as_deleted_or_nonpublic_in_the_audit():
    items = {USA_SHORTS: _rows(USA_SHORTS, "s")}
    prune = pr.compute_prune_plan(items, {}, {"s": "scheduled"}, include_nonpublic=True)
    audit = pr.compute_audit(items, {}, {"s": "scheduled"}, [], [], [], prune)
    assert (audit["deleted_entries"], audit["nonpublic_entries"]) == (0, 0)


def test_nonpublic_entry_in_unsorted_is_removed_as_unsorted_not_nonpublic():
    plan = _prune({UNSORTED: ["p"]}, {}, {"p": "private"}, include_nonpublic=True)
    assert _removed(plan) == [(UNSORTED, "p", "unsorted")]


# --- the nonpublic switch defaults from config -----------------------------------------


@pytest.mark.parametrize(
    "cfg_default, flag, expect_nonpublic",
    [(True, None, True), (False, None, False), (True, False, False), (False, True, True)],
)
def test_include_nonpublic_defaults_from_config_and_flag_overrides(
    monkeypatch, cfg_default, flag, expect_nonpublic
):
    svc = _scenario(monkeypatch)
    summary = pr.reconcile(
        svc,
        _cfg(prune_include_nonpublic=cfg_default),
        apply=False,
        prune=True,
        max_deletes=150,
        include_nonpublic=flag,
    )
    reasons = [r["reason"] for r in summary["prune"]["rows"]]
    assert ("nonpublic" in reasons) is expect_nonpublic


def test_config_defaults_nonpublic_pruning_on_and_yaml_agrees():
    from gomoveshift.config import PlaylistReconcilerConfig

    assert PlaylistReconcilerConfig().prune_include_nonpublic is True
    assert PlaylistReconcilerConfig(prune_include_nonpublic=False).prune_include_nonpublic is False

    import yaml

    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    block = yaml.safe_load(cfg_path.read_text())["playlist_reconciler"]
    assert block["prune_include_nonpublic"] is True
