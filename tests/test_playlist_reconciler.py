"""Tests for the daily playlist reconciler: the pure
desired-state diff engine plus the apply glue (``fetch_live_membership``,
``apply_inserts``, ``reconcile``) — all against a mocked service, no live
API calls."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.pipeline import playlist_reconciler as pr


def _video(video_id, title, kind, published_at="2026-06-01T00:00:00Z", privacy="public"):
    return {
        "id": video_id,
        "title": title,
        "category": kind,
        "published_at": published_at,
        "privacy_status": privacy,
    }


def test_module_mutates_playlist_items_only():
    """Prune adds ``playlistItems().delete``; nothing else may write."""
    src = Path(pr.__file__).read_text()
    writes = set(re.findall(r"\.(\w+)\(\)\s*\.(delete|update|insert)\(", src))
    assert writes == {("playlistItems", "insert"), ("playlistItems", "delete")}


@pytest.mark.parametrize(
    ("kind", "region", "state", "title", "expected"),
    [
        ("stream", None, None, "West NC Mountains Cam", [pr.PL_MOUNTAINS_CAM]),
        ("short", "ireland", None, "A Short", [pr.PL_IRELAND_SHORTS]),
        ("short", "usa", None, "A Short", [pr.PL_USA_SHORTS]),
        # A short with no region is unrouted -- never defaulted to USA Shorts.
        ("short", None, None, "A Short", []),
        ("real_lf", "ireland", None, "A Drive", [pr.PL_IRELAND_DRIVES]),
        ("real_lf", "ireland", None, "Highlights from A Drive", [pr.PL_IRELAND_HIGHLIGHTS]),
        ("real_lf", "usa", None, "Highlights from A Drive", [pr.PL_USA_HIGHLIGHTS]),
        # Highlights keep their state playlist, never Drives.
        (
            "real_lf",
            "usa",
            "tennessee",
            "Highlights from A",
            [pr.PL_USA_HIGHLIGHTS, "Tennessee Drives"],
        ),
        ("real_lf", "usa", None, "A Drive", [pr.PL_USA_DRIVES]),
        ("real_lf", "usa", "tennessee", "A Drive", [pr.PL_USA_DRIVES, "Tennessee Drives"]),
        ("real_lf", "usa", "montana", "A Drive", [pr.PL_USA_DRIVES]),  # no Montana playlist
        ("real_lf", None, None, "Untitled Drive", []),
    ],
)
def test_target_playlists(kind, region, state, title, expected):
    assert pr.target_playlists(kind, region, state, title) == expected


@pytest.mark.parametrize(
    ("title", "region", "state"),
    [
        ("Full Drive: Blue Ridge Parkway, Tennessee", "usa", "tennessee"),
        ("Port Royal to Lady's Island, SC", "usa", "south carolina"),
        ("Killarney to Dingle via N86, Kerry", "ireland", None),
        ("Some Road, Somewhere, Belgium", None, None),
        ("No comma at all here", None, None),
        # A trailing quality suffix never defeats the parse.
        ("Unicoi to Kingsport, TN in 4K", "usa", "tennessee"),
        ("Washington County, VA in 4K", "usa", "virginia"),
        ("Cline, VA in 4k ", "usa", "virginia"),
        ("Goleen, County Cork in 4K", "ireland", None),
        ("Kenmare to Glengarriff, Ireland | 4K", "ireland", None),
        ("Scenic Loop, Tennessee | 4K Relaxing Drive", "usa", "tennessee"),
        # ...but never guesses past it.
        ("West Cork in 4K", None, None),
        ("Some Road, Belgium in 4K", None, None),
        ("Ireland Scenic Drive in 4K | Conor Pass, Co. Kerry", None, None),
    ],
)
def test_derive_title_region_state(title, region, state):
    assert pr.derive_title_region_state(title) == (region, state)


def test_load_overrides_skips_comments_and_missing_path(tmp_path):
    p = tmp_path / "overrides.json"
    p.write_text(json.dumps({"_comment": "x", "abc123": {"region": "ireland", "state": None}}))
    assert pr.load_overrides(p) == {"abc123": {"region": "ireland", "state": None}}
    assert pr.load_overrides(tmp_path / "nope.json") == {}
    assert pr.load_overrides(None) == {}


def test_compute_plan_inserts_missing_membership_only():
    videos = [
        _video("v1", "A Drive, Tennessee", "real_lf"),
        _video("v2", "A Drive, Tennessee", "real_lf"),
    ]
    live = {"USA Drives": {"v1"}, "Tennessee Drives": {"v1", "v2"}}
    plan = pr.compute_plan(
        videos, live, {}, max_inserts=40, newest_first=True, backfill_since="2020-01-01"
    )
    assert plan["inserts"] == [
        {
            "video_id": "v2",
            "playlist_title": "USA Drives",
            "published_at": "2026-06-01T00:00:00Z",
            "kind": "real_lf",
        }
    ]
    assert plan["unrouted"] == []


def test_compute_plan_skips_non_public():
    videos = [_video("v1", "A Drive, Ireland", "real_lf", privacy="private")]
    plan = pr.compute_plan(
        videos, {}, {}, max_inserts=40, newest_first=True, backfill_since="2020-01-01"
    )
    assert plan["inserts"] == plan["unrouted"] == []


def test_compute_plan_reports_unrouted_never_uses_fallback():
    videos = [_video("v1", "No location signal here", "real_lf")]
    plan = pr.compute_plan(
        videos, {}, {}, max_inserts=40, newest_first=True, backfill_since="2020-01-01"
    )
    assert plan["inserts"] == []
    assert plan["unrouted"] == [
        {"video_id": "v1", "title": "No location signal here", "kind": "real_lf"}
    ]


def test_compute_plan_overrides_apply_only_when_title_yields_no_region():
    videos = [_video("v1", "No location signal here", "real_lf")]
    overrides = {"v1": {"region": "ireland", "state": None}}
    plan = pr.compute_plan(
        videos, {}, overrides, max_inserts=40, newest_first=True, backfill_since="2020-01-01"
    )
    assert plan["inserts"][0]["playlist_title"] == pr.PL_IRELAND_DRIVES


def test_compute_plan_backfill_since_bounds_shorts_only():
    videos = [
        _video("old_short", "A Short, Ireland", "short", published_at="2025-01-01T00:00:00Z"),
        _video("old_lf", "A Drive, Ireland", "real_lf", published_at="2025-01-01T00:00:00Z"),
    ]
    plan = pr.compute_plan(
        videos, {}, {}, max_inserts=40, newest_first=True, backfill_since="2026-01-01"
    )
    routed = {c["video_id"] for c in plan["inserts"]}
    assert "old_short" not in routed
    assert "old_lf" in routed


def test_compute_plan_caps_and_orders_newest_first():
    videos = [
        _video("v1", "A Drive, Ireland", "real_lf", published_at="2026-01-01T00:00:00Z"),
        _video("v2", "A Drive, Ireland", "real_lf", published_at="2026-03-01T00:00:00Z"),
    ]
    plan = pr.compute_plan(
        videos, {}, {}, max_inserts=1, newest_first=True, backfill_since="2020-01-01"
    )
    assert [c["video_id"] for c in plan["inserts"]] == ["v2"]


# ── apply glue ────────────────────────────────────────────────


class _Resp:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class _FakeService:
    """Mocked YouTube service: ``playlists()``/``playlistItems()`` list +
    insert, keyed by canned pages; ``inserted`` records every insert call."""

    def __init__(self, playlists_page=None, playlist_items_pages=None):
        self._playlists_page = playlists_page or {"items": []}
        self._playlist_items_pages = playlist_items_pages or {}
        self.inserted = []

    def playlists(self):
        outer = self

        class _Playlists:
            def list(self, **kw):
                return _Resp(outer._playlists_page)

        return _Playlists()

    def playlistItems(self):
        outer = self

        class _PlaylistItems:
            def list(self, **kw):
                return _Resp(outer._playlist_items_pages[kw["playlistId"]][kw.get("pageToken")])

            def insert(self, **kw):
                snippet = kw["body"]["snippet"]
                outer.inserted.append((snippet["playlistId"], snippet["resourceId"]["videoId"]))
                return _Resp({})

        return _PlaylistItems()


def _config():
    return SimpleNamespace(
        max_inserts_per_run=40, newest_first=True, backfill_since="2020-01-01", overrides_path=""
    )


def test_resolve_playlist_ids_maps_titles_and_fails_loudly_on_missing():
    service = _FakeService(
        playlists_page={"items": [{"snippet": {"title": "USA Drives"}, "id": "PLUSA"}]}
    )
    assert pr.resolve_playlist_ids(service, {"USA Drives"}) == {"USA Drives": "PLUSA"}
    with pytest.raises(RuntimeError, match="Ireland Drives"):
        pr.resolve_playlist_ids(service, {"USA Drives", "Ireland Drives"})


def test_fetch_live_membership_paginates_per_playlist():
    service = _FakeService(
        playlist_items_pages={
            "PL1": {
                None: {
                    "items": [{"snippet": {"resourceId": {"videoId": "v1"}}}],
                    "nextPageToken": "P2",
                },
                "P2": {"items": [{"snippet": {"resourceId": {"videoId": "v2"}}}]},
            },
            "PL2": {None: {"items": []}},
        }
    )
    membership, unscannable = pr.fetch_live_membership(service, ["PL1", "PL2"])
    assert membership == {"PL1": {"v1", "v2"}, "PL2": set()}
    assert unscannable == set()


def _make_http_error(status_code):
    from googleapiclient.errors import HttpError

    class _Resp:
        def __init__(self, status):
            self.status = status
            self.reason = "Not Found"

    body = json.dumps({"error": {"code": status_code, "message": "playlistNotFound"}}).encode()
    return HttpError(_Resp(status_code), body, uri="fake://")


class _PagingErrorService:
    """Fake service where page 1 of ``legacy_id`` succeeds with a nextPageToken,
    then any page-2+ request raises the given HttpError -- mirrors a legacy
    (13-char) playlist id's live pagination behavior."""

    def __init__(self, legacy_id, page1_items, error, first_page_error=False):
        self.legacy_id = legacy_id
        self.page1_items = page1_items
        self.error = error
        self.first_page_error = first_page_error

    def playlistItems(self):
        outer = self

        class _PlaylistItems:
            def list(self, **kw):
                page_token = kw.get("pageToken")
                if kw["playlistId"] == outer.legacy_id:
                    if page_token is None and not outer.first_page_error:
                        return _Resp(
                            {
                                "items": outer.page1_items,
                                "nextPageToken": "P2",
                            }
                        )
                    raise outer.error
                return _Resp({"items": []})

        return _PlaylistItems()


def test_fetch_live_membership_skips_legacy_playlist_that_404s_on_page_two():
    legacy_id = "PLlegacy0000"
    error = _make_http_error(404)
    service = _PagingErrorService(
        legacy_id,
        page1_items=[{"snippet": {"resourceId": {"videoId": "v1"}}}],
        error=error,
    )
    membership, unscannable = pr.fetch_live_membership(service, [legacy_id, "PLModern"])
    assert legacy_id not in membership
    assert unscannable == {legacy_id}
    assert membership["PLModern"] == set()


def test_fetch_live_membership_first_page_404_is_unchanged_and_not_unscannable():
    missing_id = "PLDoesNotExist"
    error = _make_http_error(404)
    service = _PagingErrorService(missing_id, page1_items=[], error=error, first_page_error=True)
    with pytest.raises(type(error)):
        pr.fetch_live_membership(service, [missing_id])


def test_apply_inserts_skips_live_videos_and_caps_at_max_inserts():
    service = _FakeService()
    plan = {
        "inserts": [
            {"video_id": "v1", "playlist_id": "PLUSA"},  # already live -> skipped
            {"video_id": "v2", "playlist_id": "PLUSA"},  # inserted
            {"video_id": "v3", "playlist_id": "PLUSA"},  # beyond the cap
        ],
        "live_by_id": {"PLUSA": {"v1"}},
    }
    assert pr.apply_inserts(service, plan, max_inserts=2) == {"applied": 1, "skipped": 1}
    assert service.inserted == [("PLUSA", "v2")]


def _fake_channel_service(already_live=()):
    titles = list(pr.ALL_TARGET_PLAYLISTS)
    ids = [f"PL_{i}" for i in range(len(titles))]
    items = [{"snippet": {"resourceId": {"videoId": v}}} for v in already_live]
    return _FakeService(
        playlists_page={"items": [{"snippet": {"title": t}, "id": i} for t, i in zip(titles, ids)]},
        playlist_items_pages={i: {None: {"items": items}} for i in ids},
    )


@pytest.mark.parametrize(
    ("apply", "already_live", "expected"),
    [
        (False, (), {"planned": 2, "applied": 0, "skipped": 0, "unroutable": 0}),
        (True, (), {"planned": 2, "applied": 2, "skipped": 0, "unroutable": 0}),
        # Re-run: both targets already carry v1 -- nothing left to insert.
        (True, ("v1",), {"planned": 0, "applied": 0, "skipped": 0, "unroutable": 0}),
    ],
)
def test_reconcile(monkeypatch, apply, already_live, expected):
    manifest = {"videos": [_video("v1", "A Drive, Tennessee", "real_lf")]}
    monkeypatch.setattr(pr, "build_channel_manifest", lambda service, *, now: manifest)
    service = _fake_channel_service(already_live)

    summary = pr.reconcile(service, _config(), apply=apply)

    assert summary == {**expected, "skipped_unscannable": 0, "unrouted": []}
    assert len(service.inserted) == expected["applied"]


def _unroutable_videos(n):
    return [_video(f"u{i:02d}", f"No Region Here {i}", "real_lf") for i in range(n)]


def test_reconcile_names_unrouted_video_in_summary_and_log(monkeypatch, caplog):
    manifest = {"videos": _unroutable_videos(1)}
    monkeypatch.setattr(pr, "build_channel_manifest", lambda service, *, now: manifest)
    with caplog.at_level("WARNING", logger=pr.logger.name):
        summary = pr.reconcile(_fake_channel_service(), _config(), apply=False)
    assert summary["unroutable"] == 1
    assert summary["unrouted"] == [
        {"video_id": "u00", "title": "No Region Here 0", "kind": "real_lf"}
    ]
    msgs = [r.getMessage() for r in caplog.records if r.getMessage().startswith("UNROUTED:")]
    assert len(msgs) == 1
    assert "u00" in msgs[0] and "kind=real_lf" in msgs[0] and "'No Region Here 0'" in msgs[0]


def test_reconcile_caps_unrouted_warnings_at_25(monkeypatch, caplog):
    manifest = {"videos": _unroutable_videos(30)}
    monkeypatch.setattr(pr, "build_channel_manifest", lambda service, *, now: manifest)
    with caplog.at_level("WARNING", logger=pr.logger.name):
        summary = pr.reconcile(_fake_channel_service(), _config(), apply=False)
    assert len(summary["unrouted"]) == 30
    msgs = [r.getMessage() for r in caplog.records]
    assert sum(m.startswith("UNROUTED:") for m in msgs) == 25
    assert sum("and 5 more" in m for m in msgs) == 1


def test_reconcile_excludes_unscannable_playlist_from_insert_planning(monkeypatch):
    """A legacy playlist that 404s mid-pagination must not be planned against --
    an under-counted membership would cause a duplicate insert next run."""
    manifest = {"videos": [_video("v1", "A Drive, Tennessee", "real_lf")]}
    monkeypatch.setattr(pr, "build_channel_manifest", lambda service, *, now: manifest)
    titles = list(pr.ALL_TARGET_PLAYLISTS)
    ids = [f"PL_{i}" for i in range(len(titles))]
    usa_drives_id = ids[titles.index(pr.PL_USA_DRIVES)]
    tn_id = ids[titles.index("Tennessee Drives")]

    class _Service:
        def playlists(self):
            class _Playlists:
                def list(self, **kw):
                    return _Resp(
                        {"items": [{"snippet": {"title": t}, "id": i} for t, i in zip(titles, ids)]}
                    )

            return _Playlists()

        def playlistItems(self):
            class _PlaylistItems:
                def list(self, **kw):
                    if kw["playlistId"] == tn_id:
                        if kw.get("pageToken") is None:
                            return _Resp({"items": [], "nextPageToken": "P2"})
                        raise _make_http_error(404)
                    return _Resp({"items": []})

            return _PlaylistItems()

    summary = pr.reconcile(_Service(), _config(), apply=False)
    assert summary["skipped_unscannable"] == 1
    assert summary["planned"] == 1  # only USA Drives -- Tennessee Drives excluded
    assert usa_drives_id  # sanity: title resolved


_PLAN_KW = {"max_inserts": 40, "newest_first": True, "backfill_since": "2020-01-01"}


def _plan(videos, live, overrides=None):
    return pr.compute_plan(videos, live, overrides or {}, **_PLAN_KW)


def test_live_ireland_highlight_is_never_planned_into_drives():
    """A highlight whose title lacks the 'Highlights from' prefix is
    authoritative by live Highlights membership, not by title."""
    videos = [_video("h1", "Drive Through Kerry, Ireland", "real_lf")]
    live = {pr.PL_IRELAND_HIGHLIGHTS: {"h1"}, pr.PL_IRELAND_DRIVES: {"h1"}}
    plan = _plan(videos, live)
    assert plan["inserts"] == []
    assert plan["desired"] == {"h1": [pr.PL_IRELAND_HIGHLIGHTS]}


def test_live_usa_highlight_keeps_state_playlist_and_never_drives():
    videos = [_video("h1", "Cline, VA in 4K", "real_lf")]
    live = {pr.PL_USA_HIGHLIGHTS: {"h1"}}
    plan = _plan(videos, live)
    assert [c["playlist_title"] for c in plan["inserts"]] == ["Virginia Drives"]
    assert plan["desired"] == {"h1": [pr.PL_USA_HIGHLIGHTS, "Virginia Drives"]}
    assert plan["unrouted"] == []


def test_live_usa_highlight_without_known_state_keeps_live_state_membership():
    videos = [_video("h1", "Hyperlapse Fall Colors", "real_lf")]
    live = {pr.PL_USA_HIGHLIGHTS: {"h1"}, "North Carolina Drives": {"h1"}}
    plan = _plan(videos, live)
    assert plan["inserts"] == []
    assert plan["desired"]["h1"] == ["North Carolina Drives", pr.PL_USA_HIGHLIGHTS]


def test_live_highlight_in_both_regions_is_unrouted_not_guessed():
    videos = [_video("h1", "Somewhere", "real_lf")]
    live = {pr.PL_IRELAND_HIGHLIGHTS: {"h1"}, pr.PL_USA_HIGHLIGHTS: {"h1"}}
    plan = _plan(videos, live)
    assert plan["inserts"] == []
    assert [u["video_id"] for u in plan["unrouted"]] == ["h1"]
    assert "h1" not in plan["desired"]


def test_live_region_is_last_resort_for_title_without_region():
    videos = [_video("v1", "West Cork in 4K", "real_lf")]
    plan = _plan(videos, {pr.PL_IRELAND_DRIVES: {"v1"}})
    assert plan["unrouted"] == []
    assert plan["inserts"] == []
    assert plan["desired"] == {"v1": [pr.PL_IRELAND_DRIVES]}


def test_live_usa_family_membership_routes_usa_and_keeps_every_live_state():
    videos = [_video("v1", "Tail of the Dragon", "real_lf")]
    live = {"Tennessee Drives": {"v1"}, "North Carolina Drives": {"v1"}}
    plan = _plan(videos, live)
    assert plan["unrouted"] == []
    # USA Drives is missing, the hand-curated states are kept (and not re-planned).
    assert [c["playlist_title"] for c in plan["inserts"]] == [pr.PL_USA_DRIVES]
    assert plan["desired"]["v1"] == sorted(
        [pr.PL_USA_DRIVES, "Tennessee Drives", "North Carolina Drives"]
    )


def test_title_routed_usa_video_keeps_extra_live_state_playlist():
    videos = [_video("v1", "A Drive, Tennessee", "real_lf")]
    live = {pr.PL_USA_DRIVES: {"v1"}, "Tennessee Drives": {"v1"}, "North Carolina Drives": {"v1"}}
    plan = _plan(videos, live)
    assert plan["inserts"] == []
    assert "North Carolina Drives" in plan["desired"]["v1"]


def test_live_region_ambiguous_or_absent_is_unrouted():
    videos = [
        _video("both", "No signal", "real_lf"),
        _video("none", "No signal either", "real_lf"),
    ]
    live = {pr.PL_IRELAND_DRIVES: {"both"}, pr.PL_USA_DRIVES: {"both"}}
    plan = _plan(videos, live)
    assert {u["video_id"] for u in plan["unrouted"]} == {"both", "none"}


def test_override_beats_live_region_fallback():
    videos = [_video("v1", "No signal", "real_lf")]
    live = {pr.PL_USA_DRIVES: {"v1"}}
    plan = _plan(videos, live, {"v1": {"region": "ireland", "state": None}})
    assert [c["playlist_title"] for c in plan["inserts"]] == [pr.PL_IRELAND_DRIVES]


def test_drives_family_membership_never_routes_a_short():
    """Only the two Shorts playlists are a short's live-membership signal."""
    videos = [_video("s1", "A Short", "short")]
    plan = _plan(videos, {pr.PL_IRELAND_DRIVES: {"s1"}})
    assert plan["desired"] == {}
    assert [u["video_id"] for u in plan["unrouted"]] == ["s1"]


def test_desired_includes_videos_skipped_by_backfill_window():
    videos = [_video("old", "A Short, Tennessee", "short", published_at="2025-01-01T00:00:00Z")]
    plan = pr.compute_plan(
        videos, {}, {}, max_inserts=40, newest_first=True, backfill_since="2026-01-01"
    )
    assert plan["inserts"] == []
    assert plan["desired"] == {"old": [pr.PL_USA_SHORTS]}


def test_superlong_membership_is_in_desired():
    videos = [_video("p1", "Whatever", "real_lf")]
    plan = _plan(videos, {pr.PL_USA_SUPERLONG: {"p1"}})
    assert plan["desired"] == {"p1": [pr.PL_USA_SUPERLONG]}


def test_load_overrides_warns_loudly_on_missing_configured_file(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        assert pr.load_overrides(tmp_path / "gone.json") == {}
    assert "gone.json" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING"):
        assert pr.load_overrides("") == {}
    assert caplog.text == ""


def test_resolve_overrides_path_is_cwd_independent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    monkeypatch.setenv("GMS_INPUTS_DIR", str(inputs))
    resolved = pr.resolve_overrides_path("playlist_routing_overrides.json")
    assert resolved.is_absolute()
    assert resolved.parent == inputs
    other = tmp_path / "other-inputs"
    other.mkdir()
    monkeypatch.setenv("GMS_INPUTS_DIR", str(other))
    assert pr.resolve_overrides_path("playlist_routing_overrides.json").parent == other
    assert pr.resolve_overrides_path("/abs/x.json") == Path("/abs/x.json")
    assert pr.resolve_overrides_path("") is None


def test_overrides_file_loads_and_skips_comment_keys(tmp_path):
    path = tmp_path / "playlist_routing_overrides.json"
    path.write_text(
        json.dumps(
            {
                "_comment": "keys starting with an underscore are comments",
                "vid-usa-0001": {"region": "usa", "state": "north carolina", "note": "fictional"},
                "vid-ire-0002": {"region": "ireland", "note": "fictional"},
            }
        )
    )
    overrides = pr.load_overrides(path)
    assert set(overrides) == {"vid-usa-0001", "vid-ire-0002"}
    assert all(v["region"] in ("ireland", "usa") for v in overrides.values())
    assert overrides["vid-usa-0001"] == {
        "region": "usa",
        "state": "north carolina",
        "note": "fictional",
    }


def test_reconcile_loads_overrides_relative_to_inputs_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    seen = {}
    monkeypatch.setattr(pr, "load_overrides", lambda path: seen.setdefault("path", path) and {})
    manifest = {"videos": []}
    monkeypatch.setattr(pr, "build_channel_manifest", lambda service, *, now: manifest)
    cfg = _config()
    cfg.overrides_path = "playlist_routing_overrides.json"
    pr.reconcile(_fake_channel_service(), cfg, apply=False)
    assert Path(seen["path"]).is_absolute()


# --- shorts region = title > overrides > live Shorts membership --------


def _short_plan(title, live=None, overrides=None, video_id="s1"):
    return _plan([_video(video_id, title, "short")], live or {}, overrides)


def test_irish_short_live_only_in_usa_shorts_moves_to_ireland_shorts():
    """An Irish-titled short in USA Shorts is desired in Ireland Shorts only."""
    plan = _short_plan("Just Driving — County Kerry, Ireland", {pr.PL_USA_SHORTS: {"s1"}})
    assert plan["desired"] == {"s1": [pr.PL_IRELAND_SHORTS]}
    assert [(c["video_id"], c["playlist_title"]) for c in plan["inserts"]] == [
        ("s1", pr.PL_IRELAND_SHORTS)
    ]


def test_american_short_live_only_in_ireland_shorts_moves_to_usa_shorts():
    plan = _short_plan("Colleton County — 4K 🚗", {pr.PL_IRELAND_SHORTS: {"s1"}})
    assert plan["desired"] == {"s1": [pr.PL_USA_SHORTS]}


def test_short_in_both_shorts_playlists_resolves_to_its_title_region():
    live = {pr.PL_IRELAND_SHORTS: {"s1"}, pr.PL_USA_SHORTS: {"s1"}}
    plan = _short_plan("Driving to Horn Head, Donegal", live)
    assert plan["desired"] == {"s1": [pr.PL_IRELAND_SHORTS]}
    assert plan["inserts"] == []


def test_short_without_title_region_follows_the_override():
    plan = _short_plan(
        "Quiet Roads", {pr.PL_USA_SHORTS: {"s1"}}, {"s1": {"region": "ireland", "state": None}}
    )
    assert plan["desired"] == {"s1": [pr.PL_IRELAND_SHORTS]}


def test_title_region_beats_override_and_live_membership():
    plan = _short_plan(
        "Colleton County Vibes",
        {pr.PL_IRELAND_SHORTS: {"s1"}},
        {"s1": {"region": "ireland", "state": None}},
    )
    assert plan["desired"] == {"s1": [pr.PL_USA_SHORTS]}


@pytest.mark.parametrize(
    ("live", "expected"),
    [
        ({pr.PL_IRELAND_SHORTS: {"s1"}}, pr.PL_IRELAND_SHORTS),
        ({pr.PL_USA_SHORTS: {"s1"}}, pr.PL_USA_SHORTS),
    ],
)
def test_short_without_title_or_override_region_falls_back_to_its_only_live_playlist(
    live, expected
):
    plan = _short_plan("Dirty water", live)
    assert plan["desired"] == {"s1": [expected]}
    assert plan["inserts"] == []


@pytest.mark.parametrize(
    "live",
    [
        {},
        {pr.PL_IRELAND_SHORTS: {"s1"}, pr.PL_USA_SHORTS: {"s1"}},
        {pr.PL_MOUNTAINS_CAM: {"s1"}},
    ],
)
def test_short_with_no_signal_is_unrouted_never_defaulted_to_usa_shorts(live):
    plan = _short_plan("Dirty water", live)
    assert plan["desired"] == {}
    assert plan["inserts"] == []
    assert plan["unrouted"] == [{"video_id": "s1", "title": "Dirty water", "kind": "short"}]


def test_car_cam_shorts_route_to_their_own_region_shorts_playlist():
    """A 39s "Car Cam: <place>" is a short (classify); the title's place or
    county evidence routes it to its own region's Shorts playlist with no
    override needed."""
    cases = {
        "vid-car-0001": ("Car Cam: To Clonakilty \U0001f697", pr.PL_IRELAND_SHORTS),
        "vid-car-0002": ("Car Cam: Dingle Peninsula \U0001f697", pr.PL_IRELAND_SHORTS),
        "vid-car-0003": (
            "Car Cam: Dingle Peninsula toward Gortmore \U0001f697",
            pr.PL_IRELAND_SHORTS,
        ),
        "vid-car-0004": ("Car Cam: Transylvania County \U0001f697", pr.PL_USA_SHORTS),
    }
    videos = [
        {
            "id": vid,
            "title": title,
            "duration_seconds": 39,
            "published_at": "2026-06-01T00:00:00Z",
            "privacy_status": "public",
        }
        for vid, (title, _) in cases.items()
    ]
    plan = _plan(videos, {})
    assert plan["unrouted"] == []
    got = {(i["video_id"], i["playlist_title"]) for i in plan["inserts"]}
    assert got == {(vid, target) for vid, (_, target) in cases.items()}
