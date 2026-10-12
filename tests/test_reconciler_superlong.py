"""Superlong parts must route to their own Superlong playlist, never the
region Drives playlist: a reconciler whose own hardcoded
``target_playlists`` classify a superlong part as a plain ``real_lf`` would
send it to ``PL_{IRELAND,USA}_DRIVES``, undoing the uploader-side decision
to keep a 4h part out of the 20-minute Drives playlists (autoplay).

Recognition signal: LIVE MEMBERSHIP of a Superlong playlist, not title or
duration. The uploader inserts every superlong part -- including an
unsuffixed single-part day -- into its region's Superlong playlist at
upload time, so by the next reconcile run "already a member of Ireland/USA
Superlong" is authoritative and covers the single-part case a title-suffix
heuristic (``_SUPERLONG_PART_SUFFIX_RE``, only appended when
``part_count > 1``) would silently mis-route."""

from __future__ import annotations

import pytest

from gomoveshift.pipeline import playlist_reconciler as pr

# Exact literal titles the uploader auto-creates
# (drive_phases/youtube_uploader._get_or_create_playlist) -- spelled out
# here rather than via ``pr.PL_*`` so a routing test's RED shows the WRONG
# PLAYLIST being chosen instead of an AttributeError on a not-yet-defined
# constant.
_IRELAND_SUPERLONG = "Ireland Superlong"
_USA_SUPERLONG = "USA Superlong"


def _video(video_id, title, kind, published_at="2026-06-01T00:00:00Z", privacy="public"):
    return {
        "id": video_id,
        "title": title,
        "category": kind,
        "published_at": published_at,
        "privacy_status": privacy,
    }


def _plan(videos, live_membership):
    return pr.compute_plan(
        videos, live_membership, {}, max_inserts=40, newest_first=True, backfill_since="2020-01-01"
    )


def _plan_ids(videos, live_membership):
    return {(c["video_id"], c["playlist_title"]) for c in _plan(videos, live_membership)["inserts"]}


def test_superlong_playlists_are_resolved_and_membership_fetched():
    assert _IRELAND_SUPERLONG in pr.ALL_TARGET_PLAYLISTS
    assert _USA_SUPERLONG in pr.ALL_TARGET_PLAYLISTS


def test_public_superlong_part_not_inserted_into_region_drives():
    # v1 is a single-part superlong day (real production title shape --
    # no " — Part N of M" suffix, since ``build_superlong_part_title``
    # only appends it when ``part_count > 1``) that the uploader already
    # put in "Ireland Superlong". It must never ALSO be planned for
    # "Ireland Drives" even though the title alone would classify/derive
    # it as a routable Ireland real_lf.
    videos = [_video("v1", "Driving Ring of Kerry Coastal Loop, Kerry", "real_lf")]
    live = {_IRELAND_SUPERLONG: {"v1"}}
    inserts = _plan_ids(videos, live)
    assert ("v1", pr.PL_IRELAND_DRIVES) not in inserts


def test_public_superlong_part_already_satisfied_in_its_own_playlist():
    # v2 is a MULTI-part day (the deliberate "— Part N of M" suffix), also
    # already a member of "USA Superlong". Two things must hold: nothing
    # left to insert into Superlong (already satisfied, insert-only +
    # idempotent) and it must not land in Drives/state playlists either --
    # and it must not fall into ``unrouted`` (the suffix breaks title-based
    # region derivation outright, so region-first routing would have
    # missed it entirely).
    videos = [
        _video("v2", "Driving A Long Haul Through Tennessee, Tennessee — Part 1 of 2", "real_lf")
    ]
    live = {_USA_SUPERLONG: {"v2"}}
    plan = _plan(videos, live)
    inserts = {(c["video_id"], c["playlist_title"]) for c in plan["inserts"]}
    assert ("v2", _USA_SUPERLONG) not in inserts
    assert not any(pt in (pr.PL_USA_DRIVES, "Tennessee Drives") for (_vid, pt) in inserts)
    assert plan["unrouted"] == []


def test_single_part_superlong_day_carries_no_title_suffix_but_still_routes():
    # Same single-part shape as v1 above, spelled out as its own case per
    # the "single-part day" concern: no suffix at all, membership is the
    # ONLY signal available, and it must still keep this out of Drives.
    videos = [_video("v3", "Driving A Wild Coastal Loop, Kerry", "real_lf")]
    live = {_IRELAND_SUPERLONG: {"v3"}}
    inserts = _plan_ids(videos, live)
    assert not any(pt == pr.PL_IRELAND_DRIVES for (_vid, pt) in inserts)


@pytest.mark.parametrize(
    ("video_id", "title", "kind", "expected_playlist"),
    [
        ("lf1", "A Drive, Ireland", "real_lf", pr.PL_IRELAND_DRIVES),
        ("sh1", "A Short, Kerry", "short", pr.PL_IRELAND_SHORTS),
        ("st1", "West NC Mountains Cam", "stream", pr.PL_MOUNTAINS_CAM),
        ("hl1", "Highlights from A Drive, Kerry", "real_lf", pr.PL_IRELAND_HIGHLIGHTS),
    ],
)
def test_non_superlong_kinds_route_unchanged(video_id, title, kind, expected_playlist):
    videos = [_video(video_id, title, kind)]
    inserts = _plan_ids(videos, live_membership={})
    assert (video_id, expected_playlist) in inserts


def test_regular_lf_not_in_any_superlong_playlist_still_goes_to_drives():
    # Membership recognition must never false-positive a regular LF that
    # simply isn't a member of anything yet.
    videos = [_video("lf2", "A Drive, Ireland", "real_lf")]
    live = {_IRELAND_SUPERLONG: set(), _USA_SUPERLONG: set()}
    inserts = _plan_ids(videos, live)
    assert ("lf2", pr.PL_IRELAND_DRIVES) in inserts


def test_resolve_playlist_ids_does_not_hard_fail_when_superlong_playlists_absent():
    # A channel with no superlong content ever uploaded has neither
    # playlist yet -- the nightly run must not hard-fail on that.
    class _Resp:
        def __init__(self, result):
            self._result = result

        def execute(self):
            return self._result

    class _Service:
        def playlists(self):
            class _Playlists:
                def list(self, **kw):
                    return _Resp(
                        {
                            "items": [
                                {"snippet": {"title": "Ireland Drives"}, "id": "PL1"},
                                {"snippet": {"title": "USA Drives"}, "id": "PL2"},
                            ]
                        }
                    )

            return _Playlists()

    titles = {"Ireland Drives", "USA Drives", _IRELAND_SUPERLONG, _USA_SUPERLONG}
    result = pr.resolve_playlist_ids(
        _Service(), titles, optional={_IRELAND_SUPERLONG, _USA_SUPERLONG}
    )
    assert result == {"Ireland Drives": "PL1", "USA Drives": "PL2"}
