"""Tests for the channel-manifest builder + classifier (no live creds)."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import httplib2
import pytest
from googleapiclient.errors import HttpError

from gomoveshift.pipeline.site_sync.channel_manifest import (
    build_channel_manifest,
    classify,
    enumerate_uploads,
    parse_iso8601_duration,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("PT1H2M3S", 3723),
        ("PT44M", 2640),
        ("PT90S", 90),
        ("PT0S", 0),
        ("P0D", 0),
        ("", 0),
        ("garbage", 0),
    ],
)
def test_parse_iso8601_duration(text, expected):
    assert parse_iso8601_duration(text) == expected


@pytest.mark.parametrize(
    ("seconds", "privacy", "expected"),
    [
        (90, "public", "short"),
        (91, "public", "real_lf"),
        (21600, "public", "real_lf"),
        (21601, "public", "stream"),
        (3600, "private", "held"),
        (3600, "unlisted", "held"),
    ],
)
def test_classify_boundaries(seconds, privacy, expected):
    assert classify(seconds, privacy) == expected


@pytest.mark.parametrize(
    ("seconds", "title", "expected"),
    [
        (2940, "West NC Mountains Cam", "stream"),  # 49 min cam → not real_lf
        (13620, "West NC Mountains Cam", "stream"),  # 227 min cam → not real_lf
        (780, "Scenic Drive, County Kerry, Ireland", "real_lf"),  # normal LF unaffected
        (3600, "Live Mountain Cam", "stream"),  # live + cam
        (0, "Mountains Cam", "stream"),  # live/upcoming report P0D -> still a stream
        (39, "Car Cam: Blue Ridge Parkway 🚗", "short"),  # short length wins over the title
        (46, "Blue Ridge Parkway Full Autumn Drive is Live!", "short"),
        (90, "Mountains Cam", "short"),  # boundary: <= SHORT_MAX_SECONDS is a short
        (91, "Mountains Cam", "stream"),  # first second past the boundary
    ],
)
def test_classify_cam_title(seconds, title, expected):
    assert classify(seconds, "public", title) == expected


class _Resp:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class _Resource:
    """A YouTube API resource whose ``.list(**kw)`` resolves a canned response."""

    def __init__(self, resolver):
        self._resolver = resolver

    def list(self, **kwargs):
        return _Resp(self._resolver(kwargs))


class _FakeService:
    def __init__(self, channels_result, playlist_pages, videos_by_id):
        self._channels = _Resource(lambda kw: channels_result)
        self._playlist = _Resource(lambda kw: playlist_pages[kw["pageToken"]])
        self._videos = _Resource(
            lambda kw: {
                "items": [videos_by_id[i] for i in kw["id"].split(",") if i in videos_by_id]
            }
        )

    def channels(self):
        return self._channels

    def playlistItems(self):
        return self._playlist

    def videos(self):
        return self._videos


def _playlist_item(video_id, title, published_at):
    return {
        "snippet": {
            "resourceId": {"videoId": video_id},
            "title": title,
            "publishedAt": published_at,
        }
    }


def _video(video_id, duration, privacy):
    return {
        "id": video_id,
        "contentDetails": {"duration": duration},
        "status": {"privacyStatus": privacy},
        "snippet": {"title": video_id},
    }


def _build_fake_service():
    channels_result = {"items": [{"contentDetails": {"relatedPlaylists": {"uploads": "UPLOADS"}}}]}
    playlist_pages = {
        None: {
            "items": [
                _playlist_item("vidShort", "A Short", "2026-04-01T12:00:00Z"),
                _playlist_item("vidLF", "Real Long Form", "2026-04-29T12:00:00Z"),
            ],
            "nextPageToken": "PAGE2",
        },
        "PAGE2": {
            "items": [
                _playlist_item("vidStream", "A Livestream", "2026-04-10T12:00:00Z"),
                _playlist_item("vidHeld", "A Private LF", "2026-04-20T12:00:00Z"),
            ],
        },
    }
    videos_by_id = {
        "vidShort": _video("vidShort", "PT45S", "public"),
        "vidLF": _video("vidLF", "PT13M", "public"),
        "vidStream": _video("vidStream", "PT7H", "public"),
        "vidHeld": _video("vidHeld", "PT12M", "private"),
    }
    return _FakeService(channels_result, playlist_pages, videos_by_id)


def test_build_channel_manifest():
    service = _build_fake_service()
    now = datetime(2026, 6, 27, 14, 0, 0, tzinfo=timezone.utc)

    manifest = build_channel_manifest(service, now=now)

    assert manifest["generated_at"] == "2026-06-27T14:00:00+00:00"
    assert manifest["source"] == "youtube-data-api/uploads-playlist"
    assert manifest["counts"] == {
        "total": 4,
        "short": 1,
        "stream": 1,
        "real_lf": 1,
        "held": 1,
    }

    # Pagination across both pages collected all four videos.
    by_id = {v["id"]: v for v in manifest["videos"]}
    assert set(by_id) == {"vidShort", "vidLF", "vidStream", "vidHeld"}
    assert by_id["vidShort"]["category"] == "short"
    assert by_id["vidLF"]["category"] == "real_lf"
    assert by_id["vidStream"]["category"] == "stream"
    assert by_id["vidStream"]["duration_seconds"] == 25200
    assert by_id["vidHeld"]["category"] == "held"
    assert by_id["vidLF"]["duration_seconds"] == 780

    # Sorted by published_at descending.
    order = [v["id"] for v in manifest["videos"]]
    assert order == ["vidLF", "vidHeld", "vidStream", "vidShort"]


# --- enumeration: the plain uploads listing repeats and omits ids ---------


def _http_error(status: int) -> HttpError:
    return HttpError(httplib2.Response({"status": status}), b"{}")


class _ListingService:
    """Playlist-aware fake: ``listings[playlistId]`` is a list of pages (lists of
    ids); a value that is an ``HttpError`` is raised on first fetch. Records every
    playlistId asked for."""

    def __init__(self, listings, videos_by_id=None):
        self.listings = listings
        self.asked = []
        self.video_calls = []
        self._videos = videos_by_id or {}

    def channels(self):
        resp = {
            "items": [
                {"id": "UCchan", "contentDetails": {"relatedPlaylists": {"uploads": "UUchan"}}}
            ]
        }
        return _Resource(lambda kw: resp)

    def playlistItems(self):
        def resolve(kw):
            pid, token = kw["playlistId"], kw.get("pageToken")
            if token is None:
                self.asked.append(pid)
            listing = self.listings.get(pid, _http_error(404))
            if isinstance(listing, Exception):
                raise listing
            index = int(token or 0)
            page = {
                "items": [
                    {"snippet": {"resourceId": {"videoId": i}, "title": f"t-{i}"}}
                    for i in listing[index]
                ]
            }
            if index + 1 < len(listing):
                page["nextPageToken"] = str(index + 1)
            return page

        return _Resource(resolve)

    def videos(self):
        def resolve(kw):
            self.video_calls.append(kw)
            return {"items": [self._videos[i] for i in kw["id"].split(",") if i in self._videos]}

        return _Resource(resolve)


def test_split_listings_recover_ids_the_plain_listing_omits():
    service = _ListingService(
        {
            "UUchan": [["a", "b"]],
            "UUSHchan": [["b", "c"]],
            "UULFchan": [["d"], ["e"]],
            "UULVchan": [["f"]],
        }
    )

    assert list(enumerate_uploads(service)) == ["a", "b", "c", "d", "e", "f"]
    assert service.asked == ["UUchan", "UUSHchan", "UULFchan", "UULVchan"]


def test_duplicate_ids_in_a_listing_are_deduped_and_warned(caplog):
    service = _ListingService(
        {
            "UUchan": [["a", "b", "a"], ["b", "a"]],
            "UUSHchan": [[]],
            "UULFchan": [[]],
            "UULVchan": [[]],
        }
    )

    with caplog.at_level(logging.WARNING):
        assert list(enumerate_uploads(service)) == ["a", "b"]

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("UUchan" in w and "5 entries" in w and "2 unique" in w for w in warnings)


def test_a_clean_listing_logs_no_duplicate_warning(caplog):
    service = _ListingService(
        {"UUchan": [["a"]], "UUSHchan": [["b"]], "UULFchan": [[]], "UULVchan": [[]]}
    )

    with caplog.at_level(logging.WARNING):
        enumerate_uploads(service)

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_split_playlist_that_404s_degrades_to_the_plain_listing(caplog):
    service = _ListingService({"UUchan": [["a", "b"]]})  # every split id 404s

    with caplog.at_level(logging.INFO):
        assert list(enumerate_uploads(service)) == ["a", "b"]

    assert service.asked == ["UUchan", "UUSHchan", "UULFchan", "UULVchan"]
    assert "UULVchan" in caplog.text


def test_a_non_404_error_on_a_split_playlist_is_not_swallowed():
    service = _ListingService({"UUchan": [["a"]], "UUSHchan": _http_error(500)})

    with pytest.raises(HttpError):
        enumerate_uploads(service)


def test_a_plain_listing_error_always_propagates():
    service = _ListingService({"UUchan": _http_error(404)})

    with pytest.raises(HttpError):
        enumerate_uploads(service)


def test_a_playlist_id_without_the_uploads_prefix_is_listed_plain_only():
    service = _build_fake_service()

    assert list(enumerate_uploads(service)) == ["vidShort", "vidLF", "vidStream", "vidHeld"]


def test_manifest_includes_videos_only_the_split_listings_return():
    service = _ListingService(
        {"UUchan": [["a"]], "UUSHchan": [["a", "s"]], "UULFchan": [["l"]]},
        {
            "a": _video("a", "PT30S", "public"),
            "s": _video("s", "PT40S", "public"),
            "l": _video("l", "PT20M", "public"),
        },
    )

    manifest = build_channel_manifest(service, now=datetime(2026, 10, 5, tzinfo=timezone.utc))

    assert sorted(v["id"] for v in manifest["videos"]) == ["a", "l", "s"]
    assert manifest["counts"]["total"] == 3
    assert manifest["counts"]["short"] == 2
    assert manifest["counts"]["real_lf"] == 1


# --- locally known ids: videos.list recovers what no listing returns (review fix) ---


def _owned(video_id, channel_id="UCchan", title=None):
    return {
        "id": video_id,
        "snippet": {
            "channelId": channel_id,
            "title": title or f"own-{video_id}",
            "publishedAt": "2026-01-02T03:04:05Z",
        },
    }


def _all_listings(plain):
    return {"UUchan": [plain], "UUSHchan": [[]], "UULFchan": [[]], "UULVchan": [[]]}


def test_a_locally_known_id_no_listing_returns_is_recovered_via_videos_list(caplog):
    service = _ListingService(_all_listings(["a"]), {"p": _owned("p", title="Private One")})

    with caplog.at_level(logging.INFO):
        found = enumerate_uploads(service, known_ids=["a", "p"])

    assert list(found) == ["a", "p"]
    assert found["p"]["title"] == "Private One"
    assert found["p"]["publishedAt"] == "2026-01-02T03:04:05Z"
    assert [c["id"] for c in service.video_calls] == ["p"]  # "a" was listed, not re-fetched
    assert any(
        "recovered 1" in r.getMessage() and "0 unresolved" in r.getMessage() for r in caplog.records
    )


def test_a_known_id_owned_by_another_channel_is_ignored():
    service = _ListingService(_all_listings(["a"]), {"x": _owned("x", channel_id="UCother")})

    assert list(enumerate_uploads(service, known_ids=["x"])) == ["a"]


def test_a_deleted_known_id_videos_list_does_not_return_is_ignored(caplog):
    service = _ListingService(_all_listings(["a"]), {})

    with caplog.at_level(logging.INFO):
        assert list(enumerate_uploads(service, known_ids=["gone"])) == ["a"]

    assert any(
        "recovered 0" in r.getMessage() and "1 unresolved" in r.getMessage() for r in caplog.records
    )


def test_without_known_ids_videos_list_is_never_called():
    service = _ListingService(_all_listings(["a"]), {"p": _owned("p")})

    enumerate_uploads(service)
    enumerate_uploads(service, known_ids=[])
    enumerate_uploads(service, known_ids=["a", "a"])  # already listed -> nothing to fetch

    assert service.video_calls == []


def test_known_ids_are_fetched_in_batches_of_fifty_and_deduped():
    ids = [f"k{i}" for i in range(120)]
    service = _ListingService(_all_listings([]), {i: _owned(i) for i in ids})

    found = enumerate_uploads(service, known_ids=ids + ids[:5])

    assert len(found) == 120
    assert [len(c["id"].split(",")) for c in service.video_calls] == [50, 50, 20]
    assert all(c["part"] == "snippet" for c in service.video_calls)


def test_known_ids_cannot_be_verified_without_the_channel_id():
    service = _ListingService(_all_listings(["a"]), {"p": _owned("p")})
    service.channels = lambda: _Resource(
        lambda kw: {"items": [{"contentDetails": {"relatedPlaylists": {"uploads": "UUchan"}}}]}
    )

    assert list(enumerate_uploads(service, known_ids=["p"])) == ["a"]
    assert service.video_calls == []


def test_manifest_forwards_known_ids_so_a_private_video_is_held_not_invisible():
    held = {
        **_owned("p"),
        "contentDetails": {"duration": "PT30S"},
        "status": {"privacyStatus": "private"},
    }
    service = _ListingService(
        _all_listings(["a"]), {"a": _video("a", "PT30S", "public"), "p": held}
    )

    manifest = build_channel_manifest(
        service, now=datetime(2026, 10, 5, tzinfo=timezone.utc), known_ids=["p"]
    )

    assert {v["id"]: v["category"] for v in manifest["videos"]} == {"a": "short", "p": "held"}


# --- frozen seed of ids no listing returns (private/unlisted), wired by default ---


def test_frozen_seed_is_a_unique_list_of_video_ids():
    from gomoveshift.pipeline.site_sync import known_ids

    seed = known_ids.FROZEN_KNOWN_IDS
    # The shipped default is empty: a deployment supplies its own ids (the
    # override path is exercised by the monkeypatched-seed tests below).
    assert seed == ()
    assert len(set(seed)) == len(seed)
    assert all(re.fullmatch(r"[A-Za-z0-9_-]{11}", i) for i in seed)


def test_manifest_defaults_known_ids_to_the_frozen_seed(monkeypatch):
    from gomoveshift.pipeline.site_sync import channel_manifest as cm

    monkeypatch.setattr(cm, "FROZEN_KNOWN_IDS", ("p",))
    held = {
        **_owned("p"),
        "contentDetails": {"duration": "PT30S"},
        "status": {"privacyStatus": "private"},
    }
    service = _ListingService(
        _all_listings(["a"]), {"a": _video("a", "PT30S", "public"), "p": held}
    )

    manifest = build_channel_manifest(service, now=datetime(2026, 10, 5, tzinfo=timezone.utc))

    assert {v["id"]: v["category"] for v in manifest["videos"]} == {"a": "short", "p": "held"}


def test_manifest_explicit_known_ids_replace_the_seed(monkeypatch):
    from gomoveshift.pipeline.site_sync import channel_manifest as cm

    monkeypatch.setattr(cm, "FROZEN_KNOWN_IDS", ("p",))
    service = _ListingService(_all_listings(["a"]), {"a": _video("a", "PT30S", "public")})

    build_channel_manifest(service, now=datetime(2026, 10, 5, tzinfo=timezone.utc), known_ids=[])

    assert all("p" not in kw["id"].split(",") for kw in service.video_calls)
