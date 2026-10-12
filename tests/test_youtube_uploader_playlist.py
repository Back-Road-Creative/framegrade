"""Tests for YouTubeUploader playlist auto-assignment.

The uploader has no path that uploads without calling ``playlistItems.insert``
exactly once. Playlist resolution is total via ``resolve_playlist``; the
``default`` from config makes assignment universal.

Idempotency: ``playlists.list`` is called at most once per uploader instance
(per-run cache); when a playlist with the resolved title exists, no
``playlists.insert`` happens. When none exists, ``playlists.insert`` runs
before ``playlistItems.insert``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fake_creds_dict():
    return {
        "token": "ya29.fake",
        "refresh_token": "1//fake",
        "client_id": "fake.apps.googleusercontent.com",
        "client_secret": "fakesecret",  # pragma: allowlist secret
        "token_uri": "https://oauth2.googleapis.com/token",
        "scopes": ["https://www.googleapis.com/auth/youtube.upload"],
    }


@pytest.fixture
def fake_video_file(tmp_path: Path) -> Path:
    p = tmp_path / "lf_01.mp4"
    p.write_bytes(b"\x00\x00\x00\x20ftypiso5")
    return p


def _build_mock_service(*, existing_playlists: list[dict] | None = None) -> MagicMock:
    """Build a YouTube service mock for upload + playlist flows.

    ``existing_playlists``: list of {"id": str, "snippet": {"title": str}}
    returned by playlists.list. None or [] means no playlists exist.
    """
    svc = MagicMock()

    # --- videos().insert resumable upload
    insert_request = MagicMock()
    insert_request.next_chunk.side_effect = [
        (None, {"id": "VIDEO_ID_1"}),
    ]
    svc.videos.return_value.insert.return_value = insert_request

    # --- playlists().list
    list_request = MagicMock()
    list_request.execute.return_value = {"items": list(existing_playlists or [])}
    svc.playlists.return_value.list.return_value = list_request

    # --- playlists().insert
    insert_pl_request = MagicMock()
    insert_pl_request.execute.return_value = {
        "id": "NEW_PLAYLIST_ID",
        "snippet": {"title": "Ireland Drives"},
    }
    svc.playlists.return_value.insert.return_value = insert_pl_request

    # --- playlistItems().insert
    insert_item_request = MagicMock()
    insert_item_request.execute.return_value = {"id": "PLITEM_ID"}
    svc.playlistItems.return_value.insert.return_value = insert_item_request

    return svc


def _make_uploader(svc, fake_creds_dict):
    from gomoveshift.pipeline.drive_phases import youtube_uploader as yu

    with (
        patch.object(yu, "load_youtube_credentials", return_value=fake_creds_dict),
        patch.object(
            yu, "_build_credentials_obj", return_value=MagicMock(valid=True, expired=False)
        ),
        patch.object(yu, "build", return_value=svc),
        patch.object(yu, "MediaFileUpload", return_value=MagicMock()),
    ):
        up = yu.YouTubeUploader(credentials_dir=Path("/tmp/fake-creds"))
        up._build_service()
    return up, yu


def _make_cfg() -> "object":
    from gomoveshift.config import Config

    return Config.model_validate(
        {
            "drive": {
                "youtube_playlists": {
                    "rules": [
                        {"match": {"region": "ireland"}, "playlist_name": "Ireland Drives"},
                        {"match": {"region": "usa"}, "playlist_name": "USA Drives"},
                    ],
                    "default": "All Drives",
                }
            }
        }
    )


def test_get_or_create_creates_playlist_when_missing(fake_creds_dict, fake_video_file):
    """No existing playlist → playlists.insert before playlistItems.insert."""
    svc = _build_mock_service(existing_playlists=[])
    up, yu = _make_uploader(svc, fake_creds_dict)
    cfg = _make_cfg()

    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        vid = up.upload_video_with_playlist(
            fake_video_file,
            title="Driving Dingle",
            description="d",
            tags=[],
            metadata={"location": "Dingle, Kerry, Ireland"},
            config=cfg,
        )

    assert vid == "VIDEO_ID_1"
    # playlists.list called exactly once
    assert svc.playlists.return_value.list.call_count == 1
    # playlists.insert was called (no match in list)
    assert svc.playlists.return_value.insert.call_count == 1
    insert_body = svc.playlists.return_value.insert.call_args.kwargs["body"]
    assert insert_body["snippet"]["title"] == "Ireland Drives"
    # playlistItems.insert called exactly once with the new playlist id
    assert svc.playlistItems.return_value.insert.call_count == 1
    item_body = svc.playlistItems.return_value.insert.call_args.kwargs["body"]
    assert item_body["snippet"]["playlistId"] == "NEW_PLAYLIST_ID"
    assert item_body["snippet"]["resourceId"]["videoId"] == "VIDEO_ID_1"


def test_idempotent_playlist_when_exists(fake_creds_dict, fake_video_file):
    """Existing playlist with matching title → no playlists.insert."""
    svc = _build_mock_service(
        existing_playlists=[
            {"id": "EXISTING_PL_ID", "snippet": {"title": "Ireland Drives"}},
            {"id": "OTHER_PL_ID", "snippet": {"title": "Other"}},
        ]
    )
    up, yu = _make_uploader(svc, fake_creds_dict)
    cfg = _make_cfg()

    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        up.upload_video_with_playlist(
            fake_video_file,
            title="Driving Dingle",
            description="d",
            tags=[],
            metadata={"location": "Dingle, Kerry, Ireland"},
            config=cfg,
        )

    assert svc.playlists.return_value.list.call_count == 1
    # No insert — playlist already existed
    assert svc.playlists.return_value.insert.call_count == 0
    # playlistItems.insert called once with the existing playlist id
    assert svc.playlistItems.return_value.insert.call_count == 1
    item_body = svc.playlistItems.return_value.insert.call_args.kwargs["body"]
    assert item_body["snippet"]["playlistId"] == "EXISTING_PL_ID"


def test_one_playlists_list_per_run_cache(fake_creds_dict, tmp_path):
    """playlists.list is called at most once per uploader instance, even
    across multiple uploads — the resolved name → id map is cached."""
    svc = _build_mock_service(
        existing_playlists=[{"id": "EXISTING_PL_ID", "snippet": {"title": "Ireland Drives"}}]
    )
    # Two upload requests in a row need two video-insert request objects.
    svc.videos.return_value.insert.side_effect = [
        MagicMock(next_chunk=MagicMock(side_effect=[(None, {"id": "VID_A"})])),
        MagicMock(next_chunk=MagicMock(side_effect=[(None, {"id": "VID_B"})])),
    ]
    up, yu = _make_uploader(svc, fake_creds_dict)
    cfg = _make_cfg()

    f1 = tmp_path / "lf_01.mp4"
    f1.write_bytes(b"\x00\x00\x00\x20ftypiso5")
    f2 = tmp_path / "lf_02.mp4"
    f2.write_bytes(b"\x00\x00\x00\x20ftypiso5")

    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        up.upload_video_with_playlist(
            f1,
            title="A",
            description="",
            tags=[],
            metadata={"location": "X, Ireland"},
            config=cfg,
        )
        up.upload_video_with_playlist(
            f2,
            title="B",
            description="",
            tags=[],
            metadata={"location": "Y, Ireland"},
            config=cfg,
        )

    # Cache hit on second call — only one playlists.list across both uploads.
    assert svc.playlists.return_value.list.call_count == 1
    # Two videos → two playlistItems.insert calls.
    assert svc.playlistItems.return_value.insert.call_count == 2


def test_default_playlist_used_when_no_rule_matches(fake_creds_dict, fake_video_file):
    """No matching rule → resolver returns ``default``; uploader still assigns."""
    svc = _build_mock_service(existing_playlists=[])
    # The created playlist will be "All Drives" not "Ireland Drives".
    svc.playlists.return_value.insert.return_value.execute.return_value = {
        "id": "ALL_DRIVES_PL_ID",
        "snippet": {"title": "All Drives"},
    }
    up, yu = _make_uploader(svc, fake_creds_dict)
    cfg = _make_cfg()

    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        up.upload_video_with_playlist(
            fake_video_file,
            title="Driving Belgium",
            description="d",
            tags=[],
            metadata={"location": "Somewhere, Belgium"},
            config=cfg,
        )

    insert_body = svc.playlists.return_value.insert.call_args.kwargs["body"]
    assert insert_body["snippet"]["title"] == "All Drives"
    assert svc.playlistItems.return_value.insert.call_count == 1


def test_every_upload_calls_playlistitems_insert(fake_creds_dict, fake_video_file):
    """Universal contract: for every videos.insert there is exactly one
    playlistItems.insert. No upload path bypasses playlist assignment."""
    svc = _build_mock_service(existing_playlists=[])
    up, yu = _make_uploader(svc, fake_creds_dict)
    cfg = _make_cfg()

    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        up.upload_video_with_playlist(
            fake_video_file,
            title="t",
            description="d",
            tags=[],
            metadata={"location": "X, Ireland"},
            config=cfg,
        )

    assert svc.videos.return_value.insert.call_count == 1
    assert svc.playlistItems.return_value.insert.call_count == 1
