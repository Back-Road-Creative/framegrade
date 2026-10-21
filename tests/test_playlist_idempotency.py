"""YouTube playlist insert idempotency.

If ``videos.insert`` succeeds but ``playlistItems.insert`` fails, the video
is orphaned on YouTube with no playlist. Resume must re-run only the
playlist assignment, never re-uploading the video.

Contract:
- ``upload_if_needed`` persists ``video_id`` BEFORE attempting the playlist
  insert (so a crash between the two API calls leaves a recoverable record).
- On success, ``playlist_id`` is persisted on the manifest record.
- On resume, ``video_id`` present + ``playlist_id`` missing triggers a
  playlist-only re-run (no second ``videos.insert``).
- Existing behaviour preserved: both present → full skip; both absent → full
  upload + assign.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from googleapiclient.errors import HttpError

from _state_helpers import make_state

from gomoveshift.pipeline.title_decision import decide_title


@pytest.fixture(autouse=True)
def _publication_gate_open(monkeypatch):
    # The publication gate holds every insert until the final text is bound;
    # these upload tests predate the gate, so they open it in-test only.
    monkeypatch.setattr(
        "gomoveshift.pipeline.drive_phases.youtube_uploader.FINAL_TEXT_BIND_ON", True
    )


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


def _build_mock_service(
    *,
    existing_playlists: list[dict] | None = None,
    playlistitem_insert_raises: Exception | None = None,
) -> MagicMock:
    svc = MagicMock()

    insert_request = MagicMock()
    insert_request.next_chunk.side_effect = [
        (None, {"id": "VIDEO_ID_1"}),
    ]
    svc.videos.return_value.insert.return_value = insert_request

    list_request = MagicMock()
    list_request.execute.return_value = {"items": list(existing_playlists or [])}
    svc.playlists.return_value.list.return_value = list_request

    insert_pl_request = MagicMock()
    insert_pl_request.execute.return_value = {
        "id": "NEW_PLAYLIST_ID",
        "snippet": {"title": "All Drives"},
    }
    svc.playlists.return_value.insert.return_value = insert_pl_request

    insert_item_request = MagicMock()
    if playlistitem_insert_raises is not None:
        insert_item_request.execute.side_effect = playlistitem_insert_raises
    else:
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


def _make_cfg():
    from gomoveshift.config import Config

    return Config.model_validate(
        {
            "drive": {
                "youtube_playlists": {
                    "rules": [],
                    "default": "All Drives",
                }
            }
        }
    )


def _fake_http_error(status: int = 500) -> HttpError:
    resp = MagicMock()
    resp.status = status
    resp.reason = "Internal Server Error"
    return HttpError(resp, b'{"error": {"message": "boom"}}')


# ---------------------------------------------------------------------------
# Playlist insert is now idempotent across resume


def test_playlist_failure_persists_video_id_for_recovery(fake_creds_dict, fake_video_file):
    """Playlist insert fails after video insert succeeds — record stores
    video_id (no playlist_id) so resume can complete only the playlist step.
    """
    svc = _build_mock_service(playlistitem_insert_raises=_fake_http_error(500))
    up, yu = _make_uploader(svc, fake_creds_dict)
    cfg = _make_cfg()
    state = make_state()

    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        with pytest.raises(HttpError):
            up.upload_if_needed(
                state=state,
                lf_idx=1,
                video_path=fake_video_file,
                title="t",
                description="d",
                tags=[],
                metadata={
                    "location": "Somewhere",
                    "title_decision": decide_title("longform", "t", None, suffix=""),
                },
                config=cfg,
            )

    # video_id was persisted to state BEFORE the playlist insert raised
    record = state["youtube_uploads"][1]
    assert record["video_id"] == "VIDEO_ID_1"
    assert record.get("playlist_id") is None
    # videos.insert was attempted exactly once
    assert svc.videos.return_value.insert.call_count == 1


def test_resume_runs_only_playlist_insert_after_orphan(fake_creds_dict, fake_video_file):
    """Second run on a state with video_id-but-no-playlist_id must do only
    playlistItems.insert — NOT a second videos.insert.
    """
    svc = _build_mock_service()
    up, yu = _make_uploader(svc, fake_creds_dict)
    cfg = _make_cfg()
    # Simulate the post-orphan state from the previous test
    state = make_state(
        youtube_uploads={
            1: {
                "video_id": "VIDEO_ID_1",
                "url": "https://youtu.be/VIDEO_ID_1",
                "publish_at": None,
                "title": "t",
                "uploaded_at": "2026-05-15T00:00:00+00:00",
                # playlist_id intentionally absent
            }
        },
    )

    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        record = up.upload_if_needed(
            state=state,
            lf_idx=1,
            video_path=fake_video_file,
            title="t",
            description="d",
            tags=[],
            metadata={"location": "Somewhere"},
            config=cfg,
        )

    # videos.insert MUST NOT be called (no re-upload)
    svc.videos.return_value.insert.assert_not_called()
    # playlistItems.insert ran exactly once for the recovery
    assert svc.playlistItems.return_value.insert.call_count == 1
    # playlist_id now persisted
    assert record["video_id"] == "VIDEO_ID_1"
    assert record["playlist_id"] == "NEW_PLAYLIST_ID"
    assert state["youtube_uploads"][1]["playlist_id"] == "NEW_PLAYLIST_ID"


def test_full_skip_when_both_video_and_playlist_present(fake_creds_dict, fake_video_file):
    """When both video_id and playlist_id are persisted, resume skips both
    API calls (preserves the earlier idempotency behaviour).
    """
    svc = _build_mock_service()
    up, yu = _make_uploader(svc, fake_creds_dict)
    cfg = _make_cfg()
    state = make_state(
        youtube_uploads={
            2: {
                "video_id": "VIDEO_ID_1",
                "playlist_id": "PRE_EXISTING_PL",
                "url": "https://youtu.be/VIDEO_ID_1",
                "publish_at": None,
                "title": "t",
                "uploaded_at": "2026-05-15T00:00:00+00:00",
            }
        },
    )

    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        record = up.upload_if_needed(
            state=state,
            lf_idx=2,
            video_path=fake_video_file,
            title="t",
            description="d",
            tags=[],
            metadata={"location": "Somewhere"},
            config=cfg,
        )

    # Both APIs untouched on full resume
    svc.videos.return_value.insert.assert_not_called()
    svc.playlistItems.return_value.insert.assert_not_called()
    assert record["video_id"] == "VIDEO_ID_1"
    assert record["playlist_id"] == "PRE_EXISTING_PL"


def test_first_run_persists_both_video_and_playlist_id(fake_creds_dict, fake_video_file):
    """Happy path: fresh state, both APIs succeed, both ids persist."""
    svc = _build_mock_service()
    up, yu = _make_uploader(svc, fake_creds_dict)
    cfg = _make_cfg()
    state = make_state()

    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        record = up.upload_if_needed(
            state=state,
            lf_idx=3,
            video_path=fake_video_file,
            title="t",
            description="d",
            tags=[],
            metadata={
                "location": "Somewhere",
                "title_decision": decide_title("longform", "t", None, suffix=""),
            },
            config=cfg,
        )

    assert record["video_id"] == "VIDEO_ID_1"
    assert record["playlist_id"] == "NEW_PLAYLIST_ID"
    assert state["youtube_uploads"][3]["playlist_id"] == "NEW_PLAYLIST_ID"
    assert svc.videos.return_value.insert.call_count == 1
    assert svc.playlistItems.return_value.insert.call_count == 1


def test_orphan_recovery_skipped_when_no_metadata_config(fake_creds_dict, fake_video_file):
    """If caller doesn't pass metadata+config (legacy path), the
    orphan-recovery branch must NOT fire — there is no resolvable
    playlist in that mode.
    """
    svc = _build_mock_service()
    up, yu = _make_uploader(svc, fake_creds_dict)
    state = make_state(
        youtube_uploads={
            5: {
                "video_id": "VIDEO_ID_1",
                "url": "https://youtu.be/VIDEO_ID_1",
                "publish_at": None,
                "title": "t",
                "uploaded_at": "2026-05-15T00:00:00+00:00",
            }
        },
    )

    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        record = up.upload_if_needed(
            state=state,
            lf_idx=5,
            video_path=fake_video_file,
            title="t",
            description="d",
            tags=[],
        )

    svc.videos.return_value.insert.assert_not_called()
    svc.playlistItems.return_value.insert.assert_not_called()
    assert record["video_id"] == "VIDEO_ID_1"
