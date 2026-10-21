"""Tests for upload body completeness — defaultLanguage, license,
embeddable, containsSyntheticMedia.

YouTube Data API v3 ``videos.insert`` body reference (external, not a
business fact): ``snippet.defaultLanguage`` (BCP-47 string),
``status.license`` (``"youtube"`` | ``"creativeCommon"``),
``status.embeddable`` (bool), ``status.containsSyntheticMedia`` (bool, in
``status``, part="status").
https://developers.google.com/youtube/v3/docs/videos/insert
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


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


@pytest.fixture
def mock_service():
    svc = MagicMock()
    insert_request = MagicMock()
    insert_request.next_chunk.side_effect = [
        (MagicMock(progress=lambda: 0.5), None),
        (None, {"id": "FAKEVIDEOID123"}),
    ]
    svc.videos.return_value.insert.return_value = insert_request
    svc.thumbnails.return_value.set.return_value.execute.return_value = {}
    svc.videos.return_value.update.return_value.execute.return_value = {}
    return svc


def _make_uploader(mock_service, fake_creds_dict):
    from gomoveshift.pipeline.drive_phases import youtube_uploader as yu

    with (
        patch.object(yu, "load_youtube_credentials", return_value=fake_creds_dict),
        patch.object(
            yu, "_build_credentials_obj", return_value=MagicMock(valid=True, expired=False)
        ),
        patch.object(yu, "build", return_value=mock_service) as build_mock,
        patch.object(yu, "MediaFileUpload", return_value=MagicMock()),
    ):
        up = yu.YouTubeUploader(credentials_dir=Path("/tmp/fake-creds"))
        svc = up._build_service()
        assert svc is mock_service
        assert build_mock.called
    return up, yu


def test_upload_video_sends_default_language_license_embeddable(
    mock_service, fake_creds_dict, fake_video_file
):
    """Every upload carries the three always-on fields with robust
    config-backed defaults, with no caller opt-in required."""
    up, yu = _make_uploader(mock_service, fake_creds_dict)
    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        up.upload_video(fake_video_file, title="t", description="d", tags=[])

    body = mock_service.videos.return_value.insert.call_args.kwargs["body"]
    assert body["snippet"]["defaultLanguage"] == "en"
    assert body["status"]["license"] == "youtube"
    assert body["status"]["embeddable"] is True
    assert "containsSyntheticMedia" not in body["status"]


def test_upload_video_contains_synthetic_media_true_sent(
    mock_service, fake_creds_dict, fake_video_file
):
    """Passing contains_synthetic_media=True adds it to status."""
    up, yu = _make_uploader(mock_service, fake_creds_dict)
    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        up.upload_video(
            fake_video_file,
            title="t",
            description="d",
            tags=[],
            contains_synthetic_media=True,
        )

    body = mock_service.videos.return_value.insert.call_args.kwargs["body"]
    assert body["status"]["containsSyntheticMedia"] is True


def test_upload_video_contains_synthetic_media_false_sent(
    mock_service, fake_creds_dict, fake_video_file
):
    """Passing contains_synthetic_media=False sends it explicitly (not
    omitted) — only None omits the field."""
    up, yu = _make_uploader(mock_service, fake_creds_dict)
    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        up.upload_video(
            fake_video_file,
            title="t",
            description="d",
            tags=[],
            contains_synthetic_media=False,
        )

    body = mock_service.videos.return_value.insert.call_args.kwargs["body"]
    assert body["status"]["containsSyntheticMedia"] is False


def test_upload_video_body_field_defaults_come_from_config(
    mock_service, fake_creds_dict, fake_video_file
):
    """Defaults are config-driven, not just hardcoded literals — an
    overridden config value flows into the body."""
    from gomoveshift.config import Config

    up, yu = _make_uploader(mock_service, fake_creds_dict)
    overridden_config = Config()
    overridden_config.drive.youtube_default_language = "es"
    overridden_config.drive.youtube_license = "creativeCommon"
    overridden_config.drive.youtube_embeddable = False

    with (
        patch.object(yu, "get_config", return_value=overridden_config),
        patch.object(yu, "MediaFileUpload", return_value=MagicMock()),
    ):
        up.upload_video(fake_video_file, title="t", description="d", tags=[])

    body = mock_service.videos.return_value.insert.call_args.kwargs["body"]
    assert body["snippet"]["defaultLanguage"] == "es"
    assert body["status"]["license"] == "creativeCommon"
    assert body["status"]["embeddable"] is False
