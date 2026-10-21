"""Tests for the operator end-screen checklist sidecar.

After every YouTube upload, ``upload_video_with_playlist`` MUST:

1. Format an operator-readable checklist line via
   ``format_endscreen_checklist_line`` exactly once.
2. Append the line to ``<output_base>/operator_checklist.txt`` (created
   if absent, append-only on subsequent uploads).
3. Emit the same line at INFO via the module logger.

Total by construction: there is no upload path that returns a video id
without emitting the checklist line, mirroring the playlist-assignment contract.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from _state_helpers import decided


@pytest.fixture(autouse=True)
def _publication_gate_open(monkeypatch):
    # The publication gate holds every insert until the final text is bound;
    # these upload tests predate the gate, so they open it in-test only.
    monkeypatch.setattr(
        "gomoveshift.pipeline.drive_phases.youtube_uploader.FINAL_TEXT_BIND_ON", True
    )


CHECKLIST_FILENAME = "operator_checklist.txt"


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


def _build_mock_service(video_ids: list[str]) -> MagicMock:
    svc = MagicMock()
    insert_requests = [
        MagicMock(next_chunk=MagicMock(side_effect=[(None, {"id": vid})])) for vid in video_ids
    ]
    svc.videos.return_value.insert.side_effect = insert_requests

    list_request = MagicMock()
    list_request.execute.return_value = {
        "items": [{"id": "EXISTING_PL_ID", "snippet": {"title": "Ireland Drives"}}]
    }
    svc.playlists.return_value.list.return_value = list_request

    item_request = MagicMock()
    item_request.execute.return_value = {"id": "PLITEM_ID"}
    svc.playlistItems.return_value.insert.return_value = item_request
    return svc


def _make_uploader(svc, fake_creds_dict, output_base: Path):
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
    up.set_operator_checklist_path(output_base / CHECKLIST_FILENAME)
    return up, yu


def _make_cfg(reference_video_id=None):
    from gomoveshift.config import Config

    drive_block: dict = {
        "youtube_playlists": {
            "rules": [{"match": {"region": "ireland"}, "playlist_name": "Ireland Drives"}],
            "default": "All Drives",
        },
    }
    if reference_video_id is not None:
        drive_block["endscreen"] = {"reference_video_id": reference_video_id}
    return Config.model_validate({"drive": drive_block})


def _write_video(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_bytes(b"\x00\x00\x00\x20ftypiso5")
    return p


def test_first_upload_creates_checklist_file_and_logs_line(fake_creds_dict, tmp_path, caplog):
    output_base = tmp_path / "out"
    output_base.mkdir()
    svc = _build_mock_service(["VID_A"])
    up, yu = _make_uploader(svc, fake_creds_dict, output_base)
    cfg = _make_cfg(reference_video_id="REFREF")

    checklist_path = output_base / CHECKLIST_FILENAME
    assert not checklist_path.exists()

    video_path = _write_video(tmp_path, "lf_01.mp4")
    with (
        patch.object(yu, "MediaFileUpload", return_value=MagicMock()),
        caplog.at_level(logging.INFO, logger=yu.__name__),
    ):
        up.upload_video_with_playlist(
            video_path,
            title="t",
            description="d",
            tags=[],
            metadata=decided({"location": "X, Ireland"}, "t"),
            config=cfg,
        )

    assert checklist_path.exists()
    contents = checklist_path.read_text()
    expected_line = (
        "END SCREEN: open https://studio.youtube.com/video/VID_A/editor "
        "→ Endscreen → Import from video REFREF"
    )
    assert contents.strip().splitlines() == [expected_line]
    # Logger emitted the same line at INFO.
    assert any(expected_line in rec.message for rec in caplog.records)


def test_second_upload_appends_does_not_truncate(fake_creds_dict, tmp_path):
    output_base = tmp_path / "out"
    output_base.mkdir()
    svc = _build_mock_service(["VID_A", "VID_B"])
    up, yu = _make_uploader(svc, fake_creds_dict, output_base)
    cfg = _make_cfg(reference_video_id="REFREF")

    f1 = _write_video(tmp_path, "lf_01.mp4")
    f2 = _write_video(tmp_path, "lf_02.mp4")
    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        up.upload_video_with_playlist(
            f1,
            title="t",
            description="d",
            tags=[],
            metadata=decided({"location": "X, Ireland"}, "t"),
            config=cfg,
        )
        up.upload_video_with_playlist(
            f2,
            title="t",
            description="d",
            tags=[],
            metadata=decided({"location": "X, Ireland"}, "t"),
            config=cfg,
        )

    lines = (output_base / CHECKLIST_FILENAME).read_text().strip().splitlines()
    assert len(lines) == 2
    assert "VID_A" in lines[0]
    assert "VID_B" in lines[1]
    # Both lines used the import-from-video form (reference id was set).
    for line in lines:
        assert "Import from video REFREF" in line


def test_no_reference_video_id_uses_apply_template_form(fake_creds_dict, tmp_path):
    output_base = tmp_path / "out"
    output_base.mkdir()
    svc = _build_mock_service(["VID_A"])
    up, yu = _make_uploader(svc, fake_creds_dict, output_base)
    cfg = _make_cfg(reference_video_id=None)

    f1 = _write_video(tmp_path, "lf_01.mp4")
    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        up.upload_video_with_playlist(
            f1,
            title="t",
            description="d",
            tags=[],
            metadata=decided({"location": "X, Ireland"}, "t"),
            config=cfg,
        )

    line = (output_base / CHECKLIST_FILENAME).read_text().strip()
    assert line == (
        "END SCREEN: open https://studio.youtube.com/video/VID_A/editor "
        "→ Endscreen → Apply template"
    )
