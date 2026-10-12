"""Tests for ``tools/tiktok_probe.py`` — the one-shot unaudited-path
probe. No network: adapter/auth calls are mocked. The behavior with real
teeth is the ``--yes-live`` refusal, which must hold WITHOUT ever reaching
auth or the adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import tiktok_probe  # noqa: E402

_FAKE_INIT = {
    "publish_id": "pub1",
    "upload_url": "https://open-upload.tiktokapis.com/u/1",
    "chunk_plan": mock.Mock(chunk_size=20, total_chunk_count=1, video_size=20),
}


def _video(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x" * 20)
    return str(video)


def test_refuses_without_yes_live_and_never_touches_auth(tmp_path):
    with mock.patch.object(tiktok_probe, "get_valid_tiktok_credentials") as creds:
        rc = tiktok_probe.run([_video(tmp_path), "--mode", "inbox"])
    assert rc != 0
    creds.assert_not_called()


@mock.patch.object(tiktok_probe, "upload_video_chunks")
@mock.patch.object(tiktok_probe, "query_creator_info")
@mock.patch.object(tiktok_probe, "init_direct_post", return_value=_FAKE_INIT)
@mock.patch.object(
    tiktok_probe, "get_valid_tiktok_credentials", return_value={"access_token": "tok"}
)
def test_inbox_mode_success_path_skips_creator_info_and_uploads(
    _creds, mock_init, mock_creator_info, mock_upload, tmp_path
):
    rc = tiktok_probe.run([_video(tmp_path), "--mode", "inbox", "--yes-live"])
    assert rc == 0
    assert mock_init.call_args.kwargs["mode"] == "inbox"
    mock_creator_info.assert_not_called()
    mock_upload.assert_called_once()


@mock.patch.object(tiktok_probe, "upload_video_chunks")
@mock.patch.object(tiktok_probe, "init_direct_post", return_value=_FAKE_INIT)
@mock.patch.object(tiktok_probe, "query_creator_info", return_value={"privacy_level_options": []})
@mock.patch.object(
    tiktok_probe, "get_valid_tiktok_credentials", return_value={"access_token": "tok"}
)
def test_self_only_mode_queries_creator_info_and_forces_audit_approved(
    _creds, _creator_info, mock_init, _upload, tmp_path
):
    rc = tiktok_probe.run([_video(tmp_path), "--mode", "self-only", "--yes-live"])
    assert rc == 0
    kwargs = mock_init.call_args.kwargs
    assert kwargs["mode"] == "direct"
    assert kwargs["privacy_level"] == "SELF_ONLY"
    assert kwargs["audit_approved"] is True
