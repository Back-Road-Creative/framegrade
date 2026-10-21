"""Tests for YouTubeUploader — fully mocks googleapiclient, no network."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from _state_helpers import make_state, decided


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
    # Canonical pipeline LF filename (matches the default
    # YOUTUBE_UPLOAD_FILENAME_PATTERN guard ``^(?:lf|sl|short)_\d{2}\.mp4$``).
    p = tmp_path / "lf_01.mp4"
    p.write_bytes(b"\x00\x00\x00\x20ftypiso5")
    return p


@pytest.fixture
def mock_service():
    """A mock youtube service whose insert().next_chunk() iterates to completion."""
    svc = MagicMock()
    insert_request = MagicMock()
    # next_chunk returns (status, response) — first call returns in-progress, second returns done.
    insert_request.next_chunk.side_effect = [
        (MagicMock(progress=lambda: 0.5), None),
        (None, {"id": "FAKEVIDEOID123"}),
    ]
    svc.videos.return_value.insert.return_value = insert_request
    svc.thumbnails.return_value.set.return_value.execute.return_value = {}
    svc.videos.return_value.update.return_value.execute.return_value = {}
    return svc


def _make_uploader(mock_service, fake_creds_dict):
    """Construct a YouTubeUploader with mocked build + creds loading."""
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
        # Force service init
        svc = up._build_service()
        assert svc is mock_service
        assert build_mock.called
    return up, yu


def test_upload_video_request_body(mock_service, fake_creds_dict, fake_video_file):
    up, yu = _make_uploader(mock_service, fake_creds_dict)
    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        vid = up.upload_video(
            fake_video_file,
            title="A scenic drive",
            description="A scenic drive",
            tags=["driving", "pov"],
        )

    assert vid == "FAKEVIDEOID123"

    # Inspect the call to videos().insert
    call_kwargs = mock_service.videos.return_value.insert.call_args.kwargs
    body = call_kwargs["body"]
    # Title is sent verbatim — no silent truncation. Over-cap titles raise
    # via the API-limits gate instead (see test_upload_video_rejects_*).
    assert body["snippet"]["title"] == "A scenic drive"
    assert body["snippet"]["description"] == "A scenic drive"
    assert body["snippet"]["tags"] == ["driving", "pov"]
    assert body["snippet"]["categoryId"] == "2"
    assert body["status"]["privacyStatus"] == "private"
    assert "publishAt" not in body["status"]


def test_upload_video_rejects_over_limit_metadata(mock_service, fake_creds_dict, fake_video_file):
    """The API-limits gate raises on an over-cap title/description or on the
    ``<``/``>`` YouTube rejects, rather than silently truncating (which
    desynced the state record from the metadata files/CSV) or 400-ing
    mid-batch after every gate already passed."""
    up, yu = _make_uploader(mock_service, fake_creds_dict)
    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        with pytest.raises(ValueError, match="title too long"):
            up.upload_video(fake_video_file, title="A" * 101, description="d", tags=[])
        with pytest.raises(ValueError, match="description too long"):
            up.upload_video(fake_video_file, title="t", description="x" * 5001, tags=[])
        with pytest.raises(ValueError, match="API rejects"):
            up.upload_video(fake_video_file, title="a<b>c", description="d", tags=[])


def test_upload_video_with_publish_at(mock_service, fake_creds_dict, fake_video_file):
    from datetime import datetime, timezone

    up, yu = _make_uploader(mock_service, fake_creds_dict)
    publish_at = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)

    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        up.upload_video(
            fake_video_file,
            title="Hello",
            description="d",
            tags=[],
            publish_at=publish_at,
        )

    body = mock_service.videos.return_value.insert.call_args.kwargs["body"]
    assert "publishAt" in body["status"]
    assert body["status"]["publishAt"].startswith("2026-05-04T12:00:00")


def test_upload_video_omits_recording_details_without_date(
    mock_service, fake_creds_dict, fake_video_file
):
    """No recording_date → body must not include recordingDetails, and the
    insert ``part`` string must not request the recordingDetails resource
    (sending the part without a body is harmless but adds noise to the API
    log; keeping it minimal makes payload diffs in test failures readable)."""
    up, yu = _make_uploader(mock_service, fake_creds_dict)
    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        up.upload_video(fake_video_file, title="t", description="d", tags=[])

    call_kwargs = mock_service.videos.return_value.insert.call_args.kwargs
    assert "recordingDetails" not in call_kwargs["body"]
    assert "recordingDetails" not in call_kwargs["part"]


def test_upload_video_with_recording_date(mock_service, fake_creds_dict, fake_video_file):
    """recording_date='YYYY-MM-DD' → recordingDetails.recordingDate set as
    ISO 8601 UTC midnight (the format YouTube Data API v3 requires), and
    insert ``part`` extended to include recordingDetails."""
    up, yu = _make_uploader(mock_service, fake_creds_dict)
    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        up.upload_video(
            fake_video_file,
            title="t",
            description="d",
            tags=[],
            recording_date="2025-09-14",
        )

    call_kwargs = mock_service.videos.return_value.insert.call_args.kwargs
    body = call_kwargs["body"]
    assert body["recordingDetails"] == {"recordingDate": "2025-09-14T00:00:00.000Z"}
    assert "recordingDetails" in call_kwargs["part"]


def test_upload_video_recording_date_passthrough_iso_timestamp(
    mock_service, fake_creds_dict, fake_video_file
):
    """A caller passing a full ISO 8601 UTC timestamp (already-normalized) is
    accepted verbatim — useful for callers that derive the moment with sub-day
    precision (e.g. ffprobe ``creation_time``)."""
    up, yu = _make_uploader(mock_service, fake_creds_dict)
    iso = "2025-09-14T12:00:00.000Z"
    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        up.upload_video(
            fake_video_file,
            title="t",
            description="d",
            tags=[],
            recording_date=iso,
        )

    body = mock_service.videos.return_value.insert.call_args.kwargs["body"]
    assert body["recordingDetails"]["recordingDate"] == iso


def test_upload_if_needed_forwards_recording_date(mock_service, fake_creds_dict, fake_video_file):
    """End-to-end: ``recording_date`` passed to ``upload_if_needed`` (the
    Phase-D entry point) lands on the API body's recordingDetails. Guards
    against the kwarg being dropped at any plumbing layer."""
    up, yu = _make_uploader(mock_service, fake_creds_dict)
    state = make_state()
    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        up.upload_if_needed(
            state=state,
            lf_idx=1,
            video_path=fake_video_file,
            title="t",
            metadata=decided({}, "t"),
            description="d",
            tags=[],
            recording_date="2025-09-14",
        )

    body = mock_service.videos.return_value.insert.call_args.kwargs["body"]
    assert body["recordingDetails"]["recordingDate"] == "2025-09-14T00:00:00.000Z"


def test_upload_video_recording_date_rejects_bogus_string(
    mock_service, fake_creds_dict, fake_video_file
):
    """Garbage in raises ValueError at the boundary, not silently uploads.
    Boundary check is cheap; the API would 400 hours later otherwise."""
    up, yu = _make_uploader(mock_service, fake_creds_dict)
    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        with pytest.raises(ValueError, match="recording_date"):
            up.upload_video(
                fake_video_file,
                title="t",
                description="d",
                tags=[],
                recording_date="not-a-date",
            )


def test_resume_skips_previously_uploaded(mock_service, fake_creds_dict, fake_video_file):
    """If state already has youtube_uploads[lf_idx], upload_if_needed skips."""
    up, yu = _make_uploader(mock_service, fake_creds_dict)

    state = make_state(
        youtube_uploads={
            1: {
                "video_id": "EXISTING123",
                "url": "https://youtu.be/EXISTING123",
                "publish_at": None,
                "title": "prev",
                "uploaded_at": "2026-04-12T00:00:00+00:00",
            }
        }
    )

    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        result = up.upload_if_needed(
            state=state,
            lf_idx=1,
            video_path=fake_video_file,
            title="new title",
            description="d",
            tags=[],
        )

    assert result["video_id"] == "EXISTING123"
    # insert was NOT called
    mock_service.videos.return_value.insert.assert_not_called()


def test_upload_if_needed_records_state(mock_service, fake_creds_dict, fake_video_file):
    up, yu = _make_uploader(mock_service, fake_creds_dict)
    state = make_state()

    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        result = up.upload_if_needed(
            state=state,
            lf_idx=3,
            video_path=fake_video_file,
            title="t",
            metadata=decided({}, "t"),
            description="d",
            tags=["a"],
        )

    assert result["video_id"] == "FAKEVIDEOID123"
    assert state["youtube_uploads"][3]["video_id"] == "FAKEVIDEOID123"
    assert state["youtube_uploads"][3]["url"].endswith("FAKEVIDEOID123")
    assert "uploaded_at" in state["youtube_uploads"][3]


def test_retry_on_network_error_capped(mock_service, fake_creds_dict, fake_video_file):
    """Transient HttpError / IOError should retry, capped at 5."""
    from gomoveshift.pipeline.drive_phases import youtube_uploader as yu

    insert_request = MagicMock()
    # Always fail transiently — should attempt 5 times then raise
    insert_request.next_chunk.side_effect = OSError("network blip")
    mock_service.videos.return_value.insert.return_value = insert_request

    with (
        patch.object(yu, "load_youtube_credentials", return_value=fake_creds_dict),
        patch.object(
            yu, "_build_credentials_obj", return_value=MagicMock(valid=True, expired=False)
        ),
        patch.object(yu, "build", return_value=mock_service),
        patch.object(yu, "MediaFileUpload", return_value=MagicMock()),
        patch.object(yu.time, "sleep"),
    ):
        up = yu.YouTubeUploader(credentials_dir=Path("/tmp/fake-creds"))
        with pytest.raises(OSError):
            up.upload_video(fake_video_file, title="t", description="d", tags=[])

    # 5 attempts at next_chunk
    assert insert_request.next_chunk.call_count == 5


def test_refresh_error_not_retried(mock_service, fake_creds_dict, fake_video_file):
    """RefreshError is a hard auth failure — do not retry."""
    from google.auth.exceptions import RefreshError

    from gomoveshift.pipeline.drive_phases import youtube_uploader as yu

    insert_request = MagicMock()
    insert_request.next_chunk.side_effect = RefreshError("token dead")
    mock_service.videos.return_value.insert.return_value = insert_request

    with (
        patch.object(yu, "load_youtube_credentials", return_value=fake_creds_dict),
        patch.object(
            yu, "_build_credentials_obj", return_value=MagicMock(valid=True, expired=False)
        ),
        patch.object(yu, "build", return_value=mock_service),
        patch.object(yu, "MediaFileUpload", return_value=MagicMock()),
        patch.object(yu.time, "sleep"),
    ):
        up = yu.YouTubeUploader(credentials_dir=Path("/tmp/fake-creds"))
        with pytest.raises(RefreshError):
            up.upload_video(fake_video_file, title="t", description="d", tags=[])

    assert insert_request.next_chunk.call_count == 1


def test_set_thumbnail(mock_service, fake_creds_dict, fake_video_file, tmp_path):
    up, yu = _make_uploader(mock_service, fake_creds_dict)
    thumb = tmp_path / "thumb.jpg"
    thumb.write_bytes(b"\xff\xd8\xff")

    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        up.set_thumbnail("VID1", thumb)

    mock_service.thumbnails.return_value.set.assert_called_once()
    args = mock_service.thumbnails.return_value.set.call_args.kwargs
    assert args["videoId"] == "VID1"


def test_update_privacy_unlist(mock_service, fake_creds_dict):
    up, _yu = _make_uploader(mock_service, {**fake_creds_dict})
    up.update_privacy("VID1", privacy="unlisted")
    call = mock_service.videos.return_value.update.call_args.kwargs
    assert call["part"] == "status"
    assert call["body"]["id"] == "VID1"
    assert call["body"]["status"]["privacyStatus"] == "unlisted"
    assert "publishAt" not in call["body"]["status"]


# ----------------------------------------------------------------------
# Filename guard — reject non-canonical filenames at the upload boundary.
#
# Context: legacy "episode NN FINAL" drafts from an ancient pipeline
# version were uploaded as raw episode concats. Current code paths do not
# exhibit that bug, but a defense-in-depth filename check prevents any
# future path that bypasses LF assembly from uploading the wrong asset.
#
# The guard matches against ``YOUTUBE_UPLOAD_FILENAME_PATTERN`` in
# ``gomoveshift.config`` (default ``^(?:lf|sl|short)_\d{2}\.mp4$``). Tests must read
# the pattern at call time so dev overrides take effect per-invocation.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_name",
    [
        "episode 01 FINAL.mp4",
        "episode 02 FINAL.mp4",
        "episode 03 FINAL.mp4",
    ],
)
def test_upload_rejects_episode_final_filename(mock_service, fake_creds_dict, tmp_path, bad_name):
    """Legacy 'episode NN FINAL.mp4' drafts must never reach YouTube again."""
    up, yu = _make_uploader(mock_service, fake_creds_dict)
    bad_path = tmp_path / bad_name
    bad_path.write_bytes(b"\x00\x00\x00\x20ftypiso5")

    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        with pytest.raises(ValueError) as exc:
            up.upload_video(bad_path, title="t", description="d", tags=[])

    # The error message should cite both the offending filename and the
    # expected pattern so operators can diagnose at a glance.
    assert bad_name in str(exc.value)
    assert "lf_" in str(exc.value)
    # And insert must never have been called.
    mock_service.videos.return_value.insert.assert_not_called()


@pytest.mark.parametrize(
    "bad_name",
    [
        "master.mp4",
        "drive_master.mp4",
        "lf_1.mp4",  # single-digit — canonical is zero-padded
        "lf_001.mp4",  # three-digit — too many
        "lf_01.mov",  # wrong extension
        "LF_01.mp4",  # uppercase
        "day_2026_02_20_master.mp4",
        "preview.mp4",
        "sl_1.mp4",  # single-digit — canonical superlong is zero-padded too
        "sl_001.mp4",  # three-digit — too many
        "sl_01.mov",  # wrong extension
        "SL_01.mp4",  # uppercase
        "SHORT_01.mp4",  # uppercase
    ],
)
def test_upload_rejects_arbitrary_non_lf_filename(
    mock_service, fake_creds_dict, tmp_path, bad_name
):
    """Anything that is not ``lf_NN.mp4``/``sl_NN.mp4``/``short_NN.mp4`` is blocked."""
    up, yu = _make_uploader(mock_service, fake_creds_dict)
    bad_path = tmp_path / bad_name
    bad_path.write_bytes(b"\x00\x00\x00\x20ftypiso5")

    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        with pytest.raises(ValueError):
            up.upload_video(bad_path, title="t", description="d", tags=[])

    mock_service.videos.return_value.insert.assert_not_called()


@pytest.mark.parametrize("good_name", ["lf_01.mp4", "lf_07.mp4", "lf_42.mp4"])
def test_upload_accepts_lf_filename(mock_service, fake_creds_dict, tmp_path, good_name):
    """Canonical ``lf_NN.mp4`` names pass the guard and upload normally."""
    up, yu = _make_uploader(mock_service, fake_creds_dict)
    good_path = tmp_path / good_name
    good_path.write_bytes(b"\x00\x00\x00\x20ftypiso5")

    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        vid = up.upload_video(good_path, title="t", description="d", tags=[])

    assert vid == "FAKEVIDEOID123"
    mock_service.videos.return_value.insert.assert_called_once()


@pytest.mark.parametrize("good_name", ["short_01.mp4", "short_02.mp4", "short_14.mp4"])
def test_upload_accepts_short_filename(mock_service, fake_creds_dict, tmp_path, good_name):
    """Canonical ``short_NN.mp4`` passes the guard."""
    up, yu = _make_uploader(mock_service, fake_creds_dict)
    good_path = tmp_path / good_name
    good_path.write_bytes(b"\x00\x00\x00\x20ftypiso5")

    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        vid = up.upload_video(good_path, title="t", description="d", tags=[])

    assert vid == "FAKEVIDEOID123"
    mock_service.videos.return_value.insert.assert_called_once()


@pytest.mark.parametrize("good_name", ["sl_01.mp4", "sl_02.mp4", "sl_03.mp4"])
def test_upload_accepts_superlong_filename(mock_service, fake_creds_dict, tmp_path, good_name):
    """Canonical ``sl_NN.mp4`` superlong-part names pass the guard (load-bearing:
    superlong parts must reach YouTube through the same upload path as an LF)."""
    up, yu = _make_uploader(mock_service, fake_creds_dict)
    good_path = tmp_path / good_name
    good_path.write_bytes(b"\x00\x00\x00\x20ftypiso5")

    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        vid = up.upload_video(good_path, title="t", description="d", tags=[])

    assert vid == "FAKEVIDEOID123"
    mock_service.videos.return_value.insert.assert_called_once()


def test_upload_allows_custom_pattern_via_config(mock_service, fake_creds_dict, tmp_path):
    """Overriding ``YOUTUBE_UPLOAD_FILENAME_PATTERN`` via config must work.

    A permissive ``.*`` pattern should accept a filename that the default
    strict guard would reject. Also confirms the pattern is read lazily
    (at call time, not import time) so dev overrides take effect per run.
    """
    from gomoveshift.config import Config, DriveConfig

    up, yu = _make_uploader(mock_service, fake_creds_dict)
    # "drive_master.mp4" normally rejected — but a ".*" override accepts all.
    relaxed_path = tmp_path / "drive_master.mp4"
    relaxed_path.write_bytes(b"\x00\x00\x00\x20ftypiso5")

    relaxed_cfg = Config(drive=DriveConfig(youtube_upload_filename_pattern=".*"))

    with (
        patch.object(yu, "MediaFileUpload", return_value=MagicMock()),
        patch.object(yu, "get_config", return_value=relaxed_cfg),
    ):
        vid = up.upload_video(relaxed_path, title="t", description="d", tags=[])

    assert vid == "FAKEVIDEOID123"
    mock_service.videos.return_value.insert.assert_called_once()


def test_default_pattern_is_strict():
    """Regression pin: default must be ``^(?:lf|sl|short)_\\d{2}\\.mp4$``.

    Loosening the default further is a breaking quality change and must be
    an explicit, reviewed decision. This test prevents an accidental flip.
    The ``sl_``/``short_`` halves were added deliberately (for the superlong-upload
    slot and the Shorts publishing adapter) so both can reach
    YouTube through the same guard.
    """
    from gomoveshift.config import DriveConfig

    cfg = DriveConfig()
    assert cfg.youtube_upload_filename_pattern == r"^(?:lf|sl|short)_\d{2}\.mp4$", (
        "Default YOUTUBE_UPLOAD_FILENAME_PATTERN changed — review the "
        "rationale for the upload filename guard before loosening the default."
    )


def test_upload_video_rejects_over_duration_backstop(
    mock_service, fake_creds_dict, fake_video_file
):
    """Backstop: duration >= 12h refuses the upload (unreachable given
    the superlong 4h-per-part cap)."""
    from gomoveshift.pipeline.validation.ffprobe_client import FFprobeClient

    up, yu = _make_uploader(mock_service, fake_creds_dict)
    with (
        patch.object(yu, "MediaFileUpload", return_value=MagicMock()),
        patch.object(FFprobeClient, "probe_duration", return_value=43200.0),
        pytest.raises(ValueError, match="unreachable"),
    ):
        up.upload_video(fake_video_file, title="t", description="d", tags=[])
    mock_service.videos.return_value.insert.assert_not_called()


def test_upload_video_rejects_over_size_backstop(mock_service, fake_creds_dict, fake_video_file):
    """Backstop: file size >= 256 GB refuses the upload."""
    import os

    up, yu = _make_uploader(mock_service, fake_creds_dict)
    os.truncate(fake_video_file, 256 * 1024 * 1024 * 1024)  # sparse — no real disk usage
    with (
        patch.object(yu, "MediaFileUpload", return_value=MagicMock()),
        pytest.raises(ValueError, match="unreachable"),
    ):
        up.upload_video(fake_video_file, title="t", description="d", tags=[])
    mock_service.videos.return_value.insert.assert_not_called()


# ----------------------------------------------------------------------
# YouTube quotaExceeded disambiguation.
#
# Context: the YouTube Data API returns quota issues as HTTP 403 with an
# ``error_details`` entry whose ``reason`` is ``quotaExceeded`` /
# ``userRateLimitExceeded`` / ``dailyLimitExceeded``. The original uploader
# treated any 403 as PERMANENT (never retry, never recoverable), which
# would abort a mid-Phase-3 batch with a generic "HTTP 403 — not retrying"
# and no operator guidance. The fix: parse ``HttpError.error_details`` and
# raise ``YouTubeQuotaError`` on quota reasons so Phase D can handle them
# distinctly from permanent auth failures (which remain unrecoverable).
#
# See the docstring on :class:`YouTubeQuotaError`.
# ----------------------------------------------------------------------


def _make_http_error(status_code: int, reason: str | None = None, message: str = "forbidden"):
    """Build an HttpError whose ``error_details`` carries ``reason``.

    Mirrors the real structure emitted by googleapiclient on 4xx: a JSON
    envelope at ``content`` with an ``error.errors[].reason`` entry that
    HttpError exposes as ``self.error_details``.
    """
    import json

    from googleapiclient.errors import HttpError

    class _Resp:
        def __init__(self, status):
            self.status = status
            self.reason = "Forbidden" if status == 403 else "Error"

    errors = [{"message": reason or message, "domain": "youtube.quota", "reason": reason}]
    body = json.dumps(
        {"error": {"code": status_code, "message": message, "errors": errors}}
    ).encode("utf-8")
    return HttpError(_Resp(status_code), body, uri="fake://")


@pytest.mark.parametrize(
    "reason",
    ["quotaExceeded", "userRateLimitExceeded", "dailyLimitExceeded"],
)
def test_403_quota_raises_youtube_quota_error(
    reason, mock_service, fake_creds_dict, fake_video_file
):
    """403 + quota-class ``reason`` must raise ``YouTubeQuotaError``.

    This disambiguates a daily-quota / rate-limit failure from a permanent
    auth failure: the former is recoverable after backoff (quotaExceeded
    resets at midnight Pacific Time; rate-limit resets in seconds), the latter
    requires operator credential rotation. Phase D catches the two
    categories separately.
    """
    from gomoveshift.pipeline.drive_phases import youtube_uploader as yu
    from gomoveshift.pipeline.exceptions import YouTubeQuotaError

    insert_request = MagicMock()
    insert_request.next_chunk.side_effect = _make_http_error(403, reason=reason)
    mock_service.videos.return_value.insert.return_value = insert_request

    with (
        patch.object(yu, "load_youtube_credentials", return_value=fake_creds_dict),
        patch.object(
            yu, "_build_credentials_obj", return_value=MagicMock(valid=True, expired=False)
        ),
        patch.object(yu, "build", return_value=mock_service),
        patch.object(yu, "MediaFileUpload", return_value=MagicMock()),
        patch.object(yu.time, "sleep"),
    ):
        up = yu.YouTubeUploader(credentials_dir=Path("/tmp/fake-creds"))
        with pytest.raises(YouTubeQuotaError) as exc:
            up.upload_video(fake_video_file, title="t", description="d", tags=[])

    # Operator-facing hint must appear in the message — "midnight Pacific Time"
    # is the canonical reset anchor for the YouTube Data API daily quota.
    msg = str(exc.value)
    assert "quota" in msg.lower() or "rate limit" in msg.lower(), (
        f"YouTubeQuotaError message must cite quota/rate-limit cause; got {msg!r}"
    )
    assert "midnight Pacific Time" in msg or "retry" in msg.lower(), (
        f"YouTubeQuotaError message must include recovery guidance; got {msg!r}"
    )
    # The quota reason itself must be recoverable on the exception for
    # programmatic routing (Phase D banner, telemetry).
    assert exc.value.reason == reason, f"reason attr must be {reason!r}; got {exc.value.reason!r}"
    # No retry was attempted — quota is surfaced immediately.
    assert insert_request.next_chunk.call_count == 1


def test_403_auth_failure_still_raises_http_error(mock_service, fake_creds_dict, fake_video_file):
    """Regression pin: a 403 that is NOT quota-class must still fall through
    to the original PERMANENT path — raises ``HttpError`` verbatim, no retry.

    Keeps the existing permanent-403 semantics intact for true auth failures
    (e.g. ``forbidden`` / ``insufficientPermissions``). Only quota-class
    reasons are diverted into ``YouTubeQuotaError``.
    """
    from googleapiclient.errors import HttpError

    from gomoveshift.pipeline.drive_phases import youtube_uploader as yu
    from gomoveshift.pipeline.exceptions import YouTubeQuotaError

    insert_request = MagicMock()
    # 403 with reason=forbidden — permanent auth failure, NOT quota.
    insert_request.next_chunk.side_effect = _make_http_error(403, reason="forbidden")
    mock_service.videos.return_value.insert.return_value = insert_request

    with (
        patch.object(yu, "load_youtube_credentials", return_value=fake_creds_dict),
        patch.object(
            yu, "_build_credentials_obj", return_value=MagicMock(valid=True, expired=False)
        ),
        patch.object(yu, "build", return_value=mock_service),
        patch.object(yu, "MediaFileUpload", return_value=MagicMock()),
        patch.object(yu.time, "sleep"),
    ):
        up = yu.YouTubeUploader(credentials_dir=Path("/tmp/fake-creds"))
        with pytest.raises(HttpError) as exc:
            up.upload_video(fake_video_file, title="t", description="d", tags=[])

    # Must NOT be diverted into YouTubeQuotaError — this is true auth failure.
    assert not isinstance(exc.value, YouTubeQuotaError)
    # No retry attempted — 403 remains a permanent category.
    assert insert_request.next_chunk.call_count == 1


# ----------------------------------------------------------------------
# HTTP 401 — refresh OAuth token and retry once.
#
# Context: 401 during a long-running upload often means the OAuth access
# token expired mid-flight (refresh TTL < upload duration). Previous code
# treated 401 as PERMANENT, dropping the resumable session and forcing a
# full re-upload. The fix attempts ONE token refresh via the existing
# google.oauth2 Credentials.refresh() path, retries the chunk once, and
# only fails permanent if the retry also returns 401 (genuine revocation)
# OR the refresh itself raises (refresh-side failure — logged loudly so
# the operator can distinguish from a revoked-token case).
#
# A 401 must no longer be treated as a permanent failure.
# ----------------------------------------------------------------------


def test_401_refresh_then_retry_succeeds(mock_service, fake_creds_dict, fake_video_file):
    """401 → refresh succeeds → retry chunk → 200 OK → upload completes.

    Happy path for the refresh-and-retry flow. Asserts:
    - Credentials.refresh() was called exactly once.
    - next_chunk was called exactly twice (failed call + retried call).
    - The upload completes and returns the video id from the retried call.
    """
    from gomoveshift.pipeline.drive_phases import youtube_uploader as yu

    insert_request = MagicMock()
    # First call: 401. Second call (after refresh): completes successfully.
    insert_request.next_chunk.side_effect = [
        _make_http_error(401, reason="authError", message="Invalid Credentials"),
        (None, {"id": "FAKEVIDEOID123"}),
    ]
    mock_service.videos.return_value.insert.return_value = insert_request

    fake_creds_obj = MagicMock(valid=True, expired=False)

    with (
        patch.object(yu, "load_youtube_credentials", return_value=fake_creds_dict),
        patch.object(yu, "_build_credentials_obj", return_value=fake_creds_obj),
        patch.object(yu, "build", return_value=mock_service),
        patch.object(yu, "MediaFileUpload", return_value=MagicMock()),
        patch.object(yu.time, "sleep"),
    ):
        up = yu.YouTubeUploader(credentials_dir=Path("/tmp/fake-creds"))
        vid = up.upload_video(fake_video_file, title="t", description="d", tags=[])

    assert vid == "FAKEVIDEOID123"
    # refresh was attempted exactly once
    assert fake_creds_obj.refresh.call_count == 1
    # next_chunk: one failed (401), one succeeded
    assert insert_request.next_chunk.call_count == 2


def test_401_refresh_succeeds_then_second_401_fails_permanent(
    mock_service, fake_creds_dict, fake_video_file
):
    """401 → refresh succeeds → retry → second 401 → fail-permanent.

    Genuine revoked-token case: refresh worked but the refreshed token is
    still rejected. Must NOT loop indefinitely — exactly one refresh + one
    retry, then raise the HttpError.
    """
    from googleapiclient.errors import HttpError

    from gomoveshift.pipeline.drive_phases import youtube_uploader as yu

    insert_request = MagicMock()
    insert_request.next_chunk.side_effect = [
        _make_http_error(401, reason="authError", message="Invalid Credentials"),
        _make_http_error(401, reason="authError", message="Invalid Credentials"),
        # Sentinel — if the loop ever calls next_chunk a third time the
        # test fails with a clear "no more side effects" StopIteration.
    ]
    mock_service.videos.return_value.insert.return_value = insert_request

    fake_creds_obj = MagicMock(valid=True, expired=False)

    with (
        patch.object(yu, "load_youtube_credentials", return_value=fake_creds_dict),
        patch.object(yu, "_build_credentials_obj", return_value=fake_creds_obj),
        patch.object(yu, "build", return_value=mock_service),
        patch.object(yu, "MediaFileUpload", return_value=MagicMock()),
        patch.object(yu.time, "sleep"),
    ):
        up = yu.YouTubeUploader(credentials_dir=Path("/tmp/fake-creds"))
        with pytest.raises(HttpError):
            up.upload_video(fake_video_file, title="t", description="d", tags=[])

    # Exactly ONE refresh attempt — never loop refresh on persistent 401.
    assert fake_creds_obj.refresh.call_count == 1
    # Exactly TWO next_chunk calls — original + one retry, then permanent fail.
    assert insert_request.next_chunk.call_count == 2


def test_401_refresh_itself_fails_permanent_with_loud_log(
    mock_service, fake_creds_dict, fake_video_file, caplog
):
    """401 → refresh raises → fail-permanent with WARNING/ERROR log.

    Refresh-side failure (e.g. network blip on the token endpoint, dead
    refresh_token) must be distinguishable from a revoked-token failure
    in the logs. Must NOT retry the chunk — refresh failure is itself
    fatal for this upload attempt.
    """
    import logging

    from googleapiclient.errors import HttpError

    from gomoveshift.pipeline.drive_phases import youtube_uploader as yu

    insert_request = MagicMock()
    insert_request.next_chunk.side_effect = [
        _make_http_error(401, reason="authError", message="Invalid Credentials"),
        # Sentinel — chunk must NOT be re-attempted after refresh fails.
    ]
    mock_service.videos.return_value.insert.return_value = insert_request

    # Refresh itself blows up — emulates a token-endpoint failure or
    # genuinely dead refresh_token.
    fake_creds_obj = MagicMock(valid=True, expired=False)
    fake_creds_obj.refresh.side_effect = RuntimeError("token endpoint unreachable")

    with (
        patch.object(yu, "load_youtube_credentials", return_value=fake_creds_dict),
        patch.object(yu, "_build_credentials_obj", return_value=fake_creds_obj),
        patch.object(yu, "build", return_value=mock_service),
        patch.object(yu, "MediaFileUpload", return_value=MagicMock()),
        patch.object(yu.time, "sleep"),
        caplog.at_level(logging.WARNING, logger=yu.logger.name),
    ):
        up = yu.YouTubeUploader(credentials_dir=Path("/tmp/fake-creds"))
        with pytest.raises(HttpError):
            up.upload_video(fake_video_file, title="t", description="d", tags=[])

    # Refresh was attempted exactly once
    assert fake_creds_obj.refresh.call_count == 1
    # Chunk must NOT have been retried after the refresh-side failure.
    assert insert_request.next_chunk.call_count == 1
    # A loud log must distinguish refresh-side failure from revocation.
    refresh_failure_logs = [
        rec
        for rec in caplog.records
        if rec.levelno >= logging.WARNING
        and "refresh" in rec.getMessage().lower()
        and "fail" in rec.getMessage().lower()
    ]
    assert refresh_failure_logs, (
        "Refresh-side failure must emit a WARNING/ERROR log mentioning "
        "'refresh' and 'fail' so operators can distinguish it from a "
        "genuine revoked-token (post-refresh 401) failure. "
        f"Got log records: {[r.getMessage() for r in caplog.records]}"
    )


# Partial upload failure resume
# ---------------------------------------------------------------------------
# Phase D iterates upload_if_needed across drive_plan.regular_lfs in order.
# If LF 2 of 5 fails after LF 1 succeeded, state["youtube_uploads"] retains
# LF 1's record. On resume the run must:
#   - skip LF 1 (no new videos.insert call for that LF)
#   - re-attempt LFs 2-5 (LF 2 may now succeed, LFs 3-5 are first attempts)
# This contract is the recovery story for YouTubeQuotaError + transient
# HTTP errors. test_phase_d covers the orchestrator-level abort + state
# checkpoint; this test covers the per-LF resume primitive directly.


def _make_lf_path(tmp_path: Path, idx: int) -> Path:
    p = tmp_path / f"lf_{idx:02d}.mp4"
    p.write_bytes(b"\x00\x00\x00\x20ftypiso5")
    return p


def test_partial_upload_failure_resume_does_not_reupload_completed_lfs(
    mock_service, fake_creds_dict, tmp_path
):
    """LF 2 of 5 fails mid-batch; resume must not re-upload LF 1
    and must succeed for LFs 2-5.

    Models the production batch shape: 5 LFs, LF 1 uploads cleanly, LF 2
    raises (transient or quota), the partial state is then persisted by
    Phase D. On resume, upload_if_needed is called for every LF in plan
    order — LF 1 returns the cached record without touching the API, LFs
    2-5 each go through one full upload.
    """
    from gomoveshift.pipeline.drive_phases import youtube_uploader as yu

    lf_paths = [_make_lf_path(tmp_path, i) for i in range(1, 6)]

    # ── First run: only LF 1 succeeds; LF 2 raises ────────────────────
    # Use a fresh service mock so we can count insert calls per phase.
    first_service = MagicMock()
    first_insert = MagicMock()
    first_insert.next_chunk.side_effect = [(None, {"id": "VID1"})]
    first_service.videos.return_value.insert.return_value = first_insert
    first_service.thumbnails.return_value.set.return_value.execute.return_value = {}

    state = make_state()

    with (
        patch.object(yu, "load_youtube_credentials", return_value=fake_creds_dict),
        patch.object(
            yu, "_build_credentials_obj", return_value=MagicMock(valid=True, expired=False)
        ),
        patch.object(yu, "build", return_value=first_service),
        patch.object(yu, "MediaFileUpload", return_value=MagicMock()),
    ):
        up1 = yu.YouTubeUploader(credentials_dir=Path("/tmp/fake-creds"))

        # LF 1 succeeds
        up1.upload_if_needed(
            state=state,
            lf_idx=1,
            video_path=lf_paths[0],
            title="LF 1",
            metadata=decided({}, "LF 1"),
            description="d",
            tags=[],
        )
        assert state["youtube_uploads"][1]["video_id"] == "VID1"

        # LF 2 raises mid-upload — simulate the transient failure that
        # cascades into a resume scenario. The error propagates out;
        # Phase D's caller persists state and re-raises.
        crash_insert = MagicMock()
        crash_insert.next_chunk.side_effect = OSError("network blip on LF 2")
        first_service.videos.return_value.insert.return_value = crash_insert
        # Disable inner retry sleep so the test runs instantly.
        with patch.object(yu.time, "sleep"):
            with pytest.raises(OSError):
                up1.upload_if_needed(
                    state=state,
                    lf_idx=2,
                    video_path=lf_paths[1],
                    title="LF 2",
                    metadata=decided({}, "LF 2"),
                    description="d",
                    tags=[],
                )

    # State must reflect: LF 1 recorded, LF 2 absent.
    assert 1 in state["youtube_uploads"]
    assert state["youtube_uploads"][1]["video_id"] == "VID1"
    assert 2 not in state["youtube_uploads"]

    # First-run service saw exactly two insert() calls (LF 1 + the LF 2 attempt).
    assert first_service.videos.return_value.insert.call_count == 2

    # ── Resume run: fresh service, fresh uploader, same state ───────────
    resume_service = MagicMock()

    # Each insert() call returns a unique completed response.
    resume_responses = iter(
        [
            (None, {"id": "VID2"}),
            (None, {"id": "VID3"}),
            (None, {"id": "VID4"}),
            (None, {"id": "VID5"}),
        ]
    )

    def _make_resume_insert(*args, **kwargs):
        req = MagicMock()
        req.next_chunk.side_effect = [next(resume_responses)]
        return req

    resume_service.videos.return_value.insert.side_effect = _make_resume_insert
    resume_service.thumbnails.return_value.set.return_value.execute.return_value = {}

    with (
        patch.object(yu, "load_youtube_credentials", return_value=fake_creds_dict),
        patch.object(
            yu, "_build_credentials_obj", return_value=MagicMock(valid=True, expired=False)
        ),
        patch.object(yu, "build", return_value=resume_service),
        patch.object(yu, "MediaFileUpload", return_value=MagicMock()),
    ):
        up2 = yu.YouTubeUploader(credentials_dir=Path("/tmp/fake-creds"))

        # Plan-order iteration: resume run calls upload_if_needed for ALL
        # LFs (Phase D does not pre-filter). LF 1 must short-circuit on the
        # state cache; LFs 2-5 each upload once.
        for idx in range(1, 6):
            up2.upload_if_needed(
                state=state,
                lf_idx=idx,
                video_path=lf_paths[idx - 1],
                title=f"LF {idx}",
                metadata=decided({}, f"LF {idx}"),
                description="d",
                tags=[],
            )

    # LF 1 must NOT have triggered a fresh insert call on the resume service.
    assert resume_service.videos.return_value.insert.call_count == 4, (
        "Resume must call videos.insert exactly once per non-cached LF (2,3,4,5). "
        f"Got {resume_service.videos.return_value.insert.call_count}."
    )

    # All five LFs end up recorded; LF 1 keeps its original video_id.
    assert state["youtube_uploads"][1]["video_id"] == "VID1"
    assert state["youtube_uploads"][2]["video_id"] == "VID2"
    assert state["youtube_uploads"][3]["video_id"] == "VID3"
    assert state["youtube_uploads"][4]["video_id"] == "VID4"
    assert state["youtube_uploads"][5]["video_id"] == "VID5"


def test_partial_upload_failure_resume_handles_orphan_playlist_recovery(
    mock_service, fake_creds_dict, tmp_path
):
    """An LF whose videos.insert
    succeeded but whose playlistItems.insert never persisted a
    playlist_id (the orphan-on-YouTube failure mode) must, on resume,
    skip the video re-upload AND re-run only the playlist assignment.

    This is the construction-correct guarantee: state
    with video_id present + playlist_id missing routes through the
    playlist-only recovery branch.
    """
    from gomoveshift.pipeline.drive_phases import youtube_uploader as yu

    lf_path = _make_lf_path(tmp_path, 7)

    state = make_state(
        youtube_uploads={
            7: {
                "video_id": "ORPHAN7",
                "url": "https://youtu.be/ORPHAN7",
                "publish_at": None,
                "title": "orphan",
                "uploaded_at": "2026-05-01T00:00:00+00:00",
                # NOTE: no playlist_id — the orphan signature.
            }
        }
    )

    # Construct minimal metadata + config that triggers the playlist branch.
    metadata = {"region": "ireland", "location": "Cork, Ireland"}
    cfg = MagicMock()
    cfg.drive.youtube_playlists.rules = []
    cfg.drive.youtube_playlists.default = "All Drives"

    with (
        patch.object(yu, "load_youtube_credentials", return_value=fake_creds_dict),
        patch.object(
            yu, "_build_credentials_obj", return_value=MagicMock(valid=True, expired=False)
        ),
        patch.object(yu, "build", return_value=mock_service),
        patch.object(yu, "MediaFileUpload", return_value=MagicMock()),
    ):
        up = yu.YouTubeUploader(credentials_dir=Path("/tmp/fake-creds"))
        # Stub the playlist assignment to a known id; we only care that
        # the recovery path was taken, not the real Google API surface.
        with patch.object(
            up,
            "assign_video_to_resolved_playlist",
            return_value="PL_RECOVERED",
        ) as mock_assign:
            result = up.upload_if_needed(
                state=state,
                lf_idx=7,
                video_path=lf_path,
                title="orphan retry",
                description="d",
                tags=[],
                metadata=metadata,
                config=cfg,
            )

    # videos.insert must NOT have fired — the orphan recovery only re-runs
    # the playlist assignment.
    mock_service.videos.return_value.insert.assert_not_called()
    mock_assign.assert_called_once_with("ORPHAN7", metadata=metadata, config=cfg)
    assert result["video_id"] == "ORPHAN7"
    assert result["playlist_id"] == "PL_RECOVERED"
    assert state["youtube_uploads"][7]["playlist_id"] == "PL_RECOVERED"
