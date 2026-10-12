"""Tests for the subscribe CTA gated on skip_ost, the Drive retry count (5 attempts),
and the removed dead strategy config field."""

import time
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.config import DriveConfig
from gomoveshift.pipeline.single_pass import SinglePassProcessor
from gomoveshift.pipeline.uploader import _drive_execute


# ── subscribe CTA must respect skip_ost ──────────────────────


class TestBuildVideoFilterSkipOstNoCta:
    """_build_video_filter with skip_ost=True must not include subscribe overlay."""

    def _make_processor(self):
        """Create a SinglePassProcessor with minimal config for filter tests."""
        config = MagicMock()
        config.output.audio_target_lufs = -14
        config.output.audio_target_tp = -1
        config.output.audio_target_lra = 11
        return SinglePassProcessor(config)

    def test_skip_ost_true_with_valid_font_no_subscribe(self):
        """skip_ost=True + valid font_path must NOT produce subscribe overlay."""
        proc = self._make_processor()
        vf = proc._build_video_filter(
            color_filter="null",
            location="",
            font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            duration=60.0,
            fade_video=False,
            video_width=1920,
            skip_ost=True,
        )
        assert "Subscribe" not in vf
        assert "subscribe" not in vf.lower() or "subscribe" not in vf

    def test_skip_ost_false_with_valid_font_has_subscribe(self):
        """skip_ost=False + valid font_path must produce subscribe overlay."""
        proc = self._make_processor()
        vf = proc._build_video_filter(
            color_filter="null",
            location="",
            font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            duration=60.0,
            fade_video=False,
            video_width=1920,
            skip_ost=False,
        )
        assert "gomoveshift" in vf.lower()

    def test_skip_ost_true_no_location_overlay(self):
        """skip_ost=True must also suppress location overlay when location is empty."""
        proc = self._make_processor()
        vf = proc._build_video_filter(
            color_filter="null",
            location="",
            font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            duration=60.0,
            fade_video=False,
            video_width=1920,
            skip_ost=True,
        )
        # Should only have scale filter, no drawtext
        assert "drawtext" not in vf


# ── helper: _build_fade_alpha ─────────────────────────────────


class TestBuildFadeAlpha:
    """The extracted _build_fade_alpha helper must produce valid ffmpeg alpha expressions."""

    def test_returns_string(self):
        result = SinglePassProcessor._build_fade_alpha(1.0, 9.0, 1.0)
        assert isinstance(result, str)

    def test_contains_time_boundaries(self):
        result = SinglePassProcessor._build_fade_alpha(2.0, 10.0, 1.5)
        assert "2.0" in result
        assert "10.0" in result


# ── Drive retry must attempt 5 times ─────────────────────────


class TestDriveRetryFiveAttempts:
    """_drive_execute must retry up to 5 times on HTTP 429."""

    def test_retries_five_times_on_429(self):
        """5 attempts total (1 initial + 4 retries) before raising."""
        from googleapiclient.errors import HttpError

        mock_resp = MagicMock()
        mock_resp.status = 429

        exc = HttpError(mock_resp, b"rate limited")
        mock_request = MagicMock()
        mock_request.execute.side_effect = exc

        with patch("gomoveshift.pipeline.uploader.time") as mock_time:
            mock_time.sleep = MagicMock()
            with pytest.raises(RuntimeError, match="5 attempts"):
                _drive_execute(mock_request, max_attempts=5)

        assert mock_request.execute.call_count == 5

    def test_error_message_mentions_quota(self):
        """Final error message must mention spreading uploads or increasing quota."""
        from googleapiclient.errors import HttpError

        mock_resp = MagicMock()
        mock_resp.status = 429

        exc = HttpError(mock_resp, b"rate limited")
        mock_request = MagicMock()
        mock_request.execute.side_effect = exc

        with patch("gomoveshift.pipeline.uploader.time") as mock_time:
            mock_time.sleep = MagicMock()
            with pytest.raises(RuntimeError, match="spreading uploads"):
                _drive_execute(mock_request)

    def test_default_max_attempts_is_five(self):
        """Default max_attempts constant must be 5."""
        from gomoveshift.pipeline.uploader import _DRIVE_RATE_LIMIT_MAX_ATTEMPTS

        assert _DRIVE_RATE_LIMIT_MAX_ATTEMPTS == 5

    def test_non_429_error_not_retried(self):
        """Non-429 HttpErrors must propagate immediately without retry."""
        from googleapiclient.errors import HttpError

        mock_resp = MagicMock()
        mock_resp.status = 400

        exc = HttpError(mock_resp, b"bad request")
        mock_request = MagicMock()
        mock_request.execute.side_effect = exc

        with pytest.raises(HttpError):
            _drive_execute(mock_request)

        assert mock_request.execute.call_count == 1


# ── DriveConfig must not have strategy field ───────────────


class TestDriveConfigNoStrategyField:
    """The strategy field was removed from DriveConfig."""

    def test_no_strategy_field(self):
        """DriveConfig must not have a 'strategy' attribute."""
        cfg = DriveConfig()
        assert not hasattr(cfg, "strategy"), (
            "DriveConfig still has 'strategy' field — it should be removed"
        )

    def test_no_strategy_in_model_fields(self):
        """strategy must not appear in DriveConfig's pydantic model fields."""
        assert "strategy" not in DriveConfig.model_fields

    def test_no_per_day_or_day_first_fields(self):
        """per_day and day_first were removed — single pipeline now."""
        cfg = DriveConfig()
        assert not hasattr(cfg, "per_day")
        assert not hasattr(cfg, "day_first")
