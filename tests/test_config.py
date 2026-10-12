"""Tests for gomoveshift.config — frozen encoding constants enforcement + DriveConfig."""

from gomoveshift.config import DriveConfig, OutputConfig
from gomoveshift.pipeline.encoding_constants import (
    LONGFORM_CRF,
    LONGFORM_PIX_FMT,
    LONGFORM_PRESET,
    LONGFORM_PROFILE,
)


class TestOutputConfigFreezeEncodingConstants:
    """model_post_init must override user-supplied quality fields to canonical values."""

    def test_crf_override_is_rejected(self):
        cfg = OutputConfig(crf=99)
        assert cfg.crf == LONGFORM_CRF, f"crf must be frozen to {LONGFORM_CRF}, got {cfg.crf}"

    def test_preset_override_is_rejected(self):
        cfg = OutputConfig(preset="ultrafast")
        assert cfg.preset == LONGFORM_PRESET, (
            f"preset must be frozen to {LONGFORM_PRESET!r}, got {cfg.preset!r}"
        )

    def test_profile_override_is_rejected(self):
        cfg = OutputConfig(profile="baseline")
        assert cfg.profile == LONGFORM_PROFILE, f"profile must be frozen to {LONGFORM_PROFILE!r}"

    def test_pix_fmt_override_is_rejected(self):
        cfg = OutputConfig(pix_fmt="yuv420p")
        assert cfg.pix_fmt == LONGFORM_PIX_FMT, f"pix_fmt must be frozen to {LONGFORM_PIX_FMT!r}"

    def test_non_quality_fields_not_overridden(self):
        """Fields not in encoding_constants (e.g. audio_bitrate) must remain as set."""
        cfg = OutputConfig(audio_bitrate="192k")
        assert cfg.audio_bitrate == "192k", "Non-frozen fields must not be overridden"


class TestDriveConfigNoReview:
    """DriveConfig.no_review field for review gate integration."""

    def test_no_review_defaults_false(self):
        cfg = DriveConfig()
        assert cfg.no_review is False

    def test_no_review_can_be_set_true(self):
        cfg = DriveConfig(no_review=True)
        assert cfg.no_review is True
