"""Tests for LongFormMetadataGenerator SEO init — PermissionError handling."""

import logging
from pathlib import Path
from unittest.mock import patch

from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator


class TestSEOPermissionError:
    """SEO Manager must catch PermissionError on .lock file without dumping a traceback."""

    def test_permission_error_sets_degraded(self, tmp_path):
        """PermissionError during SEOManager init sets _seo_degraded = True."""
        with patch(
            "seo_manager.SEOManager.__init__",
            side_effect=PermissionError("mock lock error"),
        ):
            gen = LongFormMetadataGenerator(
                seo_storage_path=tmp_path / "seo",
                credentials_dir=tmp_path / "creds",
            )
        assert gen._seo_degraded is True
        assert gen._manager is None
        assert gen._optimizer is None

    def test_permission_error_log_message(self, tmp_path, caplog):
        """Log must mention 'permission error' and 'lock file ownership'."""
        with (
            caplog.at_level(logging.ERROR),
            patch(
                "seo_manager.SEOManager.__init__",
                side_effect=PermissionError("mock lock error"),
            ),
        ):
            LongFormMetadataGenerator(
                seo_storage_path=tmp_path / "seo",
                credentials_dir=tmp_path / "creds",
            )

        log_text = caplog.text.lower()
        assert "permission error" in log_text
        assert "lock file ownership" in log_text

    def test_permission_error_no_traceback(self, tmp_path, caplog):
        """PermissionError must NOT trigger the generic handler's traceback dump."""
        with (
            caplog.at_level(logging.ERROR),
            patch(
                "seo_manager.SEOManager.__init__",
                side_effect=PermissionError("mock lock error"),
            ),
        ):
            LongFormMetadataGenerator(
                seo_storage_path=tmp_path / "seo",
                credentials_dir=tmp_path / "creds",
            )

        assert "Traceback" not in caplog.text, (
            "Generic exception handler fired — PermissionError was not caught specifically"
        )
