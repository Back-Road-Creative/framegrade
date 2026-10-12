"""Tests for SOPS stderr sanitization.

SOPS stderr can carry age-key paths and ``.sops.yaml`` recipient rules — key
material that must never reach a raised exception message or a WARNING log
(raw tokens are never logged; the same applies to key material).

These tests use SYNTHETIC fake age paths only — no real secret is referenced.
"""

import json
import logging
from unittest.mock import MagicMock, patch

from gomoveshift.pipeline._yt_creds import _sanitize_sops_stderr, load_youtube_credentials

# Synthetic stderr: an age-key path + a recipient rule + a generic category
# line. None of these reference a real key — the path/recipient are fabricated.
_FAKE_AGE_PATH = "/tmp/fake/.config/sops/age/keys.txt"
_FAKE_RECIPIENT = "age1faketestrecipient0000000000000000000000000000000000000000"
_SAMPLE_STDERR = (
    "Failed to get the data key required to decrypt the SOPS file.\n"
    "age: no identity matched any of the recipients\n"
    f"loaded age identity from {_FAKE_AGE_PATH}\n"
    f"recipient {_FAKE_RECIPIENT} could not be matched\n"
    "config: /tmp/fake/.sops.yaml\n"
)


class TestSanitizeSopsStderr:
    def test_age_path_not_present_verbatim(self):
        out = _sanitize_sops_stderr(_SAMPLE_STDERR)
        assert _FAKE_AGE_PATH not in out
        assert _FAKE_RECIPIENT not in out
        assert ".sops.yaml" not in out

    def test_generic_category_survives(self):
        out = _sanitize_sops_stderr(_SAMPLE_STDERR)
        assert "Failed to get the data key" in out

    def test_empty_input(self):
        assert _sanitize_sops_stderr("") == ""

    def test_all_sensitive_lines_yields_placeholder(self):
        sensitive = f"loaded age key from {_FAKE_AGE_PATH}\nrecipient {_FAKE_RECIPIENT}\n"
        out = _sanitize_sops_stderr(sensitive)
        assert _FAKE_AGE_PATH not in out
        assert _FAKE_RECIPIENT not in out
        assert out  # non-empty placeholder rather than a misleading blank


class TestLoadCredsWarningSanitized:
    """The WARNING emitted on a failed SOPS attempt must not leak key material."""

    def test_warning_omits_age_path(self, tmp_path, caplog, monkeypatch):
        monkeypatch.setenv("GMS_SOPS_SUDO_USER", "keyowner")
        enc_file = tmp_path / "youtube_upload_token.json.enc"
        enc_file.write_text("encrypted-content")
        creds_dir = tmp_path / "credentials"
        creds_dir.mkdir()
        fake_creds = {"token": "abc"}

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            if cmd[0] == "sops":
                result.returncode = 1
                result.stderr = _SAMPLE_STDERR
                result.stdout = ""
            else:  # opt-in sudo fallback succeeds so we don't reach BrandCredentials
                result.returncode = 0
                result.stdout = json.dumps(fake_creds)
                result.stderr = ""
            return result

        with caplog.at_level(logging.WARNING):
            with patch("subprocess.run", side_effect=mock_run):
                creds = load_youtube_credentials(credentials_dir=creds_dir)

        assert creds == fake_creds
        joined = caplog.text
        assert _FAKE_AGE_PATH not in joined
        assert _FAKE_RECIPIENT not in joined
        assert ".sops.yaml" not in joined


class TestSopsEncryptErrorSanitized:
    """uploader.sops_encrypt_json must not embed raw stderr in the RuntimeError."""

    def test_runtime_error_omits_age_path(self, tmp_path):
        import pytest

        from gomoveshift.pipeline import uploader

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 1
            result.stderr = _SAMPLE_STDERR
            result.stdout = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            with pytest.raises(RuntimeError) as excinfo:
                uploader.sops_encrypt_json({"k": "v"}, tmp_path / "out.json.enc")

        msg = str(excinfo.value)
        assert _FAKE_AGE_PATH not in msg
        assert _FAKE_RECIPIENT not in msg
        assert ".sops.yaml" not in msg
        assert "sops encrypt failed" in msg
