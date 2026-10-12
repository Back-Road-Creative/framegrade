"""Tests for PipelineUploader early credential detection."""

import logging


from gomoveshift.pipeline.uploader import PipelineUploader


class TestUploaderEarlyCredentialDetection:
    """PipelineUploader must detect missing credentials at __init__ time (warn, not raise)."""

    def test_no_raise_at_init_when_legacy_credentials_missing(self, tmp_path):
        """Construction must not raise even when the credentials file doesn't exist."""
        missing = tmp_path / "nonexistent_creds.json"
        # Should not raise — warning only
        PipelineUploader(credentials_path=missing)

    def test_warns_at_init_when_legacy_credentials_missing(self, tmp_path, caplog):
        """A warning must be logged when credentials file is absent at construction."""
        missing = tmp_path / "nonexistent_creds.json"
        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.uploader"):
            PipelineUploader(credentials_path=missing)
        low = caplog.text.lower()
        assert "credential" in low or "cred" in low or "not found" in low, (
            f"Expected a credentials warning at init, got: {caplog.text!r}"
        )

    def test_no_raise_at_init_when_sops_credentials_missing(self, tmp_path):
        """SOPS mode: construction must not raise when .enc file doesn't exist."""
        missing_enc = tmp_path / "drive_credentials.json.enc"
        missing_tok = tmp_path / "drive_token.json.enc"
        # Should not raise — warning only
        PipelineUploader(credentials_enc=missing_enc, token_enc=missing_tok)
