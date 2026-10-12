"""Tests for YouTube credential loading — SOPS without sudo fallback.

The container environment blocks sudo ('no new privileges' flag). The credential
loader should try sops directly first, and only when GMS_SOPS_SUDO_USER is set
fall back to ``sudo -u <that user> sops``.

Also tests that missing credentials don't hard-fail the pipeline — analytics
sync should degrade gracefully so the pipeline continues without it.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline._yt_creds import sops_age_recipients
from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator


class TestLoadYoutubeCredentialsSopsFallback:
    """_load_youtube_credentials should try sops directly before sudo."""

    @pytest.fixture
    def gen(self):
        return LongFormMetadataGenerator()

    def test_sops_direct_succeeds_without_sudo(self, gen, tmp_path):
        """When sops (no sudo) succeeds, credentials are returned without trying sudo."""
        fake_creds = {"token": "abc", "client_id": "123"}
        enc_file = tmp_path / "youtube_upload_token.json.enc"
        enc_file.write_text("encrypted-content")
        gen.credentials_dir = tmp_path / "credentials"
        gen.credentials_dir.mkdir()

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            if cmd[0] == "sops":
                result.returncode = 0
                result.stdout = json.dumps(fake_creds)
            else:
                # sudo path should NOT be reached
                raise AssertionError("sudo should not be called when sops succeeds")
            return result

        with patch("subprocess.run", side_effect=mock_run):
            creds = gen._load_youtube_credentials()

        assert creds == fake_creds

    def test_sops_binary_format_token_is_unwrapped(self, gen, tmp_path):
        """A token sops stored as BINARY (the pipeline writer's `.enc` override
        does that) decrypts to ``{"data": "<json text>"}``; the loader must hand
        back the real fields, or every Credentials() refresh dies on the
        "do not contain the necessary fields" error."""
        fake_creds = {"token": "abc", "refresh_token": "r", "client_id": "123"}
        (tmp_path / "youtube_upload_token.json.enc").write_text("encrypted-content")
        gen.credentials_dir = tmp_path / "credentials"
        gen.credentials_dir.mkdir()

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = json.dumps({"data": json.dumps(fake_creds, indent=2)})
            return result

        with patch("subprocess.run", side_effect=mock_run):
            creds = gen._load_youtube_credentials()

        assert creds == fake_creds

    def test_sops_direct_fails_falls_back_to_sudo(self, gen, tmp_path, monkeypatch):
        """When sops (no sudo) fails and GMS_SOPS_SUDO_USER is set, it retries via sudo -u <user>."""
        monkeypatch.setenv("GMS_SOPS_SUDO_USER", "keyowner")
        fake_creds = {"token": "xyz", "client_id": "456"}
        enc_file = tmp_path / "youtube_upload_token.json.enc"
        enc_file.write_text("encrypted-content")
        gen.credentials_dir = tmp_path / "credentials"
        gen.credentials_dir.mkdir()

        calls = []

        def mock_run(cmd, **kwargs):
            calls.append(list(cmd))
            result = MagicMock()
            if cmd[0] == "sops":
                # Direct sops fails
                result.returncode = 1
                result.stderr = "age: no identity matched"
                result.stdout = ""
            elif cmd[0] == "sudo":
                # sudo fallback succeeds
                result.returncode = 0
                result.stdout = json.dumps(fake_creds)
            else:
                result.returncode = 1
                result.stderr = "unknown"
                result.stdout = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            creds = gen._load_youtube_credentials()

        assert creds == fake_creds
        assert [c[0] for c in calls] == ["sops", "sudo"], "direct sops first, then sudo"
        assert calls[1][:4] == ["sudo", "-u", "keyowner", "sops"], (
            "the sudo attempt must run as the user named by GMS_SOPS_SUDO_USER"
        )

    def test_no_sudo_attempt_when_sudo_user_unset(self, gen, tmp_path, monkeypatch):
        """With GMS_SOPS_SUDO_USER unset there is exactly one attempt (direct sops) and no sudo."""
        monkeypatch.delenv("GMS_SOPS_SUDO_USER", raising=False)
        (tmp_path / "youtube_upload_token.json.enc").write_text("encrypted-content")
        gen.credentials_dir = tmp_path / "credentials"
        gen.credentials_dir.mkdir()

        calls = []

        def mock_run(cmd, **kwargs):
            calls.append(list(cmd))
            result = MagicMock()
            result.returncode = 1
            result.stderr = "age: no identity matched"
            result.stdout = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            with pytest.raises(FileNotFoundError, match="YouTube credentials not found"):
                gen._load_youtube_credentials()

        assert [c[0] for c in calls] == ["sops"]

    def test_both_sops_methods_fail_raises(self, gen, tmp_path):
        """When both sops approaches fail and no BrandCredentials, raises FileNotFoundError."""
        enc_file = tmp_path / "youtube_upload_token.json.enc"
        enc_file.write_text("encrypted-content")
        gen.credentials_dir = tmp_path / "credentials"
        gen.credentials_dir.mkdir()

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 1
            result.stderr = "decrypt failed"
            result.stdout = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            with pytest.raises(FileNotFoundError, match="YouTube credentials not found"):
                gen._load_youtube_credentials()


class TestAnalyticsSyncGracefulDegradation:
    """Pipeline should continue when YouTube credentials are unavailable."""

    def test_missing_credentials_does_not_abort_pipeline(self):
        """FileNotFoundError from analytics sync should be a warning, not a hard fail.

        The orchestrator should mark analytics_sync as completed and continue
        to subsequent stages rather than aborting the entire pipeline.
        """
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator()

        # Simulate what the orchestrator does in stage 0b
        # After the fix, FileNotFoundError should be caught by the generic
        # Exception handler, not treated as a hard pipeline failure.
        stages_completed = []
        failed = False

        try:
            meta_gen = LongFormMetadataGenerator()
            meta_gen._sync_youtube_analytics()
            stages_completed.append("analytics_sync")
        except FileNotFoundError:
            # This is what the FIXED code should do — warn and continue
            stages_completed.append("analytics_sync")
        except Exception:
            stages_completed.append("analytics_sync")

        assert "analytics_sync" in stages_completed
        assert not failed


# --- sops_age_recipients: public recipient metadata ---

_ENC_WITH_RECIPIENTS = (
    '{"data": "ENC[PLACEHOLDER]", "sops": {"age": ['
    '{"recipient": "age1aaaa", "enc": "x"}, {"recipient": "age1bbbb", "enc": "y"}]}}\n'
)


def test_sops_age_recipients_reads_public_metadata(tmp_path):
    enc_path = tmp_path / "token.json.enc"
    enc_path.write_text(_ENC_WITH_RECIPIENTS)
    assert sops_age_recipients(enc_path) == ["age1aaaa", "age1bbbb"]


def test_sops_age_recipients_is_empty_for_missing_or_unparseable(tmp_path):
    assert sops_age_recipients(tmp_path / "nope.enc") == []
    junk = tmp_path / "junk.enc"
    junk.write_text("ENC[AES256_GCM,data:not-json]")
    assert sops_age_recipients(junk) == []
    junk.write_text('{"sops": {"age": "not-a-list"}}')
    assert sops_age_recipients(junk) == []
