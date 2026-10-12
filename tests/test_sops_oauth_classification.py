"""SOPS / OAuth refresh failure classification.

Before this classification existed:

  * ``google.auth.exceptions.RefreshError`` was not an ``HttpError`` so
    ``is_permanent_error`` returned False and ``_drive_execute`` retried
    the upload ``_DRIVE_RATE_LIMIT_MAX_ATTEMPTS`` (5) times — 30+
    seconds per file — before propagating with no actionable hint.
  * ``sops_decrypt_json`` raised a bare ``RuntimeError`` regardless of
    why SOPS failed, forcing operators to grep logs to figure out the
    recovery path.

Now:

  * ``RefreshError`` at any credential refresh site (``_get_credentials``
    or ``_refresh_credentials_if_needed``) is re-raised as
    ``DriveCredentialError`` with the SOPS token path and the recovery
    command in the message.
  * ``sops_decrypt_json`` recognises a known-recoverable failure pattern
    set (``no age identity``, ``DECRYPTION_FAILED``) and raises
    ``SopsDecryptError`` with the age-key path and a recovery hint naming
    ``GMS_AGE_KEY_FILE``. Unknown failures still raise generic
    ``RuntimeError`` so we never misclassify.
  * Both new exception types are recognised by ``is_permanent_error``
    so the retry loop fast-fails instead of burning the full budget.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.uploader import (
    DriveCredentialError,
    SopsDecryptError,
    is_permanent_error,
    sops_decrypt_json,
)


# ──────────────────────────────────────────────────────────────────────
# 1. is_permanent_error classifies the new types as permanent
# ──────────────────────────────────────────────────────────────────────


def test_is_permanent_error_recognises_drive_credential_error():
    exc = DriveCredentialError("token revoked")
    assert is_permanent_error(exc) is True


def test_is_permanent_error_recognises_sops_decrypt_error():
    exc = SopsDecryptError("age identity missing")
    assert is_permanent_error(exc) is True


def test_is_permanent_error_recognises_raw_refresh_error():
    """Raw ``RefreshError`` (un-wrapped) is also classified permanent
    so any future call site that lets it bubble doesn't burn retries."""
    from google.auth.exceptions import RefreshError

    exc = RefreshError("invalid_grant: Token has been expired or revoked.")
    assert is_permanent_error(exc) is True


def test_is_permanent_error_leaves_generic_exception_transient():
    """Unrelated RuntimeError stays non-permanent (retryable)."""
    exc = RuntimeError("connection reset by peer")
    assert is_permanent_error(exc) is False


# ──────────────────────────────────────────────────────────────────────
# 2. sops_decrypt_json classifies known-recoverable stderr patterns
# ──────────────────────────────────────────────────────────────────────


def _mock_sops_run(returncode: int, stderr: str):
    """Build a CompletedProcess stand-in for ``subprocess.run``."""
    mock = MagicMock(spec=subprocess.CompletedProcess)
    mock.returncode = returncode
    mock.stderr = stderr
    mock.stdout = ""
    return mock


@pytest.fixture
def existing_enc(tmp_path: Path) -> Path:
    enc = tmp_path / "drive_token.json.enc"
    enc.write_text("placeholder — content irrelevant, sops is mocked")
    return enc


def test_sops_decrypt_no_age_identity_raises_sops_decrypt_error(monkeypatch, existing_enc):
    fake = _mock_sops_run(1, "Failed to decrypt: no age identity matched any of the recipients")
    monkeypatch.setattr("gomoveshift.pipeline.uploader.subprocess.run", lambda *a, **kw: fake)

    with pytest.raises(SopsDecryptError) as exc:
        sops_decrypt_json(existing_enc)

    msg = str(exc.value)
    assert "no age identity" in msg
    assert "GMS_AGE_KEY_FILE" in msg
    assert "re-encrypt" in msg
    assert "Age key:" in msg


def test_sops_decrypt_decryption_failed_raises_sops_decrypt_error(monkeypatch, existing_enc):
    fake = _mock_sops_run(1, "DECRYPTION_FAILED_OR_BAD_RECORD_MAC")
    monkeypatch.setattr("gomoveshift.pipeline.uploader.subprocess.run", lambda *a, **kw: fake)

    with pytest.raises(SopsDecryptError) as exc:
        sops_decrypt_json(existing_enc)

    assert "DECRYPTION_FAILED" in str(exc.value)


def test_sops_decrypt_unknown_failure_keeps_generic_runtimeerror(monkeypatch, existing_enc):
    """Failures NOT matching the known-recoverable patterns must keep
    bubbling as the original RuntimeError so unfamiliar failures stay
    visible (we'd rather see a noisy error than miscategorise)."""
    fake = _mock_sops_run(1, "sops: binary not found")
    monkeypatch.setattr("gomoveshift.pipeline.uploader.subprocess.run", lambda *a, **kw: fake)

    with pytest.raises(RuntimeError) as exc:
        sops_decrypt_json(existing_enc)

    # Generic RuntimeError, NOT our subclass
    assert not isinstance(exc.value, SopsDecryptError)
    assert "sops: binary not found" in str(exc.value)


# ──────────────────────────────────────────────────────────────────────
# 3. _refresh_credentials_if_needed wraps RefreshError
# ──────────────────────────────────────────────────────────────────────


def _make_uploader_with_creds(creds_stub):
    """Build a minimal PipelineUploader instance with a creds stub.

    We bypass ``__init__`` because the real constructor performs filesystem
    + Drive-API setup we don't need for the refresh-path test.
    """
    from gomoveshift.pipeline.uploader import PipelineUploader

    inst = PipelineUploader.__new__(PipelineUploader)
    inst._creds = creds_stub
    import threading

    inst._refresh_lock = threading.Lock()
    inst._legacy_mode = False
    inst.token_enc = Path("/tmp/secrets/drive_token.json.enc")
    inst._token_path = Path("/dev/null")
    return inst


def test_refresh_credentials_wraps_refresh_error_as_drive_credential_error():
    from google.auth.exceptions import RefreshError

    creds = MagicMock()
    creds.expired = True
    creds.valid = False
    creds.refresh.side_effect = RefreshError("invalid_grant: token revoked")

    inst = _make_uploader_with_creds(creds)

    with pytest.raises(DriveCredentialError) as exc:
        inst._refresh_credentials_if_needed()

    msg = str(exc.value)
    assert "drive_token.json.enc" in msg
    assert "--init-oauth" in msg
    assert "GMS_AGE_KEY_FILE" in msg
    # __cause__ preserves the original for log forensics
    assert isinstance(exc.value.__cause__, RefreshError)


def test_refresh_credentials_noop_when_valid():
    creds = MagicMock()
    creds.expired = False
    creds.valid = True

    inst = _make_uploader_with_creds(creds)
    inst._refresh_credentials_if_needed()

    creds.refresh.assert_not_called()


def test_refresh_credentials_noop_when_creds_none():
    inst = _make_uploader_with_creds(None)
    inst._refresh_credentials_if_needed()  # no raise, no-op


# ──────────────────────────────────────────────────────────────────────
# 4. DriveCredentialError is a DriveUploadError (catch-broader callers
#    still see it as an upload failure)
# ──────────────────────────────────────────────────────────────────────


def test_drive_credential_error_is_drive_upload_error():
    """Existing call sites that catch ``DriveUploadError`` continue to
    catch this — the new subclass narrows the message, not the lineage."""
    from gomoveshift.pipeline.uploader import DriveUploadError

    exc = DriveCredentialError("token revoked")
    assert isinstance(exc, DriveUploadError)
