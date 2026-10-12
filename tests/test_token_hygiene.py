"""Token-hygiene tests.

Two narrow OAuth-token write paths used to leak credential bytes:

1. ``LongFormMetadataGenerator._sync_youtube_analytics`` dropped a JSON
   token to a predictable ``/tmp/yt_creds_*`` path via ``tempfile.mkstemp``
   and removed it in a ``finally`` block — a SIGKILL between mkstemp and
   unlink left the credentials on disk.
2. ``PipelineUploader._get_credentials`` (legacy plaintext path) wrote
   refreshed tokens with plain ``open(..., "w")``, inheriting the process
   umask (typically ``0o022`` → ``0o644``, world-readable). OAuth refresh
   tokens grant indefinite YouTube/Drive API access; they must be ``0o600``.

Plus a deprecation-warning regression check: ``_get_credentials`` must
log a warning when ``self._legacy_mode`` is True, so operators know they
are on a path that should be retired.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import stat
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ──────────────────────────────────────────────────────────────────────
# Fix A — _sync_youtube_analytics leaves no /tmp/yt_creds_* file behind
# ──────────────────────────────────────────────────────────────────────


def _make_meta_gen(monkeypatch, tmp_path: Path):
    """Build a minimal LongFormMetadataGenerator that bypasses real init."""
    from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator

    gen = LongFormMetadataGenerator.__new__(LongFormMetadataGenerator)
    # _manager must be non-None or _sync_youtube_analytics short-circuits.
    gen._manager = MagicMock()
    gen._optimizer = None
    gen._ledger = None
    gen.storage_path = tmp_path
    gen.credentials_dir = tmp_path

    # Stub credential loading so we don't hit SOPS / Fernet stores.
    monkeypatch.setattr(
        gen,
        "_load_youtube_credentials",
        lambda: {
            "token": "fake-token",
            "refresh_token": "fake-refresh",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "fake.apps.googleusercontent.com",
            "client_secret": "fake-secret",  # pragma: allowlist secret
            "scopes": ["https://www.googleapis.com/auth/youtube.readonly"],
        },
    )
    return gen


def test_sync_analytics_does_not_call_mkstemp(monkeypatch, tmp_path):
    """The leak-prone tempfile.mkstemp must no longer be used.

    Patch ``tempfile.mkstemp`` to a sentinel that records calls; after a
    successful ``_sync_youtube_analytics`` run, call count must be zero.
    NamedTemporaryFile is the approved replacement.
    """
    import tempfile

    mkstemp_calls = []

    def _record_mkstemp(*args, **kwargs):
        mkstemp_calls.append((args, kwargs))
        # If anything still calls it, fall back to the real impl so the
        # rest of the run can proceed and surface a clearer failure.
        return tempfile.mkstemp.__wrapped__(*args, **kwargs)  # pragma: no cover

    # Wrap so the sentinel still works if .__wrapped__ is missing.
    real_mkstemp = tempfile.mkstemp

    def _sentinel(*args, **kwargs):
        mkstemp_calls.append((args, kwargs))
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(tempfile, "mkstemp", _sentinel)

    # No-op the actual sync target so we never reach Google APIs.
    monkeypatch.setattr(
        "seo_manager.analytics.youtube_sync.YouTubeAnalyticsSync.__init__",
        lambda self, credentials_path: None,
    )

    gen = _make_meta_gen(monkeypatch, tmp_path)
    monkeypatch.setattr(gen, "_fetch_channel_videos", lambda sync: {})

    gen._sync_youtube_analytics()

    assert mkstemp_calls == [], (
        f"_sync_youtube_analytics must not call tempfile.mkstemp; "
        f"got {len(mkstemp_calls)} call(s): {mkstemp_calls!r}"
    )


def test_sync_analytics_leaves_no_temp_token(monkeypatch, tmp_path):
    """After a sync run, no ``/tmp/yt_creds_*.json`` file may survive.

    Captures the set of yt_creds_*.json files in the system temp dir
    before and after the call. The set must be unchanged regardless of
    success path. Belt-and-braces against any future tempfile choice
    that doesn't auto-clean.
    """
    import tempfile

    tmp_root = Path(tempfile.gettempdir())
    pattern = str(tmp_root / "yt_creds_*.json")

    before = set(glob.glob(pattern))

    monkeypatch.setattr(
        "seo_manager.analytics.youtube_sync.YouTubeAnalyticsSync.__init__",
        lambda self, credentials_path: None,
    )

    gen = _make_meta_gen(monkeypatch, tmp_path)
    monkeypatch.setattr(gen, "_fetch_channel_videos", lambda sync: {})

    gen._sync_youtube_analytics()

    after = set(glob.glob(pattern))
    leaked = after - before
    assert not leaked, f"_sync_youtube_analytics leaked credential temp file(s): {leaked!r}"


# ──────────────────────────────────────────────────────────────────────
# Fix B — _get_credentials writes legacy token with mode 0o600
# ──────────────────────────────────────────────────────────────────────


def _stub_credentials(monkeypatch, creds) -> None:
    """Make ``Credentials(...)`` return ``creds`` for the rest of the test.

    ``_get_credentials`` imports the name from the module at call time, so
    the module attribute is the thing to replace. Patching ``__new__`` on
    the class instead breaks the real class for the rest of the process
    (see ``test_no_dunder_new_patching.py``).
    """
    from google.oauth2 import credentials as google_creds_mod

    monkeypatch.setattr(google_creds_mod, "Credentials", MagicMock(return_value=creds))


def _make_uploader_legacy(token_path: Path):
    """Construct a minimal PipelineUploader in legacy mode without
    running the real __init__ (which performs filesystem and Drive-API
    setup we don't need)."""
    from gomoveshift.pipeline.uploader import PipelineUploader

    inst = PipelineUploader.__new__(PipelineUploader)
    inst._creds = None
    inst._legacy_mode = True
    inst._credentials_path = token_path  # any-existing-file is fine
    inst._token_path = token_path
    inst._refresh_lock = threading.Lock()
    inst._client = None
    return inst


def _write_seed_token(path: Path) -> None:
    """Write a placeholder token JSON so _get_credentials can parse it."""
    path.write_text(
        json.dumps(
            {
                "token": "stale",
                "refresh_token": "fake-refresh",
                "token_uri": "https://oauth2.googleapis.com/token",
                "client_id": "fake.apps.googleusercontent.com",
                "client_secret": "fake-secret",  # pragma: allowlist secret
                "scopes": ["https://www.googleapis.com/auth/drive.file"],
            }
        )
    )


def test_uploader_legacy_token_file_mode_is_0o600(monkeypatch, tmp_path):
    """After ``_get_credentials`` writes a refreshed token, the file
    must be mode 0o600 — never world- or group-readable."""
    token_path = tmp_path / "drive_token.json"
    _write_seed_token(token_path)
    # Pre-create with a deliberately wide mode so we can detect the
    # atomic-replace path tightens it.
    os.chmod(token_path, 0o644)

    inst = _make_uploader_legacy(token_path)

    # Patch Credentials to a stub that reports as expired and accepts
    # refresh() without doing real network IO.
    refreshed_creds = MagicMock()
    refreshed_creds.expired = True
    refreshed_creds.valid = False
    refreshed_creds.token = "fresh-token"  # noqa: S105 — test stub, not a secret
    refreshed_creds.expiry = None

    def _refresh(_req):
        refreshed_creds.expired = False
        refreshed_creds.valid = True

    refreshed_creds.refresh.side_effect = _refresh

    _stub_credentials(monkeypatch, refreshed_creds)

    inst._get_credentials()

    mode = stat.S_IMODE(os.stat(token_path).st_mode)
    assert mode == 0o600, (
        f"Refreshed token written by _get_credentials must be 0o600; got 0o{mode:o}"
    )


def test_legacy_mode_logs_deprecation(monkeypatch, tmp_path, caplog):
    """When ``self._legacy_mode`` is True, ``_get_credentials`` must
    emit a warning that names the legacy path, so operators see they
    are on a path scheduled for removal."""
    token_path = tmp_path / "drive_token.json"
    _write_seed_token(token_path)
    inst = _make_uploader_legacy(token_path)

    # Stub Credentials so refresh path is a no-op (still valid → no rewrite).
    valid_creds = MagicMock()
    valid_creds.expired = False
    valid_creds.valid = True
    _stub_credentials(monkeypatch, valid_creds)

    with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.uploader"):
        inst._get_credentials()

    text = caplog.text.lower()
    assert "deprecat" in text, (
        f"Expected DEPRECATED warning for legacy token mode, got: {caplog.text!r}"
    )
    assert "legacy" in text or "plaintext" in text or "token_path" in text, (
        f"Expected the warning to identify the legacy plaintext path, got: {caplog.text!r}"
    )


def test_stubbing_credentials_leaves_the_real_class_constructible():
    """The stub must not outlive its test.

    Otherwise four tests in
    ``test_collector_analytics_shared_creds.py`` fail with
    ``object.__new__() takes exactly one argument`` because they shared an
    xdist worker with the two tests above.
    """
    from google.oauth2 import credentials as google_creds_mod

    with pytest.MonkeyPatch.context() as mp:
        _stub_credentials(mp, MagicMock())

    creds = google_creds_mod.Credentials(token=None, refresh_token=None)
    assert isinstance(creds, google_creds_mod.Credentials)
