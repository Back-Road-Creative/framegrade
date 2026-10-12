"""``load_youtube_credentials`` must return a stored ``scopes`` list
unchanged — the consent CLI writes force-ssl into it and
nothing downstream may silently narrow it back to upload-only."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from gomoveshift.pipeline import _yt_creds


def test_load_youtube_credentials_preserves_stored_scopes(tmp_path, monkeypatch):
    creds_dir = tmp_path / "credentials"
    enc_path = creds_dir.parent / "youtube_upload_token.json.enc"
    enc_path.parent.mkdir(parents=True, exist_ok=True)
    enc_path.write_text("encrypted", encoding="utf-8")

    stored = {
        "token": "t",  # noqa: S106
        "refresh_token": "r",  # noqa: S106
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": "cid",
        "client_secret": "cs",  # noqa: S106  # pragma: allowlist secret
        "scopes": [
            "https://www.googleapis.com/auth/youtube.upload",
            "https://www.googleapis.com/auth/youtube.force-ssl",
        ],
    }

    fake_result = MagicMock(returncode=0, stdout=json.dumps(stored), stderr="")
    monkeypatch.setattr(_yt_creds.subprocess, "run", lambda *a, **k: fake_result)

    result = _yt_creds.load_youtube_credentials(creds_dir)
    assert result["scopes"] == stored["scopes"]
