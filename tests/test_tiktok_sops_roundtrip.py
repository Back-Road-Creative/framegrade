"""Exercise token persistence against SOPS, with offline throwaway age keys."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from freezegun import freeze_time

from gomoveshift.publishing import tiktok_auth

FIXED_NOW = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)


@pytest.mark.skipif(
    not shutil.which("sops") or not shutil.which("age-keygen"),
    reason="real SOPS/age roundtrip requires their local binaries",
)
@pytest.mark.parametrize("rule", [r"tiktok_publishing\.json$", r"unrelated\.json$"])
def test_refresh_persists_with_existing_recipients_despite_temporary_filename(
    tmp_path, monkeypatch, rule
):
    """Explicit recipients must survive a discovered config that rejects the temp name."""
    key = tmp_path / "age-key.txt"
    subprocess.run(["age-keygen", "-o", str(key)], check=True, capture_output=True)
    recipient = subprocess.run(
        ["age-keygen", "-y", str(key)], check=True, capture_output=True, text=True
    ).stdout.strip()
    monkeypatch.setenv("SOPS_AGE_KEY_FILE", str(key))
    monkeypatch.delenv("SOPS_CONFIG", raising=False)
    monkeypatch.delenv("SOPS_AGE_RECIPIENTS", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".sops.yaml").write_text(
        f"creation_rules:\n  - path_regex: '{rule}'\n    age: {recipient}\n"
    )
    target = tmp_path / "tiktok_publishing.json.enc"
    target.write_text(json.dumps({"sops": {"age": [{"recipient": recipient}]}}))
    now = FIXED_NOW
    old = {
        "access_token": "fixture-old-access",
        "refresh_token": "fixture-old-refresh",
        "client_key": "fixture-client",  # pragma: allowlist secret
        "client_secret": "fixture-secret",  # pragma: allowlist secret
        "access_expires_at": (now - timedelta(days=1)).isoformat(),
        "refresh_expires_at": (now + timedelta(days=30)).isoformat(),
    }
    monkeypatch.setattr(tiktok_auth, "load_tiktok_credentials", lambda *a, **kw: old)
    monkeypatch.setattr(
        tiktok_auth.requests,
        "post",
        lambda *a, **kw: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "access_token": "fixture-new-access",
                "refresh_token": "fixture-rotated-refresh",
                "expires_in": 86400,
            },
        ),
    )

    with freeze_time(FIXED_NOW):
        updated = tiktok_auth.get_valid_tiktok_credentials(tmp_path / "credentials")

    decrypted = subprocess.run(
        ["sops", "-d", "--output-type", "json", str(target)],
        check=True,
        capture_output=True,
        text=True,
    )
    saved = json.loads(decrypted.stdout)
    if set(saved) == {"data"}:
        saved = json.loads(saved["data"])
    assert saved == updated
    assert saved["refresh_token"] == "fixture-rotated-refresh"  # noqa: S105 -- test fixture
    assert saved["access_token"] == "fixture-new-access"  # noqa: S105 -- test fixture
    assert "fixture-new-access" not in target.read_text()
    assert not list(tmp_path.glob("*.tmp"))
