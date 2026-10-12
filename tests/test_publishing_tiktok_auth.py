"""TDD for TikTok credential loading + rotated-refresh-token persistence.
sops subprocess and ``requests.post`` are both patched — no real decrypt/network."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import requests
from freezegun import freeze_time

from gomoveshift.publishing import tiktok_auth

FIXED_NOW = datetime(2026, 8, 23, 12, 0, 0, tzinfo=timezone.utc)


def _write_enc(path, creds: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(creds))


def _base_creds(**overrides) -> dict:
    creds = {
        "client_key": "ck_test",  # pragma: allowlist secret
        "client_secret": "cs_test",  # pragma: allowlist secret
        "open_id": "open123",
        "scopes": ["video.publish"],
        "access_token": "old_access_token",
        "access_expires_at": (FIXED_NOW - timedelta(minutes=1)).isoformat(),
        "refresh_token": "old_refresh_token",
        "refresh_expires_at": (FIXED_NOW + timedelta(days=300)).isoformat(),
        "redirect_uri": "https://example.com/callback",
        "environment": "sandbox",
    }
    creds.update(overrides)
    return creds


def _fake_sops_run(cmd, **kwargs):
    if "-d" in cmd:
        return SimpleNamespace(returncode=0, stdout=open(cmd[-1]).read(), stderr="")
    if "-e" in cmd:
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    raise AssertionError(f"unexpected sops invocation: {cmd}")


def _patch_post(monkeypatch, payload=None, exc=None):
    def _fake_post(*a, **kw):
        if exc:
            raise exc
        return type("R", (), {"raise_for_status": lambda s: None, "json": lambda s: payload})()

    monkeypatch.setattr(tiktok_auth.requests, "post", _fake_post)


@pytest.fixture(autouse=True)
def _patch_env(monkeypatch):
    monkeypatch.setattr(tiktok_auth.subprocess, "run", _fake_sops_run)
    with freeze_time(FIXED_NOW):
        yield


def _enc_path(tmp_path):
    return tmp_path / tiktok_auth._TIKTOK_CREDENTIALS_FILENAME


def test_load_unwraps_sops_binary_data_field(tmp_path):
    """A credentials file stored in sops BINARY format decrypts, under this
    loader's forced ``--output-type json``, to ``{"data": "<the json text>"}``
    -- one key, no credentials. Every caller then dies on
    ``KeyError: 'access_expires_at'``.

    A file encrypted in binary format keeps every real key inside that one
    string, so the loader must unwrap it.
    """
    creds = _base_creds()
    enc_path = _enc_path(tmp_path)
    enc_path.parent.mkdir(parents=True, exist_ok=True)
    enc_path.write_text(json.dumps({"data": json.dumps(creds)}))

    loaded = tiktok_auth.load_tiktok_credentials(tmp_path / "credentials")

    assert loaded == creds, "binary-format credentials must load like JSON ones"


def test_load_leaves_a_real_data_key_alone(tmp_path):
    """Only the sops-binary shape is unwrapped: a genuine credentials payload
    that happens to carry a ``data`` field alongside others is untouched."""
    creds = _base_creds(data="not-a-wrapper")
    enc_path = _enc_path(tmp_path)
    enc_path.parent.mkdir(parents=True, exist_ok=True)
    enc_path.write_text(json.dumps(creds))

    assert tiktok_auth.load_tiktok_credentials(tmp_path / "credentials") == creds


@pytest.mark.parametrize(
    "response_extra,expected_refresh,crash",
    [
        (
            {"refresh_token": "rotated_refresh_token", "refresh_expires_in": 31536000},
            "rotated_refresh_token",
            False,
        ),
        ({}, "old_refresh_token", False),
        ({}, "old_refresh_token", True),
    ],
)
def test_refresh_persistence(tmp_path, monkeypatch, response_extra, expected_refresh, crash):
    enc_path = _enc_path(tmp_path)
    original = _base_creds()
    _write_enc(enc_path, original)
    _patch_post(
        monkeypatch, {"access_token": "new_access_token", "expires_in": 86400, **response_extra}
    )
    if crash:
        monkeypatch.setattr(
            tiktok_auth.os,
            "replace",
            lambda *a, **kw: (_ for _ in ()).throw(OSError("simulated crash")),
        )
        with pytest.raises(OSError):
            tiktok_auth.get_valid_tiktok_credentials(tmp_path / "credentials")
        assert json.loads(enc_path.read_text()) == original
        return
    tiktok_auth.get_valid_tiktok_credentials(tmp_path / "credentials")
    on_disk = json.loads(enc_path.read_text())
    assert on_disk["refresh_token"] == expected_refresh  # noqa: S105 — test fixture, not a secret
    assert on_disk["access_token"] == "new_access_token"  # noqa: S105


def test_expired_refresh_token_raises_actionable_error(tmp_path):
    _write_enc(
        _enc_path(tmp_path),
        _base_creds(refresh_expires_at=(FIXED_NOW - timedelta(days=1)).isoformat()),
    )
    with pytest.raises(tiktok_auth.TikTokAuthError) as exc_info:
        tiktok_auth.get_valid_tiktok_credentials(tmp_path / "credentials")
    assert "re-create the TikTok credential record" in str(exc_info.value)


def test_no_token_or_secret_leaks_on_refresh_failure(tmp_path, monkeypatch, caplog):
    _write_enc(_enc_path(tmp_path), _base_creds())
    _patch_post(
        monkeypatch, exc=requests.RequestException("boom secret_token=old_refresh_token leaked")
    )
    with caplog.at_level("DEBUG"), pytest.raises(tiktok_auth.TikTokAuthError) as exc_info:
        tiktok_auth.get_valid_tiktok_credentials(tmp_path / "credentials")
    assert "old_refresh_token" not in str(exc_info.value)
    assert "cs_test" not in str(exc_info.value)
    for record in caplog.records:
        assert "old_refresh_token" not in record.getMessage()
        assert "cs_test" not in record.getMessage()


def test_refresh_encrypt_uses_explicit_age_recipients_not_temp_filename(tmp_path, monkeypatch):
    """The encrypt call must not depend on the mkstemp-generated filename
    matching a .sops.yaml creation rule (it never does — random suffix)."""
    enc_path = _enc_path(tmp_path)
    _write_enc(enc_path, _base_creds())
    (tmp_path / ".sops.yaml").write_text(
        "creation_rules:\n"
        "  - path_regex: (youtube_(credentials|upload_token)|meta_publishing|"
        "tiktok_publishing)\\.json$\n"
        "    age: age1recipientone,age1recipienttwo\n"
    )
    _patch_post(monkeypatch, {"access_token": "new_access_token", "expires_in": 86400})

    seen_encrypt_cmds = []
    real_fake = _fake_sops_run

    def _capturing_run(cmd, **kwargs):
        if "-e" in cmd:
            seen_encrypt_cmds.append(cmd)
        return real_fake(cmd, **kwargs)

    monkeypatch.setattr(tiktok_auth.subprocess, "run", _capturing_run)

    tiktok_auth.get_valid_tiktok_credentials(tmp_path / "credentials")

    assert seen_encrypt_cmds, "expected an encrypt invocation"
    (cmd,) = seen_encrypt_cmds
    assert "--age" in cmd
    age_value = cmd[cmd.index("--age") + 1]
    assert age_value == "age1recipientone,age1recipienttwo"


def test_refresh_encrypt_keeps_existing_recipients_when_yaml_does_not_match(tmp_path, monkeypatch):
    """A refresh whose file doesn't match any ``.sops.yaml`` path_regex must
    keep the file's OWN existing recipients, not fall through to a plain
    ``sops -e -i`` with no ``--age`` at all (the exact bug shape
    ``uploader.sops_encrypt_json`` already fixed on the YouTube side)."""
    enc_path = _enc_path(tmp_path)
    creds = _base_creds()
    creds["sops"] = {"age": [{"recipient": "age1realfilerecipient"}]}
    _write_enc(enc_path, creds)
    (tmp_path / ".sops.yaml").write_text(
        "creation_rules:\n  - path_regex: some_other_file\\.json$\n    age: age1wrongrecipient\n"
    )
    _patch_post(monkeypatch, {"access_token": "new_access_token", "expires_in": 86400})

    seen_encrypt_cmds = []
    real_fake = _fake_sops_run

    def _capturing_run(cmd, **kwargs):
        if "-e" in cmd:
            seen_encrypt_cmds.append(cmd)
        return real_fake(cmd, **kwargs)

    monkeypatch.setattr(tiktok_auth.subprocess, "run", _capturing_run)

    tiktok_auth.get_valid_tiktok_credentials(tmp_path / "credentials")

    assert seen_encrypt_cmds, "expected an encrypt invocation"
    (cmd,) = seen_encrypt_cmds
    assert "--age" in cmd, "must not fall through to a plain sops -e -i with no --age"
    age_value = cmd[cmd.index("--age") + 1]
    assert age_value == "age1realfilerecipient"


def test_refresh_encrypt_failure_does_not_lose_rotated_token(tmp_path, monkeypatch):
    """If the encrypt step fails, the rotated token must not vanish: the
    error surfaces and the original on-disk credentials are untouched."""
    enc_path = _enc_path(tmp_path)
    original = _base_creds()
    _write_enc(enc_path, original)
    (tmp_path / ".sops.yaml").write_text(
        "creation_rules:\n"
        "  - path_regex: (youtube_(credentials|upload_token)|meta_publishing|"
        "tiktok_publishing)\\.json$\n"
        "    age: age1recipientone\n"
    )
    _patch_post(
        monkeypatch,
        {
            "access_token": "new_access_token",
            "expires_in": 86400,
            "refresh_token": "rotated_refresh_token",
        },
    )

    def _failing_run(cmd, **kwargs):
        if "-d" in cmd:
            return SimpleNamespace(returncode=0, stdout=open(cmd[-1]).read(), stderr="")
        if "-e" in cmd:
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="config file not found, or has no creation rules",
            )
        raise AssertionError(f"unexpected sops invocation: {cmd}")

    monkeypatch.setattr(tiktok_auth.subprocess, "run", _failing_run)

    with pytest.raises(tiktok_auth.TikTokAuthError):
        tiktok_auth.get_valid_tiktok_credentials(tmp_path / "credentials")

    assert json.loads(enc_path.read_text()) == original
    leftover_tmp = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftover_tmp == []


def test_sops_decrypt_failure_does_not_leak_key_material(tmp_path, monkeypatch):
    stderr = (
        "Failed to get the data key required to decrypt the data key: "
        "age recipient age1abcdefghijklmnop, key file /home/example/.config/sops/age/keys.txt"
    )
    monkeypatch.setattr(
        tiktok_auth.subprocess,
        "run",
        lambda cmd, **kw: SimpleNamespace(returncode=1, stderr=stderr),
    )
    _write_enc(_enc_path(tmp_path), _base_creds())
    with pytest.raises(FileNotFoundError) as exc_info:
        tiktok_auth.load_tiktok_credentials(tmp_path / "credentials")
    message = str(exc_info.value)
    assert "age1abcdefghijklmnop" not in message
    assert "keys.txt" not in message
    assert "/home/example" not in message
