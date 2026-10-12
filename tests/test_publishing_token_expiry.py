"""A token-expiry report the daily digest can render. Returns
day counts and labels only -- SECRETS RULE: never a token value, never SOPS
stderr text. sops subprocess is patched, exactly like
test_publishing_tiktok_auth.py -- no real decrypt."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gomoveshift.publishing import meta_auth, tiktok_auth, token_expiry

FIXED_NOW = datetime(2026, 9, 24, 12, 0, 0, tzinfo=timezone.utc)


def _fake_sops_run(cmd, **kwargs):
    if "-d" in cmd:
        return SimpleNamespace(returncode=0, stdout=open(cmd[-1]).read(), stderr="")
    raise AssertionError(f"unexpected sops invocation: {cmd}")


def _fake_sops_fail(cmd, **kwargs):
    return SimpleNamespace(
        returncode=1,
        stdout="",
        stderr="age recipient age1abcdefghijklmnop, key file /home/example/.config/sops/age/keys.txt",
    )


def _write_meta_enc(creds_dir, creds):
    path = creds_dir.parent / "meta_publishing.json.enc"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(creds))


def _write_tiktok_enc(creds_dir, creds):
    path = creds_dir.parent / "tiktok_publishing.json.enc"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(creds))


def _meta_creds(**overrides):
    creds = {
        "app_id": "app1",
        "app_secret": "shh",  # pragma: allowlist secret
        "page_id": "page1",
        "ig_user_id": "ig1",
        "page_access_token": "PAGE_TOKEN_VALUE",  # pragma: allowlist secret
        "token_type": "bearer",
        "scopes": ["pages_manage_posts"],
        "user_token_expires_at": (FIXED_NOW + timedelta(days=3)).isoformat(),
        "issued_at": (FIXED_NOW - timedelta(days=57)).isoformat(),
        "graph_version": "v21.0",
    }
    creds.update(overrides)
    return creds


def _tiktok_creds(**overrides):
    creds = {
        "client_key": "ck",  # pragma: allowlist secret
        "client_secret": "cs",  # pragma: allowlist secret
        "open_id": "open1",
        "access_token": "ACCESS_TOKEN_VALUE",  # pragma: allowlist secret
        "access_expires_at": (FIXED_NOW + timedelta(hours=2)).isoformat(),
        "refresh_token": "REFRESH_TOKEN_VALUE",  # pragma: allowlist secret
        "refresh_expires_at": (FIXED_NOW + timedelta(days=40)).isoformat(),
    }
    creds.update(overrides)
    return creds


@pytest.fixture(autouse=True)
def _patch_sops(monkeypatch):
    monkeypatch.setattr(meta_auth.subprocess, "run", _fake_sops_run)
    monkeypatch.setattr(tiktok_auth.subprocess, "run", _fake_sops_run)


def test_days_left_for_both_platforms(tmp_path):
    _write_meta_enc(tmp_path / "credentials", _meta_creds())
    _write_tiktok_enc(tmp_path / "credentials", _tiktok_creds())

    rows = token_expiry.token_expiry_report(now=FIXED_NOW, credentials_dir=tmp_path / "credentials")

    by_key = {(r["platform"], r["what"]): r for r in rows}
    assert by_key[("meta", "user token")]["days_left"] == 3
    assert by_key[("tiktok", "refresh token")]["days_left"] == 40


def test_overdue_is_a_negative_day_count(tmp_path):
    _write_meta_enc(
        tmp_path / "credentials",
        _meta_creds(user_token_expires_at=(FIXED_NOW - timedelta(days=2)).isoformat()),
    )
    _write_tiktok_enc(tmp_path / "credentials", _tiktok_creds())

    rows = token_expiry.token_expiry_report(now=FIXED_NOW, credentials_dir=tmp_path / "credentials")

    row = next(r for r in rows if r["platform"] == "meta")
    assert row["days_left"] == -2


def test_missing_credentials_file_is_a_status_row_not_an_exception(tmp_path):
    rows = token_expiry.token_expiry_report(now=FIXED_NOW, credentials_dir=tmp_path / "credentials")

    meta_rows = [r for r in rows if r["platform"] == "meta"]
    tiktok_rows = [r for r in rows if r["platform"] == "tiktok"]
    assert all(r["status"] == "missing" for r in meta_rows)
    assert all(r["status"] == "missing" for r in tiktok_rows)
    assert len(tiktok_rows) == 1


def test_decrypt_failure_is_unreadable_never_raises(tmp_path, monkeypatch):
    _write_meta_enc(tmp_path / "credentials", _meta_creds())
    monkeypatch.setattr(meta_auth.subprocess, "run", _fake_sops_fail)

    rows = token_expiry.token_expiry_report(now=FIXED_NOW, credentials_dir=tmp_path / "credentials")

    meta_rows = [r for r in rows if r["platform"] == "meta"]
    assert [r["status"] for r in meta_rows] == ["unreadable", "unreadable"]


def test_no_token_value_or_sops_stderr_ever_appears_in_output(tmp_path, monkeypatch):
    _write_meta_enc(tmp_path / "credentials", _meta_creds())
    _write_tiktok_enc(tmp_path / "credentials", _tiktok_creds())
    monkeypatch.setattr(tiktok_auth.subprocess, "run", _fake_sops_fail)

    rows = token_expiry.token_expiry_report(now=FIXED_NOW, credentials_dir=tmp_path / "credentials")
    dumped = json.dumps(rows)

    assert "PAGE_TOKEN_VALUE" not in dumped
    assert "ACCESS_TOKEN_VALUE" not in dumped
    assert "REFRESH_TOKEN_VALUE" not in dumped
    assert "keys.txt" not in dumped
    assert "age1abcdefghijklmnop" not in dumped


def test_cli_prints_json_rows(tmp_path, monkeypatch, capsys):
    _write_meta_enc(tmp_path / "credentials", _meta_creds())
    _write_tiktok_enc(tmp_path / "credentials", _tiktok_creds())
    monkeypatch.setattr(token_expiry, "DEFAULT_CREDENTIALS_DIR_OVERRIDE", None, raising=False)

    token_expiry.main(argv=["--credentials-dir", str(tmp_path / "credentials")])

    out = json.loads(capsys.readouterr().out)
    assert isinstance(out, list)
    assert {row["platform"] for row in out} == {"meta", "tiktok"}


def _rows_by_key(tmp_path, now=FIXED_NOW, **meta_overrides):
    _write_meta_enc(tmp_path / "credentials", _meta_creds(**meta_overrides))
    _write_tiktok_enc(tmp_path / "credentials", _tiktok_creds())
    rows = token_expiry.token_expiry_report(now=now, credentials_dir=tmp_path / "credentials")
    return {(r["platform"], r["what"]): r for r in rows}


def test_never_expiring_user_token_is_a_status_row_not_a_crash(tmp_path):
    # A record whose user token never expires stores "never" (/debug_token
    # reports expires_at 0). It must be a status row: an unhandled
    # fromisoformat("never") would raise and take every row, TikTok's included,
    # with it.
    by_key = _rows_by_key(tmp_path, user_token_expires_at="never")  # noqa: S106 -- an expiry, not a secret

    assert by_key[("meta", "user token")] == {
        "platform": "meta",
        "what": "user token",
        "status": "never",
    }
    assert by_key[("tiktok", "refresh token")]["days_left"] == 40


def test_unparsable_expiry_is_unreadable_for_that_row_only(tmp_path):
    by_key = _rows_by_key(tmp_path, user_token_expires_at="soon")  # noqa: S106 -- an expiry, not a secret

    assert by_key[("meta", "user token")]["status"] == "unreadable"
    assert by_key[("tiktok", "refresh token")]["days_left"] == 40


def test_instagram_data_access_row_uses_the_recorded_date(tmp_path):
    by_key = _rows_by_key(
        tmp_path, data_access_expires_at=(FIXED_NOW + timedelta(days=20)).isoformat()
    )

    row = by_key[("meta", "Instagram data access")]
    assert row["days_left"] == 20
    assert row["warn_days"] == 14
    assert "re-create the Meta credential record" in row["fix"]


def test_instagram_data_access_falls_back_to_issued_at_plus_90_days(tmp_path):
    # Records that lack data_access_expires_at. issued_at is the login date and may be
    # date-only; Meta's data access runs 90 days from that login.
    now = datetime(2026, 12, 20, 12, 0, 0, tzinfo=timezone.utc)
    by_key = _rows_by_key(tmp_path, now=now, issued_at="2026-10-02")

    row = by_key[("meta", "Instagram data access")]
    assert row["expires_at"].startswith("2026-12-31")
    assert row["days_left"] == 10


def test_instagram_data_access_missing_when_nothing_dates_the_login(tmp_path):
    by_key = _rows_by_key(tmp_path, issued_at=None)

    assert by_key[("meta", "Instagram data access")]["status"] == "missing"


def test_tiktok_access_token_row_is_gone(tmp_path):
    # The 24 h access token is refreshed on use (tiktok_auth.get_valid_tiktok_credentials),
    # so its expiry is never something to act on: it read 0 days every day.
    by_key = _rows_by_key(tmp_path)

    assert ("tiktok", "access token") not in by_key
