"""Console Settings tab -- GMS_* registry set/unset flags, credential
presence-only status, and the secret-scan proof: no env value ever appears
in the rendered HTML."""

import json

from fastapi.testclient import TestClient

from gomoveshift.console import settings as console_settings
from gomoveshift.console.app import create_app

_SECRET = "sk-verySecretValue-12345"  # noqa: S105 -- fixture, not real  # pragma: allowlist secret


def _client() -> TestClient:
    return TestClient(create_app())


def test_registry_status_flags_set_and_unset(monkeypatch):
    monkeypatch.setenv("GMS_STAGING_DIR", "/tmp/whatever")
    monkeypatch.delenv("GMS_SITE_PATH", raising=False)
    rows = {r["name"]: r for r in console_settings.registry_status()}
    assert rows["GMS_STAGING_DIR"]["is_set"] is True
    assert rows["GMS_SITE_PATH"]["is_set"] is False
    assert "value" not in rows["GMS_STAGING_DIR"]


def test_credential_status_reports_presence(tmp_path, monkeypatch):
    present = tmp_path / "drive_credentials.json.enc"
    present.write_text("{}")
    monkeypatch.setenv("GMS_DRIVE_CREDENTIALS_ENC", str(present))
    monkeypatch.setenv("GMS_DRIVE_TOKEN_ENC", str(tmp_path / "missing.enc"))
    rows = {r["name"]: r for r in console_settings.credential_status()}
    assert rows["GMS_DRIVE_CREDENTIALS_ENC"]["present"] is True
    assert rows["GMS_DRIVE_TOKEN_ENC"]["present"] is False


def test_settings_route_renders_registry_and_credential_rows(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GMS_STAGING_DIR", _SECRET)
    resp = _client().get("/settings")
    assert resp.status_code == 200
    assert "GMS_STAGING_DIR" in resp.text
    assert "GMS_DRIVE_CREDENTIALS_ENC" in resp.text


def test_settings_page_never_leaks_a_secret_looking_env_value(monkeypatch, tmp_path):
    """Secret-scan the rendered HTML: no env value may appear in the page."""
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    for name, *_rest in [(row.name,) for row in console_settings.load_registry()]:
        monkeypatch.setenv(name, _SECRET)
    for cred_name, _default, _desc in console_settings.CREDENTIAL_SOURCES:
        monkeypatch.setenv(cred_name, _SECRET)
    resp = _client().get("/settings")
    assert resp.status_code == 200
    assert _SECRET not in resp.text


def test_settings_form_prefills_only_from_brand_override_never_env(monkeypatch, tmp_path):
    """The form must never source a value from os.environ -- only from the
    current brand's override file -- even when both are set."""
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GMS_OUTPUT_DIR", _SECRET)
    console_settings.save_brand_settings(
        "gomoveshift", {"GMS_OUTPUT_DIR": "/brand/from-file"}, data_dir=tmp_path
    )

    resp = _client().get("/settings")

    assert resp.status_code == 200
    assert "/brand/from-file" in resp.text
    assert _SECRET not in resp.text
    assert 'name="GMS_OUTPUT_DIR"' in resp.text


def test_settings_post_persists_and_empty_value_removes_override(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    console_settings.save_brand_settings(
        "gomoveshift", {"GMS_OUTPUT_DIR": "/brand/old"}, data_dir=tmp_path
    )
    client = _client()

    resp = client.post(
        "/settings",
        data={"GMS_OUTPUT_DIR": "  /brand/new  ", "GMS_STAGING_DIR": ""},
        follow_redirects=False,
    )

    assert resp.status_code == 303
    assert resp.headers["location"] == "/settings"
    reloaded = console_settings.load_brand_settings("gomoveshift", data_dir=tmp_path)
    assert reloaded["GMS_OUTPUT_DIR"] == "/brand/new"
    assert reloaded["GMS_STAGING_DIR"] is None


def test_youtube_identity_status_reports_per_brand(tmp_path, monkeypatch):
    creds_dir = tmp_path / "credentials"
    monkeypatch.setattr(console_settings, "known_brands", lambda: ["gomoveshift"])
    from gomoveshift.pipeline import _yt_creds

    monkeypatch.setattr(_yt_creds, "DEFAULT_CREDENTIALS_DIR", creds_dir)

    not_connected = console_settings.youtube_identity_status()
    assert not_connected == [
        {"brand": "gomoveshift", "channel_id": None, "channel_title": "not connected"}
    ]

    # A token minted before identity capture has no sidecar: it IS connected,
    # the channel is just unknown -- say so and name the command that fixes it.
    (tmp_path / "youtube_upload_token.json.enc").write_text("encrypted", encoding="utf-8")
    legacy = console_settings.youtube_identity_status()
    assert legacy == [
        {
            "brand": "gomoveshift",
            "channel_id": None,
            "channel_title": "token present, channel unknown -- run "
            "`gomoveshift youtube-auth --brand gomoveshift`",
        }
    ]

    (tmp_path / "youtube_upload_token.identity.json").write_text(
        json.dumps({"channel_id": "UCabc123", "channel_title": "GoMoveShift Official"}),
        encoding="utf-8",
    )
    connected = console_settings.youtube_identity_status()
    assert connected == [
        {"brand": "gomoveshift", "channel_id": "UCabc123", "channel_title": "GoMoveShift Official"}
    ]


def test_settings_route_renders_youtube_identity_and_never_leaks_secret(monkeypatch, tmp_path):
    """The live ``GET /settings`` route shows each brand's channel identity,
    and the raw token contents never reach the page even when a
    secret-looking string sits in the encrypted file."""
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))
    creds_dir = tmp_path / "credentials"
    (tmp_path / "youtube_upload_token.json.enc").write_text(_SECRET, encoding="utf-8")
    (tmp_path / "youtube_upload_token.identity.json").write_text(
        json.dumps({"channel_id": "UCabc123", "channel_title": "GoMoveShift Official"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(console_settings, "known_brands", lambda: ["gomoveshift"])
    from gomoveshift.pipeline import _yt_creds

    monkeypatch.setattr(_yt_creds, "DEFAULT_CREDENTIALS_DIR", creds_dir)

    response = _client().get("/settings")

    assert response.status_code == 200
    assert "GoMoveShift Official" in response.text
    assert "UCabc123" in response.text
    assert _SECRET not in response.text


def test_settings_post_rejects_unknown_name(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    client = _client()

    resp = client.post("/settings", data={"NOT_A_REGISTRY_NAME": "x"})

    assert resp.status_code == 400
