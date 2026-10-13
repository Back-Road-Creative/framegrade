"""``gomoveshift youtube-auth`` — consent flow that mints the upload token
with the ``youtube.upload`` + ``youtube.force-ssl`` scopes.

The OAuth flow itself is never exercised — ``flow_factory`` is a seam so
tests never open a browser or hit a real Google endpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from gomoveshift.cli.handlers import youtube_auth
from gomoveshift.cli.handlers.youtube_auth import run_youtube_auth
from gomoveshift.pipeline.drive_phases.youtube_uploader import (
    YOUTUBE_COMMENT_SCOPE,
    YOUTUBE_UPLOAD_SCOPE,
)


_ANALYTICS_SCOPE = "https://www.googleapis.com/auth/yt-analytics.readonly"

_SERVICE_DIR = Path(__file__).resolve().parents[1]


class _FakeCreds:
    def __init__(self, scopes):
        self.token = "fake-token"  # noqa: S105 — test fixture, not a real credential
        self.refresh_token = "fake-refresh"  # noqa: S105
        self.token_uri = "https://oauth2.googleapis.com/token"  # noqa: S105
        self.client_id = "fake-client-id"
        self.client_secret = "fake-client-secret"  # noqa: S105  # pragma: allowlist secret
        self.scopes = scopes


class _FakeFlow:
    def __init__(self, client_config, scopes):
        self.client_config = client_config  # Path (downloaded file) or dict (from the shared token)
        self.scopes = scopes

    def run_local_server(self, port=0):
        return _FakeCreds(self.scopes)


class _FakeChannelsList:
    def __init__(self, response):
        self._response = response

    def execute(self):
        return self._response


class _FakeChannelsResource:
    def __init__(self, response):
        self._response = response

    def list(self, **kwargs):
        return _FakeChannelsList(self._response)


class _FakeYouTubeService:
    """Fake ``googleapiclient`` service -- ``channels().list(mine=True).execute()``
    only, matching the one call ``_fetch_channel_identity`` makes."""

    def __init__(self, response=None):
        self._response = response or {
            "items": [{"id": "UCfakechannel", "snippet": {"title": "Fake Channel"}}]
        }

    def channels(self):
        return _FakeChannelsResource(self._response)


def _fake_service_factory(_credentials):
    return _FakeYouTubeService()


def _boom_service_factory(_credentials):
    raise RuntimeError("network unreachable")


def test_subparser_registered():
    from gomoveshift.cli import _COMMAND_MAP, _build_parser

    assert "youtube-auth" in _COMMAND_MAP
    parser = _build_parser()
    args = parser.parse_args(["youtube-auth"])
    assert args.command == "youtube-auth"
    assert args.brand is None
    assert args.dry_run is False


def test_subparser_accepts_brand_and_dry_run():
    from gomoveshift.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["youtube-auth", "--brand", "examplebrand", "--dry-run"])
    assert args.brand == "examplebrand"
    assert args.dry_run is True


def test_dry_run_prints_path_and_scopes_without_flow(tmp_path, capsys):
    creds_dir = tmp_path / "credentials"
    rc = run_youtube_auth(
        brand=None,
        credentials_dir=creds_dir,
        dry_run=True,
        flow_factory=None,
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert str(tmp_path / "youtube_upload_token.json.enc") in captured.out
    assert YOUTUBE_UPLOAD_SCOPE in captured.out
    assert YOUTUBE_COMMENT_SCOPE in captured.out
    assert _ANALYTICS_SCOPE in captured.out


def test_dry_run_never_calls_flow_factory(tmp_path):
    creds_dir = tmp_path / "credentials"
    calls = []

    def _boom(client_secret_path, scopes):
        calls.append((client_secret_path, scopes))
        raise AssertionError("flow_factory must not be called in --dry-run")

    rc = run_youtube_auth(brand=None, credentials_dir=creds_dir, dry_run=True, flow_factory=_boom)
    assert rc == 0
    assert calls == []


def test_shared_token_written_through_sops_encrypt(tmp_path, monkeypatch):
    creds_dir = tmp_path / "credentials"
    client_secret_path = tmp_path / "client_secret.json"
    client_secret_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "gomoveshift.cli.handlers.youtube_auth._resolve_client_secret_path",
        lambda: client_secret_path,
    )

    written = {}

    def _fake_sops_encrypt_json(data, enc_path, *, age_recipients=None):
        written["data"] = data
        written["enc_path"] = enc_path
        enc_path.parent.mkdir(parents=True, exist_ok=True)
        enc_path.write_text("encrypted", encoding="utf-8")

    monkeypatch.setattr(
        "gomoveshift.cli.handlers.youtube_auth.sops_encrypt_json", _fake_sops_encrypt_json
    )

    rc = run_youtube_auth(
        brand=None,
        credentials_dir=creds_dir,
        dry_run=False,
        flow_factory=_FakeFlow,
        service_factory=_fake_service_factory,
    )
    assert rc == 0
    expected_path = creds_dir.parent / "youtube_upload_token.json.enc"
    assert written["enc_path"] == expected_path
    assert written["data"]["token"] == "fake-token"  # noqa: S105
    assert written["data"]["scopes"] == [
        YOUTUBE_UPLOAD_SCOPE,
        YOUTUBE_COMMENT_SCOPE,
        _ANALYTICS_SCOPE,
    ]
    assert "channel_id" not in written["data"]
    assert "channel_title" not in written["data"]
    assert expected_path.exists()


def test_brand_scoped_token_path(tmp_path, monkeypatch):
    creds_dir = tmp_path / "credentials"
    brand_dir = creds_dir / "brands" / "examplebrand" / "creds"
    brand_dir.mkdir(parents=True)
    client_secret_path = tmp_path / "client_secret.json"
    client_secret_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "gomoveshift.cli.handlers.youtube_auth._resolve_client_secret_path",
        lambda: client_secret_path,
    )

    written = {}

    def _fake_sops_encrypt_json(data, enc_path, *, age_recipients=None):
        written["enc_path"] = enc_path
        enc_path.parent.mkdir(parents=True, exist_ok=True)
        enc_path.write_text("encrypted", encoding="utf-8")

    monkeypatch.setattr(
        "gomoveshift.cli.handlers.youtube_auth.sops_encrypt_json", _fake_sops_encrypt_json
    )

    rc = run_youtube_auth(
        brand="examplebrand",
        credentials_dir=creds_dir,
        dry_run=False,
        flow_factory=_FakeFlow,
        service_factory=_fake_service_factory,
    )
    assert rc == 0
    expected_path = creds_dir / "brands" / "examplebrand" / "youtube_upload_token.json.enc"
    assert written["enc_path"] == expected_path


def _no_shared_token(**_kw):
    raise FileNotFoundError("no youtube_upload_token.json.enc anywhere")


def test_missing_client_secret_errors_cleanly(tmp_path, monkeypatch, capsys):
    """No downloaded client_secret.json AND no shared token to borrow the
    OAuth client from → exit 2 with the download hint (nothing to fall back to)."""
    creds_dir = tmp_path / "credentials"
    missing_secret = tmp_path / "nope" / "client_secret.json"

    monkeypatch.setattr(
        "gomoveshift.cli.handlers.youtube_auth._resolve_client_secret_path",
        lambda: missing_secret,
    )
    monkeypatch.setattr(
        "gomoveshift.cli.handlers.youtube_auth._load_shared_token", _no_shared_token
    )
    monkeypatch.setattr(
        "gomoveshift.cli.handlers.youtube_auth._config_manager_path_unset", lambda: False
    )

    rc = run_youtube_auth(
        brand=None, credentials_dir=creds_dir, dry_run=False, flow_factory=_FakeFlow
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert str(missing_secret) in err
    assert "console.cloud.google.com" in err


def test_missing_config_names_the_launch_directory_not_a_download(tmp_path, monkeypatch, capsys):
    """Launched outside the service directory no config.yaml loads, so
    ``seo.manager_path`` stays "." and both lookups go cwd-relative. The cause
    is the launch directory: the error must say so and give the command, and
    must not send the operator to download a client secret they already have."""
    monkeypatch.setattr(
        "gomoveshift.cli.handlers.youtube_auth._resolve_client_secret_path",
        lambda: tmp_path / "client_secret.json",
    )
    monkeypatch.setattr(
        "gomoveshift.cli.handlers.youtube_auth._load_shared_token", _no_shared_token
    )
    monkeypatch.setattr(
        "gomoveshift.cli.handlers.youtube_auth._config_manager_path_unset", lambda: True
    )

    rc = run_youtube_auth(
        brand=None, credentials_dir=tmp_path / "credentials", dry_run=False, flow_factory=_FakeFlow
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "config.yaml" in err
    assert str(Path.cwd()) in err
    assert f"cd {_SERVICE_DIR} && .venv/bin/gomoveshift youtube-auth" in err
    assert "console.cloud.google.com" not in err


def _launch_directory_error(tmp_path, monkeypatch, capsys) -> str:
    """The stderr of a run that loaded no config.yaml and found no credentials."""
    monkeypatch.setattr(youtube_auth, "_resolve_client_secret_path", lambda: tmp_path / "cs.json")
    monkeypatch.setattr(youtube_auth, "_load_shared_token", _no_shared_token)
    monkeypatch.setattr(youtube_auth, "_config_manager_path_unset", lambda: True)
    rc = run_youtube_auth(
        brand=None, credentials_dir=tmp_path / "credentials", dry_run=False, flow_factory=_FakeFlow
    )
    assert rc == 2
    return capsys.readouterr().err


def _wheel_install(root: Path, monkeypatch) -> Path:
    """Make the handler look installed as a wheel under ``root/.venv``; returns
    the site-packages directory it then lives in."""
    site = root / ".venv" / "lib" / "python3.12" / "site-packages"
    handler = site / "gomoveshift" / "cli" / "handlers" / "youtube_auth.py"
    handler.parent.mkdir(parents=True)
    monkeypatch.setattr(youtube_auth, "__file__", str(handler))
    monkeypatch.setattr(sys, "prefix", str(root / ".venv"))
    return site


def test_launch_directory_command_survives_a_wheel_install(tmp_path, monkeypatch, capsys):
    """Installed as a wheel the handler lives under the venv's site-packages,
    not the checkout. The command must still name the directory that holds
    ``config.yaml`` -- ``cd .../site-packages`` sends the operator nowhere."""
    service = tmp_path / "svc"
    _wheel_install(service, monkeypatch)
    (service / "config.yaml").write_text("seo: {}\n", encoding="utf-8")

    err = _launch_directory_error(tmp_path, monkeypatch, capsys)

    assert f"cd {service} && .venv/bin/gomoveshift youtube-auth" in err
    assert "site-packages" not in err


def test_launch_directory_error_never_invents_a_directory(tmp_path, monkeypatch, capsys):
    """No ``config.yaml`` beside the code or beside the venv: the error names
    the file to launch next to and gives no ``cd`` at all, rather than a path
    that only looks like the service directory."""
    _wheel_install(tmp_path / "elsewhere", monkeypatch)

    err = _launch_directory_error(tmp_path, monkeypatch, capsys)

    assert " cd " not in err
    assert "site-packages" not in err
    assert "config.yaml" in err
    assert "gomoveshift youtube-auth" in err


def test_config_manager_path_unset_follows_the_loaded_config(tmp_path, monkeypatch):
    """The seam reads the LOADED config: unset in a directory with no
    config.yaml, set once a config.yaml there names ``seo.manager_path``."""
    from gomoveshift.cli.handlers.youtube_auth import _config_manager_path_unset

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    assert _config_manager_path_unset() is True

    (tmp_path / "config.yaml").write_text(f"seo:\n  manager_path: {tmp_path}\n")
    assert _config_manager_path_unset() is False


def test_missing_client_secret_falls_back_to_shared_token_client(tmp_path, monkeypatch, capsys):
    """The 09-26 brand-consent failure: no client_secret.json on the box, but
    the shared GMS token already carries the OAuth client id+secret. The flow
    must reuse that client in memory (never a plaintext file on disk) so a
    brand mint works wherever the shared token does."""
    creds_dir = tmp_path / "credentials"
    (creds_dir / "brands" / "examplebrand" / "creds").mkdir(parents=True)
    missing_secret = tmp_path / "nope" / "client_secret.json"

    monkeypatch.setattr(
        "gomoveshift.cli.handlers.youtube_auth._resolve_client_secret_path",
        lambda: missing_secret,
    )
    loader_calls = []

    def _shared_token(**kw):
        loader_calls.append(kw)
        return {
            "token": "shared-token",  # noqa: S106
            "refresh_token": "shared-refresh",  # noqa: S106
            "client_id": "shared-client-id.apps.googleusercontent.com",
            "client_secret": "shared-client-secret",  # noqa: S106  # pragma: allowlist secret
            "scopes": [YOUTUBE_UPLOAD_SCOPE],
        }

    monkeypatch.setattr("gomoveshift.cli.handlers.youtube_auth._load_shared_token", _shared_token)

    seen = {}

    class _ConfigFlow(_FakeFlow):
        def __init__(self, client_config, scopes):
            seen["client_config"] = client_config
            super().__init__(client_config, scopes)

    written = {}

    def _fake_sops_encrypt_json(data, enc_path, *, age_recipients=None):
        written["data"] = data
        written["enc_path"] = enc_path
        enc_path.parent.mkdir(parents=True, exist_ok=True)
        enc_path.write_text("encrypted", encoding="utf-8")

    monkeypatch.setattr(
        "gomoveshift.cli.handlers.youtube_auth.sops_encrypt_json", _fake_sops_encrypt_json
    )

    rc = run_youtube_auth(
        brand="examplebrand",
        credentials_dir=creds_dir,
        dry_run=False,
        flow_factory=_ConfigFlow,
        service_factory=_fake_service_factory,
    )
    assert rc == 0
    # The shared (brand=None) token is the source of the client, even for a brand mint.
    assert loader_calls == [{"credentials_dir": creds_dir, "brand": None}]
    installed = seen["client_config"]["installed"]
    assert installed["client_id"] == "shared-client-id.apps.googleusercontent.com"
    assert installed["client_secret"] == "shared-client-secret"  # noqa: S105  # pragma: allowlist secret
    assert installed["token_uri"] == "https://oauth2.googleapis.com/token"  # noqa: S105 — endpoint URL, not a secret
    assert not missing_secret.exists()
    assert not list(tmp_path.rglob("client_secret.json"))
    assert (
        written["enc_path"]
        == creds_dir / "brands" / "examplebrand" / "youtube_upload_token.json.enc"
    )
    assert "reusing the OAuth client" in capsys.readouterr().out


def test_shared_token_without_client_fields_errors_cleanly(tmp_path, monkeypatch, capsys):
    creds_dir = tmp_path / "credentials"
    missing_secret = tmp_path / "nope" / "client_secret.json"
    monkeypatch.setattr(
        "gomoveshift.cli.handlers.youtube_auth._resolve_client_secret_path",
        lambda: missing_secret,
    )
    monkeypatch.setattr(
        "gomoveshift.cli.handlers.youtube_auth._load_shared_token",
        lambda **_kw: {"token": "t", "refresh_token": "r"},  # noqa: S106
    )
    rc = run_youtube_auth(
        brand=None, credentials_dir=creds_dir, dry_run=False, flow_factory=_FakeFlow
    )
    assert rc == 2
    assert "client_id" in capsys.readouterr().err


_SHARED_ENC = (
    '{"data": "ENC[PLACEHOLDER]", "sops": {"age": ['
    '{"recipient": "age1shared1", "enc": "x"}, {"recipient": "age1shared2", "enc": "y"}]}}\n'
)


def test_brand_mint_inherits_the_shared_tokens_recipients(tmp_path, monkeypatch):
    """A new brand token has no file to inherit recipients from; it must take
    the SHARED token's, or the pipeline sops config's catch-all rule encrypts
    it for keys the dev keyring cannot open."""
    creds_dir = tmp_path / "credentials"
    (creds_dir / "brands" / "examplebrand" / "creds").mkdir(parents=True)
    shared_enc = creds_dir.parent / "youtube_upload_token.json.enc"
    shared_enc.write_text(_SHARED_ENC, encoding="utf-8")
    client_secret_path = tmp_path / "client_secret.json"
    client_secret_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "gomoveshift.cli.handlers.youtube_auth._resolve_client_secret_path",
        lambda: client_secret_path,
    )
    written = {}

    def _fake_sops_encrypt_json(data, enc_path, *, age_recipients=None):
        written["enc_path"] = enc_path
        written["age_recipients"] = age_recipients
        enc_path.parent.mkdir(parents=True, exist_ok=True)
        enc_path.write_text("encrypted", encoding="utf-8")

    monkeypatch.setattr(
        "gomoveshift.cli.handlers.youtube_auth.sops_encrypt_json", _fake_sops_encrypt_json
    )

    rc = run_youtube_auth(
        brand="examplebrand",
        credentials_dir=creds_dir,
        dry_run=False,
        flow_factory=_FakeFlow,
        service_factory=_fake_service_factory,
    )
    assert rc == 0
    assert (
        written["enc_path"]
        == creds_dir / "brands" / "examplebrand" / "youtube_upload_token.json.enc"
    )
    assert written["age_recipients"] == ["age1shared1", "age1shared2"]


def test_shared_remint_passes_no_explicit_recipients(tmp_path, monkeypatch):
    """The shared re-mint replaces an existing file — sops_encrypt_json inherits
    that file's recipients itself; the handler must not second-guess it."""
    creds_dir = tmp_path / "credentials"
    client_secret_path = tmp_path / "client_secret.json"
    client_secret_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "gomoveshift.cli.handlers.youtube_auth._resolve_client_secret_path",
        lambda: client_secret_path,
    )
    written = {}

    def _fake_sops_encrypt_json(data, enc_path, *, age_recipients=None):
        written["age_recipients"] = age_recipients
        enc_path.parent.mkdir(parents=True, exist_ok=True)
        enc_path.write_text("encrypted", encoding="utf-8")

    monkeypatch.setattr(
        "gomoveshift.cli.handlers.youtube_auth.sops_encrypt_json", _fake_sops_encrypt_json
    )
    rc = run_youtube_auth(
        brand=None,
        credentials_dir=creds_dir,
        dry_run=False,
        flow_factory=_FakeFlow,
        service_factory=_fake_service_factory,
    )
    assert rc == 0
    assert written["age_recipients"] is None


def test_mint_captures_and_stores_channel_identity(tmp_path, monkeypatch, capsys):
    """Channel id/title come from the same ``channels().list(mine=True)``
    pattern ``youtube_adapter.py`` already uses, and land in a plain JSON
    sidecar NEXT TO the token -- never inside the encrypted token dict, and
    no token field (token/refresh_token/client_secret) is ever printed."""
    creds_dir = tmp_path / "credentials"
    client_secret_path = tmp_path / "client_secret.json"
    client_secret_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "gomoveshift.cli.handlers.youtube_auth._resolve_client_secret_path",
        lambda: client_secret_path,
    )

    def _fake_sops_encrypt_json(data, enc_path, *, age_recipients=None):
        enc_path.parent.mkdir(parents=True, exist_ok=True)
        enc_path.write_text("encrypted", encoding="utf-8")

    monkeypatch.setattr(
        "gomoveshift.cli.handlers.youtube_auth.sops_encrypt_json", _fake_sops_encrypt_json
    )

    rc = run_youtube_auth(
        brand=None,
        credentials_dir=creds_dir,
        dry_run=False,
        flow_factory=_FakeFlow,
        service_factory=_fake_service_factory,
    )
    assert rc == 0

    enc_path = creds_dir.parent / "youtube_upload_token.json.enc"
    sidecar = creds_dir.parent / "youtube_upload_token.identity.json"
    assert sidecar.exists()
    identity = json.loads(sidecar.read_text(encoding="utf-8"))
    assert identity == {"channel_id": "UCfakechannel", "channel_title": "Fake Channel"}
    # The encrypted-token fake write is a plain marker string -- the identity
    # fields never went anywhere near it.
    assert enc_path.read_text(encoding="utf-8") == "encrypted"

    captured = capsys.readouterr()
    assert "Fake Channel" in captured.out
    assert "UCfakechannel" in captured.out
    for secret in ("fake-token", "fake-refresh", "fake-client-secret"):
        assert secret not in captured.out
        assert secret not in captured.err


def test_channel_identity_fetch_failure_does_not_block_mint(tmp_path, monkeypatch, capsys):
    """A channel-identity lookup failure (network, quota, whatever) is
    best-effort: the mint still succeeds and writes the token, it just skips
    the sidecar -- doctor's WARN path picks this case up later."""
    creds_dir = tmp_path / "credentials"
    client_secret_path = tmp_path / "client_secret.json"
    client_secret_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "gomoveshift.cli.handlers.youtube_auth._resolve_client_secret_path",
        lambda: client_secret_path,
    )
    written = {}

    def _fake_sops_encrypt_json(data, enc_path, *, age_recipients=None):
        written["enc_path"] = enc_path
        enc_path.parent.mkdir(parents=True, exist_ok=True)
        enc_path.write_text("encrypted", encoding="utf-8")

    monkeypatch.setattr(
        "gomoveshift.cli.handlers.youtube_auth.sops_encrypt_json", _fake_sops_encrypt_json
    )

    rc = run_youtube_auth(
        brand=None,
        credentials_dir=creds_dir,
        dry_run=False,
        flow_factory=_FakeFlow,
        service_factory=_boom_service_factory,
    )
    assert rc == 0
    assert written["enc_path"].exists()
    sidecar = creds_dir.parent / "youtube_upload_token.identity.json"
    assert not sidecar.exists()

    captured = capsys.readouterr()
    for secret in ("fake-token", "fake-refresh", "fake-client-secret"):
        assert secret not in captured.out
        assert secret not in captured.err


def test_flow_factory_requested_scopes_include_analytics_read(tmp_path, monkeypatch):
    creds_dir = tmp_path / "credentials"
    client_secret_path = tmp_path / "client_secret.json"
    client_secret_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "gomoveshift.cli.handlers.youtube_auth._resolve_client_secret_path",
        lambda: client_secret_path,
    )
    monkeypatch.setattr(
        "gomoveshift.cli.handlers.youtube_auth.sops_encrypt_json",
        lambda data, enc_path, *, age_recipients=None: None,
    )
    seen = {}

    def _spy_flow(client_config, scopes):
        seen["scopes"] = list(scopes)
        return _FakeFlow(client_config, scopes)

    rc = run_youtube_auth(
        brand=None,
        credentials_dir=creds_dir,
        dry_run=False,
        flow_factory=_spy_flow,
        service_factory=_fake_service_factory,
    )
    assert rc == 0
    assert set(seen["scopes"]) == {YOUTUBE_UPLOAD_SCOPE, YOUTUBE_COMMENT_SCOPE, _ANALYTICS_SCOPE}
