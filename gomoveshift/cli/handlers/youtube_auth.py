"""Handler for ``gomoveshift youtube-auth``: the operator consent flow
that mints the YouTube upload token with the ``youtube.upload`` +
``youtube.force-ssl`` + ``yt-analytics.readonly`` scopes.

``yt-analytics.readonly`` is what ``tracking/collector.py`` needs for every
YouTube Analytics read; a token minted without it 403s insufficientPermissions.

``youtube.force-ssl`` is needed so first comments can post
(``publishing/youtube_comments.py`` records a 403 today without it) and so a
second brand channel can get its own token via
``--brand``.

The browser-consent step is a human act — ``flow_factory`` is a seam so
tests never launch a browser or reach a real Google endpoint. Never prints
or logs a token, refresh token or client secret.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional, Union

from ...pipeline._yt_creds import sops_age_recipients, write_channel_identity
from ...pipeline.drive_phases.youtube_uploader import (
    YOUTUBE_ANALYTICS_SCOPE,
    YOUTUBE_COMMENT_SCOPE,
    YOUTUBE_UPLOAD_SCOPE,
)
from ...pipeline.uploader import sops_encrypt_json
from ...publishing.credential_paths import resolve_sops_file

_YOUTUBE_TOKEN_FILENAME = "youtube_upload_token.json.enc"  # noqa: S105 — filename, not a secret

FlowFactory = Callable[[Union[Path, dict], list], object]
ServiceFactory = Callable[[object], object]


def _resolve_client_secret_path() -> Path:
    """The same client-secret source ``tracking.collector.YouTubeCollector``
    already resolves from ``config.seo.manager_path`` — analytics sync and
    this consent flow share one client secret."""
    from ...tracking.collector import _resolve_default_cred_paths

    client_secret_path, _token_path = _resolve_default_cred_paths()
    return client_secret_path


def _service_dir() -> Optional[Path]:
    """The service directory holding the project ``config.yaml``, or None when it
    cannot be found. A ``config.yaml`` under the config home is overlaid from any
    launch directory; relative credential paths resolve against the launch
    directory. An editable install has it three levels above this file; a wheel
    install puts this file under the venv's site-packages, so the venv's parent is
    tried next. Never a guess: a candidate counts only when ``config.yaml`` is
    actually there."""
    for candidate in (Path(__file__).resolve().parents[3], Path(sys.prefix).resolve().parent):
        if (candidate / "config.yaml").is_file():
            return candidate
    return None


def _config_manager_path_unset() -> bool:
    """True when no loaded config.yaml set ``seo.manager_path`` (it is still the
    "." default) -- the process was launched outside the service directory, so
    the client secret and the shared token were both looked up cwd-relative."""
    from ...config import get_config

    return str(get_config().seo.manager_path) == "."


def _load_shared_token(*, credentials_dir: Path, brand: Optional[str]) -> dict:
    """The one loader every other YouTube consumer uses (seam for tests)."""
    from ...pipeline._yt_creds import load_youtube_credentials

    return load_youtube_credentials(credentials_dir=credentials_dir, brand=brand)


def _client_config_from_token(token_data: dict) -> dict:
    """Rebuild an ``installed``-app client config from the OAuth client a
    minted token already carries (same shape ``from_client_secrets_file``
    reads). Kept in memory — never written to disk as plaintext."""
    client_id = token_data.get("client_id")
    client_secret = token_data.get("client_secret")
    if not client_id or not client_secret:
        raise KeyError("client_id/client_secret")
    return {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": token_data.get("token_uri") or "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }


def _default_flow_factory(client_config: Union[Path, dict], scopes: list):
    from google_auth_oauthlib.flow import InstalledAppFlow

    if isinstance(client_config, dict):
        return InstalledAppFlow.from_client_config(client_config, scopes)
    return InstalledAppFlow.from_client_secrets_file(str(client_config), scopes)


def _default_service_factory(credentials):
    """Real YouTube Data API v3 service, built the same way
    ``youtube_uploader.YouTubeUploader._build_service`` does. A seam so
    tests never reach a real Google endpoint (see ``ServiceFactory``)."""
    from googleapiclient.discovery import build

    return build("youtube", "v3", credentials=credentials, cache_discovery=False)


def _fetch_channel_identity(
    credentials, service_factory: ServiceFactory, err
) -> tuple[Optional[str], Optional[str]]:
    """Best-effort ``channel_id``/``channel_title`` lookup via the same
    ``channels().list(mine=True)`` pattern ``youtube_adapter.py`` already
    uses for its own channel lookups. Never blocks the mint: on any failure
    (network, an unexpected API shape, a fake factory in a test) this WARNs
    to ``err`` -- with no token field or exception detail in the message,
    only the two public fields this function ever handles -- and returns
    ``(None, None)`` so the caller skips writing an identity sidecar.
    """
    try:
        service = service_factory(credentials)
        resp = service.channels().list(part="snippet", mine=True).execute()
        items = resp.get("items") or []
        if not items:
            return None, None
        channel_id = items[0].get("id")
        channel_title = (items[0].get("snippet") or {}).get("title")
        if not channel_id or not channel_title:
            return None, None
        return channel_id, channel_title
    except Exception:  # noqa: BLE001 — best-effort; doctor WARNs on a missing sidecar
        print(
            "Warning: could not fetch channel identity from YouTube; "
            "doctor will WARN on this token until a re-mint succeeds.",
            file=err,
        )
        return None, None


def run_youtube_auth(
    *,
    brand: Optional[str] = None,
    credentials_dir: Optional[Path] = None,
    dry_run: bool = False,
    flow_factory: Optional[FlowFactory] = None,
    service_factory: Optional[ServiceFactory] = None,
    out=None,
    err=None,
) -> int:
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr

    if credentials_dir is None:
        from ...pipeline._yt_creds import DEFAULT_CREDENTIALS_DIR

        credentials_dir = DEFAULT_CREDENTIALS_DIR
    creds_dir = Path(credentials_dir)

    scopes = [YOUTUBE_UPLOAD_SCOPE, YOUTUBE_COMMENT_SCOPE, YOUTUBE_ANALYTICS_SCOPE]
    enc_path = resolve_sops_file(_YOUTUBE_TOKEN_FILENAME, brand, creds_dir)

    if dry_run:
        print(f"Would write: {enc_path}", file=out)
        print(f"Scopes: {', '.join(scopes)}", file=out)
        return 0

    client_secret_path = _resolve_client_secret_path()
    client_config: Union[Path, dict]
    if client_secret_path.exists():
        client_config = client_secret_path
    else:
        # No downloaded client_secret.json on this box: borrow the OAuth client
        # the SHARED token was minted with (brand=None on purpose — a brand mint
        # runs under the same Google Cloud OAuth client, only the channel differs).
        try:
            client_config = _client_config_from_token(
                _load_shared_token(credentials_dir=creds_dir, brand=None)
            )
        except FileNotFoundError:
            if _config_manager_path_unset():
                # The launch directory is the cause, not a missing download.
                service_dir = _service_dir()
                rerun = (
                    f"Run from the service directory: cd {service_dir} && "
                    ".venv/bin/gomoveshift youtube-auth"
                    if service_dir is not None
                    else "Run gomoveshift youtube-auth from the project directory, "
                    "the one that holds config.yaml"
                )
                print(
                    f"Error: no config.yaml was loaded from {Path.cwd()} "
                    "(seo.manager_path is unset), so the client secret and the shared "
                    f"token were looked up relative to this directory. {rerun}",
                    file=err,
                )
                return 2
            print(
                f"Error: client secret not found at {client_secret_path} and no shared "
                "youtube_upload_token.json.enc to reuse the OAuth client from. "
                "Download it from https://console.cloud.google.com/apis/credentials "
                "and place it there before re-running.",
                file=err,
            )
            return 2
        except KeyError:
            print(
                f"Error: client secret not found at {client_secret_path} and the shared "
                "token carries no client_id/client_secret to reuse. Download it from "
                "https://console.cloud.google.com/apis/credentials and place it there.",
                file=err,
            )
            return 2
        print(
            f"client_secret.json not found at {client_secret_path}; "
            "reusing the OAuth client embedded in the shared upload token.",
            file=out,
        )

    factory = flow_factory or _default_flow_factory
    flow = factory(client_config, scopes)
    creds = flow.run_local_server(port=0)

    granted_scopes = list(creds.scopes or scopes)
    token_data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": granted_scopes,
    }
    # Channel identity is public, non-secret data -- fetched now (while the
    # freshly-minted credentials are in hand) but never mixed into
    # token_data: it is written to a plain JSON sidecar, never inside the
    # encrypted token, after the token itself is safely on disk below.
    channel_id, channel_title = _fetch_channel_identity(
        creds, service_factory or _default_service_factory, err
    )
    enc_path.parent.mkdir(parents=True, exist_ok=True)
    # A first brand mint has no file to inherit recipients from, so it takes
    # the shared token's; otherwise the pipeline config's catch-all rule would
    # encrypt it for keys the dev keyring cannot open. A re-mint of an existing
    # file inherits that file's recipients inside sops_encrypt_json.
    age_recipients = None
    if brand and not enc_path.exists():
        shared_enc = resolve_sops_file(_YOUTUBE_TOKEN_FILENAME, None, creds_dir)
        age_recipients = sops_age_recipients(shared_enc) or None
    sops_encrypt_json(token_data, enc_path, age_recipients=age_recipients)

    if channel_id and channel_title:
        write_channel_identity(enc_path, channel_id, channel_title)
        print(f"Channel: {channel_title} ({channel_id})", file=out)

    print(f"Wrote: {enc_path}", file=out)
    print(f"Scopes: {', '.join(granted_scopes)}", file=out)
    return 0


def cmd_youtube_auth(args, config):
    """Argparse entry point for ``gomoveshift youtube-auth``."""
    return run_youtube_auth(
        brand=getattr(args, "brand", None),
        credentials_dir=getattr(args, "credentials_dir", None),
        dry_run=getattr(args, "dry_run", False),
    )
