"""TikTok credential loading + rotated-refresh-token persistence (mirrors
``gomoveshift.publishing.meta_auth``); written back atomically under a lock."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import subprocess  # noqa: S404 — intentional sops subprocess
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import yaml

from gomoveshift.pipeline._file_lock import make_blocking_file_lock
from gomoveshift.pipeline._yt_creds import _unwrap_sops_binary, sops_age_recipients
from gomoveshift.publishing.credential_paths import configured_base, resolve_sops_file

logger = logging.getLogger(__name__)

_SOPS_SENSITIVE_LINE = re.compile(
    r"(?i)(?:\bage\b|age1[0-9a-z]+|recipient|identity|keys?\.txt|key file"
    r"|\.sops|(?:^|\s)[~/]\S+|[A-Za-z]:[\\/])"
)
_SOPS_REDACTED = "[sops stderr redacted — see DEBUG log]"

_TIKTOK_CREDENTIALS_FILENAME = "tiktok_publishing.json.enc"
DEFAULT_CREDENTIALS_DIR = configured_base()

TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"  # noqa: S105 — endpoint, not a secret

ACCESS_TOKEN_REFRESH_MARGIN = timedelta(minutes=30)
_REAUTH_HINT = (
    "re-create the TikTok credential record (see docs/social-publishing-tiktok.md, Reconnecting)"
)

_lock_state = threading.local()
_file_lock = make_blocking_file_lock(_lock_state)


class TikTokAuthError(RuntimeError):
    """Credential load/refresh failure — message is always generic."""


def _run_sops(args: list) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603, S607 — fixed args, sops must be on PATH
        ["sops", *args], capture_output=True, text=True, timeout=30, check=False
    )


def _age_recipients_for(sops_file: Path) -> str | None:
    """Read the age recipients that govern ``sops_file`` from the
    ``.sops.yaml`` creation rules, sitting alongside it. Returns ``None``
    if the config or a matching rule can't be found.

    Config-rule fallback ONLY — a brand-new file with no prior ciphertext
    has no real recipients to preserve yet. A file that already exists
    must never take this path: see ``get_valid_tiktok_credentials``, which
    prefers the file's own real recipients (``sops_age_recipients``) so a
    ``.sops.yaml`` regex miss can't drop them to a wrong/catch-all key
    (the shape ``uploader.sops_encrypt_json`` already fixed for YouTube
    tokens).

    Read fresh every call rather than hardcoded: a hardcoded recipient
    list silently rots the day the store is rekeyed.
    """
    sops_yaml = sops_file.parent / ".sops.yaml"
    try:
        rules = yaml.safe_load(sops_yaml.read_text()).get("creation_rules", [])
    except (OSError, yaml.YAMLError, AttributeError):
        return None
    # ``.sops.yaml`` rules match the un-suffixed ``*.json`` name the store was
    # first encrypted under (e.g. ``\.json$``) — ``.stem`` strips only the
    # outer ``.enc``, leaving that name for the regex to match against.
    match_name = sops_file.stem if sops_file.suffix == ".enc" else sops_file.name
    for rule in rules:
        pattern = rule.get("path_regex")
        age = rule.get("age")
        if pattern and age and re.search(pattern, match_name):
            return str(age).strip()
    return None


def _sops_error(action: str, stderr: str) -> str:
    logger.debug("  TikTok %s failed — raw stderr: %s", action, stderr)
    if not stderr:
        return ""
    kept = [
        ln.strip()
        for ln in stderr.splitlines()
        if ln.strip() and not _SOPS_SENSITIVE_LINE.search(ln)
    ]
    return " ".join(kept) if kept else _SOPS_REDACTED


def load_tiktok_credentials(credentials_dir: Path | None = None, brand: str | None = None) -> dict:
    """Decrypt ``tiktok_publishing.json.enc``; raises ``FileNotFoundError`` if missing/bad.

    ``brand`` routes through ``credential_paths.resolve_sops_file``;
    ``brand=None`` (the default) is byte-for-byte the pre-brand shared path."""
    creds_dir = credentials_dir or DEFAULT_CREDENTIALS_DIR
    sops_file = resolve_sops_file(_TIKTOK_CREDENTIALS_FILENAME, brand, creds_dir)
    if not sops_file.exists():
        raise FileNotFoundError(f"TikTok credentials not found: {sops_file} does not exist.")
    result = _run_sops(["-d", "--output-type", "json", str(sops_file)])
    if result.returncode == 0:
        logger.info("  TikTok credentials loaded from SOPS")
        return _unwrap_sops_binary(json.loads(result.stdout))
    raise FileNotFoundError(
        f"TikTok credentials decrypt failed for {sops_file}: "
        f"{_sops_error('decrypt', result.stderr)}"
    )


def get_valid_tiktok_credentials(
    credentials_dir: Path | None = None, brand: str | None = None
) -> dict:
    """Load credentials, refresh + atomically persist a rotated token once
    near ``access_expires_at``; raises past ``refresh_expires_at``.

    ``brand=None`` (the default) is byte-for-byte the pre-brand shared path."""
    creds_dir = credentials_dir or DEFAULT_CREDENTIALS_DIR
    sops_file = resolve_sops_file(_TIKTOK_CREDENTIALS_FILENAME, brand, creds_dir)
    creds = load_tiktok_credentials(creds_dir, brand=brand)
    now = datetime.now(timezone.utc)
    if now < datetime.fromisoformat(creds["access_expires_at"]) - ACCESS_TOKEN_REFRESH_MARGIN:
        return creds
    if now >= datetime.fromisoformat(creds["refresh_expires_at"]):
        raise TikTokAuthError(f"TikTok refresh token has expired — {_REAUTH_HINT} to re-authorize.")
    try:
        resp = requests.post(
            TOKEN_URL,
            data={
                "client_key": creds["client_key"],
                "client_secret": creds["client_secret"],
                "grant_type": "refresh_token",
                "refresh_token": creds["refresh_token"],
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        access_token = data["access_token"]
        expires_in = int(data["expires_in"])
    except (requests.RequestException, KeyError, ValueError, TypeError):
        raise TikTokAuthError("TikTok token refresh request failed") from None
    updated = dict(creds)
    updated["access_token"] = access_token
    updated["access_expires_at"] = (now + timedelta(seconds=expires_in)).isoformat()
    rotated_refresh = data.get("refresh_token")
    if rotated_refresh:
        updated["refresh_token"] = rotated_refresh
    refresh_expires_in = data.get("refresh_expires_in")
    if refresh_expires_in is not None:
        updated["refresh_expires_at"] = (
            now + timedelta(seconds=int(refresh_expires_in))
        ).isoformat()
    sops_file.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(sops_file):
        fd, tmp = tempfile.mkstemp(dir=str(sops_file.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(updated, f)
            # The file's own real recipients (already-encrypted-for) win over
            # a `.sops.yaml` creation-rule guess — the same priority
            # `uploader.sops_encrypt_json` uses. Only a brand-new file with
            # no prior ciphertext (no real recipients yet) falls back to the
            # config rule.
            existing_recipients = sops_age_recipients(sops_file)
            recipients = (
                ",".join(existing_recipients)
                if existing_recipients
                else (_age_recipients_for(sops_file))
            )
            # SOPS still discovers .sops.yaml with --age: a creation rule
            # rejecting the random .tmp name aborts before those explicit
            # recipients are used. Once readers are resolved, isolate that
            # encryption from filename rules, including SOPS_CONFIG in the
            # caller's environment. Keep config discovery for the no-reader
            # fallback; never invent or drop recipients.
            encrypt_args = ["--config", os.devnull, "--age", recipients] if recipients else []
            result = _run_sops([*encrypt_args, "-e", "-i", tmp])
            if result.returncode != 0:
                raise TikTokAuthError(
                    f"TikTok credential re-encrypt failed: {_sops_error('encrypt', result.stderr)}"
                )
            os.replace(tmp, sops_file)
        except BaseException:
            with contextlib.suppress(FileNotFoundError, OSError):
                os.unlink(tmp)
            raise
    return updated
