"""``gomoveshift day-reels [--apply]`` and ``day-reels authorize --folder-id <id>``.

``authorize`` is the one-time read-only Drive sign-in; the token is SOPS-encrypted and
never printed. See ``docs/social-publishing-queue.md`` ("Day reels from Drive").
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ...paths import gms_data_home
from ...publishing.day_reels import (
    DriveReadError,
    Outcome,
    RcloneDrive,
    parse_authorize_output,
    process_requests,
)

__all__ = ["ClientUnavailableError", "Outcome", "cmd_day_reels", "credentials_path"]

_FOLDER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{9,127}")


def credentials_path() -> Path:
    """``GMS_DAY_REELS_DRIVE_ENC``, else ``day_reels_drive.json.enc`` beside
    the pipeline's Drive token (same directory, same keyring)."""
    raw = os.environ.get("GMS_DAY_REELS_DRIVE_ENC", "").strip()
    if raw:
        return Path(raw).expanduser()
    from ...pipeline import uploader

    return Path(uploader.DEFAULT_TOKEN_ENC).parent / "day_reels_drive.json.enc"


def _decrypt(path: Path) -> dict:
    from ...pipeline.uploader import sops_decrypt_json

    return sops_decrypt_json(path)


def _encrypt(data: dict, path: Path) -> None:
    from ...pipeline import uploader
    from ...pipeline._yt_creds import sops_age_recipients

    recipients = sops_age_recipients(Path(uploader.DEFAULT_TOKEN_ENC)) or None
    uploader.sops_encrypt_json(data, path, age_recipients=recipients)


class ClientUnavailableError(RuntimeError):
    """The pipeline's own Google OAuth client could not be read."""


def _pipeline_client() -> tuple:
    """``(client_id, client_secret)`` of the pipeline's own Google client, read from its Drive
    token (``GMS_DRIVE_TOKEN_ENC``). Signing in through it keeps day reels off rclone's
    built-in client, whose quota every rclone user shares."""
    from ...pipeline import uploader

    try:
        data = _decrypt(Path(uploader.DEFAULT_TOKEN_ENC))
    except Exception as exc:  # unreadable for any reason: refuse, never fall back
        raise ClientUnavailableError(
            f"pipeline Drive token unreadable ({type(exc).__name__})"
        ) from None
    client = (str(data.get("client_id") or ""), str(data.get("client_secret") or ""))
    if not all(client):
        raise ClientUnavailableError("pipeline Drive token has no client_id/client_secret")
    return client


def _authorize(args) -> int:
    folder_id = args.folder_id or ""
    if not _FOLDER_ID.fullmatch(folder_id):
        print("Error: --folder-id must be the Drive folder id of the day-reels request folder.")
        return 2
    # rclone decodes the blob as URL-safe, unpadded base64 (padding is "illegal").
    blob = json.dumps({"scope": "drive.readonly"}).encode()
    scope = base64.urlsafe_b64encode(blob).decode().rstrip("=")
    try:
        client_id, client_secret = _pipeline_client()
    except ClientUnavailableError as exc:
        print(
            f"Error: cannot sign in through the pipeline's own Google client: {exc}; nothing was saved."
        )
        return 1
    # The client travels in the child's environment, never argv (rclone reads RCLONE_DRIVE_*).
    env = dict(
        os.environ, RCLONE_DRIVE_CLIENT_ID=client_id, RCLONE_DRIVE_CLIENT_SECRET=client_secret
    )
    print("Open the link rclone prints below, choose the Google account, and tap Allow.")
    result = subprocess.run(
        ["rclone", "authorize", "drive", scope, "--auth-no-open-browser"],
        stdout=subprocess.PIPE,
        text=True,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        print(f"Error: rclone authorize exited {result.returncode}; nothing was saved.")
        return 1
    try:
        token = parse_authorize_output(result.stdout)
    except ValueError as exc:
        print(f"Error: {exc}; nothing was saved.")
        return 1
    path = credentials_path()
    # The token only refreshes through the client that issued it, so the pair is stored together.
    _encrypt(
        {
            "rclone_token": token,
            "folder_id": folder_id,
            "client_id": client_id,
            "client_secret": client_secret,
        },
        path,
    )
    print(f"Saved the read-only Drive sign-in (encrypted) to {path}")
    return 0


def cmd_day_reels(args, config) -> int:
    if args.action == "authorize":
        return _authorize(args)

    path = credentials_path()
    if not path.exists():
        print(
            f"DAY_REELS skipped: no Drive sign-in at {path} (run: gomoveshift day-reels authorize)"
        )
        return 0
    creds = _decrypt(path)
    drive = RcloneDrive(
        folder_id=creds["folder_id"],
        token=creds["rclone_token"],
        client_id=creds.get("client_id", ""),
        client_secret=creds.get("client_secret", ""),
    )
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    try:
        outcomes = process_requests(
            drive,
            state_dir=gms_data_home() / "day_reels",
            storage_path=Path(config.seo.storage_path),
            apply=bool(args.apply),
            now_utc=now,
        )
    except DriveReadError as exc:
        print(f"DAY_REELS failed: {exc}")
        return 1
    if not outcomes:
        print("DAY_REELS no requests")
    for outcome in outcomes:
        ids = ",".join(str(i) for i in outcome.delivery_ids)
        suffix = f" ids={ids}" if ids else ""
        print(f"DAY_REEL {outcome.state} {outcome.name}: {outcome.reason}{suffix}")
    return 0
