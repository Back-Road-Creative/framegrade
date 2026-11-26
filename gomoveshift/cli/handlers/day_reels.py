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

__all__ = ["Outcome", "cmd_day_reels", "credentials_path"]

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


def _authorize(args) -> int:
    folder_id = args.folder_id or ""
    if not _FOLDER_ID.fullmatch(folder_id):
        print("Error: --folder-id must be the Drive folder id of the day-reels request folder.")
        return 2
    # rclone decodes the blob as URL-safe, unpadded base64 (padding is "illegal").
    blob = json.dumps({"scope": "drive.readonly"}).encode()
    scope = base64.urlsafe_b64encode(blob).decode().rstrip("=")
    print("Open the link rclone prints below, choose the Google account, and tap Allow.")
    result = subprocess.run(
        ["rclone", "authorize", "drive", scope, "--auth-no-open-browser"],
        stdout=subprocess.PIPE,
        text=True,
        check=False,
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
    _encrypt({"rclone_token": token, "folder_id": folder_id}, path)
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
    drive = RcloneDrive(folder_id=creds["folder_id"], token=creds["rclone_token"])
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
