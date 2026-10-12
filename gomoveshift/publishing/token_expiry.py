"""Read-only token-expiry report for the daily digest. Wraps the
existing Meta/TikTok credential loaders and returns day counts and labels
only -- SECRETS RULE: never a token value, never raw SOPS stderr. A missing
credentials file or a decrypt failure is a ``"missing"``/``"unreadable"``
status row, never an exception; so is a date a row cannot parse, and a token
the record says never expires is a ``"never"`` row."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from gomoveshift.publishing.meta_auth import load_meta_credentials
from gomoveshift.publishing.tiktok_auth import load_tiktok_credentials

# Meta: an app's access to a user's data lasts 90 days from their last login,
# and Meta's list of permissions that "do not expire" names the Page ones but
# no instagram_* one -- so when it lapses Instagram publishing, comments and
# insights stop while Facebook keeps working.
# https://developers.facebook.com/docs/facebook-login/auth-vs-data
DATA_ACCESS = "Instagram data access"
_DATA_ACCESS_DAYS = 90
_DATA_ACCESS_WARN_DAYS = 14
_META_SETUP_FIX = (
    "re-create the Meta credential record (see docs/social-publishing-meta.md, Reconnecting)"
)
_TIKTOK_SETUP_FIX = (
    "re-create the TikTok credential record (see docs/social-publishing-tiktok.md, Reconnecting)"
)


def _status_for(exc: FileNotFoundError) -> str:
    return "missing" if "does not exist" in str(exc) else "unreadable"


def _parse(value: str) -> datetime | None:
    """ISO date or datetime -> aware datetime, else None. ``issued_at`` may be
    date-only, so a value without an offset is read as UTC."""
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _row(platform: str, what: str, expires_at: str | None, now: datetime, **hint) -> dict:
    """A dated row, or a ``missing``/``never``/``unreadable`` status row.
    ``hint`` (``warn_days``, ``fix``) rides on dated rows only."""
    if not expires_at:
        return {"platform": platform, "what": what, "status": "missing"}
    if expires_at == "never":
        return {"platform": platform, "what": what, "status": "never"}
    parsed = _parse(expires_at)
    if parsed is None:
        return {"platform": platform, "what": what, "status": "unreadable"}
    return {
        "platform": platform,
        "what": what,
        "expires_at": expires_at,
        "days_left": (parsed - now).days,
        **hint,
    }


def _data_access_row(creds: dict, now: datetime) -> dict:
    """The ``data_access_expires_at`` date in the record (from /debug_token),
    else the login date (``issued_at``) plus 90 days for records that lack it."""
    hint = {"warn_days": _DATA_ACCESS_WARN_DAYS, "fix": _META_SETUP_FIX}
    expires_at = creds.get("data_access_expires_at")
    if not expires_at and creds.get("issued_at"):
        issued = _parse(creds["issued_at"])
        if issued is None:
            return {"platform": "meta", "what": DATA_ACCESS, "status": "unreadable"}
        expires_at = (issued + timedelta(days=_DATA_ACCESS_DAYS)).isoformat()
    return _row("meta", DATA_ACCESS, expires_at, now, **hint)


def token_expiry_report(
    now: datetime | None = None, credentials_dir: Path | None = None
) -> list[dict]:
    """Meta ``user token`` + ``Instagram data access`` and TikTok ``refresh
    token`` rows. Each row is ``{"platform", "what", "expires_at",
    "days_left"}`` (plus ``warn_days``/``fix`` where the digest should say
    what to do), or ``{"platform", "what", "status": "missing"|"unreadable"
    |"never"}``. Never includes a token value. TikTok's 24 h access token has
    no row: ``tiktok_auth.get_valid_tiktok_credentials`` refreshes it on use."""
    now = now or datetime.now(timezone.utc)
    rows: list[dict] = []

    try:
        meta_creds = load_meta_credentials(credentials_dir)
    except FileNotFoundError as exc:
        status = _status_for(exc)
        rows.append({"platform": "meta", "what": "user token", "status": status})
        rows.append({"platform": "meta", "what": DATA_ACCESS, "status": status})
    else:
        rows.append(
            _row(
                "meta",
                "user token",
                meta_creds.get("user_token_expires_at"),
                now,
                fix=_META_SETUP_FIX,
            )
        )
        rows.append(_data_access_row(meta_creds, now))

    try:
        tiktok_creds = load_tiktok_credentials(credentials_dir)
    except FileNotFoundError as exc:
        rows.append({"platform": "tiktok", "what": "refresh token", "status": _status_for(exc)})
    else:
        rows.append(
            _row(
                "tiktok",
                "refresh token",
                tiktok_creds.get("refresh_expires_at"),
                now,
                fix=_TIKTOK_SETUP_FIX,
            )
        )

    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Meta/TikTok token expiry report (JSON rows).")
    parser.add_argument("--credentials-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    print(json.dumps(token_expiry_report(credentials_dir=args.credentials_dir)))


if __name__ == "__main__":
    main()
