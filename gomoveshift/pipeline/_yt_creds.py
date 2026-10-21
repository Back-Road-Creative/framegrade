"""
Shared YouTube credential loader.

Extracted from longform_metadata.LongFormMetadataGenerator._load_youtube_credentials
so both analytics sync and YouTube upload can share a single source of truth.

Credentials are never written to disk — decrypted content is returned as a dict
and held in memory only.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess  # noqa: S404 — intentional sops subprocess
import tempfile
from pathlib import Path
from typing import Optional

from gomoveshift.publishing.credential_paths import configured_base, resolve_sops_file

logger = logging.getLogger(__name__)

# Lines matching this pattern are dropped from any SOPS stderr before it
# reaches a user-visible exception message or a WARNING log. SOPS stderr can
# carry age-key file paths and ``.sops.yaml`` recipient rules — key material
# that must never be logged (raw tokens are never logged; the same
# spirit applies to key paths/recipients). The pattern targets actual leak
# vectors (filesystem paths, ``age1...`` recipients, ``.sops`` config refs,
# key-file/identity/recipient phrasing) WITHOUT swallowing the generic
# "data key required to decrypt" category line — note ``age\b`` (the age
# *backend*) only, so the word "data key" survives.
_SOPS_SENSITIVE_LINE = re.compile(
    r"(?i)(?:\bage\b|age1[0-9a-z]+|recipient|identity|keys?\.txt|key file"
    r"|\.sops|(?:^|\s)[~/]\S+|[A-Za-z]:[\\/])"
)
_SOPS_REDACTED = "[sops stderr redacted — see DEBUG log]"


def _sanitize_sops_stderr(stderr: str) -> str:
    """Strip key material from SOPS stderr before surfacing it.

    Drops every line referencing age keys, recipients, ``.sops.yaml`` config,
    or filesystem paths, keeping only generic category lines (e.g.
    ``"Failed to get the data key ..."``) so the message stays useful. When
    every line is sensitive a fixed placeholder is returned, so the caller
    still emits a non-empty, non-leaking signal. Empty input returns ``""``.

    The full, unredacted stderr is logged at DEBUG by callers for operators
    who need it; only the user-visible string is sanitized here.
    """
    if not stderr:
        return ""
    kept = [
        ln.strip()
        for ln in stderr.splitlines()
        if ln.strip() and not _SOPS_SENSITIVE_LINE.search(ln)
    ]
    if not kept:
        return _SOPS_REDACTED
    return " ".join(kept)


# Default credentials directory — resolved from the user's config.yaml
# ``seo.credentials_dir`` at import time. Previously this was sourced from
# ``SEOConfig()`` bare defaults (``./credentials``, cwd-relative), causing
# the SOPS file lookup at ``creds_dir.parent / "youtube_upload_token.json.enc"``
# to resolve to ``./youtube_upload_token.json.enc`` — a path that never
# existed and surfaced as ``exists=False`` in pipeline logs. The fix wires
# the import-time constant to the live ``get_config()`` value so the user's
# ``seo:`` block actually drives behaviour. Tests still monkeypatch this
# constant directly via ``conftest.isolate_seo_storage_from_workspace``.
DEFAULT_CREDENTIALS_DIR = configured_base()


def sops_age_recipients(enc_path: Path) -> list[str]:
    """Return the age recipients an existing SOPS file is encrypted for.

    Reads only the public ``sops.age[].recipient`` metadata — no key, no
    decrypt. Empty list when the file is missing, not JSON, or not a SOPS
    age file, so callers fall through to the config's creation rules.
    """
    try:
        doc = json.loads(Path(enc_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(doc, dict):
        return []
    entries = (doc.get("sops") or {}).get("age") or []
    out: list[str] = []
    for entry in entries:
        recipient = entry.get("recipient") if isinstance(entry, dict) else None
        if isinstance(recipient, str) and recipient and recipient not in out:
            out.append(recipient)
    return out


def channel_identity_sidecar_path(enc_path: Path) -> Path:
    """The plain-JSON sidecar next to a token's SOPS ``.enc`` file that holds
    the channel's PUBLIC, non-secret ``channel_id``/``channel_title`` --
    never written inside the encrypted token, never itself encrypted, so
    ``doctor`` and the console Settings page can read identity with no sops
    call and no network call (see ``read_channel_identity``)."""
    name = enc_path.name
    suffix = ".json.enc"
    stem = name[: -len(suffix)] if name.endswith(suffix) else enc_path.stem
    return enc_path.parent / f"{stem}.identity.json"


def read_channel_identity(enc_path: Path) -> Optional[dict]:
    """Offline read of the channel identity sidecar for ``enc_path`` -- no
    sops call, no network call, just a filesystem read.

    Returns ``None`` when the sidecar is missing, unreadable, not JSON, or
    missing either field (covers a partially written or pre-sidecar token,
    which ``doctor`` reports as WARN rather than crashing here).
    """
    path = channel_identity_sidecar_path(enc_path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    channel_id = doc.get("channel_id")
    channel_title = doc.get("channel_title")
    if not channel_id or not channel_title:
        return None
    return {"channel_id": channel_id, "channel_title": channel_title}


def write_channel_identity(enc_path: Path, channel_id: str, channel_title: str) -> Path:
    """Write the plain-JSON identity sidecar next to ``enc_path``.

    Atomic (temp file in the same directory + ``os.replace``) so a
    concurrent reader (``doctor``, the console) never observes a partial
    write. Contains only the two public fields -- never a token field.
    """
    path = channel_identity_sidecar_path(enc_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix=".identity-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"channel_id": channel_id, "channel_title": channel_title}, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    return path


def _unwrap_sops_binary(payload: dict) -> dict:
    """Return the real credentials from a sops BINARY-format decrypt.

    sops stores a file it did not parse as structured data under a single
    ``data`` key. Decrypting that with ``--output-type json`` (which this
    loader forces) yields ``{"data": "<the original json text>"}`` -- one key,
    no credentials -- so every caller dies on ``KeyError:
    'access_expires_at'`` while the credentials sit intact inside the string.

    A ``tiktok_publishing.json.enc`` file can be in that shape, and so is
    every token ``uploader.sops_encrypt_json`` writes:
    its ``--filename-override`` ends in ``.enc``, which sops encrypts as
    binary. A brand YouTube token minted that way fails its first refresh with
    "credentials do not contain the necessary fields". Both
    loaders unwrap here, where the format is known, rather than teaching every
    caller about it.

    Anything else is returned untouched -- including a payload that genuinely
    carries a ``data`` field alongside real keys.
    """
    if sorted(payload) != ["data"] or not isinstance(payload["data"], str):
        return payload
    try:
        inner = json.loads(payload["data"])
    except (TypeError, ValueError):
        return payload
    return inner if isinstance(inner, dict) else payload


def load_youtube_credentials(credentials_dir: Path | None = None, brand: str | None = None) -> dict:
    """
    Load YouTube OAuth credentials from available sources.

    Tries in order:
    1. SOPS-encrypted file (youtube_upload_token.json.enc in the credentials directory)
    2. BrandCredentials (Fernet-encrypted in credentials/<brand>/)

    Args:
        credentials_dir: Override the default credentials directory.
        brand: Routes source 1 through ``credential_paths.resolve_sops_file``
            and source 2's ``BrandCredentials`` name (default ``"gomoveshift"``
            when omitted). ``brand=None`` is byte-for-byte the pre-brand path.

    Returns:
        Dict with OAuth token fields (token, refresh_token, client_id, etc.)

    Raises:
        FileNotFoundError: If no credentials found anywhere.
    """
    creds_dir = credentials_dir or DEFAULT_CREDENTIALS_DIR

    # Source 1: SOPS-encrypted token in the credentials directory
    sops_file = resolve_sops_file("youtube_upload_token.json.enc", brand, creds_dir)
    if sops_file.exists():
        sops_commands = [
            ["sops", "-d", "--output-type", "json", str(sops_file)],
        ]
        # Optional second attempt as another OS user that owns the key file.
        # No default: set GMS_SOPS_SUDO_USER to enable it.
        sudo_user = os.environ.get("GMS_SOPS_SUDO_USER", "").strip()
        if sudo_user:
            sops_commands.append(
                ["sudo", "-u", sudo_user, "sops", "-d", "--output-type", "json", str(sops_file)]
            )
        for sops_cmd in sops_commands:
            try:
                result = subprocess.run(  # noqa: S603 — args are fixed, not user input
                    sops_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                if result.returncode == 0:
                    logger.info("  YouTube credentials loaded from SOPS")
                    return _unwrap_sops_binary(json.loads(result.stdout))
                # Each SOPS attempt failure is surfaced at WARNING so an
                # auth-blocker doesn't hide behind debug noise. A later
                # attempt may still succeed — the final "all attempts
                # failed" warning remains the definitive signal. The stderr
                # is sanitized first: raw SOPS stderr can carry age-key paths
                # / recipient rules; the full text goes to DEBUG only.
                logger.debug(
                    "  SOPS attempt failed (%s) — raw stderr: %s",
                    sops_cmd[0],
                    result.stderr.strip(),
                )
                logger.warning(
                    "  SOPS attempt failed (%s): %s — trying next source",
                    sops_cmd[0],
                    _sanitize_sops_stderr(result.stderr),
                )
            except FileNotFoundError:
                logger.warning("  SOPS binary %s not found — trying next source", sops_cmd[0])
                continue
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "  SOPS attempt error (%s): %s — trying next source",
                    sops_cmd[0],
                    e,
                )
                continue
        logger.warning("  All SOPS decrypt attempts failed — trying BrandCredentials")

    # Source 2: BrandCredentials (Fernet-encrypted)
    brand_slug = brand or "gomoveshift"
    gomoveshift_creds = creds_dir / brand_slug
    if gomoveshift_creds.exists():
        try:
            from seo_manager.security.brand_credentials import BrandCredentials

            bc = BrandCredentials(brand_slug, credentials_dir=creds_dir)
            logger.info("  YouTube credentials loaded from BrandCredentials")
            return bc.load_youtube_credentials()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  BrandCredentials load failed: {e}")

    raise FileNotFoundError(
        "YouTube credentials not found. Checked:\n"
        f"  1. SOPS: {sops_file} (exists={sops_file.exists()})\n"
        f"  2. BrandCredentials: {gomoveshift_creds}/\n"
        "YouTube credentials are MANDATORY for upload and analytics sync."
    )
