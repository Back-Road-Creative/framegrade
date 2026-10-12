"""Shared Meta (Instagram + Facebook) credential loader. Mirrors
``gomoveshift.pipeline._yt_creds.load_youtube_credentials``: SOPS-encrypted
JSON, sanitized stderr, credentials held in memory only. Both Meta adapters
import ``load_meta_credentials`` from here — neither owns a copy."""

from __future__ import annotations

import json
import logging
import re
import subprocess  # noqa: S404 — intentional sops subprocess
from pathlib import Path

from gomoveshift.config import get_config as _get_config
from gomoveshift.publishing.credential_paths import resolve_sops_file

logger = logging.getLogger(__name__)

# Same redaction pattern as ``pipeline._yt_creds`` — drops path/key lines.
_SOPS_SENSITIVE_LINE = re.compile(
    r"(?i)(?:\bage\b|age1[0-9a-z]+|recipient|identity|keys?\.txt|key file"
    r"|\.sops|(?:^|\s)[~/]\S+|[A-Za-z]:[\\/])"
)
_SOPS_REDACTED = "[sops stderr redacted — see DEBUG log]"

#: Used only when a credential record omits ``graph_version``.
DEFAULT_GRAPH_VERSION = "v21.0"
_META_CREDENTIALS_FILENAME = "meta_publishing.json.enc"
DEFAULT_CREDENTIALS_DIR = Path(_get_config().seo.credentials_dir)


def _sanitize_sops_stderr(stderr: str) -> str:
    if not stderr:
        return ""
    kept = [
        ln.strip()
        for ln in stderr.splitlines()
        if ln.strip() and not _SOPS_SENSITIVE_LINE.search(ln)
    ]
    return " ".join(kept) if kept else _SOPS_REDACTED


def load_meta_credentials(credentials_dir: Path | None = None, brand: str | None = None) -> dict:
    """Decrypt ``meta_publishing.json.enc`` via SOPS (fields: ``app_id``,
    ``app_secret``, ``page_id``, ``ig_user_id``, ``page_access_token``,
    ``token_type``, ``scopes``, ``user_token_expires_at``,
    ``data_access_expires_at`` (null or absent in older records),
    ``issued_at``, ``graph_version``); raises ``FileNotFoundError`` if
    undecryptable.

    ``brand`` routes through ``credential_paths.resolve_sops_file``: a brand
    with its own ``brands/<brand>/creds`` directory reads from there;
    ``brand=None`` (the default) is byte-for-byte the pre-brand shared path."""
    creds_dir = credentials_dir or DEFAULT_CREDENTIALS_DIR
    sops_file = resolve_sops_file(_META_CREDENTIALS_FILENAME, brand, creds_dir)
    if not sops_file.exists():
        raise FileNotFoundError(f"Meta credentials not found: {sops_file} does not exist.")
    result = subprocess.run(  # noqa: S603, S607 — fixed args, sops must be on PATH
        ["sops", "-d", "--output-type", "json", str(sops_file)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode == 0:
        logger.info("  Meta credentials loaded from SOPS")
        return json.loads(result.stdout)
    logger.debug("  SOPS decrypt failed — raw stderr: %s", result.stderr)
    raise FileNotFoundError(
        f"Meta credentials decrypt failed for {sops_file}: {_sanitize_sops_stderr(result.stderr)}"
    )
