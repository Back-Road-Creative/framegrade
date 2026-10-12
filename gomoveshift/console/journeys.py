"""Console Journeys & site tab (minimal slice): lists open PRs on the
site repo and builds argv for the two read-only journal/site sync jobs the
CLI already exposes (`sync-site`, `publish-journal --replay-pending`). A
missing/failing `gh` never 500s -- it degrades to an empty PR list plus an
error message the template renders.

The site-sync job's own path follows the same tenant-boundary rule as the
Media library (``library.resolve_site_path``): a brand's ``GMS_SITE_PATH``
is config, not something an operator types into a form each run -- a
form-supplied path could name another brand's checkout.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from .library import resolve_site_path

__all__ = ["build_publish_journal_replay_argv", "build_sync_site_argv", "resolve_site_path"]


def site_repo() -> str:
    """``GMS_SITE_REPO`` (``owner/repo`` of the site `sync-site`/`publish-journal`
    write into; read at call time for testability). Empty when not configured --
    there is no default repo."""
    raw = os.environ.get("GMS_SITE_REPO")
    return raw.strip() if raw and raw.strip() else ""


def list_site_prs(repo: "str | None" = None, runner=subprocess.run) -> "tuple[list[dict], str]":
    """Open PRs on the site repo via ``gh pr list``. Never raises: a missing
    or failing ``gh``, or unparseable output, degrades to an empty list plus
    an error message."""
    repo = repo or site_repo()
    if not repo:
        return [], "GMS_SITE_REPO is not set; set it to the site's owner/repo to list its open PRs"
    try:
        result = runner(
            ["gh", "pr", "list", "--repo", repo, "--json", "number,title,url"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], f"gh unavailable: {exc}"
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "gh pr list failed").strip()
        return [], message
    try:
        prs = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return [], "gh returned unparseable output"
    return prs, ""


def build_sync_site_argv(site_path: str) -> list[str]:
    """``gomoveshift sync-site --site-path <site_path>`` -- dry-run by
    default (no ``--apply``), matching the parser's own default."""
    if not site_path or not site_path.strip():
        raise ValueError("site_path is required")
    return [sys.executable, "-m", "gomoveshift", "sync-site", "--site-path", site_path]


def build_publish_journal_replay_argv() -> list[str]:
    """``gomoveshift publish-journal --replay-pending`` -- dry-run by
    default (no ``--apply``); needs no operator-supplied path."""
    return [sys.executable, "-m", "gomoveshift", "publish-journal", "--replay-pending"]
