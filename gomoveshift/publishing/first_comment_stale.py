"""A first comment found live with a replaced text: the ``first_comment_stale`` health incident
(shown by ``status --json``) and the automatic in-place fix, run at the start of each drain."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .queue import (
    fail_first_comment_correction,
    first_comment_correction_ids,
    first_comment_corrections,
    pending_first_comment_corrections,
    resolve_first_comment_correction,
)

logger = logging.getLogger(__name__)

KIND = "first_comment_stale"
_DETAIL_MAX = 500


class CorrectionUnavailableError(Exception):
    """No client call for this correction, or the live comment id is unknown."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _key(delivery_id: int) -> str:
    return f"{KIND}:{delivery_id}"


def _describe(row: Any) -> str:
    steps = []
    for item in first_comment_corrections(row):
        if item.get("action") == "edit":
            steps.append(f"edit comment {item.get('live_id') or '?'} to {item.get('to')!r}")
        else:
            steps.append(f"delete duplicate comment {item.get('live_id') or '?'}")
    return (
        f"{row['platform']}#{row['id']} {row['project_name']} short {row['short_index']}: "
        f"{row['first_comment_error']}; the drain will {'; '.join(steps)}"
    )[:_DETAIL_MAX]


def raise_stale_incident(
    conn: sqlite3.Connection, delivery_id: int, incidents_dir: Optional[Path] = None
) -> None:
    """File (or refresh) the one open incident for ``delivery_id``. Never fatal."""
    from ..health.store import (  # lazy: health imports the pipeline
        default_data_root,
        default_incidents_dir,
        new_incident,
        safe_name,
        write_row,
    )

    row = conn.execute(
        "SELECT id, project_name, short_index, platform, first_comment_error, "
        "first_comment_correction FROM deliveries WHERE id=?",
        (delivery_id,),
    ).fetchone()
    if row is None or not first_comment_corrections(row):
        return
    directory = incidents_dir or default_incidents_dir(default_data_root())
    filename = f"{safe_name(_key(delivery_id))}.json"
    detail = _describe(row)
    try:
        path = directory / filename
        prior = json.loads(path.read_text()) if path.is_file() else None
        if isinstance(prior, dict) and not prior.get("resolved_at"):
            prior.update(detail=detail, error=row["first_comment_error"])
            write_row(directory, prior)
            return
        write_row(
            directory,
            new_incident(
                KIND,
                _key(delivery_id),
                filename,
                _now_iso(),
                delivery_id=delivery_id,
                platform=row["platform"],
                project=row["project_name"],
                live_comments=first_comment_corrections(row),
                error=row["first_comment_error"],
                detail=detail,
            ),
        )
    except (OSError, ValueError):
        logger.exception(
            "first comment: could not write the stale-comment incident %s", delivery_id
        )


def _resolve_incidents(
    platforms: Iterable[str], still_pending: set, incidents_dir: Optional[Path]
) -> None:
    """Resolve open incidents of ``platforms`` whose delivery has no correction left."""
    from ..health.store import default_data_root, default_incidents_dir, load_all, write_row

    wanted = set(platforms)
    directory = incidents_dir or default_incidents_dir(default_data_root())
    try:
        rows, _ = load_all(directory)
        for row in rows:
            if (
                row.get("kind") == KIND
                and not row.get("resolved_at")
                and row.get("platform") in wanted
                and row.get("delivery_id") not in still_pending
            ):
                row["resolved_at"] = _now_iso()
                write_row(directory, row)
    except OSError:
        logger.exception("first comment: could not resolve stale-comment incidents")


def run_corrections(
    conn: sqlite3.Connection,
    *,
    platforms: Iterable[str],
    client_for: Callable[[str], object],
    refused: Callable[[Exception], bool],
    result: Any,
    incidents_dir: Optional[Path] = None,
) -> None:
    """Apply every pending correction through ``client_for(platform)`` (built lazily; has
    ``edit_comment(id, text)`` / ``delete_comment(id)``). ``result`` gains ``corrected`` per fix
    and ``stale`` plus an error line per failure; ``refused(exc)`` counts toward the cap."""
    platforms = tuple(platforms)
    _resolve_incidents(platforms, first_comment_correction_ids(conn, platforms), incidents_dir)
    for row in pending_first_comment_corrections(conn, platforms):
        for item in first_comment_corrections(row):
            try:
                client = client_for(row["platform"])
                call = getattr(client, f"{item.get('action')}_comment", None)
                if call is None or not item.get("live_id"):
                    raise CorrectionUnavailableError(
                        f"{row['platform']} cannot {item.get('action')} comment "
                        f"{item.get('live_id') or '(id unknown)'}"
                    )
                if item["action"] == "edit":
                    call(item["live_id"], item["to"])
                else:
                    call(item["live_id"])
            except Exception as exc:  # noqa: BLE001 -- graded per correction
                detail = str(exc).strip() or exc.__class__.__name__
                fail_first_comment_correction(
                    conn,
                    row["id"],
                    detail,
                    refused=refused(exc) and not isinstance(exc, CorrectionUnavailableError),
                )
                result.stale += 1
                result.errors.append(f"delivery {row['id']}: stale comment not corrected: {detail}")
                logger.warning(
                    "first comment: delivery %s correction failed: %s", row["id"], detail
                )
                raise_stale_incident(conn, row["id"], incidents_dir)
                continue
            resolve_first_comment_correction(conn, row["id"], item, _now_iso())
            result.corrected += 1
    _resolve_incidents(platforms, first_comment_correction_ids(conn, platforms), incidents_dir)
