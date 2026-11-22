"""Relink a drive's queued first comments and sidecars to a replacement long-form upload.

Replacing a still-scheduled long-form video uploads the re-render as a NEW video and deletes
the old one, so the old id is dead. Four places still name it:

- the queue's ``deliveries.first_comment_text`` ("Watch the full drive: https://youtu.be/<id>");
- the drive's ``related_video_worklist_state.json`` (operator lines);
- the drive's ``<project>_publish_plan.json`` (``parent_lf_video_id`` and URLs in the text);
- the drive's ``pending_journal.json`` (``youtube_id`` of an entry not flushed yet).

:func:`relink_replaced_lf` rewrites exactly those, old id -> new id, and nothing else: it never
touches the journal site, its API or KV, or any /journal-photo/ file. Queue rows whose first
comment is already posted, or whose status is terminal / ``awaiting_manual_publish``, are left
alone and listed with the reason. Every file write is atomic; a second run finds nothing to
change and reports zero; ``dry_run`` reports without writing.

``pending_journal.json`` records no "sent" flag: the replay flush REMOVES an entry once written
to the site, so every entry still in the file has not been sent and is rewritten.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from gomoveshift.pipeline._atomic_write import atomic_write_text
from gomoveshift.pipeline.pending_journal import (
    PENDING_FILENAME,
    load_pending,
    pending_lock,
    save_pending,
)
from gomoveshift.publishing import queue as _queue

WORKLIST_STATE_FILENAME = "related_video_worklist_state.json"
_ID_CHAR = "A-Za-z0-9_-"


@dataclass(frozen=True)
class QueueRewrite:
    delivery_id: int
    short_index: int
    platform: str
    status: str
    written: bool


@dataclass(frozen=True)
class QueueSkip:
    delivery_id: int
    short_index: int
    platform: str
    status: str
    reason: str


@dataclass(frozen=True)
class FileRewrite:
    path: str
    replacements: int
    written: bool
    note: str = ""


@dataclass
class RelinkReport:
    project: str
    dry_run: bool
    queue_rewritten: List[QueueRewrite] = field(default_factory=list)
    queue_skipped: List[QueueSkip] = field(default_factory=list)
    files: List[FileRewrite] = field(default_factory=list)

    @property
    def changes(self) -> int:
        """Rewrites made (or, in a dry run, that would be made): queue rows plus file edits."""
        return len(self.queue_rewritten) + sum(f.replacements for f in self.files)


def _validate(id_map: Mapping[str, str]) -> None:
    if not id_map:
        raise ValueError("id_map is empty: nothing to relink")
    for old, new in id_map.items():
        if not old or not new:
            raise ValueError(f"id_map holds an empty id ({old!r} -> {new!r})")
        if old == new:
            raise ValueError(f"old id equals new id ({old!r}): refusing a no-op relink")
    chained = sorted(set(id_map) & set(id_map.values()))
    if chained:
        raise ValueError(f"id_map chains (a new id is also an old id): {chained}")


def _pattern(id_map: Mapping[str, str]) -> "re.Pattern[str]":
    alt = "|".join(re.escape(o) for o in sorted(id_map, key=len, reverse=True))
    return re.compile(rf"(?<![{_ID_CHAR}])(?:{alt})(?![{_ID_CHAR}])")


def _sub(text: str, pat: "re.Pattern[str]", id_map: Mapping[str, str]) -> Tuple[str, int]:
    return pat.subn(lambda m: id_map[m.group(0)], text)


def _walk(node: Any, pat, id_map) -> Tuple[Any, int]:
    if isinstance(node, str):
        return _sub(node, pat, id_map)
    if isinstance(node, list):
        out, n = [], 0
        for item in node:
            new, k = _walk(item, pat, id_map)
            out.append(new)
            n += k
        return out, n
    if isinstance(node, dict):
        out_d, n = {}, 0
        for key, item in node.items():
            new, k = _walk(item, pat, id_map)
            out_d[key] = new
            n += k
        return out_d, n
    return node, 0


def _relink_json_file(path: Path, pat, id_map, *, dry_run: bool, sort_keys: bool) -> FileRewrite:
    if not path.is_file():
        return FileRewrite(str(path), 0, False, "absent")
    raw = json.loads(path.read_text(encoding="utf-8"))
    new, count = _walk(raw, pat, id_map)
    if count == 0:
        return FileRewrite(str(path), 0, False)
    if not dry_run:
        atomic_write_text(path, json.dumps(new, sort_keys=sort_keys, indent=2) + "\n")
    return FileRewrite(str(path), count, not dry_run)


def _relink_pending(path: Path, id_map, *, dry_run: bool) -> FileRewrite:
    if not path.is_file():
        return FileRewrite(str(path), 0, False, "absent")
    if dry_run:  # a read only: the lock file would be the one thing a dry run left behind
        count = sum(e.youtube_id in id_map for e in load_pending(path))
    else:
        with pending_lock(path):
            entries = load_pending(path)
            count = 0
            for entry in entries:
                if entry.youtube_id in id_map:
                    entry.youtube_id = id_map[entry.youtube_id]
                    count += 1
            if count:
                save_pending(path, entries)
    note = "no sent flag is recorded; every entry still in the file is unsent" if count else ""
    return FileRewrite(str(path), count, bool(count) and not dry_run, note)


def _relink_queue(conn, project, pat, id_map, now, report) -> None:
    rows = _queue.first_comments_naming(conn, project, id_map)
    for row in rows:
        text = row["first_comment_text"]
        new_text, n = _sub(text, pat, id_map)
        if n == 0:  # the old id only inside a longer token
            continue
        base = (row["id"], row["short_index"], row["platform"], row["status"])
        if row["first_comment_posted_at"] is not None:
            reason = "first comment already posted"
        elif row["status"] in _queue.FIRST_COMMENT_FROZEN_STATUSES:
            reason = f"status {row['status']}"
        else:
            written = False
            if not report.dry_run:
                written = _queue.rewrite_unposted_first_comment(
                    conn, row["id"], text, new_text, now.isoformat()
                )
                if not written:
                    report.queue_skipped.append(
                        QueueSkip(*base, "row changed while relinking; rerun")
                    )
                    continue
            report.queue_rewritten.append(QueueRewrite(*base, written))
            continue
        report.queue_skipped.append(QueueSkip(*base, reason))


def relink_replaced_lf(
    project: str,
    output_dir: Path,
    id_map: Dict[str, str],
    *,
    conn: sqlite3.Connection,
    now: datetime,
    dry_run: bool = False,
) -> RelinkReport:
    """Rewrite every old long-form id in ``id_map`` to its replacement (see the module doc).

    Raises ``ValueError`` on an empty map, an empty id, an old id equal to its new id, or a new
    id that is also an old id (a chain would break idempotence). Idempotent: once rewritten,
    the old ids are gone, so a rerun reports zero changes and writes nothing."""
    _validate(id_map)
    out = Path(output_dir)
    pat = _pattern(id_map)
    report = RelinkReport(project=project, dry_run=dry_run)
    _relink_queue(conn, project, pat, id_map, now, report)
    report.files.append(
        _relink_json_file(
            out / WORKLIST_STATE_FILENAME, pat, id_map, dry_run=dry_run, sort_keys=False
        )
    )
    report.files.append(
        _relink_json_file(
            out / f"{project}_publish_plan.json", pat, id_map, dry_run=dry_run, sort_keys=True
        )
    )
    report.files.append(_relink_pending(out / PENDING_FILENAME, id_map, dry_run=dry_run))
    return report
