"""Relink a drive's queued first comments and sidecars to a replacement long-form upload.

The old video id is dead after a replace, yet four places name it: the queue's
``first_comment_text``, ``related_video_worklist_state.json``, ``<project>_publish_plan.json``
and ``pending_journal.json``. :func:`relink_replaced_lf` rewrites old -> new in those and
nowhere else: never the journal site / API / KV / photos, never a live comment. File stores run
first, the queue last, each in its own ``try`` (failures land in ``RelinkReport.errors``).
See docs/social-publishing-queue.md for each store's rule.

Reported as ``operator_items``, never rewritten: a posted comment still linking the old id, other
projects' rows naming an old id, a rewritten plan (Gate C/D must be re-confirmed) and journeys to
check by hand. A stale comment a drain found live is raised by the drain itself (a
``first_comment_stale`` health incident, corrected in place on the next pass), never only by a
relink rerun; a rewritten row of a published short lands in ``drain_will_verify``. A rerun
reports zero changes; ``dry_run`` reports without writing.

The #517 parent-public gate holds for a relink too. ``id_map`` is ``{old_id: (new_id,
new_publish_at_iso_utc)}``: the replacement goes public at that time (a scheduled upload is
private until then), and ``parent_is_public_at(<the row's slot>, new_publish_at)`` must hold
before a row links it. A row that fails drops its link (an unposted row's text becomes NULL, as
``enqueue`` does) and is reported; a live comment with a pending edit gets a ``delete`` instead.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from gomoveshift.pipeline._atomic_write import atomic_write_json
from gomoveshift.pipeline.pending_journal import (
    PENDING_FILENAME,
    load_pending,
    pending_lock,
    save_pending,
)
from gomoveshift.publishing import queue as _queue
from gomoveshift.publishing.first_comment import _aware, parent_is_public_at
from gomoveshift.publishing.plan_io import load_publish_plan
from gomoveshift.publishing.queue_short_text import SidecarChangedError

WORKLIST_STATE_FILENAME = "related_video_worklist_state.json"
CHECKLIST_FILENAME = "operator_checklist.txt"
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
    #: Things only a human can settle (a live comment that still links the old id, rows of
    #: other projects, a Gate C/D to re-confirm, journey files to check).
    operator_items: List[str] = field(default_factory=list)
    #: Rewritten rows whose comment may already be live with the old text (the short is
    #: published): the next first-comment drain looks, and on a hit corrects it and files a
    #: ``first_comment_stale`` incident. Nothing for a human to do.
    drain_will_verify: List[str] = field(default_factory=list)
    #: One line per store that failed; the other stores' results above are still true.
    errors: List[str] = field(default_factory=list)

    @property
    def changes(self) -> int:
        """Rewrites made (or, in a dry run, that would be made): queue rows plus file edits."""
        return len(self.queue_rewritten) + sum(f.replacements for f in self.files)


def _split(id_map: Mapping[str, Tuple[str, str]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """``({old: new}, {old: new_publish_at})`` after validating ``id_map``. Raises ``TypeError``
    for a value that is not an ``(new_id, publish_at)`` pair (a bare string included) and
    ``ValueError`` for an empty map or id, a no-op or chained relink, or a missing, unparseable
    or unzoned publish time (an unknown time cannot be ordered, so it never links)."""
    if not id_map:
        raise ValueError("id_map is empty: nothing to relink")
    ids: Dict[str, str] = {}
    times: Dict[str, str] = {}
    for old, value in id_map.items():
        if isinstance(value, str) or not isinstance(value, tuple) or len(value) != 2:
            raise TypeError(
                f"id_map[{old!r}] must be (new_video_id, new_publish_at_iso_utc), got {value!r}"
            )
        new, at = value
        if not old or not new:
            raise ValueError(f"id_map holds an empty id ({old!r} -> {new!r})")
        if old == new:
            raise ValueError(f"old id equals new id ({old!r}): refusing a no-op relink")
        if _aware(at) is None:
            raise ValueError(f"id_map[{old!r}] has no usable publish time ({at!r}): refusing")
        ids[old], times[old] = new, at
    chained = sorted(set(ids) & set(ids.values()))
    if chained:
        raise ValueError(f"id_map chains (a new id is also an old id): {chained}")
    return ids, times


def _alt(id_map: Mapping[str, str]) -> str:
    return "|".join(re.escape(o) for o in sorted(id_map, key=len, reverse=True))


def _pattern(id_map: Mapping[str, str]) -> "re.Pattern[str]":
    return re.compile(rf"(?<![{_ID_CHAR}])(?:{_alt(id_map)})(?![{_ID_CHAR}])")


def _url_pattern(id_map: Mapping[str, str]) -> "re.Pattern[str]":
    """An old id only where a URL names it: ``youtu.be/<id>``, ``watch?v=<id>``, ``/shorts/<id>``."""
    return re.compile(rf"(?P<ctx>youtu\.be/|[?&]v=|/shorts/)(?P<id>{_alt(id_map)})(?![{_ID_CHAR}])")


def _sub(text: str, pat: "re.Pattern[str]", id_map: Mapping[str, str]) -> Tuple[str, int]:
    return pat.subn(lambda m: id_map[m.group(0)], text)


def _walk_plan(node: Any, url_pat, id_map, key: str = "", times=None) -> Tuple[Any, int]:
    """Rewrite ``parent_lf_video_id`` (and, beside it, ``parent_lf_publish_at``) and URL-context
    ids; every other string is left alone."""
    if isinstance(node, str):
        if key == "parent_lf_video_id" and node in id_map:
            return id_map[node], 1
        return url_pat.subn(lambda m: m.group("ctx") + id_map[m.group("id")], node)
    if isinstance(node, list):
        got = [_walk_plan(item, url_pat, id_map, "", times) for item in node]
        return [g[0] for g in got], sum(g[1] for g in got)
    if isinstance(node, dict):
        got_d = {k: _walk_plan(v, url_pat, id_map, k, times) for k, v in node.items()}
        out = {k: g[0] for k, g in got_d.items()}
        parent = node.get("parent_lf_video_id")
        if times and isinstance(parent, str) and parent in times:
            out["parent_lf_publish_at"] = times[parent]
        return out, sum(g[1] for g in got_d.values())
    return node, 0


def _relink_plan(path: Path, id_map, times, *, dry_run: bool) -> FileRewrite:
    if not path.is_file():
        return FileRewrite(str(path), 0, False, "absent")
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    new, count = _walk_plan(json.loads(data), _url_pattern(id_map), id_map, "", times)
    if count == 0:
        return FileRewrite(str(path), 0, False)
    if dry_run:
        return FileRewrite(str(path), count, False)
    tmp = path.with_name(path.name + ".relink-tmp")
    try:
        tmp.write_text(json.dumps(new, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        load_publish_plan(tmp)  # parse + consistency gates, before anything replaces the plan
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise SidecarChangedError(f"{path.name} changed while the relink ran")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return FileRewrite(str(path), count, True)


def _relink_worklist(out: Path, pat, id_map, *, dry_run: bool) -> FileRewrite:
    """Append the NEW ``RELATED VIDEO`` lines (and a ``SUPERSEDED`` line per replaced id) to the
    operator checklist; the dedup state only gains the new lines, its old ones are kept."""
    state_path = out / WORKLIST_STATE_FILENAME
    if not state_path.is_file():
        return FileRewrite(str(state_path), 0, False, "absent")
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{state_path.name}: expected a JSON list, got {type(raw).__name__}")
    emitted = {line for line in raw if isinstance(line, str)}
    new_lines: List[str] = []
    pairs: Dict[str, str] = {}
    for line in sorted(emitted):
        new_line, n = _sub(line, pat, id_map)
        if n == 0 or new_line in emitted or new_line in new_lines:
            continue
        new_lines.append(new_line)
        pairs.update({m.group(0): id_map[m.group(0)] for m in pat.finditer(line)})
    if not new_lines:
        return FileRewrite(str(state_path), 0, False)
    if dry_run:
        return FileRewrite(str(state_path), len(new_lines), False)
    appended = new_lines + [f"SUPERSEDED: {old} replaced by {new}" for old, new in pairs.items()]
    with (out / CHECKLIST_FILENAME).open("a", encoding="utf-8") as fh:
        fh.writelines(line + "\n" for line in appended)
    atomic_write_json(state_path, sorted(emitted | set(new_lines)))
    return FileRewrite(
        str(state_path), len(new_lines), True, f"new lines appended to {CHECKLIST_FILENAME}"
    )


def _relink_pending(path: Path, id_map, report: RelinkReport, *, dry_run: bool) -> FileRewrite:
    if not path.is_file():
        return FileRewrite(str(path), 0, False, "absent")
    with contextlib.nullcontext() if dry_run else pending_lock(path):
        entries = load_pending(path)
        hit = [e for e in entries if e.youtube_id in id_map]
        if hit and not dry_run:
            for entry in hit:
                entry.youtube_id = id_map[entry.youtube_id]
            # merge_pending semantics: the later entry wins, the first slot is kept
            save_pending(path, {(e.journey_path, e.youtube_id): e for e in entries}.values())
    for journey in sorted({e.journey_path for e in hit}):
        report.operator_items.append(
            f"pending journal entry for {journey} now names the replacement id; the journey "
            "file was not read, so if it already lists the OLD id, remove that entry by hand"
        )
    note = "no sent flag is recorded; every entry still in the file is unsent" if hit else ""
    return FileRewrite(str(path), len(hit), bool(hit) and not dry_run, note)


_PRIVATE_NOTE = (
    ": the replacement is not public by this short's slot ({slot}), so no comment links it "
    "(backfill adds it once the plan allows)"
)


def _row_label(row) -> str:
    return f"delivery {row['id']} (short {row['short_index']}, {row['platform']})"


def _drain_corrects(row) -> bool:
    """Whether the next drain fixes this row's stale comment: a pending correction under the cap,
    every item naming its live comment, and (Instagram) a known media id. Else a hand step."""
    items = _queue.first_comment_corrections(row)
    if not items or not (row["first_comment_error"] or "").startswith("stale"):
        return False
    if row["first_comment_correction_attempts"] >= _queue.FIRST_COMMENT_MAX_ATTEMPTS:
        return False
    if any(not i.get("live_id") for i in items):
        return False
    if row["platform"] != "instagram":  # only an Instagram edit is a delete + post on the media
        return True
    target = _queue.registered_platform_id(
        row["platform"], row["platform_upload_id"], row["platform_post_id"]
    )
    return bool(target) or all(i.get("action") != "edit" for i in items)


def _drain_checks(row) -> bool:
    """Whether the first-comment drain ever reads this unposted row's post."""
    target = _queue.registered_platform_id(
        row["platform"], row["platform_upload_id"], row["platform_post_id"]
    )
    return bool(target) and row["upload_kind"] != _queue.FACEBOOK_PHOTO_KIND


def _gate(row, pat, times, text) -> Tuple[Any, bool]:
    """``(new_publish_at, public)`` for the old id ``text`` names: whether the replacement is
    public by this row's own slot (``parent_is_public_at``; an unknown slot is never public)."""
    found = pat.search(text or "")
    at = times[found.group(0)] if found else None
    return at, parent_is_public_at(row["scheduled_at"], at)


def _relink_pending_corrections(conn, row, pat, id_map, times, now, report) -> None:
    """A pending ``edit``'s ``to`` text naming an old id is rewritten too (a second replace), so
    the drain never writes a dead link; it records the replacement's publish time (``parent_at``)
    for the drain's own gate re-check. An edit whose replacement is not yet public becomes a
    ``delete`` of the live comment (and the row's intended text goes), never a link to a private
    video."""
    base = (row["id"], row["short_index"], row["platform"], row["status"])
    rewrite = lambda t: _sub(t, pat, id_map)[0]  # noqa: E731
    gate = lambda t: _gate(row, pat, times, t)  # noqa: E731
    items = [
        i
        for i in _queue.first_comment_corrections(row)
        if i.get("action") == "edit" and _sub(i.get("to") or "", pat, id_map)[1]
    ]
    if not items:
        return
    written = False
    if not report.dry_run:
        outcome = _queue.rewrite_first_comment_corrections(
            conn, row["id"], rewrite, now.isoformat(), gate=gate
        )
        if not outcome:
            report.queue_skipped.append(QueueSkip(*base, "row changed while relinking; rerun"))
            return
        written = True
    for item in items:
        if not gate(item["to"])[1]:
            report.operator_items.append(
                f"{_row_label(row)}: the replacement is not public by this short's slot "
                f"({row['scheduled_at']}); the edit of live comment {item.get('live_id') or '?'} "
                "became a delete, so it links no private video"
            )
    report.queue_rewritten.append(QueueRewrite(*base, written))


def _relink_queue(conn, project, pat, id_map, times, now, report) -> None:
    for row in _queue.first_comments_naming(conn, project, id_map):
        _relink_pending_corrections(conn, row, pat, id_map, times, now, report)
        text = row["first_comment_text"] or ""
        _, public = _gate(row, pat, times, text)
        new_text, n = _sub(text, pat, id_map)
        if n == 0:  # the old id only inside a longer token, or only in a pending correction
            continue
        base = (row["id"], row["short_index"], row["platform"], row["status"])
        # A lease is live by this process's clock, like the CAS in ``_queue``, never the caller's.
        leased = (
            row["lease_owner"] is not None and (row["lease_expires_at"] or "") > _queue._now_iso()
        )
        if row["first_comment_posted_at"] is not None:
            reason = "comment already live"
            stale = row["first_comment_error"] or ""
            report.operator_items.append(
                f"{_row_label(row)}: comment already live, links OLD id; a live comment is "
                "never edited by a relink"
                + (
                    f" ({stale}); the drain corrects it"
                    if _drain_corrects(row)
                    else ", fix it by hand"
                )
                + ("" if public else _PRIVATE_NOTE.format(slot=row["scheduled_at"]))
            )
        elif row["status"] in _queue.FIRST_COMMENT_FROZEN_STATUSES:
            reason = f"status {row['status']}"
        elif row["first_comment_attempts"] >= _queue.FIRST_COMMENT_MAX_ATTEMPTS:
            reason = f"attempts exhausted ({row['first_comment_attempts']} refusals)"
        elif leased:
            reason = f"lease held by {row['lease_owner']} until {row['lease_expires_at']}; rerun"
        elif not public:  # the replacement is private at this slot: no link at all
            written = False
            if not report.dry_run:
                written = _queue.rewrite_unposted_first_comment(
                    conn, row["id"], text, None, now.isoformat()
                )
                if not written:
                    report.queue_skipped.append(
                        QueueSkip(*base, "row changed while relinking; rerun")
                    )
                    continue
            report.queue_rewritten.append(QueueRewrite(*base, written))
            report.operator_items.append(
                f"{_row_label(row)}: comment link dropped"
                + _PRIVATE_NOTE.format(slot=row["scheduled_at"])
            )
            continue
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
            if row["status"] == "published":
                report.drain_will_verify.append(
                    f"{_row_label(row)}: rewritten while published; the first-comment drain "
                    "checks whether the old text already went live and corrects it"
                    if _drain_checks(row)
                    else f"{_row_label(row)}: rewritten while published, but the drain never "
                    "checks it (no post id to read comments from): look at the post by hand"
                )
            continue
        report.queue_skipped.append(QueueSkip(*base, reason))


def _report_other_projects(conn, project, pat, id_map, report: RelinkReport) -> None:
    """Rows of OTHER projects whose comment names an old id: reported, never rewritten."""
    for row in _queue.first_comments_naming_elsewhere(conn, project, id_map):
        if not pat.search(row["first_comment_text"]):  # the old id only inside a longer token
            continue
        state = "already live" if row["first_comment_posted_at"] is not None else "not yet posted"
        report.operator_items.append(
            f"delivery {row['id']} (project {row['project_name']}, short {row['short_index']}, "
            f"{row['platform']}): comment {state}, links OLD id; another project's row, "
            "not rewritten"
        )


def relink_replaced_lf(
    project: str,
    output_dir: Path,
    id_map: Mapping[str, Tuple[str, str]],
    *,
    conn: sqlite3.Connection,
    now: datetime,
    dry_run: bool = False,
) -> RelinkReport:
    """Rewrite every old long-form id in ``id_map`` to its replacement (see the module doc).

    ``id_map`` is ``{old_video_id: (new_video_id, new_publish_at_iso_utc)}``; a bare-string value
    raises ``TypeError``. Raises ``ValueError`` on an empty map, an empty id, an old id equal to
    its new id, a new id that is also an old id (a chain would break idempotence) or a missing
    or unzoned publish time. Past that, a failing store is
    recorded in ``report.errors`` and the rest still run. Idempotent: once rewritten,
    the old ids are gone, so a rerun reports zero changes and writes nothing."""
    id_map, times = _split(id_map)
    out = Path(output_dir)
    pat = _pattern(id_map)
    report = RelinkReport(project=project, dry_run=dry_run)
    plan_path = out / f"{project}_publish_plan.json"
    steps = (
        (WORKLIST_STATE_FILENAME, lambda: _relink_worklist(out, pat, id_map, dry_run=dry_run)),
        (plan_path.name, lambda: _relink_plan(plan_path, id_map, times, dry_run=dry_run)),
        (
            PENDING_FILENAME,
            lambda: _relink_pending(out / PENDING_FILENAME, id_map, report, dry_run=dry_run),
        ),
    )
    for name, step in steps:  # file stores first, the queue last
        try:
            done = step()
            report.files.append(done)
        except Exception as exc:  # noqa: BLE001 -- one store failing must not hide the others
            report.errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if name == plan_path.name and done.replacements:
            report.operator_items.append(
                f"{name} {'would be' if dry_run else 'was'} rewritten: re-confirm Gate C/D "
                "(enqueue-plan refuses a plan whose reviewed content changed since it was approved)"
            )
    try:
        _relink_queue(conn, project, pat, id_map, times, now, report)
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"queue: {type(exc).__name__}: {exc}")
    try:
        _report_other_projects(conn, project, pat, id_map, report)
    except Exception as exc:  # noqa: BLE001 -- a report failure is recorded, never raised
        report.errors.append(f"other projects: {type(exc).__name__}: {exc}")
    return report
