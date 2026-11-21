"""Health rules 10 and 11: the manifest, the store and the footage must account for each other.

* ``backlog_store_orphan``: a ``ready_drives`` store project that is in no ``drive_backlog.yaml``
  row and no exclusion entry's ``project``.
* ``backlog_unlisted_footage``: a camera file sitting in a folder the manifest or the exclusions
  use that no row's ``sources`` and no exclusion's ``sources`` names.

The exclusions live in ``drive_backlog_excluded.yaml`` beside the manifest, a SEPARATE file on
purpose: seo_manager's backlog loader (and ``backlog-next``) never read it, so an excluded source
can never be launched. Contract: a top-level list of mappings ``{sources: [absolute path, ...]
(non-empty), reason: non-empty str, project: optional str}``, no other keys; a source appears once
across the file and in no manifest row. An entry that breaks it is reported (``backlog_exclusion_invalid``)
and excludes nothing (so its footage shows up as unlisted). Listing only: a media file is never
opened, probed or decoded.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from .store import new_incident, safe_name

#: Case-insensitive suffixes of a camera chapter file.
CAMERA_SUFFIXES = (".mp4", ".insv", ".mov")
_KEYS = {"sources", "reason", "project"}

#: Total seconds the folder listings of one tick may take: ``bin/gms-health-cron.sh`` holds a
#: flock with no timeout, so a hung ``/mnt`` share must not stall the tick. A folder not listed
#: in time is treated as unreadable (its open incidents stay live).
LIST_DEADLINE_S = 20.0

_DRIVE_RE = re.compile(r"^(/mnt/[A-Za-z])(?:/|$)")


def _is_drvfs(path: str) -> bool:
    """Under ``/mnt/<single letter>/``: a Windows drive, case-insensitive."""
    return _DRIVE_RE.match(path) is not None


def norm(path: str) -> str:
    """The comparison form of a path: normalised, and case-folded on a Windows drive."""
    out = os.path.normpath(str(path))
    return out.casefold() if _is_drvfs(out) else out


def _is_unmounted(folder: str) -> bool:
    """A ``/mnt/<letter>`` folder with nothing mounted there is an offline drive, not an empty one."""
    m = _DRIVE_RE.match(folder)
    return bool(m) and not os.path.ismount(m.group(1))


@dataclass(frozen=True)
class Exclusion:
    sources: Tuple[str, ...]
    reason: str
    project: Optional[str] = None


def _is_absolute(text: str) -> bool:
    """POSIX absolute only: a Windows path (``D:\\x``) names nothing the launcher can open."""
    return text.startswith("/")


def _problem(
    entry: Optional[int],
    detail: str,
    tag: Optional[str] = None,
    sources: Sequence[str] = (),
) -> Dict[str, Any]:
    return {"entry": entry, "detail": detail, "tag": tag, "sources": list(sources)}


def _entry_sources(raw: Any) -> List[str]:
    """The absolute POSIX string sources of an entry (valid or not): folders rule 11 still lists
    for an invalid entry, which excludes nothing."""
    if isinstance(raw, dict) and isinstance(raw.get("sources"), list):
        return [s.strip() for s in raw["sources"] if isinstance(s, str) and _is_absolute(s.strip())]
    return []


def _entry_tag(raw: Any) -> Optional[str]:
    """The entry's first usable source path: its key, so inserting an entry above does not
    shift every key below it."""
    if isinstance(raw, dict) and isinstance(raw.get("sources"), list):
        for src in raw["sources"]:
            if isinstance(src, str) and src.strip():
                return src.strip()
    return None


def _entry_problem(raw: Any, row_sources: Dict[str, str], seen: set) -> Optional[str]:
    """Why one entry breaks the contract, or ``None``. ``seen`` carries earlier entries' sources."""
    if not isinstance(raw, dict):
        return "expected a mapping"
    if unknown := sorted(str(k) for k in set(raw) - _KEYS):
        return f"unknown key(s) {unknown}"
    reason = raw.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return "reason must be a non-empty string"
    sources = raw.get("sources")
    if not isinstance(sources, list) or not sources:
        return "sources must be a non-empty list"
    if not all(isinstance(s, str) and s.strip() for s in sources):
        return "sources must be a list of path strings"
    if relative := [s for s in sources if not _is_absolute(s)]:
        return f"sources must be absolute paths: {relative}"
    project = raw.get("project")
    if project is not None and (not isinstance(project, str) or not project.strip()):
        return "project must be a non-empty string when given"
    normed = [norm(s) for s in sources]
    if twice := sorted({s for s, n in zip(sources, normed) if normed.count(n) > 1}):
        return f"source listed twice: {twice}"
    if dup := sorted(s for s, n in zip(sources, normed) if n in seen):
        return f"source already excluded by an earlier entry: {dup}"
    if clash := sorted(s for s, n in zip(sources, normed) if n in row_sources):
        return "source is also a drive_backlog.yaml row source: " + ", ".join(
            f"{s} (row {row_sources[norm(s)]})" for s in clash
        )
    return None


def load_exclusions(
    path: Path, rows: Iterable[Any]
) -> Tuple[List[Exclusion], List[Dict[str, Any]]]:
    """``(valid entries, problems)``. A missing file is ``([], [])``; nothing here raises."""
    row_sources = {
        norm(str(s)): str(getattr(row, "project", "?"))
        for row in rows
        for s in (getattr(row, "sources", ()) or ())
    }
    try:
        if not path.is_file():
            return [], []
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        return [], [_problem(None, f"{path} is unreadable: {type(exc).__name__}: {exc}")]
    if raw is None:
        return [], []
    if not isinstance(raw, list):
        return [], [_problem(None, f"{path}: top level must be a list of entries")]
    entries: List[Exclusion] = []
    problems: List[Dict[str, Any]] = []
    seen: set = set()
    for n, item in enumerate(raw, start=1):
        why = _entry_problem(item, row_sources, seen)
        if why:
            problems.append(_problem(n, why, _entry_tag(item), _entry_sources(item)))
            continue
        entries.append(
            Exclusion(tuple(item["sources"]), item["reason"].strip(), item.get("project"))
        )
        seen.update(norm(x) for x in item["sources"])
    return entries, problems


def exclusion_problem_incidents(
    problems: Sequence[Dict[str, Any]], now_iso: str
) -> List[Dict[str, Any]]:
    """One incident per problem, keyed by the entry's first source path (the entry number only
    when it has no usable source; ``file`` for a file-level problem)."""
    out = []
    used: Dict[str, int] = {}
    for p in problems:
        tag = "file" if p["entry"] is None else (p.get("tag") or str(p["entry"]))
        used[tag] = used.get(tag, 0) + 1
        if used[tag] > 1:
            tag = f"{tag}#{used[tag]}"
        out.append(
            new_incident(
                "backlog_exclusion_invalid",
                f"backlog_exclusion_invalid:{tag}",
                f"backlog_exclusion_invalid_{safe_name(tag)}.json",
                now_iso,
                entry=p["entry"],
                detail=p["detail"],
            )
        )
    return out


def store_projects(
    store_path: Path, tz: Any, regions: Iterable[str], notes: Optional[List[str]] = None
) -> Optional[List[str]]:
    """Every ``ready_drives`` project, or ``None`` (noted) when the store cannot be read."""
    try:
        from seo_manager.scheduling.store import ScheduleStore

        return [d.project for d in ScheduleStore(store_path, tz, list(regions)).all_drives()]
    except Exception as exc:  # noqa: BLE001 -- an unreadable store proves nothing either way
        if notes is not None:
            notes.append(
                f"backlog_store_orphan not evaluated: {store_path}: {type(exc).__name__}: {exc}"
            )
        return None


def store_orphan_incidents(
    projects: Iterable[str], rows: Iterable[Any], exclusions: Iterable[Exclusion], now_iso: str
) -> List[Dict[str, Any]]:
    known = {str(getattr(r, "project", "")) for r in rows}
    known.update(e.project for e in exclusions if e.project)
    return [
        new_incident(
            "backlog_store_orphan",
            f"backlog_store_orphan:{p}",
            f"backlog_store_orphan_{safe_name(p)}.json",
            now_iso,
            project=p,
        )
        for p in sorted(set(projects))
        if p not in known
    ]


def _key_hash(path: str) -> str:
    return hashlib.sha256(f"backlog_unlisted_footage:{path}".encode("utf-8")).hexdigest()[:10]


def _scan(folder: str) -> List[str]:
    """The camera files directly in ``folder`` (names only; nothing is opened)."""
    with os.scandir(folder) as it:
        return sorted(
            e.name for e in it if e.name.lower().endswith(CAMERA_SUFFIXES) and e.is_file()
        )


def _list_folders(
    folders: Sequence[str], notes: List[str]
) -> Tuple[Dict[str, List[str]], List[str]]:
    """``({folder: camera file names}, unreadable folders)``. Each folder is listed in a daemon
    thread against one shared deadline; a folder that raises, sits on an unmounted drive or is
    not done in time is unreadable (noted), never empty."""
    results: Dict[str, Any] = {}
    threads: Dict[str, threading.Thread] = {}

    def work(folder: str) -> None:
        try:
            results[folder] = _scan(folder)
        except OSError as exc:
            results[folder] = exc

    unreadable: List[str] = []
    for folder in folders:
        if _is_unmounted(folder):
            notes.append(f"backlog_unlisted_footage not evaluated for {folder}: drive not mounted")
            unreadable.append(folder)
            continue
        threads[folder] = threading.Thread(target=work, args=(folder,), daemon=True)
        threads[folder].start()
    deadline = time.monotonic() + LIST_DEADLINE_S
    listed: Dict[str, List[str]] = {}
    for folder, thread in threads.items():
        thread.join(max(0.0, deadline - time.monotonic()))
        got = results.get(folder)
        if thread.is_alive() or got is None:
            notes.append(
                f"backlog_unlisted_footage not evaluated for {folder}: "
                f"listing exceeded {LIST_DEADLINE_S:g} s"
            )
            unreadable.append(folder)
        elif isinstance(got, OSError):
            notes.append(
                f"backlog_unlisted_footage not evaluated for {folder}: {type(got).__name__}: {got}"
            )
            unreadable.append(folder)
        else:
            listed[folder] = got
    return listed, unreadable


def unlisted_footage_incidents(
    rows: Iterable[Any],
    exclusions: Iterable[Exclusion],
    now_iso: str,
    notes: List[str],
    unreadable: Optional[List[str]] = None,
    listed_only: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """One incident per camera file (top level of a folder holding a row or exclusion source)
    that no row and no exclusion names, keyed by its path so each opens, resolves and notifies
    on its own. A folder that cannot be listed is a note and is appended to ``unreadable``
    (comparison form, see :func:`norm`) so the caller keeps its open incidents live.
    ``listed_only`` are sources of invalid exclusion entries: their folders are listed but they
    account for nothing."""
    accounted = {norm(str(s)) for r in rows for s in (getattr(r, "sources", ()) or ())}
    accounted.update(norm(s) for e in exclusions for s in e.sources)
    originals: Dict[str, str] = {}
    for src in sorted(
        {str(s) for r in rows for s in (getattr(r, "sources", ()) or ())}
        | {s for e in exclusions for s in e.sources}
        | {s for s in listed_only if _is_absolute(s)}
    ):
        originals.setdefault(
            norm(os.path.dirname(os.path.normpath(src))), os.path.dirname(os.path.normpath(src))
        )
    listed, bad = _list_folders([originals[k] for k in sorted(originals)], notes)
    if unreadable is not None:
        unreadable.extend(norm(f) for f in bad)
    out: List[Dict[str, Any]] = []
    for folder in sorted(listed):
        for name in listed[folder]:
            path = os.path.join(folder, name)
            if norm(path) in accounted:
                continue
            out.append(
                new_incident(
                    "backlog_unlisted_footage",
                    f"backlog_unlisted_footage:{path}",
                    # safe_name alone flattens /a_b/x and /a/b_x to one file: add a key hash.
                    f"backlog_unlisted_footage_{safe_name(path)}_{_key_hash(path)}.json",
                    now_iso,
                    directory=folder,
                    name=name,
                    path=path,
                )
            )
    return out
