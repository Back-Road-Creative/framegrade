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

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from .store import new_incident, safe_name

#: Case-insensitive suffixes of a camera chapter file.
CAMERA_SUFFIXES = (".mp4", ".insv", ".mov")
_KEYS = {"sources", "reason", "project"}


@dataclass(frozen=True)
class Exclusion:
    sources: Tuple[str, ...]
    reason: str
    project: Optional[str] = None


def _is_absolute(text: str) -> bool:
    return PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute()


def _problem(entry: Optional[int], detail: str) -> Dict[str, Any]:
    return {"entry": entry, "detail": detail}


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
    if twice := sorted({s for s in sources if sources.count(s) > 1}):
        return f"source listed twice: {twice}"
    if dup := sorted(s for s in sources if s in seen):
        return f"source already excluded by an earlier entry: {dup}"
    if clash := sorted(s for s in sources if s in row_sources):
        return "source is also a drive_backlog.yaml row source: " + ", ".join(
            f"{s} (row {row_sources[s]})" for s in clash
        )
    return None


def load_exclusions(
    path: Path, rows: Iterable[Any]
) -> Tuple[List[Exclusion], List[Dict[str, Any]]]:
    """``(valid entries, problems)``. A missing file is ``([], [])``; nothing here raises."""
    row_sources = {
        str(s): str(getattr(row, "project", "?"))
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
            problems.append(_problem(n, why))
            continue
        entries.append(
            Exclusion(tuple(item["sources"]), item["reason"].strip(), item.get("project"))
        )
        seen.update(item["sources"])
    return entries, problems


def exclusion_problem_incidents(
    problems: Sequence[Dict[str, Any]], now_iso: str
) -> List[Dict[str, Any]]:
    out = []
    for p in problems:
        tag = "file" if p["entry"] is None else str(p["entry"])
        out.append(
            new_incident(
                "backlog_exclusion_invalid",
                f"backlog_exclusion_invalid:{tag}",
                f"backlog_exclusion_invalid_{tag}.json",
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


def unlisted_footage_incidents(
    rows: Iterable[Any], exclusions: Iterable[Exclusion], now_iso: str, notes: List[str]
) -> List[Dict[str, Any]]:
    """One incident per folder holding a row or exclusion source, listing its camera files
    (top level, files only) that no row and no exclusion names. A folder that cannot be listed
    is a note."""
    accounted = {str(s) for r in rows for s in (getattr(r, "sources", ()) or ())}
    accounted.update(s for e in exclusions for s in e.sources)
    folders = sorted({os.path.dirname(s) for s in accounted})
    out: List[Dict[str, Any]] = []
    for folder in folders:
        try:
            with os.scandir(folder) as it:
                names = sorted(
                    e.name for e in it if e.name.lower().endswith(CAMERA_SUFFIXES) and e.is_file()
                )
        except OSError as exc:
            notes.append(
                f"backlog_unlisted_footage not evaluated for {folder}: {type(exc).__name__}: {exc}"
            )
            continue
        missing = [n for n in names if os.path.join(folder, n) not in accounted]
        if missing:
            out.append(
                new_incident(
                    "backlog_unlisted_footage",
                    f"backlog_unlisted_footage:{folder}",
                    f"backlog_unlisted_footage_{safe_name(folder)}.json",
                    now_iso,
                    directory=folder,
                    files=missing,
                )
            )
    return out
