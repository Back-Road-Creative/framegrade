"""Rule 1: a failed drive, read from job-runner ``job/1`` records (JSON only, no ``bin/`` import)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from . import resume
from .store import new_incident, safe_name, utc_iso

MARKER = "Drive pipeline failed"
# The drive orchestrator's logger name: how a launcher-script job (argv `bash x.sh audit`, no
# "drive" anywhere) is still recognized as a drive from its own log.
DRIVE_LOGGER = "gomoveshift.pipeline.drive_orchestrator"
_TAIL_BYTES = 2_000_000
_DRIVE_NAME_RE = re.compile(r"\b(?:project|drive)[ _]name\s*[=:]\s*['\"]?([A-Za-z0-9._-]+)", re.I)
_PHASE_RE = re.compile(r"\b(phase_[a-z0-9_]+|Phase [A-F])\b")
_GATE_RE = re.compile(r"([A-Za-z][A-Za-z /_-]{2,60}?)\s+HARD FAIL|\bGATE[ _]([A-Z_]+)")
_TS_PREFIX_RE = re.compile(r"^.*?(?=" + re.escape(MARKER) + ")")


def _tail(path: Path) -> Optional[List[str]]:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - _TAIL_BYTES))
            return fh.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return None


def parse_log(lines: List[str]) -> Dict[str, Optional[str]]:
    """Drive name, phase, gate and error line from a drive log's lines (last occurrence wins)."""
    out: Dict[str, Optional[str]] = {"drive": None, "phase": None, "gate": None, "error": None}
    for line in lines:
        if (m := _DRIVE_NAME_RE.search(line)) and out["drive"] is None:
            out["drive"] = m.group(1)
        if MARKER in line:
            out["error"] = _TS_PREFIX_RE.sub("", line).strip()
        if m := _PHASE_RE.search(line):
            out["phase"] = m.group(1)
        if m := _GATE_RE.search(line):
            out["gate"] = (m.group(1) or m.group(2)).strip()
    return out


def _expect_logs(rec: Dict[str, Any]) -> List[Path]:
    return [Path(p) for p in rec.get("expect") or [] if isinstance(p, str)]


def _is_drive_job(rec: Dict[str, Any]) -> bool:
    argv = [str(a) for a in rec.get("argv") or []]
    if "drive" in argv or "drive" in str(rec.get("name") or ""):
        return True
    for lp in _expect_logs(rec):
        lines = _tail(lp)
        if lines and any(DRIVE_LOGGER in ln or MARKER in ln for ln in lines):
            return True
    return False


def _links(rec: Dict[str, Any]) -> Tuple[Set[str], Optional[str]]:
    """``(expect log paths, stages dir)`` of a job: what ties it to a sibling run of one drive."""
    logs = {str(p) for p in _expect_logs(rec)}
    stages: Optional[str] = None
    for lp in _expect_logs(rec):
        lines = _tail(lp)
        found = resume.parse_recipe_path(lines) if lines else None
        if found:
            stages = str(Path(found).parent)
    return logs, stages


def detect(
    jobs_dir: Path, staging_sha: Optional[str], now_iso: str
) -> Tuple[List[Dict[str, Any]], Set[str], int]:
    """``(incidents, succeeded_job_names, unreadable)``.

    A drive's incident is its LATEST job record only; ``succeeded_job_names`` are job names whose
    latest run succeeded (the caller resolves their open incidents)."""
    latest: Dict[str, Dict[str, Any]] = {}
    succeeded: Set[str] = set()
    unreadable = 0
    records: List[Dict[str, Any]] = []
    if jobs_dir.is_dir():
        for path in sorted(jobs_dir.glob("*/job.json")):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(rec, dict) or rec.get("schema") != "job/1":
                    raise ValueError("not job/1")
                float(rec.get("created_at") or 0)
            except (OSError, ValueError, TypeError):
                unreadable += 1
                continue
            records.append(rec)
    for rec in sorted(records, key=lambda r: float(r.get("created_at") or 0)):
        if _is_drive_job(rec):
            latest[str(rec.get("name"))] = rec
    incidents: List[Dict[str, Any]] = []
    # (created_at, log paths, stages dir) of each job that SUCCEEDED, under any name: a later
    # one sharing a failed drive's log or stages dir recovered it (audit failed -> publish ok).
    wins: List[Tuple[float, Set[str], Optional[str]]] = []
    for rec in latest.values():
        if (rec.get("result") or {}).get("state") == "succeeded":
            wins.append((float(rec.get("created_at") or 0), *_links(rec)))
    for name, rec in latest.items():
        result = rec.get("result") or {}
        state = result.get("state")
        exit_code = result.get("exit_code")
        logs = _expect_logs(rec)
        parsed: Dict[str, Optional[str]] = {}
        log_path: Optional[Path] = None
        marked = False
        recipe: Optional[str] = None
        for lp in logs:
            lines = _tail(lp)
            if lines is None:
                continue
            p = parse_log(lines)
            if p["error"] or log_path is None:
                parsed, log_path = p, lp
            marked = marked or bool(p["error"])
            recipe = resume.parse_recipe_path(lines) or recipe
        failed = state == "failed" or (exit_code not in (None, 0) and marked)
        run_id = str(rec.get("run_id") or "")
        if not failed:
            if state in ("succeeded",):
                succeeded.add(name)
            continue
        created = float(rec.get("created_at") or 0)
        mine, stages = _links(rec)
        if any(c > created and ((mine & m) or (stages and stages == s)) for c, m, s in wins):
            continue  # recovered by a later run under another job name
        drive = parsed.get("drive") or name
        key = f"drive_failed:{safe_name(drive)}:{run_id}"
        incidents.append(
            new_incident(
                "drive_failed",
                key,
                f"{safe_name(drive)}.json",
                now_iso,
                drive=drive,
                phase=parsed.get("phase"),
                gate=parsed.get("gate"),
                error=parsed.get("error") or f"job {name} ended {state} (exit {exit_code})",
                log_path=str(log_path) if log_path else None,
                resume_recipe=recipe,
                job_name=name,
                job_run_id=run_id,
                failed_at=utc_iso(result.get("ended_at")) if result.get("ended_at") else None,
                staging_sha=staging_sha,
                resume_attempts=[],
            )
        )
    return incidents, succeeded, unreadable


def summary(row: Dict[str, Any]) -> str:
    where = " / ".join(x for x in (row.get("phase"), row.get("gate")) if x)
    return f"Drive {row['drive']} failed" + (f" at {where}" if where else "") + f": {row['error']}"
