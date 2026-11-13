"""``gomoveshift status`` — print the live snapshot (``gomoveshift/status/snapshot.py``).

Read-only. All reading lives in ``build_snapshot``; this module only formats.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from ...status.snapshot import build_snapshot

_GB = 1024**3


def _head(title: str, section: Dict[str, Any]) -> List[str]:
    lines = [f"{title}  [{section['source']}]"]
    if "unavailable" in section:
        lines.append(f"  unavailable: {section['unavailable']}")
    return lines


def _render(snap: Dict[str, Any]) -> str:
    out = [f"GoMoveShift status (as of {snap['as_of']})", ""]

    jobs = snap["jobs"]
    out += _head("Jobs", jobs)
    if "unavailable" not in jobs:
        for job in jobs["running"]:
            where = " / ".join(x for x in (job["drive"], job["phase"]) if x) or "-"
            out.append(f"  RUNNING  {job['name']}  {where}  heartbeat {job['heartbeat_at']}")
        if not jobs["running"]:
            out.append("  nothing running")
        for job in jobs["recent"]:
            out.append(
                f"  {job['state']:<9} exit {job['exit_code']}  {job['name']}  {job['ended_at']}"
            )
        if jobs["unreadable"]:
            out.append(f"  {jobs['unreadable']} unreadable record(s)")

    incidents = snap["incidents"]
    out += ["", *_head("Open incidents", incidents)]
    if "unavailable" not in incidents:
        for row in incidents["open"]:
            out.append(f"  {row.get('kind')}  {row.get('key')}  {row.get('error') or ''}".rstrip())
        if not incidents["open"]:
            out.append("  none open")

    queue = snap["queue"]
    out += ["", *_head("Publishing", queue)]
    if "unavailable" not in queue:
        for platform, states in queue["counts"].items():
            out.append(f"  {platform}: " + ", ".join(f"{k} {v}" for k, v in states.items()))
        out.append(f"  manual publishes overdue: {len(queue['overdue_manual'])}")
        for row in queue["next"]:
            out.append(
                f"  next {row['scheduled_at']}  {row['platform']}  "
                f"{row['project_name']}#{row['short_index']}"
            )

    runway = snap["runway"]
    out += ["", *_head("Runway", runway)]
    if "unavailable" not in runway:
        out.append(f"  queue ends: {runway.get('queue_ends') or '(nothing booked)'}")

    backlog = snap["backlog"]
    out += ["", *_head("Drive backlog", backlog)]
    if "unavailable" not in backlog:
        regions = ", ".join(f"{k} {v}" for k, v in backlog["by_region"].items())
        out.append(f"  {backlog['rows']} row(s): {regions}")

    machine = snap["machine"]
    out += ["", *_head("Machine", machine)]
    if "unavailable" not in machine:
        for vol in machine["volumes"]:
            if "unavailable" in vol:
                out.append(f"  {vol['label']} unavailable: {vol['unavailable']}")
            else:
                out.append(
                    f"  {vol['label']} {vol['free_bytes'] / _GB:.0f} GB free of "
                    f"{vol['total_bytes'] / _GB:.0f} GB"
                )
        gpu = machine["gpu"]
        if "unavailable" in gpu:
            out.append(f"  GPU lock unavailable: {gpu['unavailable']}")
        else:
            out.append(f"  GPU lock: {'held by ' + str(gpu['owner']) if gpu['held'] else 'free'}")
        cron = machine["cron"]
        if isinstance(cron, dict):
            out.append(f"  cron logs unavailable: {cron['unavailable']}")
        else:
            for row in cron:
                if "unavailable" in row:
                    out.append(f"  {row['name']} unavailable: {row['unavailable']}")
                    continue
                failure = row["last_failure"]
                tail = f"  last failure {failure['at']}: {failure['line']}" if failure else ""
                out.append(f"  {row['name']}  last line {row['last_at']}{tail}")
    return "\n".join(out)


def cmd_status(args, config) -> int:
    """Print the snapshot. Exit 0 always: this reports, it does not judge."""
    snap = build_snapshot(config)
    print(json.dumps(snap, indent=2) if getattr(args, "json", False) else _render(snap))
    return 0
