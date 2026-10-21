"""``gomoveshift schedule`` — read-only publishing-schedule report.

Renders ``pipeline.schedule_report.build_schedule_report``. All computation
lives there (and is unit-tested without a CLI); this module only formats.
"""

from __future__ import annotations

import json

from ...pipeline.content_ledger import ContentLedger
from ...pipeline.schedule_report import build_schedule_report
from ...publishing.title_kit import build_title_kit


def _when(iso: str) -> str:
    return iso[:16].replace("T", " ")


def _count(value) -> str:
    return "unknown (Waypost decision store unreadable)" if value is None else str(value)


def _render(report: dict) -> str:
    booked = report["scheduled_longforms"]
    lines = [
        f"Publishing schedule (as of {_when(report['as_of'])})",
        f"  Weekly long-form capacity: {report['weekly_longform_capacity']}"
        f" ({', '.join(report['designed_days'])})",
        "",
        f"Scheduled long-forms: {len(booked)}",
    ]
    for e in booked:
        kind = "SL" if e["kind"] == "superlong_part" else "LF"
        lines.append(
            f"  {e['date']} {e['time'][:5]}  {kind}  [{e['region'] or '-'}]  {e['title'][:44]}"
        )
    if not booked:
        lines.append("  (nothing booked)")
    lines += [
        f"Scheduled shorts: {report['scheduled_shorts_count']}",
        f"Queue ends: {report['queue_ends'] or '(nothing booked)'}",
        "",
        "Next available long form (* = soonest for that region):",
    ]
    if not report["next_available"]:
        lines.append("  (no free slot inside the search horizon)")
    current = None
    for row in report["next_available"]:
        if row["region"] != current:
            current = row["region"]
            lines.append(f"  {current}:")
        mark = "*" if row["earliest_for_region"] else " "
        windows = ", ".join(row["shorts_windows"]) or "(none configured)"
        lines.append(
            f"   {mark} {row['day']}  {_when(row['slot'])}"
            f"   shorts: {_count(row['shorts_required'])} of {row['shorts_slots']} slots"
            f" at {windows}"
        )
    lines += ["", "Next available superlong part:"]
    if not report["next_available_superlong"]:
        lines.append("  (no free superlong slot inside the search horizon)")
    for row in report["next_available_superlong"]:
        lines.append(f"  {row['region']}:  {row['day']}  {_when(row['slot'])}")
    return "\n".join(lines)


def cmd_schedule(args, config) -> int:
    """Print what is scheduled, when it ends, and the next free slot per region."""
    ledger = ContentLedger.load(getattr(args, "ledger_path", None))
    report = build_schedule_report(
        ledger, config, per_location_cap=build_title_kit().per_location_cap()
    )
    print(json.dumps(report, indent=2) if getattr(args, "json", False) else _render(report))
    return 0
