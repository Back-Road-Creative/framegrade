"""CLI adapter for re-checking a Route Story receipt offline.

All the actual verification logic lives in ``pipeline.route_story.verify``;
this handler only prints a compact table and maps the report to an exit
code -- 0 only when every check passed.
"""

from __future__ import annotations

from ...pipeline.route_story import verify


def cmd_route_story_verify(args, config) -> int:
    """Re-check every fact in a Route Story receipt and print a pass/fail table."""
    report = verify.verify_receipt(args.receipt)

    print(f"kind: {report.kind}")
    for check in report.checks:
        status = "PASS" if check.ok else "FAIL"
        line = f"  [{status}] {check.name}"
        if check.detail:
            line += f": {check.detail}"
        print(line)

    return 0 if report.ok else 1
