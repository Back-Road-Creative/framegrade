#!/usr/bin/env python3
"""Scan every tracked file for credentials; any finding fails.

There is no baseline file and no excluded file type: a finding is fixed, or
the string is made obviously fake, never waved through. The scanner must
already be installed (``DETECT_SECRETS`` names it); this step downloads
nothing. Findings are printed as ``path:line: type`` and never include the
matched text.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Left on, these two filters skip a file by its name alone: known lock-file
# names, and a list of extensions that includes ``.lock``. An index URL or a
# token pasted into a lock file must still be found, so both are switched off;
# a file that is not text fails to decode and is skipped by the scanner itself.
NAME_FILTERS = (
    "detect_secrets.filters.heuristic.is_lock_file",
    "detect_secrets.filters.heuristic.is_non_text_file",
)


class MalformedReport(Exception):
    """The scanner's output cannot be trusted to mean "nothing found"."""


def findings(report: object) -> list[str]:
    """``path:line: type`` for every hit; raises unless the report is complete."""
    if not isinstance(report, dict):
        raise MalformedReport("report is not an object")
    if not report.get("plugins_used"):
        raise MalformedReport("report names no plugins: nothing was scanned for")
    results = report.get("results")
    if not isinstance(results, dict):
        raise MalformedReport("report has no results object")
    out = []
    for path, hits in sorted(results.items()):
        if not isinstance(hits, list) or not hits:
            raise MalformedReport(f"results for {path} are not a non-empty list")
        for hit in hits:
            if (
                not isinstance(hit, dict)
                or "type" not in hit
                or "line_number" not in hit
            ):
                raise MalformedReport(
                    f"a result for {path} lacks its type or line number"
                )
            out.append(f"{path}:{hit['line_number']}: {hit['type']}")
    return out


def scan(root: Path) -> list[str]:
    tool = os.environ.get("DETECT_SECRETS", "detect-secrets")
    # The scanner reads the list of tracked files. Outside a checkout, or in an
    # empty one, it would report nothing found without having read anything.
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, text=True
    )
    if tracked.returncode != 0 or not tracked.stdout:
        raise MalformedReport("no tracked files to scan")
    command = [tool, "scan", ".", "--no-verify"]
    for name in NAME_FILTERS:
        command += ["--disable-filter", name]
    done = subprocess.run(command, cwd=root, capture_output=True, text=True)
    if done.returncode != 0:
        raise MalformedReport(f"scanner exited {done.returncode}")
    try:
        report = json.loads(done.stdout)
    except json.JSONDecodeError as exc:
        raise MalformedReport("scanner output is not JSON") from exc
    return findings(report)


def main() -> int:
    root = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parent.parent
    )
    try:
        found = scan(root)
    except (MalformedReport, OSError) as exc:
        print(f"secret scan: could not complete: {exc}", file=sys.stderr)
        return 2
    for line in found:
        print(line)
    if found:
        print(f"secret scan: {len(found)} finding(s)", file=sys.stderr)
        return 1
    print("secret scan: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
