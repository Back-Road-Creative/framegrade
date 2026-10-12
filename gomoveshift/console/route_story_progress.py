"""Parse a route-story-film job's log tail (film.py's own progress logging)
into a small summary for the job detail page: map-leg progress, frame-QA
fault/evidence lines, and the finished marker (film.py's final bare
``print(args.output)`` of the delivered file path)."""

from __future__ import annotations

import re

_LEG_RE = re.compile(
    r"map leg (?P<leg>\d+)/(?P<legs>\d+) frames, [\d.]+ s/frame, "
    r"~(?P<seconds_left>\d+) s left in this leg"
)
_QA_REFUSED_RE = re.compile(r"route-story frame QA refused a frame; evidence: .+$")
_QA_WRITE_FAILED_RE = re.compile(r"route-story frame QA could not write .+: .+$")
_QA_EVIDENCE_RE = re.compile(r"route-story frame QA evidence: (?P<path>.+)$")
_OUTPUT_RE = re.compile(r"^(?P<path>\S+\.(?:mp4|mov|mkv))$")


def parse_progress(lines: list[str]) -> dict:
    """``{"leg", "legs", "seconds_left", "faults", "evidence", "finished",
    "output"}`` -- ``output`` is the delivered film path once ``finished``."""
    result: dict = {
        "leg": None,
        "legs": None,
        "seconds_left": None,
        "faults": [],
        "evidence": [],
        "finished": False,
        "output": None,
    }
    for raw_line in lines:
        line = raw_line.rstrip("\n")

        leg_match = _LEG_RE.search(line)
        if leg_match:
            result["leg"] = int(leg_match.group("leg"))
            result["legs"] = int(leg_match.group("legs"))
            result["seconds_left"] = int(leg_match.group("seconds_left"))
            continue

        if _QA_REFUSED_RE.search(line) or _QA_WRITE_FAILED_RE.search(line):
            result["faults"].append(line.strip())
            continue

        evidence_match = _QA_EVIDENCE_RE.search(line)
        if evidence_match:
            result["evidence"].append(evidence_match.group("path"))
            continue

        output_match = _OUTPUT_RE.match(line.strip())
        if output_match:
            result["finished"] = True
            result["output"] = output_match.group("path")

    return result
