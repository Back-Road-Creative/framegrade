"""Re-measure the route record a Short's publish-plan sidecar lacks.

A Short planned before the zero-tolerance place gate carries no ``route_record``
(or one with no ``bins``), so ``recheck_place_text`` refuses its title, description
and tags as unverifiable and ``recover-terminal`` can never re-queue it. This
re-derives the record from the sources a verifier re-measures from
(``route_claims.TIMESTAMP_SOURCE``): the day's persisted GPS points
(``gps_points_<date>.json``) and the Short's clip range
(``shorts_ost_audit_<date>.json``), through the same measurement the pipeline used
(:func:`route_claims.measure_short_route`).

It never defaults, estimates or copies a record: a missing source, a clip that does not
match the Short's media, or a clip with no GPS fix is a refusal. It writes the sidecar's
``route_record`` only; the place gate and ``recover-terminal`` still decide.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping

from gomoveshift.pipeline.route_claims import measure_short_route

from .plan_io import load_publish_plan

#: The clip range and the media's duration may differ by an encode's rounding only.
DURATION_TOLERANCE_S = 1.0
_RANK = re.compile(r"short_(\d+)\.", re.IGNORECASE)


class RemeasureRefusedError(RuntimeError):
    """The route cannot be re-measured honestly; nothing was written."""


def _read_json(path: Path, what: str):
    if not path.is_file():
        raise RemeasureRefusedError(f"source gone: {what} {path} does not exist")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RemeasureRefusedError(f"source unreadable: {what} {path}: {exc}") from exc


def _find_short(raw: Mapping[str, Any], row, sidecar: Path) -> dict:
    for short in raw.get("shorts") or []:
        if (
            short.get("project_name") == row["project_name"]
            and short.get("short_index") == row["short_index"]
        ):
            return short
    raise RemeasureRefusedError(f"{sidecar}: no short {row['project_name']}#{row['short_index']}")


def remeasure_short_route(row, stages_dir, geocoder, *, apply: bool = False) -> Dict[str, Any]:
    """Measure the Short ``row`` points at and, with ``apply``, write its ``route_record``.

    ``stages_dir`` holds ``gps_points_<date>.json`` and ``shorts_ost_audit_<date>.json``.
    Returns a summary; raises :class:`RemeasureRefusedError` (nothing written)."""
    sidecar = Path(row["sidecar_path"] or "")
    if not sidecar.is_file():
        raise RemeasureRefusedError(f"publish-plan sidecar not readable: {sidecar}")
    raw = json.loads(sidecar.read_text(encoding="utf-8"))
    short = _find_short(raw, row, sidecar)
    if "bins" in (short.get("route_record") or {}):
        raise RemeasureRefusedError("the route record already has 'bins'; nothing to re-measure")
    media = short.get("media") or {}
    found = _RANK.search(Path(str(media.get("path", ""))).name)
    if not found:
        raise RemeasureRefusedError(f"cannot read the clip rank from media {media.get('path')!r}")
    rank = int(found.group(1))
    date = short.get("recording_date")
    stages = Path(stages_dir)
    points = _read_json(stages / f"gps_points_{date}.json", "day GPS points")
    audit = _read_json(stages / f"shorts_ost_audit_{date}.json", "shorts audit")
    entry = next((e for e in audit.get("shorts") or [] if e.get("rank") == rank), None)
    if entry is None:
        raise RemeasureRefusedError(f"the shorts audit for {date} has no rank {rank}")
    if entry.get("parent_lf_index") != short.get("lf_index"):
        raise RemeasureRefusedError(
            f"rank {rank} in the audit belongs to parent LF {entry.get('parent_lf_index')}, "
            f"the Short to LF {short.get('lf_index')}"
        )
    start, end = float(entry["start_sec"]), float(entry["end_sec"])
    if abs((end - start) - float(media.get("duration_seconds", -1))) > DURATION_TOLERANCE_S:
        raise RemeasureRefusedError(
            f"clip range {start:g}-{end:g} s does not match the media duration "
            f"{media.get('duration_seconds')} s (duration mismatch)"
        )
    if not any(start <= float(p[0]) < end for p in points):
        raise RemeasureRefusedError(
            f"no GPS fix inside {start:g}-{end:g} s; bins cannot be measured"
        )
    evidence = measure_short_route(rank, start, end, [tuple(p) for p in points], geocoder)
    record = evidence.record()
    summary = {
        "rank": rank,
        "bins": len(record["bins"]),
        "retained_seconds": record["retained_seconds"],
        "unsettled_seconds": record["unsettled_seconds"],
        "gap_seconds": record["gap_seconds"],
        "windows_key": record["windows_key"],
        "applied": False,
    }
    short["route_record"] = record
    tmp = sidecar.with_name(sidecar.name + ".remeasure-tmp")
    tmp.write_text(json.dumps(raw, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    try:
        load_publish_plan(tmp)  # the new sidecar must still parse before it replaces the old
    except Exception:
        tmp.unlink()
        raise
    if apply:
        os.replace(tmp, sidecar)
        summary["applied"] = True
    else:
        tmp.unlink()
    return summary
