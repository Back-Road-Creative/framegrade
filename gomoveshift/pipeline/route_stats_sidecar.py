"""Operator-side per-LF route stats sidecar.

The pipeline already surfaces per-LF distance, elevation gain, and regions
in the YouTube description body via the distance/elevation/regions lines in
`longform_metadata.py` (`🛣️ Distance:`, `⛰️ Elevation gain:`, `🗺️ Through:`).
That handles the *viewer* surface tastefully.

Missing was an **operator** deliverable: a single JSON file per drive run
that aggregates the same per-LF route facts in one place, suitable for
thumbnail picking, post-run review, and future debugging.

This module is the simplest possible additive surface:

- Reads existing state maps populated by `_assert_post_encoding_invariants`
  in phase_bc_day_first (no new computation, no GPS work).
- Writes `<output_base>/route_stats.json` once per drive run.
- Emits whatever state has — missing values become `null`, not failures.
  The sidecar is best-effort visibility; the distance/elevation/regions description lines
  remain authoritative for viewer-facing data.

Unlike `ost_audit_sidecar.py`, this is **not gated** — there is no
`OstAuditPendingError` analog. The viewer-facing surfaces (title +
description) have their own gates; the operator sidecar exists purely
for inspection.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ROUTE_STATS_FILENAME = "route_stats.json"


def build_route_stats_payload(state) -> dict:
    """Aggregate the per-LF route stats already present on ``state``.

    Pulls every LF index from the union of the four source maps the
    pipeline populates (titles, distance, elevation, regions). Returns a
    dict shaped for one-shot operator inspection:

    ::

        {
            "generated_at": "<iso utc>",
            "lfs": [
                {
                    "lf_index": 0,
                    "title": "Killarney to Dingle via N86, Kerry",
                    "recording_date": "2024-01-03",
                    "distance_km": 47.2,
                    "elevation_gain_m": 823.0,
                    "regions": ["County Kerry", "Munster"],
                    "trim_ratio": 0.61,
                },
                ...
            ],
        }

    Missing entries become ``null`` rather than raising — the sidecar is
    a snapshot of what the pipeline computed, including gaps. Gaps
    themselves are useful operator signal (an LF with `distance_km: null`
    indicates GPS coverage was insufficient).

    Numeric values are rounded to a single decimal place to keep the file
    diff-friendly across reruns of the same drive.
    """
    titles: dict = state.per_lf_titles or {}
    recording_date: dict = state.per_lf_recording_date or {}
    distance: dict = state.per_lf_distance_km or {}
    elevation: dict = state.per_lf_elevation_gain_m or {}
    regions: dict = state.per_lf_regions or {}
    trim_ratio: dict = state.per_lf_trim_ratio or {}

    # Union of LF indices across all source maps. State round-trips
    # through JSON turn int keys into strings; coerce both for safety.
    indices: set[int] = set()
    for src in (titles, recording_date, distance, elevation, regions, trim_ratio):
        for k in src or {}:
            try:
                indices.add(int(k))
            except (TypeError, ValueError):
                continue

    lfs: list[dict] = []
    for idx in sorted(indices):
        lfs.append(
            {
                "lf_index": idx,
                "title": _coerce_str(titles, idx),
                "recording_date": _coerce_str(recording_date, idx),
                "distance_km": _coerce_float(distance, idx),
                "elevation_gain_m": _coerce_float(elevation, idx),
                "regions": _coerce_list(regions, idx),
                "trim_ratio": _coerce_float(trim_ratio, idx),
            }
        )

    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "lfs": lfs,
    }


def write_route_stats_sidecar(output_base: Path, state) -> Optional[Path]:
    """Write `<output_base>/route_stats.json` from ``state``.

    Returns the written path on success. Returns ``None`` and logs at
    WARNING when the payload would carry no LF entries — emitting an
    empty sidecar would be confusing operator signal. Never raises; the
    sidecar is best-effort.
    """
    payload = build_route_stats_payload(state)
    if not payload.get("lfs"):
        logger.debug("route_stats_sidecar: no per-LF state to serialise (skipping write)")
        return None
    output_base = Path(output_base)
    try:
        output_base.mkdir(parents=True, exist_ok=True)
        path = output_base / ROUTE_STATS_FILENAME
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        logger.info("  Wrote route stats sidecar: %s", path)
        return path
    except OSError as exc:
        logger.warning("route_stats_sidecar: write failed at %s: %s", output_base, exc)
        return None


def _coerce_str(mapping: dict, idx: int) -> Optional[str]:
    for k in (idx, str(idx)):
        if k in mapping:
            v = mapping[k]
            return str(v) if v else None
    return None


def _coerce_float(mapping: dict, idx: int) -> Optional[float]:
    for k in (idx, str(idx)):
        if k in mapping:
            v = mapping[k]
            if v is None:
                return None
            try:
                return round(float(v), 1)
            except (TypeError, ValueError):
                return None
    return None


def _coerce_list(mapping: dict, idx: int) -> Optional[list]:
    for k in (idx, str(idx)):
        if k in mapping:
            v = mapping[k]
            if not v:
                return None
            if isinstance(v, (list, tuple)):
                return [str(item) for item in v if item]
            return [str(v)]
    return None
