"""Review set: pair scan windows with output times.

Joins a ``review_set_scan`` document with a ``review-set-stops/1`` document
(see :mod:`.review_set_stops`) and emits, for each candidate window, the
``[source_t, output_t]`` pair ``review-set render --pairs`` consumes.

A window's midpoint is looked up in the stops runs. It is paired only when it
falls inside a *kept* run whose receipt-mapped ``output_ranges`` cover it;
the output time is then ``output_start_s + (midpoint - source_start_s)``.
Anything else is listed under ``unpaired`` with a reason -- output times are
never estimated:

* ``cut``              -- the midpoint is in a run the edit removed;
* ``no_output_times``  -- the stops document was built without a receipt;
* ``outside_runs``     -- the midpoint is in no run at all;
* ``not_in_output``    -- a kept run, but the receipt carries no output for that second.

Pure and deterministic: conditions in name order, windows in start order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Union

from .review_set_stops import load_stops

SCHEMA_ID = "review-set-pairs/1"


def _window(w: Any, condition: str) -> tuple[float, float]:
    try:
        return float(w["start_s"]), float(w["end_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"scan: {condition}: window needs numeric start_s/end_s") from exc


def _locate(stops: Mapping[str, Any], t: float) -> tuple[str, Any]:
    """``("paired", output_t)`` or ``("unpaired", reason)`` for source second ``t``."""
    for run in stops["runs"]:
        if not run["start_s"] <= t < run["end_s"]:
            continue
        if run["decision"] == "cut":
            return "unpaired", "cut"
        if not stops["output_times"].get("available"):
            return "unpaired", "no_output_times"
        for rng in run.get("output_ranges") or []:
            if rng["source_start_s"] <= t < rng["source_end_s"]:
                return "paired", rng["output_start_s"] + (t - rng["source_start_s"])
        return "unpaired", "not_in_output"
    return "unpaired", "outside_runs"


def build_pairs(scan: Mapping[str, Any], stops_source: Union[str, Path, Mapping[str, Any]]) -> dict:
    """Build the pairs document from a scan document and a stops document (or path)."""
    conditions = scan.get("conditions") if isinstance(scan, Mapping) else None
    if not isinstance(conditions, Mapping):
        raise ValueError("scan: 'conditions' must be an object")
    stops = load_stops(stops_source)
    pairs: list = []
    detail: list = []
    unpaired: list = []
    for name in sorted(conditions):
        windows = (conditions[name] or {}).get("windows") or []
        for a, b in sorted(_window(w, name) for w in windows):
            mid = (a + b) / 2.0
            kind, value = _locate(stops, mid)
            if kind == "paired":
                pairs.append([mid, float(value)])
                detail.append(
                    {
                        "condition": name,
                        "window": [a, b],
                        "source_t": mid,
                        "output_t": float(value),
                    }
                )
            else:
                unpaired.append({"condition": name, "window": [a, b], "reason": value})
    return {"schema": SCHEMA_ID, "pairs": pairs, "detail": detail, "unpaired": unpaired}
