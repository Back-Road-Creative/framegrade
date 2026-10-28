"""Review set: per-sequence stop states and edit decisions.

Reads one ``stop_mask_<date>.npz`` (``mask`` bool, ``states`` int8
:class:`~.stop_mask.MotionState`, 1 Hz, 0-based day-master seconds) and, when
given, the keycut assembly receipt, and emits one JSON document per sequence:

* contiguous runs of equal state with source seconds ``[start_s, end_s)``;
* the kept/cut decision for each run, read from ``mask`` (a run is cut exactly
  where ``mask`` is ``True``; CREEPING is never ``True`` there, so it is never
  cut). A run also splits where ``mask`` and ``states`` disagree, so a decision
  never covers a mixed span;
* the ``mask_summary`` totals;
* with a receipt, each kept run's output time ranges, mapped through
  :class:`~.edit_timeline.EditTimeline`. Without one, output times are omitted
  and the document says why; they are never estimated.

Pure and deterministic. Library only; no CLI. The schema is versioned and the
version is checked on load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np

from ._atomic_write import atomic_write_json
from .edit_timeline import AssemblyReceipt, EditTimeline, load_receipt
from .stop_mask import MotionState, mask_summary

SCHEMA_ID = "review-set-stops/1"
NO_RECEIPT_REASON = "no assembly receipt supplied"

_REQUIRED_KEYS = (
    "sequence",
    "states_source",
    "total_seconds",
    "runs",
    "state_seconds",
    "mask_summary",
    "output_times",
    "receipt",
)


class StopsSchemaError(ValueError):
    """A stops document is not a ``review-set-stops/1`` document."""


def _read_npz(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with np.load(path, allow_pickle=False) as data:
        if "mask" not in data.files:
            raise ValueError(f"{path.name}: npz has no 'mask' array")
        mask = np.asarray(data["mask"]).astype(bool)
        states = np.asarray(data["states"]) if "states" in data.files else None
    if mask.ndim != 1:
        raise ValueError(f"{path.name}: 'mask' must be 1-D")
    if states is None:
        # Pre-states npz: the mask is the only record, so it defines the states.
        derived = np.where(mask, int(MotionState.STOPPED), int(MotionState.MOVING))
        return mask, derived.astype(np.int8), "derived_from_mask"
    if states.shape != mask.shape:
        raise ValueError(f"{path.name}: 'states' length {states.size} != 'mask' length {mask.size}")
    valid = {int(s) for s in MotionState}
    bad = sorted({int(v) for v in np.unique(states)} - valid)
    if bad:
        raise ValueError(f"{path.name}: 'states' holds values outside MotionState: {bad}")
    return mask, states.astype(np.int8), "npz"


def _runs(mask: np.ndarray, states: np.ndarray) -> list[tuple[int, int, int, bool]]:
    """Maximal runs of equal ``(state, cut)`` as ``(start, end, state, cut)``."""
    out: list[tuple[int, int, int, bool]] = []
    n = int(mask.size)
    i = 0
    while i < n:
        s, c = int(states[i]), bool(mask[i])
        j = i + 1
        while j < n and int(states[j]) == s and bool(mask[j]) == c:
            j += 1
        out.append((i, j, s, c))
        i = j
    return out


def _output_ranges(timeline: EditTimeline, receipt: AssemblyReceipt, a: float, b: float) -> list:
    """Where source ``[a, b)`` lands in the output, one entry per receipt span it touches."""
    found = []
    for span in sorted(receipt.spans, key=lambda s: s.output_start):
        lo, hi = max(a, span.source_start), min(b, span.source_end)
        if lo >= hi:
            continue
        found.append(
            {
                "span_index": span.index,
                "source_start_s": lo,
                "source_end_s": hi,
                "output_start_s": timeline.source_to_output(lo),
                "output_end_s": timeline.source_to_output_snapped(hi),
            }
        )
    return found


def build_stops(
    npz_path: Union[str, Path],
    receipt: Union[None, str, Path, Mapping[str, Any], AssemblyReceipt] = None,
    *,
    sequence: Optional[str] = None,
) -> dict:
    """Build the stops document for one sequence.

    ``receipt`` may be a path, a parsed dict or a loaded receipt. ``sequence``
    defaults to the date in ``stop_mask_<date>.npz`` (else the file stem).
    """
    path = Path(npz_path)
    mask, states, states_source = _read_npz(path)
    if sequence is None:
        stem = path.stem
        sequence = stem[len("stop_mask_") :] if stem.startswith("stop_mask_") else stem

    loaded: Optional[AssemblyReceipt] = None
    timeline: Optional[EditTimeline] = None
    if receipt is not None:
        loaded = receipt if isinstance(receipt, AssemblyReceipt) else load_receipt(receipt)
        timeline = EditTimeline.from_receipt(loaded)

    runs = []
    seconds = {m.name: 0 for m in MotionState}
    for start, end, state, cut in _runs(mask, states):
        name = MotionState(state).name
        seconds[name] += end - start
        run: dict[str, Any] = {
            "state": name,
            "start_s": start,
            "end_s": end,
            "decision": "cut" if cut else "kept",
        }
        if timeline is not None and loaded is not None and not cut:
            run["output_ranges"] = _output_ranges(timeline, loaded, float(start), float(end))
        runs.append(run)

    return {
        "schema": SCHEMA_ID,
        "sequence": sequence,
        "states_source": states_source,
        "total_seconds": int(mask.size),
        "runs": runs,
        "state_seconds": seconds,
        "mask_summary": mask_summary(mask),
        "output_times": (
            {"available": True, "reason": None}
            if loaded is not None
            else {"available": False, "reason": NO_RECEIPT_REASON}
        ),
        "receipt": (
            None
            if loaded is None
            else {
                "schema": loaded.schema,
                "provenance": loaded.provenance,
                "output_duration": loaded.output_duration,
            }
        ),
    }


def _check(doc: Any) -> dict:
    if not isinstance(doc, Mapping):
        raise StopsSchemaError("stops document: must be a JSON object")
    if doc.get("schema") != SCHEMA_ID:
        raise StopsSchemaError(f"schema: expected {SCHEMA_ID!r}, got {doc.get('schema')!r}")
    for key in _REQUIRED_KEYS:
        if key not in doc:
            raise StopsSchemaError(f"{key}: missing")
    if not isinstance(doc["runs"], list):
        raise StopsSchemaError("runs: must be a list")
    return dict(doc)


def load_stops(source: Union[str, Path, Mapping[str, Any]]) -> dict:
    """Load a stops document from a JSON path or parsed dict, checking the schema version."""
    if isinstance(source, (str, Path)):
        try:
            data = json.loads(Path(source).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StopsSchemaError(f"not valid JSON ({exc.msg})") from exc
        return _check(data)
    return _check(source)


def write_stops(doc: Mapping[str, Any], path: Union[str, Path]) -> None:
    """Write ``doc`` atomically after checking it carries this schema version."""
    atomic_write_json(Path(path), _check(doc))
