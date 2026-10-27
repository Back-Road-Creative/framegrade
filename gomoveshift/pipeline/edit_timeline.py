"""Assembly receipt schema (``keycut-assembly-receipt/1``) and its loader.

The cutting library writes one receipt per assembly describing what it
actually cut: which source ranges went into which output ranges, and whether
each span was stream-copied (snapped forward to a keyframe) or re-encoded.
This module is the framegrade side of that contract: frozen dataclasses plus
a strict validator. Every rule violation raises :class:`ReceiptError` naming
the failing field.

All intervals are half-open ``[start, end)`` in seconds.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

SCHEMA_ID = "keycut-assembly-receipt/1"
PROVENANCES = ("keycut", "planned")
METHODS = ("copy", "reencode")
REASONS = ("stream_copy", "keyframes_unknown", "snap_back_over_excluded")
DURATION_SOURCES = ("probed", "planned")
CONTIGUITY_TOL = 1e-3


class ReceiptError(ValueError):
    """A receipt violated the schema; the message names the failing field."""


@dataclass(frozen=True)
class ReceiptSpan:
    index: int
    requested_indices: tuple[int, ...]
    merged_start: float
    merged_end: float
    source_start: float
    source_end: float
    method: str
    reason: str
    output_start: float
    output_end: float
    duration_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "requested_indices": list(self.requested_indices),
            "merged_start": self.merged_start,
            "merged_end": self.merged_end,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "method": self.method,
            "reason": self.reason,
            "output_start": self.output_start,
            "output_end": self.output_end,
            "duration_source": self.duration_source,
        }


@dataclass(frozen=True)
class AssemblyReceipt:
    provenance: str
    keycut_version: Optional[str]
    master_path: str
    master_size_bytes: Optional[int]
    time_base: str
    max_gap_sec: float
    requested: tuple[tuple[float, float], ...]
    spans: tuple[ReceiptSpan, ...]
    output_path: str
    output_duration: Optional[float]
    schema: str = SCHEMA_ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "provenance": self.provenance,
            "keycut_version": self.keycut_version,
            "master": {"path": self.master_path, "size_bytes": self.master_size_bytes},
            "time_base": self.time_base,
            "max_gap_sec": self.max_gap_sec,
            "requested": [[s, e] for s, e in self.requested],
            "spans": [s.to_dict() for s in self.spans],
            "output": {"path": self.output_path, "duration": self.output_duration},
        }


def _fail(field: str, msg: str) -> ReceiptError:
    return ReceiptError(f"{field}: {msg}")


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(field, "must be an object")
    return value


def _require(obj: Mapping[str, Any], key: str, field: str) -> Any:
    if key not in obj:
        raise _fail(f"{field}.{key}" if field else key, "missing")
    return obj[key]


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(field, "must be a number")
    if not math.isfinite(value):
        raise _fail(field, "must be finite")
    return float(value)


def _int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(field, "must be an integer")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise _fail(field, "must be a string")
    return value


def _enum(value: Any, field: str, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise _fail(field, f"must be one of {list(allowed)}, got {value!r}")
    return value


def _interval(start: float, end: float, field: str) -> None:
    if start > end:
        raise _fail(field, f"start {start} exceeds end {end} (half-open [start, end))")


def _parse_span(raw: Any, i: int, n_requested: int) -> ReceiptSpan:
    f = f"spans[{i}]"
    span = _mapping(raw, f)
    index = _int(_require(span, "index", f), f"{f}.index")
    if index != i:
        raise _fail(f"{f}.index", f"must equal its position {i}, got {index}")
    ri_raw = _require(span, "requested_indices", f)
    if not isinstance(ri_raw, (list, tuple)) or not ri_raw:
        raise _fail(f"{f}.requested_indices", "must be a non-empty list")
    ri = tuple(_int(v, f"{f}.requested_indices[{k}]") for k, v in enumerate(ri_raw))
    for v in ri:
        if not 0 <= v < n_requested:
            raise _fail(
                f"{f}.requested_indices", f"{v} outside requested range 0..{n_requested - 1}"
            )
    nums = {
        k: _number(_require(span, k, f), f"{f}.{k}")
        for k in (
            "merged_start",
            "merged_end",
            "source_start",
            "source_end",
            "output_start",
            "output_end",
        )
    }
    method = _enum(_require(span, "method", f), f"{f}.method", METHODS)
    reason = _enum(_require(span, "reason", f), f"{f}.reason", REASONS)
    dsrc = _enum(_require(span, "duration_source", f), f"{f}.duration_source", DURATION_SOURCES)
    _interval(nums["merged_start"], nums["merged_end"], f"{f}.merged")
    _interval(nums["source_start"], nums["source_end"], f"{f}.source")
    _interval(nums["output_start"], nums["output_end"], f"{f}.output")
    if nums["source_end"] != nums["merged_end"]:
        raise _fail(f"{f}.source_end", "must equal merged_end")
    if method == "reencode" and nums["source_start"] != nums["merged_start"]:
        raise _fail(f"{f}.source_start", "must equal merged_start for a reencode span")
    if method == "copy" and nums["source_start"] < nums["merged_start"]:
        raise _fail(f"{f}.source_start", "a copy span snaps forward, never before merged_start")
    return ReceiptSpan(
        index=index,
        requested_indices=ri,
        method=method,
        reason=reason,
        duration_source=dsrc,
        **nums,
    )


def _parse(data: Any) -> AssemblyReceipt:
    root = _mapping(data, "receipt")
    schema = _require(root, "schema", "")
    if schema != SCHEMA_ID:
        raise _fail("schema", f"must be {SCHEMA_ID!r}, got {schema!r}")
    provenance = _enum(_require(root, "provenance", ""), "provenance", PROVENANCES)
    kv = _require(root, "keycut_version", "")
    if kv is not None:
        kv = _string(kv, "keycut_version")
    master = _mapping(_require(root, "master", ""), "master")
    master_path = _string(_require(master, "path", "master"), "master.path")
    size = _require(master, "size_bytes", "master")
    if size is not None:
        size = _int(size, "master.size_bytes")
        if size < 0:
            raise _fail("master.size_bytes", "must be >= 0")
    time_base = _require(root, "time_base", "")
    if time_base != "seconds":
        raise _fail("time_base", f"must be 'seconds', got {time_base!r}")
    max_gap = _number(_require(root, "max_gap_sec", ""), "max_gap_sec")
    if max_gap < 0:
        raise _fail("max_gap_sec", "must be >= 0")
    req_raw = _require(root, "requested", "")
    if not isinstance(req_raw, (list, tuple)):
        raise _fail("requested", "must be a list")
    requested = []
    for k, pair in enumerate(req_raw):
        f = f"requested[{k}]"
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise _fail(f, "must be a [start, end] pair")
        s, e = _number(pair[0], f"{f}[0]"), _number(pair[1], f"{f}[1]")
        _interval(s, e, f)
        requested.append((s, e))
    spans_raw = _require(root, "spans", "")
    if not isinstance(spans_raw, (list, tuple)):
        raise _fail("spans", "must be a list")
    spans = tuple(_parse_span(r, i, len(requested)) for i, r in enumerate(spans_raw))
    if spans and abs(spans[0].output_start) > CONTIGUITY_TOL:
        raise _fail("spans[0].output_start", "must be 0")
    for i in range(1, len(spans)):
        if abs(spans[i].output_start - spans[i - 1].output_end) > CONTIGUITY_TOL:
            raise _fail(
                f"spans[{i}].output_start",
                f"not contiguous with spans[{i - 1}].output_end "
                f"({spans[i].output_start} vs {spans[i - 1].output_end})",
            )
    out = _mapping(_require(root, "output", ""), "output")
    out_path = _string(_require(out, "path", "output"), "output.path")
    dur = _require(out, "duration", "output")
    if dur is not None:
        dur = _number(dur, "output.duration")
        if dur < 0:
            raise _fail("output.duration", "must be >= 0")
    return AssemblyReceipt(
        provenance=provenance,
        keycut_version=kv,
        master_path=master_path,
        master_size_bytes=size,
        time_base="seconds",
        max_gap_sec=max_gap,
        requested=tuple(requested),
        spans=spans,
        output_path=out_path,
        output_duration=dur,
    )


def load_receipt(source: Union[str, Path, Mapping[str, Any]]) -> AssemblyReceipt:
    """Load and validate a receipt from a JSON file path or an already-parsed dict."""
    if not isinstance(source, (str, Path)):
        return _parse(source)
    try:
        data = json.loads(Path(source).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _fail("receipt", f"not valid JSON ({exc.msg})") from exc
    return _parse(data)
