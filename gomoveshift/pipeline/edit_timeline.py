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
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

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


# --------------------------------------------------------------------------
# Edit timeline: output <-> source mapping
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Crossfade:
    """An overlap of ``duration`` seconds between span ``after_span`` and the next.

    The incoming span starts ``duration`` before the outgoing span ends, so the
    output shrinks by ``duration`` per seam (sum of durations minus the overlaps).
    """

    after_span: int
    duration: float


@dataclass(frozen=True)
class Insert:
    """An output-time hold with no source (for example a still).

    ``at`` is measured on the timeline before any insert (receipt spans with
    crossfades applied). Everything at or after ``at`` moves later by ``duration``.
    """

    at: float
    duration: float


@dataclass(frozen=True)
class SourcePosition:
    """A source second that contributes to an output instant, with its weight."""

    time: float
    weight: float
    span_index: int


@dataclass(frozen=True)
class _Piece:
    out_start: float
    out_end: float
    src_start: Optional[float]  # None for an insert hold
    src_end: Optional[float]
    span_index: int

    def to_source(self, t: float) -> float:
        if self.src_start is None or self.src_end is None:
            raise ValueError("an insert hold has no source")
        out_dur = self.out_end - self.out_start
        src_dur = self.src_end - self.src_start
        if out_dur == src_dur or out_dur <= 0:
            return self.src_start + (t - self.out_start)
        return self.src_start + (t - self.out_start) * (src_dur / out_dur)

    def to_output(self, t: float) -> float:
        if self.src_start is None or self.src_end is None:
            raise ValueError("an insert hold has no source")
        out_dur = self.out_end - self.out_start
        src_dur = self.src_end - self.src_start
        if out_dur == src_dur or src_dur <= 0:
            return self.out_start + (t - self.src_start)
        return self.out_start + (t - self.src_start) * (out_dur / src_dur)


def _positive(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field}: must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field}: must be finite and > 0, got {value!r}")
    return float(value)


class EditTimeline:
    """Maps edited (output) time to source time and back from what was assembled.

    Built from an :class:`AssemblyReceipt` plus optional :class:`Crossfade` and
    :class:`Insert` extras. Intervals are half-open. A cut is never interpolated
    across: ``source_to_output`` returns ``None`` for a source second that was
    cut, and ``output_to_source`` returns no position inside an insert hold or
    outside the timeline.
    """

    def __init__(self, pieces: Sequence[_Piece]):
        self._pieces: tuple[_Piece, ...] = tuple(pieces)
        self._src = tuple(p for p in self._pieces if p.src_start is not None)
        for i in range(len(self._src) - 2):
            if self._src[i + 2].out_start < self._src[i].out_end:
                raise ValueError("crossfade: a span cannot overlap both neighbours past its length")

    # -- construction -----------------------------------------------------

    @classmethod
    def from_receipt(
        cls,
        receipt: AssemblyReceipt,
        crossfades: Iterable[Crossfade] = (),
        inserts: Iterable[Insert] = (),
    ) -> "EditTimeline":
        items = [
            (s.source_start, s.source_end, s.output_end - s.output_start, s.index)
            for s in receipt.spans
        ]
        starts = [s.output_start for s in receipt.spans]
        return cls._build(items, crossfades, inserts, starts)

    @classmethod
    def from_segments(
        cls, segments: Sequence[tuple[float, float]], crossfade: float = 0.0
    ) -> "EditTimeline":
        """Planned timeline from kept ``(start, end)`` source segments in output order.

        ``crossfade`` is the seam overlap applied between every pair of adjacent
        non-empty segments. A seam is clamped to half of the shorter neighbour so
        a short segment never overlaps both sides past its length. Zero-length
        segments contribute nothing.
        """
        kept = [(s, e) for s, e in segments if e - s > 0]
        items = [(s, e, e - s, i) for i, (s, e) in enumerate(kept)]
        fades: list[Crossfade] = []
        if crossfade > 0:
            for i in range(len(kept) - 1):
                shorter = min(items[i][2], items[i + 1][2])
                fades.append(Crossfade(i, min(float(crossfade), shorter / 2)))
        return cls._build(items, fades, (), None)

    @classmethod
    def _build(
        cls,
        items: Sequence[tuple[float, float, float, int]],
        crossfades: Iterable[Crossfade],
        inserts: Iterable[Insert],
        starts: Optional[Sequence[float]],
    ) -> "EditTimeline":
        n = len(items)
        fade_after: dict[int, float] = {}
        for cf in crossfades:
            d = _positive(cf.duration, f"crossfade[{cf.after_span}].duration")
            if not 0 <= cf.after_span < n - 1:
                raise ValueError(f"crossfade.after_span: {cf.after_span} has no following span")
            if cf.after_span in fade_after:
                raise ValueError(f"crossfade.after_span: {cf.after_span} given twice")
            if d > min(items[cf.after_span][2], items[cf.after_span + 1][2]):
                raise ValueError(
                    f"crossfade[{cf.after_span}].duration: longer than an adjacent span"
                )
            fade_after[cf.after_span] = d
        for i in range(1, n - 1):
            if fade_after.get(i - 1, 0.0) + fade_after.get(i, 0.0) > items[i][2]:
                raise ValueError(f"crossfade: overlaps at span {i} exceed its length")

        pieces: list[_Piece] = []
        running = 0.0
        shift = 0.0  # receipt starts are used verbatim; crossfades pull later spans back
        for i, (ss, se, dur, idx) in enumerate(items):
            if starts is None:
                os_ = running
                running += dur - fade_after.get(i, 0.0)
            else:
                os_ = starts[i] - shift
                shift += fade_after.get(i, 0.0)
            pieces.append(_Piece(os_, os_ + dur, ss, se, idx))

        total = max((p.out_end for p in pieces), default=0.0)
        ins = sorted(((i.at, _positive(i.duration, "insert.duration")) for i in inserts))
        for at, _ in ins:
            if not math.isfinite(at) or not 0 <= at <= total:
                raise ValueError(f"insert.at: {at!r} outside the timeline 0..{total}")
            for k in range(1, len(pieces)):
                a, b = pieces[k - 1], pieces[k]
                if b.out_start <= at < a.out_end:
                    raise ValueError(f"insert.at: {at!r} falls inside a crossfade")
        if ins:
            pieces = cls._apply_inserts(pieces, ins)
        return cls(pieces)

    @staticmethod
    def _apply_inserts(pieces: list[_Piece], ins: list[tuple[float, float]]) -> list[_Piece]:
        cuts = sorted({at for at, _ in ins})
        split: list[_Piece] = []
        for p in pieces:
            cur = p
            for at in cuts:
                if cur.out_start < at < cur.out_end:
                    mid = cur.to_source(at)
                    split.append(_Piece(cur.out_start, at, cur.src_start, mid, cur.span_index))
                    cur = _Piece(at, cur.out_end, mid, cur.src_end, cur.span_index)
            split.append(cur)

        def moved(t: float, inclusive: bool) -> float:
            return t + sum(d for at, d in ins if (at <= t if inclusive else at < t))

        out: list[_Piece] = []
        for p in split:
            # A hold at ``at`` sits before a piece starting at ``at`` and after one ending there.
            out.append(
                _Piece(
                    moved(p.out_start, True),
                    moved(p.out_end, False),
                    p.src_start,
                    p.src_end,
                    p.span_index,
                )
            )
        for k, (at, d) in enumerate(ins):
            # Holds at the same instant stack: each starts where the previous ended.
            start = moved(at, False) + sum(dd for aa, dd in ins[:k] if aa == at)
            out.append(_Piece(start, start + d, None, None, -1))
        out.sort(key=lambda p: (p.out_start, p.src_start is None))
        return out

    # -- queries ----------------------------------------------------------

    def output_to_source(self, t: float) -> tuple[SourcePosition, ...]:
        """Source positions behind output second ``t``.

        Empty inside an insert hold or outside ``[0, duration)``; two positions
        (outgoing first, weights summing to 1) inside a crossfade; else one.
        """
        hits = [p for p in self._src if p.out_start <= t < p.out_end]
        if not hits:
            return ()
        if len(hits) == 1:
            p = hits[0]
            return (SourcePosition(p.to_source(t), 1.0, p.span_index),)
        a, b = hits[0], hits[1]
        x = (t - b.out_start) / (a.out_end - b.out_start)
        return (
            SourcePosition(a.to_source(t), 1.0 - x, a.span_index),
            SourcePosition(b.to_source(t), x, b.span_index),
        )

    def source_to_output(self, t: float) -> Optional[float]:
        """Output second of source second ``t``; ``None`` if that second was cut.

        A source second used twice answers with its first use in output order.
        """
        for p in self._src:
            if p.src_start <= t < p.src_end:  # type: ignore[operator]
                return p.to_output(t)
        return None

    def source_to_output_snapped(self, t: float) -> float:
        """Like :meth:`source_to_output` but a cut second snaps to the end of the
        footage before it (``0`` before the first span, the total after the last).

        The ends are inclusive here (a marker on a span's last frame stays on it),
        which is what the chapter placement has always done.
        """
        prev_end = 0.0
        for k, p in enumerate(self._src):
            if t <= p.src_start:  # type: ignore[operator]
                return p.out_start if k == 0 or t == p.src_start else prev_end
            if t <= p.src_end:  # type: ignore[operator]
                return p.to_output(t)
            prev_end = p.out_end
        return prev_end

    @property
    def duration(self) -> float:
        return max((p.out_end for p in self._pieces), default=0.0)


@lru_cache(maxsize=64)
def _planned_timeline(segments: tuple[tuple[float, float], ...], crossfade: float) -> EditTimeline:
    return EditTimeline.from_segments(segments, crossfade)


def planned_timeline(
    segments: Sequence[tuple[float, float]], crossfade: float = 0.0
) -> EditTimeline:
    """Cached :meth:`EditTimeline.from_segments` for the per-marker mappers."""
    return _planned_timeline(tuple((float(s), float(e)) for s, e in segments), float(crossfade))
