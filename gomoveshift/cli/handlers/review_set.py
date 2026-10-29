"""Handler for ``gomoveshift review-set {scan,stops,pairs,render}``.

Thin wrappers over ``pipeline/review_set_scan``, ``review_set_stops``,
``review_set_pairs`` and ``review_set_render``: parse the inputs, call the library, write the result.
Any expected failure (missing file, non-empty output directory, schema error)
prints one ``Error:`` line to stderr and returns 1 -- never a traceback.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_USER_ERRORS = (OSError, ValueError, KeyError)


def parse_pair(text: str) -> tuple[float, float]:
    """``"S:O"`` -> ``(source_t, output_t)``; raises ``ValueError`` otherwise."""
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"--pair must look like SOURCE_S:OUTPUT_S, got {text!r}")
    return float(parts[0]), float(parts[1])


def _read_json(path: Path, what: str):
    if not path.is_file():
        raise FileNotFoundError(f"{what} not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{what} is not valid JSON ({exc.msg}): {path}") from exc


def _emit(doc: dict, out: Optional[Path]) -> None:
    text = json.dumps(doc, indent=2, sort_keys=False) + "\n"
    if out is None:
        sys.stdout.write(text)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)


def _cmd_scan(args) -> int:
    from ...pipeline.review_set_scan import scan_review_set

    if not args.scores.is_file():
        raise FileNotFoundError(f"score cache not found: {args.scores}")
    with np.load(args.scores, allow_pickle=False) as data:
        if "scores" not in data.files:
            raise ValueError(f"{args.scores.name}: npz has no 'scores' array")
        scores = np.asarray(data["scores"])
    states = None
    if args.stop_mask is not None:
        if not args.stop_mask.is_file():
            raise FileNotFoundError(f"stop-mask npz not found: {args.stop_mask}")
        with np.load(args.stop_mask, allow_pickle=False) as data:
            if "states" not in data.files:
                raise ValueError(f"{args.stop_mask.name}: npz has no 'states' array")
            states = np.asarray(data["states"])
    gps = _read_json(args.gps, "GPS points") if args.gps is not None else None
    tonemap = _read_json(args.tonemap, "tonemap sidecar") if args.tonemap is not None else None
    _emit(scan_review_set(scores, states=states, gps_points=gps, tonemap=tonemap), args.out)
    return 0


def _cmd_stops(args) -> int:
    from ...pipeline.review_set_stops import build_stops, write_stops

    if not args.stop_mask.is_file():
        raise FileNotFoundError(f"stop-mask npz not found: {args.stop_mask}")
    if args.receipt is not None and not args.receipt.is_file():
        raise FileNotFoundError(f"keycut receipt not found: {args.receipt}")
    doc = build_stops(args.stop_mask, args.receipt, sequence=args.sequence)
    write_stops(doc, args.out)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


def _load_pairs(path: Path) -> list[tuple[float, float]]:
    """Read ``--pairs``: a JSON list of 2-lists, or a ``review-set pairs`` document."""
    raw = _read_json(path, "pairs file")
    if isinstance(raw, dict) and "pairs" in raw:
        raw = raw["pairs"]
    if not isinstance(raw, list) or not all(isinstance(p, list) and len(p) == 2 for p in raw):
        raise ValueError(f"pairs file must be a JSON list of [source_t, output_t]: {path}")
    return [(float(s), float(o)) for s, o in raw]


def _cmd_pairs(args) -> int:
    from ...pipeline.review_set_pairs import build_pairs

    scan = _read_json(args.scan, "scan file")
    if not args.stops.is_file():
        raise FileNotFoundError(f"stops file not found: {args.stops}")
    _emit(build_pairs(scan, args.stops), args.out)
    return 0


def _cmd_render(args, config) -> int:
    from ...pipeline.review_set_render import render_review_set

    pairs = [parse_pair(p) for p in args.pair or []]
    if args.pairs is not None:
        pairs = _load_pairs(args.pairs) + pairs
    if not pairs:
        raise ValueError("give at least one pair (--pairs FILE or --pair S:O)")
    result = render_review_set(
        args.source,
        args.output,
        pairs,
        args.out_dir,
        excerpt_seconds=args.excerpt_seconds,
        config=config,
        synthetic=args.synthetic,
    )
    print(f"wrote {len(result['pairs'])} pair(s) to {args.out_dir}", file=sys.stderr)
    return 0


def cmd_review_set(args, config) -> int:
    try:
        if args.review_set_action == "scan":
            return _cmd_scan(args)
        if args.review_set_action == "stops":
            return _cmd_stops(args)
        if args.review_set_action == "pairs":
            return _cmd_pairs(args)
        if args.review_set_action == "render":
            return _cmd_render(args, config)
    except _USER_ERRORS as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Error: unknown review-set action {args.review_set_action!r}", file=sys.stderr)
    return 2
