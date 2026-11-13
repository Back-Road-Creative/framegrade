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

Stages from a RE-RENDER (the original stages are gone) re-plan the day's Shorts, so a
rank, a parent LF and a duration no longer name the published clip. With
``rendered_dir`` the clip is identified by its frames instead: the published media must
frame-match exactly one of the re-render's Shorts (:func:`media_signature`), and that
Short's clip range is the one measured.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from gomoveshift.pipeline.route_claims import measure_short_route

from .plan_io import load_publish_plan

#: The clip range and the media's duration may differ by an encode's rounding only.
DURATION_TOLERANCE_S = 1.0
_RANK = re.compile(r"short_(\d+)\.", re.IGNORECASE)
#: Frame-signature distance (0..1) at or under which two Shorts are the same clip, and
#: at or over which they are clearly different clips. Measured 2026-11-13 on the four
#: published mizen Shorts: the same clip re-captioned, re-encoded and trimmed 0.3 s
#: scores 0.043; different clips score 0.111 to 0.316. Between the two is a refusal.
FRAME_MATCH_MAX = 0.06
FRAME_DISTINCT_MIN = 0.09
_SIG_FRAMES = 8
_SIG_SIDE = 16


class RemeasureRefusedError(RuntimeError):
    """The route cannot be re-measured honestly; nothing was written."""


def _read_json(path: Path, what: str):
    if not path.is_file():
        raise RemeasureRefusedError(f"source gone: {what} {path} does not exist")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RemeasureRefusedError(f"source unreadable: {what} {path}: {exc}") from exc


def media_signature(path) -> bytes:
    """``_SIG_FRAMES`` evenly spaced frames of ``path``, each a ``_SIG_SIDE`` square of
    grey bytes. Small enough that a different burned caption or a re-encode barely
    moves it; a different stretch of road does."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0"]
            + [str(path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        duration = float(probe.stdout.strip())
        out = b""
        for i in range(_SIG_FRAMES):
            at = duration * (i + 0.5) / _SIG_FRAMES
            frame = subprocess.run(
                ["ffmpeg", "-v", "error", "-ss", f"{at:.3f}", "-i", str(path), "-frames:v", "1"]
                + ["-vf", f"scale={_SIG_SIDE}:{_SIG_SIDE},format=gray", "-f", "rawvideo", "-"],
                capture_output=True,
                check=True,
                timeout=300,
            )
            out += frame.stdout
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise RemeasureRefusedError(f"cannot read frames of {path}: {exc}") from exc
    return out


def signature_distance(a: bytes, b: bytes) -> float:
    """Mean absolute difference of two signatures, 0 (identical) to 1."""
    if not a or len(a) != len(b):
        raise RemeasureRefusedError(
            f"frame signature sizes differ ({len(a)} vs {len(b)} bytes); cannot compare"
        )
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a) / 255.0


def _frame_matched_entry(short, entries, rendered_dir: Path, signature_fn) -> tuple[dict, float]:
    """The one audit entry whose re-rendered Short is the published media, by frames."""
    media = short.get("media") or {}
    published = Path(str(media.get("path", "")))
    if not published.is_file():
        raise RemeasureRefusedError(f"published media {published} is gone; cannot frame-match")
    duration = float(media.get("duration_seconds", -1))
    target = signature_fn(published)
    scored = []
    for entry in entries:
        length = float(entry["end_sec"]) - float(entry["start_sec"])
        if entry.get("parent_lf_index") != short.get("lf_index"):
            continue
        if abs(length - duration) > DURATION_TOLERANCE_S:
            continue
        files = sorted(Path(rendered_dir).rglob(f"short_{int(entry['rank']):02d}.mp4"))
        if len(files) != 1:
            raise RemeasureRefusedError(
                f"{rendered_dir} holds {len(files)} file(s) for rank {entry['rank']}, need 1"
            )
        scored.append((signature_distance(target, signature_fn(files[0])), entry))
    scored.sort(key=lambda pair: pair[0])
    seen = ", ".join(f"rank {e['rank']}={d:.3f}" for d, e in scored) or "no candidate"
    matches = [pair for pair in scored if pair[0] <= FRAME_MATCH_MAX]
    if len(matches) != 1:
        raise RemeasureRefusedError(
            f"{len(matches)} re-rendered Short(s) frame-match {published.name} "
            f"(need exactly 1, distance <= {FRAME_MATCH_MAX}): {seen}"
        )
    if any(FRAME_MATCH_MAX < d < FRAME_DISTINCT_MIN for d, _ in scored):
        raise RemeasureRefusedError(
            f"ambiguous frame-match for {published.name}: a clip sits between "
            f"{FRAME_MATCH_MAX} and {FRAME_DISTINCT_MIN}: {seen}"
        )
    return matches[0][1], matches[0][0]


def _find_short(raw: Mapping[str, Any], row, sidecar: Path) -> dict:
    for short in raw.get("shorts") or []:
        if (
            short.get("project_name") == row["project_name"]
            and short.get("short_index") == row["short_index"]
        ):
            return short
    raise RemeasureRefusedError(f"{sidecar}: no short {row['project_name']}#{row['short_index']}")


def remeasure_short_route(
    row,
    stages_dir,
    geocoder,
    *,
    apply: bool = False,
    rendered_dir: Optional[Path] = None,
    signature_fn: Optional[Callable[[Path], bytes]] = None,
) -> Dict[str, Any]:
    """Measure the Short ``row`` points at and, with ``apply``, write its ``route_record``.

    ``stages_dir`` holds ``gps_points_<date>.json`` and ``shorts_ost_audit_<date>.json``.
    ``rendered_dir`` (stages from a re-render) is that render's Shorts directory: the clip
    is then the one whose rendered Short frame-matches the published media, not the one
    with the media's rank. Returns a summary; raises :class:`RemeasureRefusedError`
    (nothing written)."""
    sidecar = Path(row["sidecar_path"] or "")
    if not sidecar.is_file():
        raise RemeasureRefusedError(f"publish-plan sidecar not readable: {sidecar}")
    raw = json.loads(sidecar.read_text(encoding="utf-8"))
    short = _find_short(raw, row, sidecar)
    if "bins" in (short.get("route_record") or {}):
        raise RemeasureRefusedError("the route record already has 'bins'; nothing to re-measure")
    media = short.get("media") or {}
    date = short.get("recording_date")
    stages = Path(stages_dir)
    points = _read_json(stages / f"gps_points_{date}.json", "day GPS points")
    audit = _read_json(stages / f"shorts_ost_audit_{date}.json", "shorts audit")
    frame_distance = None
    if rendered_dir is not None:
        entry, frame_distance = _frame_matched_entry(
            short, audit.get("shorts") or [], Path(rendered_dir), signature_fn or media_signature
        )
        rank = int(entry["rank"])
    else:
        found = _RANK.search(Path(str(media.get("path", ""))).name)
        if not found:
            raise RemeasureRefusedError(
                f"cannot read the clip rank from media {media.get('path')!r}"
            )
        rank = int(found.group(1))
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
    if frame_distance is not None:
        summary["frame_distance"] = frame_distance
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
