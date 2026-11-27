"""Publish gate: a day's blur mask must be built on the same frame count as its source.

``redactcam``'s detect pass decoded the source through an ffmpeg pipe that emitted more frames
than the container holds (306,385 vs 306,377 for Kerry-to-Cork), so the mask -- and the
``source_frame_count`` in ``day_<date>_blur_regions.json`` -- ran longer than the picture it is
paired with, and the blur lagged the picture. A drive rendered on that code is not safe to
publish until a re-render rewrites the sidecar.

``mask_frame_refusal`` is the one check every upload entry point calls (``health publish`` and
the backlog republish through it, and Phase D's upload run). It compares, per ``day_<date>``
stage dir, the sidecar's ``source_frame_count`` with the container's ``nb_frames`` (the day's raw
concat, else its master). A mismatch, a missing sidecar or count, or a container that cannot be
probed all refuse (fail closed). When the raw and the master are both purged, the count the
pairing check persisted (``day_<date>_source_frames.json``) stands in for the container, but only
while the sidecar's sha256 and the mask's size/mtime still match disk; otherwise the day stays
unknown. The same proof stands in when the raw is gone and the master runs at another frame rate
than the sidecar's ``source_fps``: the mask pairs with source frames and the CFR conversion runs
after the composite, so a 29.97 fps master of a 59.94 fps day holds about half the mask's frames
and its count measures nothing (Galley Head, 2026-11-27). With no valid proof such a day is
unknown, naming both rates. There is no override: the refusal ends when the counts agree.
A refused day gets one ``mask_index_mismatch`` incident (``health/store.py`` row) per drive-day,
and the row is resolved once the day passes."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..health.store import (
    default_data_root,
    default_incidents_dir,
    new_incident,
    safe_name,
    utc_iso,
    write_row,
)

# ``persisted_source_count`` lives engine-side (``cleanup`` needs it and the engine may not import
# publishing); it and the names it reads are re-exported here so the gate's callers and tests
# are unchanged.
from ..pipeline.blur_gate import (  # noqa: F401
    SOURCE_FRAMES_SCHEMA,
    can_measure_without_master,
    file_sha256,
    mask_identity,
    persisted_source_count,
    probe_frames,
)

logger = logging.getLogger(__name__)

KIND = "mask_index_mismatch"
REASON_UNKNOWN = "mask_frame_count_unknown"
REASON_MISMATCH = "mask_frame_count_mismatch"
#: The sidecar keeps ``source_frame_count`` in its header; no need to parse the whole file.
_HEADER_BYTES = 4096
_COUNT_RE = re.compile(r'"source_frame_count"\s*:\s*(\d+)')
_FPS_RE = re.compile(r'"source_fps"\s*:\s*(\d+(?:\.\d+)?)')
#: Rates closer than this are one cadence (``color_filters.source_output_fps`` uses the same).
_RATE_TOLERANCE = 0.01
_RATE_PROBE_TIMEOUT_S = 180
#: Container candidates, in order: the raw concat the detect pass read, then the master.
_CONTAINER_NAMES = ("day_{date}_raw.mp4", "day_{date}_master.mp4")


class MaskFrameCountError(RuntimeError):
    """Raised by ``assert_mask_frame_counts_agree``: the drive must not publish."""


def _sidecar_header(sidecar: Path) -> str:
    try:
        with open(sidecar, "rb") as fh:
            return fh.read(_HEADER_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return ""


def sidecar_frame_count(sidecar: Path) -> Optional[int]:
    """``source_frame_count`` from the sidecar header, or None (missing, unreadable, absent)."""
    found = _COUNT_RE.search(_sidecar_header(sidecar))
    return int(found.group(1)) if found else None


def sidecar_source_fps(sidecar: Path) -> Optional[float]:
    """``source_fps`` from the sidecar header (the rate the mask's frames run at), or None."""
    found = _FPS_RE.search(_sidecar_header(sidecar))
    return float(found.group(1)) if found else None


def probe_rate(path: Path) -> Optional[float]:
    """The video stream's ``avg_frame_rate`` via ffprobe, or None when unreadable or 0/0."""
    try:
        out = subprocess.run(  # noqa: S603,S607 -- fixed argv
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=avg_frame_rate", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=_RATE_PROBE_TIMEOUT_S, check=True,
        ).stdout.strip()  # fmt: skip
        num, _, den = out.partition("/")
        rate = float(num) / float(den or 1)
    except (OSError, subprocess.SubprocessError, ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def container_frame_count(day_dir: Path, date: str) -> Optional[int]:
    for pattern in _CONTAINER_NAMES:
        path = day_dir / pattern.format(date=date)
        if path.is_file():
            count = probe_frames(path)
            if count:
                return count
    return None


def _has_finals(day_dir: Path) -> bool:
    """True when the dir holds any rendered final (an LF, short or superlong part)."""
    return any(any(day_dir.glob(f"{sub}/*.mp4")) for sub in ("regular_lfs", "shorts", "superlong"))


def day_findings(stages_dir: Path) -> List[Dict[str, Any]]:
    """One row per rendered ``day_<date>`` stage dir (one holding any ``day_<date>_*`` file) that
    must not publish; ``[]`` when every such day agrees."""
    findings: List[Dict[str, Any]] = []
    for day_dir in sorted(Path(stages_dir).glob("day_*")):
        if not day_dir.is_dir():
            continue
        date = day_dir.name[len("day_") :]
        if not any(day_dir.glob(f"day_{date}_*")):
            if _has_finals(day_dir):
                # Renders with no sidecar, mask or container to measure: unmeasured, so refused.
                findings.append({"date": date, "reason": REASON_UNKNOWN,
                                 "sidecar_frames": None, "container_frames": None})  # fmt: skip
            else:
                logger.warning("mask gate: %s holds no day_%s_* file and no finals; skipped",
                               day_dir, date)  # fmt: skip
            continue
        sidecar_n = sidecar_frame_count(day_dir / f"day_{date}_blur_regions.json")
        container_n = container_frame_count(day_dir, date)
        if container_n is None and not any(
            os.path.lexists(day_dir / name.format(date=date)) for name in _CONTAINER_NAMES
        ):
            container_n = persisted_source_count(day_dir, date)
        rate_note = None
        if container_n is not None and not os.path.lexists(day_dir / f"day_{date}_raw.mp4"):
            # the count came from the master: it measures source frames only at the source rate
            master_fps = probe_rate(day_dir / f"day_{date}_master.mp4")
            src_fps = sidecar_source_fps(day_dir / f"day_{date}_blur_regions.json")
            if master_fps and src_fps and abs(master_fps - src_fps) >= _RATE_TOLERANCE:
                container_n = persisted_source_count(day_dir, date)
                if container_n is None:
                    rate_note = (f"master is {master_fps:.3f} fps, the mask's source {src_fps:.3f} "
                                 f"fps, and no valid day_{date}_source_frames.json")  # fmt: skip
        if sidecar_n is None or container_n is None:
            row = {"date": date, "reason": REASON_UNKNOWN,
                   "sidecar_frames": sidecar_n, "container_frames": container_n}  # fmt: skip
            findings.append({**row, "rate_note": rate_note} if rate_note else row)
        elif sidecar_n != container_n:
            findings.append({"date": date, "reason": REASON_MISMATCH, "sidecar_frames": sidecar_n,
                             "container_frames": container_n, "difference": sidecar_n - container_n})  # fmt: skip
    return findings


def _describe(drive: str, row: Dict[str, Any]) -> str:
    if row["reason"] == REASON_UNKNOWN:
        note = f"; {row['rate_note']}" if row.get("rate_note") else ""
        return (f"{REASON_UNKNOWN}: {drive} day {row['date']} (blur sidecar frames "
                f"{row['sidecar_frames']}, container frames {row['container_frames']}{note})")  # fmt: skip
    return (f"{REASON_MISMATCH}: {drive} day {row['date']} blur mask has {row['sidecar_frames']} "
            f"frames, source has {row['container_frames']} (difference {row['difference']:+d}); "
            "re-render the day on the fixed renderer")  # fmt: skip


def _key(drive: str, date: str) -> str:
    return f"{KIND}:{safe_name(drive)}:{date}"


def _row_path(incidents_dir: Path, drive: str, date: str) -> Path:
    return incidents_dir / f"{KIND}_{safe_name(drive)}_{date}.json"


def _record(incidents_dir: Path, drive: str, row: Dict[str, Any], text: str) -> None:
    """One open incident per drive-day: an existing unresolved row is left as it is."""

    path = _row_path(incidents_dir, drive, row["date"])
    try:
        prior = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(prior, dict) and not prior.get("resolved_at"):
            return
    except (OSError, ValueError):
        pass
    write_row(
        incidents_dir,
        new_incident(KIND, _key(drive, row["date"]), path.name, utc_iso(), drive=drive,
                     day=row["date"], error=text, detail=text, **{k: v for k, v in row.items() if k != "date"}),
    )  # fmt: skip


def _resolve_passed(incidents_dir: Path, drive: str, failing: set, stages_dir: Path) -> None:

    for day_dir in Path(stages_dir).glob("day_*"):
        date = day_dir.name[len("day_") :]
        path = _row_path(incidents_dir, drive, date)
        if date in failing or not path.is_file():
            continue
        try:
            prior = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(prior, dict) and not prior.get("resolved_at"):
            prior["resolved_at"] = utc_iso()
            write_row(incidents_dir, prior)


def mask_frame_refusal(
    stages_dir: Path,
    *,
    drive: str,
    incidents_dir: Optional[Path] = None,
    record: bool = True,
    findings_fn: Callable[[Path], List[Dict[str, Any]]] = day_findings,
) -> Optional[str]:
    """None when every day's mask matches its source, else the blocking reason (one line per
    day). ``record`` writes the ``mask_index_mismatch`` incidents (a dry run passes False)."""
    findings = findings_fn(Path(stages_dir))
    if record:
        inc = incidents_dir or default_incidents_dir(default_data_root())
        for row in findings:
            _record(inc, drive, row, _describe(drive, row))
        _resolve_passed(inc, drive, {r["date"] for r in findings}, Path(stages_dir))
    if not findings:
        return None
    return "; ".join(_describe(drive, row) for row in findings)


def assert_mask_frame_counts_agree(stages_dir: Path, *, drive: str) -> None:
    reason = mask_frame_refusal(stages_dir, drive=drive)
    if reason:
        raise MaskFrameCountError(f"Publish blocked, {reason}")
