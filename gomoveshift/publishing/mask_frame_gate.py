"""Publish gate: a day's blur mask must be built on the same frame count as its source.

``redactcam``'s detect pass decoded the source through an ffmpeg pipe that emitted more frames
than the container holds (306,385 vs 306,377 for Kerry-to-Cork), so the mask -- and the
``source_frame_count`` in ``day_<date>_blur_regions.json`` -- ran longer than the picture it is
paired with, and the blur lagged the picture. A drive rendered on that code is not safe to
publish until a re-render rewrites the sidecar.

``mask_frame_refusal`` is the one check every upload entry point calls (``health publish`` and
the backlog republish through it, and Phase D's upload run). It compares, per ``day_<date>``
stage dir, the sidecar's ``source_frame_count`` with the frames of the container the detect pass
read: the day's raw concat. The encode deletes the raw and cleanup's master-drop deletes the master
once the finals exist, so a finished day usually has neither. The raw is then rebuilt from its
inputs: the ``episode_*_<date>.mp4`` files whose size:mtime fingerprints the day's colour cache
recorded at encode (the concat is a stream copy, so its frames are theirs summed). The master
counts only when its frame rate is the mask's: a master encoded at another rate (half-rate for a
59.94 fps source) is in a different frame space; when nothing else can count such a day, its
``mask_frame_count_unknown`` reason names both rates. A mismatch, a missing sidecar or count, episodes
that are not the recorded ones, or no container that can be probed all refuse (fail closed). There is no override: the refusal ends when the counts agree.
A refused day gets one ``mask_index_mismatch`` incident (``health/store.py`` row) per drive-day,
and the row is resolved once the day passes."""

from __future__ import annotations

import json
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
from ..pipeline.cache import file_fingerprint

KIND = "mask_index_mismatch"
REASON_UNKNOWN = "mask_frame_count_unknown"
REASON_MISMATCH = "mask_frame_count_mismatch"
#: The sidecar keeps ``source_frame_count`` in its header; no need to parse the whole file.
_HEADER_BYTES = 4096
_COUNT_RE = re.compile(r'"source_frame_count"\s*:\s*(\d+)')
_FPS_RE = re.compile(r'"source_fps"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
#: A master within 1% of the mask's rate shares its frame space (the sidecar's rate is measured).
_RATE_TOLERANCE = 0.01
_PROBE_TIMEOUT_S = 180
#: Container candidates, in order: the raw concat the detect pass read, then the master.
_RAW_NAME = "day_{date}_raw.mp4"
_MASTER_NAME = "day_{date}_master.mp4"
_COLOUR_CACHE_NAME = "day_{date}_color_cache.json"


class MaskFrameCountError(RuntimeError):
    """Raised by ``assert_mask_frame_counts_agree``: the drive must not publish."""


def _sidecar_head(sidecar: Path) -> str:
    try:
        with open(sidecar, "rb") as fh:
            return fh.read(_HEADER_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return ""


def sidecar_frame_count(sidecar: Path) -> Optional[int]:
    """``source_frame_count`` from the sidecar header, or None (missing, unreadable, absent)."""
    found = _COUNT_RE.search(_sidecar_head(sidecar))
    return int(found.group(1)) if found else None


def sidecar_frame_rate(sidecar: Path) -> Optional[float]:
    """``source_fps`` from the sidecar header, or None."""
    found = _FPS_RE.search(_sidecar_head(sidecar))
    return float(found.group(1)) if found else None


def probe_frames(path: Path) -> Optional[int]:
    """The video stream's container ``nb_frames`` via ffprobe, or None when unreadable."""
    try:
        out = subprocess.run(  # noqa: S603,S607 -- fixed argv
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=nb_frames", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    return int(out) if out.isdigit() else None


def probe_rate(path: Path) -> Optional[float]:
    """The video stream's average frame rate via ffprobe, or None when unreadable."""
    try:
        out = subprocess.run(  # noqa: S603,S607 -- fixed argv
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=avg_frame_rate", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S, check=True,
        ).stdout.strip()  # fmt: skip
        num, _, den = out.partition("/")
        rate = float(num) / float(den or 1)
    except (OSError, subprocess.SubprocessError, ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def episode_frame_count(day_dir: Path, date: str) -> Optional[int]:
    """The raw concat's frames rebuilt from its inputs, or None unless the stage dir's
    ``episode_*_<date>.mp4`` files are exactly the ones the colour cache fingerprinted at encode
    and each one probes."""
    try:
        cache = json.loads((day_dir / _COLOUR_CACHE_NAME.format(date=date)).read_text("utf-8"))
        recorded = cache["source_fingerprints"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    episodes = sorted(day_dir.parent.glob(f"episode_*_{date}.mp4"))
    if not isinstance(recorded, list) or not recorded or not episodes:
        return None
    try:
        found = [file_fingerprint(p) for p in episodes]
    except OSError:
        return None
    if sorted(found) != sorted(recorded):
        return None
    total = 0
    for path in episodes:
        count = probe_frames(path)
        if not count:
            return None
        total += count
    return total


def container_frame_count(
    day_dir: Path, date: str, source_fps: Optional[float] = None
) -> Optional[int]:
    """Frames of the container the mask was detected on: the raw concat, else its recorded
    episodes, else a master at the mask's frame rate; None when none of them can say."""
    raw = day_dir / _RAW_NAME.format(date=date)
    if raw.is_file():
        count = probe_frames(raw)
        if count:
            return count
    count = episode_frame_count(day_dir, date)
    if count:
        return count
    master = day_dir / _MASTER_NAME.format(date=date)
    if source_fps and master.is_file():
        rate = probe_rate(master)
        if rate and abs(rate / source_fps - 1) <= _RATE_TOLERANCE:
            return probe_frames(master) or None
    return None


def _rate_note(day_dir: Path, date: str, source_fps: Optional[float]) -> Optional[str]:
    """Why an unknown day's master was not counted, when it was set aside for its frame rate."""
    master = day_dir / _MASTER_NAME.format(date=date)
    if not source_fps or not master.is_file():
        return None
    rate = probe_rate(master)
    if not rate or abs(rate / source_fps - 1) <= _RATE_TOLERANCE:
        return None
    return (f"master is {rate:.3f} fps, the mask's source {source_fps:.3f} fps, so the master "
            "cannot count it; the raw or the recorded episodes can")  # fmt: skip


def day_findings(stages_dir: Path) -> List[Dict[str, Any]]:
    """One row per rendered ``day_<date>`` stage dir (one holding any ``day_<date>_*`` file) that
    must not publish; ``[]`` when every such day agrees."""
    findings: List[Dict[str, Any]] = []
    for day_dir in sorted(Path(stages_dir).glob("day_*")):
        if not day_dir.is_dir():
            continue
        date = day_dir.name[len("day_") :]
        if not any(day_dir.glob(f"day_{date}_*")):
            continue  # no render artefacts (only shorts/work dirs): not a rendered day
        sidecar = day_dir / f"day_{date}_blur_regions.json"
        sidecar_n = sidecar_frame_count(sidecar)
        source_fps = sidecar_frame_rate(sidecar)
        container_n = container_frame_count(day_dir, date, source_fps)
        if sidecar_n is None or container_n is None:
            row = {"date": date, "reason": REASON_UNKNOWN,
                   "sidecar_frames": sidecar_n, "container_frames": container_n}  # fmt: skip
            note = _rate_note(day_dir, date, source_fps) if container_n is None else None
            findings.append({**row, "rate_note": note} if note else row)
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
