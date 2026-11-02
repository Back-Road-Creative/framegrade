"""Slice one whole-day GPX into per-file ``<stem>.gpx`` sidecars (sidecar contract).

The generic ingest reads a ``<stem>.gpx`` beside each source video as that
episode's GPS (``create_ingest_plan``). Cameras with no GPMF (Insta360) are
fed from a day-long GPX; this module cuts that file into one sidecar per
video so each sidecar passes the existing Phase-A correspondence gates
(``assert_gpx_video_correspondence``) UNCHANGED.

Window per video, in UTC: ``[start - HEAD_PAD, start + duration + TAIL_PAD]``.

* ``start`` is the camera-local time in the ``VID_YYYYMMDD_HHMMSS_...``
  filename shifted by the camera clock's UTC offset. With no offset given it is
  MEASURED: per recording, filename local time minus the first chapter's ffprobe
  ``creation_time`` (UTC). Every recording must agree, the value must be a
  whole multiple of 15 min within 60 s and inside -12..+14 h, and the slice must
  then pass the Phase-A gate. Any failure refuses, naming the clips and values;
  an offset is never guessed. A given offset is cross-checked the same way and
  refused when ``creation_time`` disagrees by more than 60 s.
* A long recording is split into chapters that all carry the recording's
  ``HHMMSS``; only the trailing sequence number moves. A chapter starts where
  the one before it ends, so its ``start`` adds the durations of the earlier
  chapters. A chapter whose predecessor is missing is refused: its start
  cannot be known. (The Phase-A gate allows 60 min of start skew, so a
  30-min chapter sliced at the recording start would pass it.)
* ``TAIL_PAD`` runs past the clip end so the last segment keeps an
  authoritative endpoint (A2B-ENDPOINT-LOSS: a slice stopping exactly at the
  clip end loses it).
* HEAD+TAIL must stay inside the 25% / 120 s span gate, so a very short clip
  is refused loudly by the real gate, never by a loosened one.

Only ``VID_YYYYMMDD_HHMMSS_NN_NNN.mp4`` names carry a usable clock (strict
filename contract); ``LRV_*`` proxies are skipped and any other ``.mp4`` is
refused. A ``creation_time`` equal to the filename time measures 0 h (a container
clock that is local-as-Z); it is accepted only if the GPX gate passes.

A sidecar this module wrote carries ``creator="gms-day-gpx-slicer"``; only such
files are rewritten. Any other existing ``<stem>.gpx`` is the operator's and
is never overwritten.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ...analysis.gps_parser import parse_gpx_file
from ..gopro_gps import GpxCorrespondenceError, assert_gpx_video_correspondence
from .gopro_parser import _ffprobe_creation_time, get_clip_duration

logger = logging.getLogger(__name__)

HEAD_PAD = 15.0  # seconds of lead-in
TAIL_PAD = 120.0  # seconds past clip end (A2B-ENDPOINT-LOSS)
MARKER = "gms-day-gpx-slicer"
CREATION_TIME_TOLERANCE = 60.0  # seconds; beyond this a clock disagreement is refused
OFFSET_GRID = 900.0  # seconds; real UTC offsets are whole multiples of 15 min
OFFSET_RANGE_H = (-12.0, 14.0)  # inclusive; every inhabited UTC offset

VID_RE = re.compile(r"^VID_(\d{8})_(\d{6})_(\d{2})_(\d{3})\.mp4$", re.IGNORECASE)
TRKPT = re.compile(r"<trkpt\b.*?</trkpt>", re.DOTALL)
TIME = re.compile(r"<time>([^<]+)</time>")


class DayGpxSliceError(RuntimeError):
    """The day GPX cannot be sliced for a source file; the run must stop."""


@dataclass(frozen=True)
class SliceResult:
    sidecar: Path
    points: int
    start_utc: datetime
    duration: float


def _parse_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _load_points(path: Path) -> list[tuple[datetime, str]]:
    pts = []
    for m in TRKPT.finditer(path.read_text()):
        t = TIME.search(m.group(0))
        if t:
            pts.append((_parse_iso(t.group(1)), m.group(0)))
    pts.sort(key=lambda p: p[0])
    return pts


def _is_own(sidecar: Path) -> bool:
    try:
        return f'creator="{MARKER}"' in sidecar.read_text()[:512]
    except OSError:
        return False


def _fmt_h(hours: float) -> str:
    return f"{hours:+g} h"


def _zero_hint(measured: bool, hours: float) -> str:
    return (
        "; the measured offset is 0 h, which may be a local-as-Z container clock"
        if measured and hours == 0
        else ""
    )


def _measure_offset(firsts: list[tuple[Path, str, str, float]]) -> float:
    """The camera UTC offset (hours) from each recording's first chapter, or refuse.

    ``firsts`` is ``(video, date, hms, ...)`` per recording; offset = filename local time minus
    ffprobe ``creation_time`` in UTC. Every value must sit within ``CREATION_TIME_TOLERANCE`` of
    the ``OFFSET_GRID``, all must agree, and the result must be inside ``OFFSET_RANGE_H``."""
    raw: list[tuple[Path, float]] = []
    missing = []
    for video, date, hms, _lead in firsts:
        ct = _ffprobe_creation_time(video)
        if ct is None:
            missing.append(video.name)
            continue
        ct = ct if ct.tzinfo else ct.replace(tzinfo=timezone.utc)
        local = datetime.strptime(date + hms, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        raw.append((video, (local - ct).total_seconds()))
    if missing:
        raise DayGpxSliceError(
            f"cannot measure the camera clock offset: no ffprobe creation_time for "
            f"{', '.join(missing)}; pass --camera-utc-offset-hours"
        )
    off_grid = [
        f"{v.name} ({_fmt_h(sec / 3600)})"
        for v, sec in raw
        if abs(sec - round(sec / OFFSET_GRID) * OFFSET_GRID) > CREATION_TIME_TOLERANCE
    ]
    if off_grid:
        raise DayGpxSliceError(
            "measured camera clock offset is not a whole multiple of 15 min (within "
            f"{CREATION_TIME_TOLERANCE:.0f} s): {', '.join(off_grid)}"
        )
    snapped = {v.name: round(sec / OFFSET_GRID) * OFFSET_GRID / 3600 for v, sec in raw}
    if len(set(snapped.values())) > 1:
        detail = ", ".join(f"{n} {_fmt_h(h)}" for n, h in snapped.items())
        raise DayGpxSliceError(f"measured camera clock offsets disagree: {detail}")
    hours = next(iter(snapped.values()))
    lo, hi = OFFSET_RANGE_H
    if not lo <= hours <= hi:
        raise DayGpxSliceError(
            f"measured camera clock offset {_fmt_h(hours)} is outside {lo:+g}..{hi:+g} h "
            f"({', '.join(snapped)})"
        )
    return hours


def slice_day_gpx(
    day_gpx: Path, source_dir: Path, utc_offset_hours: float | None = None
) -> list[SliceResult]:
    """Write ``<stem>.gpx`` beside every ``VID_*.mp4`` in ``source_dir``.

    ``utc_offset_hours`` is the camera clock's UTC offset (local = UTC + N, so
    US Eastern daylight time is ``-4``). Left ``None`` it is measured from the first
    chapters' ffprobe ``creation_time`` (see the module docstring); given, it must agree with
    that ``creation_time`` to within 60 s. Raises :class:`DayGpxSliceError`
    before writing anything it cannot stand behind; all sidecars are validated
    before any is written.
    """
    day_gpx, source_dir = Path(day_gpx), Path(source_dir)
    if not day_gpx.is_file():
        raise DayGpxSliceError(f"day GPX not found: {day_gpx}")
    # LRV_* are Insta360 low-res proxies; ingest never reads them, so they get no sidecar.
    videos = sorted(
        p
        for p in source_dir.iterdir()
        if p.suffix.lower() == ".mp4" and not p.name.upper().startswith("LRV_")
    )
    if not videos:
        raise DayGpxSliceError(f"no .mp4 files in {source_dir}")
    points = _load_points(day_gpx)

    clips = []
    for video in videos:
        m = VID_RE.match(video.name)
        if not m:
            raise DayGpxSliceError(
                f"{video.name}: cannot read camera time (expected VID_YYYYMMDD_HHMMSS_...)"
            )
        dur = get_clip_duration(video)
        if dur <= 0:
            raise DayGpxSliceError(f"{video.name}: ffprobe duration unavailable")
        clips.append((video, m.group(1, 2, 3), int(m.group(4)), dur))
    # Seconds of earlier chapters of the same recording before each clip.
    lead: dict[Path, float] = {}
    for key in {k for _, k, _, _ in clips}:
        chapters = sorted((seq, video, dur) for video, k, seq, dur in clips if k == key)
        elapsed = 0.0
        for (seq, video, dur), prev in zip(chapters, [None, *chapters]):
            if prev is not None and seq != prev[0] + 1:
                raise DayGpxSliceError(
                    f"{video.name}: chapter of a recording whose previous chapter "
                    f"(sequence {prev[0] + 1:03d}) is missing; its start cannot be known"
                )
            lead[video] = elapsed
            elapsed += dur

    measured = utc_offset_hours is None
    if measured:
        firsts = [(v, d, h, 0.0) for v, (d, h, _l), _s, _u in clips if lead[v] == 0.0]
        utc_offset_hours = _measure_offset(firsts)
        logger.info(
            "day-gpx: measured camera UTC offset %+g h from creation_time of %s",
            utc_offset_hours, ", ".join(v.name for v, *_ in firsts),
        )  # fmt: skip
    planned: list[tuple[Path, str, SliceResult]] = []
    for video, (date, hms, _lens), _seq, dur in clips:
        local = datetime.strptime(date + hms, "%Y%m%d%H%M%S")
        start = (local - timedelta(hours=utc_offset_hours)).replace(tzinfo=timezone.utc)
        start += timedelta(seconds=lead[video])
        ct = _ffprobe_creation_time(video)
        if ct is not None and not measured:
            if ct.tzinfo is None:
                ct = ct.replace(tzinfo=timezone.utc)
            skew = abs((ct - start).total_seconds())
            if skew > CREATION_TIME_TOLERANCE:
                # start = local - offset + lead, so offset = local + lead - creation_time.
                implied = (start + timedelta(hours=utc_offset_hours) - ct).total_seconds() / 3600
                raise DayGpxSliceError(
                    f"{video.name}: ffprobe creation_time {ct.isoformat()} differs from the "
                    f"filename-derived UTC start {start.isoformat()} by {skew:.0f} s; given offset "
                    f"{_fmt_h(utc_offset_hours)}, measured {_fmt_h(implied)} (or the container "
                    "clock is local-as-Z)"
                )
        lo = start - timedelta(seconds=HEAD_PAD)
        hi = start + timedelta(seconds=dur + TAIL_PAD)
        keep = [blob for ts, blob in points if lo <= ts <= hi]
        if not keep:
            raise DayGpxSliceError(
                f"{video.name}: no trackpoints in {lo.isoformat()}..{hi.isoformat()} "
                f"(camera offset {utc_offset_hours:+g} h wrong, or day GPX is another day)"
                + _zero_hint(measured, utc_offset_hours)
            )
        sidecar = video.with_suffix(".gpx")
        if sidecar.exists() and not _is_own(sidecar):
            raise DayGpxSliceError(
                f"{sidecar.name}: existing sidecar not written by {MARKER}; refusing to overwrite"
            )
        doc = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<gpx version="1.1" creator="{MARKER}">\n<trk><trkseg>\n'
            + "\n".join(keep)
            + "\n</trkseg></trk>\n</gpx>\n"
        )
        planned.append((sidecar, doc, SliceResult(sidecar, len(keep), start, dur)))

    # Validate every slice against the real Phase-A gate before writing any.
    for sidecar, doc, res in planned:
        tmp = sidecar.with_name(sidecar.name + ".slice-check")
        tmp.write_text(doc)
        try:
            assert_gpx_video_correspondence(parse_gpx_file(tmp), res.start_utc, res.duration)
        except GpxCorrespondenceError as exc:
            raise DayGpxSliceError(
                f"{sidecar.name}: sliced window fails Phase-A gate: {exc}"
                + _zero_hint(measured, utc_offset_hours)
            ) from exc
        finally:
            tmp.unlink(missing_ok=True)

    for sidecar, doc, res in planned:
        sidecar.write_text(doc)
        logger.info("day-gpx slice: %s: %d pts (start %s)", sidecar.name, res.points, res.start_utc)
    return [r for _, _, r in planned]
