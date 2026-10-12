"""Slice one whole-day GPX into per-file ``<stem>.gpx`` sidecars (sidecar contract).

The generic ingest reads a ``<stem>.gpx`` beside each source video as that
episode's GPS (``create_ingest_plan``). Cameras with no GPMF (Insta360) are
fed from a day-long GPX; this module cuts that file into one sidecar per
video so each sidecar passes the existing Phase-A correspondence gates
(``assert_gpx_video_correspondence``) UNCHANGED.

Window per video, in UTC: ``[start - HEAD_PAD, start + duration + TAIL_PAD]``.

* ``start`` is the camera-local time in the ``VID_YYYYMMDD_HHMMSS_...``
  filename shifted by the operator-supplied UTC offset. The offset is never
  guessed: a wrong one yields an empty window (refused) or a gate failure.
* ``TAIL_PAD`` runs past the clip end so the last segment keeps an
  authoritative endpoint (A2B-ENDPOINT-LOSS: a slice stopping exactly at the
  clip end loses it).
* HEAD+TAIL must stay inside the 25% / 120 s span gate, so a very short clip
  is refused loudly by the real gate, never by a loosened one.

Only ``VID_YYYYMMDD_HHMMSS_NN_NNN.mp4`` names carry a usable clock (strict
filename contract); ``LRV_*`` proxies are skipped and any other ``.mp4`` is
refused. ffprobe ``creation_time`` is cross-checked: a >60 s disagreement with
the filename-derived start is logged (Phase A gates on ``creation_time``).

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
CREATION_TIME_TOLERANCE = 60.0  # seconds; beyond this a clock disagreement is logged

VID_RE = re.compile(r"^VID_(\d{8})_(\d{6})_\d{2}_\d{3}\.mp4$", re.IGNORECASE)
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


def slice_day_gpx(day_gpx: Path, source_dir: Path, utc_offset_hours: float) -> list[SliceResult]:
    """Write ``<stem>.gpx`` beside every ``VID_*.mp4`` in ``source_dir``.

    ``utc_offset_hours`` is the camera clock's UTC offset (local = UTC + N, so
    US Eastern daylight time is ``-4``). Raises :class:`DayGpxSliceError`
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

    planned: list[tuple[Path, str, SliceResult]] = []
    for video in videos:
        m = VID_RE.match(video.name)
        if not m:
            raise DayGpxSliceError(
                f"{video.name}: cannot read camera time (expected VID_YYYYMMDD_HHMMSS_...)"
            )
        local = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        start = (local - timedelta(hours=utc_offset_hours)).replace(tzinfo=timezone.utc)
        ct = _ffprobe_creation_time(video)
        if ct is not None:
            if ct.tzinfo is None:
                ct = ct.replace(tzinfo=timezone.utc)
            skew = abs((ct - start).total_seconds())
            if skew > CREATION_TIME_TOLERANCE:
                logger.warning(
                    "%s: ffprobe creation_time %s differs from filename-derived UTC start %s "
                    "by %.0f s (offset %+g h wrong, or the container clock is local-as-Z); "
                    "Phase A checks against creation_time",
                    video.name,
                    ct.isoformat(),
                    start.isoformat(),
                    skew,
                    utc_offset_hours,
                )
        dur = get_clip_duration(video)
        if dur <= 0:
            raise DayGpxSliceError(f"{video.name}: ffprobe duration unavailable")
        lo = start - timedelta(seconds=HEAD_PAD)
        hi = start + timedelta(seconds=dur + TAIL_PAD)
        keep = [blob for ts, blob in points if lo <= ts <= hi]
        if not keep:
            raise DayGpxSliceError(
                f"{video.name}: no trackpoints in {lo.isoformat()}..{hi.isoformat()} "
                f"(camera offset {utc_offset_hours:+g} h wrong, or day GPX is another day)"
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
            ) from exc
        finally:
            tmp.unlink(missing_ok=True)

    for sidecar, doc, res in planned:
        sidecar.write_text(doc)
        logger.info("day-gpx slice: %s: %d pts (start %s)", sidecar.name, res.points, res.start_utc)
    return [r for _, _, r in planned]
