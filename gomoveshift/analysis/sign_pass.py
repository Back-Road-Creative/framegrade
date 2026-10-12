"""Per-day road-sign OCR evidence pass (Phase A).

Reads road signs out of the RAW, pre-blur episode footage and records what it
saw. Blur covers faces, plates and vehicle occupants — never signs — so this
pass is safe to run on the ingested episodes before any masking happens, and
running it there is what makes the evidence independent of every derived
surface.

**This module records evidence ONLY — it never writes a location anywhere.**
It produces :class:`~gomoveshift.analysis.sign_evidence.SignHit` objects keyed
by calendar day; the comparison against each LF's ``ResolvedLocation`` (and any
verdict drawn from it) lives downstream. Nothing here touches a title, an OST
string, a label, or a region.

Shape of the pass, per calendar day:

1. Group episodes with :func:`..pipeline.day_grouper.group_episodes_by_day` —
   the same grouper the day master itself is built from, so the day-master
   offsets computed here agree with the real master by construction rather
   than by a parallel reimplementation.
2. Walk the day's episodes in batch order with a running cursor
   (``ep_master_start = day_cursor; day_cursor += ep.total_duration``), exactly
   as ``gps_enrichment._build_gps_points_by_date`` does, and resolve each
   episode's GLOBAL index through ``episode_files`` so ``gps_per_episode`` is
   read at the right slot.
3. Keep only MOVING seconds (median ground speed over a short window around the
   second, via ``gps_quality.window_median_speed_mps``). A GPS-LESS episode
   keeps all its seconds — a day with no trace is precisely where independent
   sign evidence is worth the most — with ``lat``/``lon`` recorded as ``None``.
4. Stride the survivors evenly and cap the total per DAY, so the evidence is
   spread across the whole drive instead of clustering in its first minutes.
5. Grab ONE native-resolution frame per sampled second (targeted seek), find
   the sign region with the cheap colour/shape filter already in
   ``analysis.signs``, OCR only that crop, and parse claims out of the text.
   The region filter costs no extra decode (the frame is already in hand) and
   is what keeps OCR off the overwhelming majority of frames.

The OCR path is **purely local** — the frame never leaves the box. We call
``OCREngine._run_easyocr`` directly rather than ``process_frame`` so the sign
path can never route an image off-box even if the (currently disabled) Claude
fallback is ever revived.

Every seam is injectable (``grab_frame_fn`` / ``ocr_fn`` / ``parse_claims_fn``)
so tests never decode video, shell out, load EasyOCR, or hit the network.

Failure policy: the pass is ADVISORY. A dead frame grab, an OCR blow-up, or a
broken episode is logged and skipped — a multi-hour run must never die for a
piece of corroborating evidence. Only a setup failure (no usable detector /
config) propagates, and its single caller in ``phase_a`` catches it, warns, and
leaves ``state.sign_hits_by_date`` unset so the next resume retries.
"""

from __future__ import annotations

import bisect
import logging
import subprocess
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Optional

from .sign_evidence import SignClaim, SignHit

logger = logging.getLogger(__name__)

# Fallbacks for the ``config.drive`` knobs (read defensively so this module
# works with a config that predates them).
_DEFAULT_MAX_SAMPLES_PER_DAY = 120
_DEFAULT_MIN_SPEED_MPS = 5.0

# Half-width (seconds) of the window the moving/stationary median is taken
# over. Wide enough to span several 1 Hz fixes, narrow enough that a brief
# junction crawl inside a moving stretch does not veto its neighbours.
_SPEED_WINDOW_SEC = 5.0

# One targeted seek + single-frame decode. Generous enough for a cold seek deep
# into a 4K HEVC episode on a slow mount, bounded so a wedged ffmpeg cannot
# stall the pass.
_FRAME_GRAB_TIMEOUT_SEC = 60.0

# Matches the padding ``SignDetector.detect_candidates`` puts around a detected
# sign before handing the crop to OCR.
_SIGN_CROP_PAD_PX = 10


@dataclass
class _Candidate:
    """One sampled second, before any decode."""

    episode_index: int
    video_path: str
    t_local: float
    ep_master_start: float
    lat: Optional[float]
    lon: Optional[float]


def _default_grab_frame(video_path: Any, t_local: float):
    """One native-resolution BGR frame at ``t_local`` via a targeted seek.

    Fixed binary, argument list, no shell. A non-zero exit, an empty pipe, a
    timeout, or an undecodable payload yields ``None`` — a missed frame is a
    skipped sample, never an exception into Phase A.
    """
    from .frame_range import setrange_filter

    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-ss",
                f"{float(t_local):.3f}",
                "-i",
                str(video_path),
                # State the source range before the YUV->RGB hop the PNG
                # encode forces — the absolute HSV thresholds downstream
                # otherwise read a domain the footage never used.
                "-vf",
                setrange_filter(video_path),
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-",
            ],
            capture_output=True,
            timeout=_FRAME_GRAB_TIMEOUT_SEC,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("sign pass: frame grab failed at %s +%.1fs: %s", video_path, t_local, exc)
        return None
    if proc.returncode != 0 or not proc.stdout:
        logger.debug(
            "sign pass: empty frame grab at %s +%.1fs (rc=%d)",
            video_path,
            t_local,
            proc.returncode,
        )
        return None
    import cv2
    import numpy as np

    frame = cv2.imdecode(np.frombuffer(proc.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or getattr(frame, "size", 0) == 0:
        logger.debug("sign pass: undecodable frame at %s +%.1fs", video_path, t_local)
        return None
    return frame


class _LazyEasyOCR:
    """Default ``ocr_fn`` — ``OCREngine._run_easyocr`` on the sign crop.

    Constructed lazily so a run with an injected ``ocr_fn`` (and every test)
    never imports EasyOCR. ``_run_easyocr`` is called directly rather than
    ``process_frame`` so no image can ever be routed off-box.
    """

    def __init__(self, config) -> None:
        self._config = config
        self._engine = None

    def __call__(self, roi):
        if self._engine is None:
            from .ocr import OCREngine

            self._engine = OCREngine(self._config)
        return self._engine._run_easyocr(roi)


class _LazyClaimParser:
    """Default ``parse_claims_fn`` — ``ocr.parse_sign_claims``, imported at
    first CALL so this module never depends on that symbol's import timing."""

    def __init__(self, fn) -> None:
        self._fn = fn

    def __call__(self, text: str):
        if self._fn is None:
            from .ocr import parse_sign_claims

            self._fn = parse_sign_claims
        return self._fn(text)


def _claim_dicts(claims) -> list:
    """Parsed claims as plain JSON-native dicts (``SignHit.claims`` contract)."""
    out: list = []
    for claim in claims or []:
        if isinstance(claim, SignClaim) or (is_dataclass(claim) and not isinstance(claim, type)):
            out.append(asdict(claim))
        elif isinstance(claim, dict):
            out.append(dict(claim))
    return out


def _episode_points(gps_entry, duration: float) -> list[tuple[float, float, float]]:
    """Episode-local ``[(t_local, lat, lon), ...]``, sorted, null-island dropped.

    Mirrors ``gps_enrichment._build_gps_points_by_date``: the stored offset is
    corrected through ``shorts_models.effective_offset`` before
    ``t_local = timestamp - offset``, and points outside ``[0, duration]`` are
    dropped (a GPX routinely covers a wider range than the clipped episode).
    ``(0, 0)`` sentinels are rejected the same way every other GPS consumer
    here rejects them — a cold-start fix must never be recorded as the place a
    sign was read.
    """
    if not gps_entry:
        return []
    gps_data = gps_entry[0]
    if gps_data is None:
        return []
    stored_offset = float(gps_entry[1]) if len(gps_entry) > 1 and gps_entry[1] is not None else 0.0

    from .shorts_models import effective_offset

    offset = effective_offset(gps_data, stored_offset)

    trackpoints = getattr(gps_data, "trackpoints", None)
    if trackpoints is None and isinstance(gps_data, dict):
        trackpoints = gps_data.get("trackpoints") or []

    points: list[tuple[float, float, float]] = []
    for tp in trackpoints or []:
        ts = getattr(tp, "timestamp", None)
        lat = getattr(tp, "lat", None)
        lon = getattr(tp, "lon", None)
        if ts is None and isinstance(tp, dict):
            ts = tp.get("timestamp")
            lat = tp.get("lat")
            lon = tp.get("lon")
        if ts is None or lat is None or lon is None:
            continue
        try:
            t_local = float(ts) - offset
            lat_f = float(lat)
            lon_f = float(lon)
        except (TypeError, ValueError):
            continue
        if lat_f == 0.0 and lon_f == 0.0:
            continue  # null-island cold-start sentinel
        if t_local < 0.0 or t_local > duration:
            continue
        points.append((t_local, lat_f, lon_f))
    points.sort()
    return points


def _nearest_fix(points, times, t: float) -> tuple[Optional[float], Optional[float]]:
    """``(lat, lon)`` of the fix nearest ``t``, or ``(None, None)``."""
    if not points:
        return None, None
    i = bisect.bisect_left(times, t)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(points):
            if best is None or abs(points[j][0] - t) < abs(points[best][0] - t):
                best = j
    if best is None:
        return None, None
    return points[best][1], points[best][2]


def _moving_seconds(points, times, duration: float, min_speed_mps: float) -> list[int]:
    """Whole seconds of an episode whose window median ground speed clears the knob.

    The window is sliced with ``bisect`` before the call so the shared
    ``window_median_speed_mps`` runs over a handful of fixes per second rather
    than rescanning a whole day's trace — same measure, linear cost.

    An UNMEASURABLE window (fewer than two in-window fixes → ``None``) is kept:
    this pass is advisory evidence, so a GPS dropout should cost decode time,
    not evidence. Only a measured, genuinely-below-threshold second is vetoed.
    """
    from ..pipeline.gps_quality import window_median_speed_mps

    kept: list[int] = []
    for sec in range(max(0, int(duration))):
        t = float(sec)
        w0 = t - _SPEED_WINDOW_SEC
        w1 = t + _SPEED_WINDOW_SEC
        lo = bisect.bisect_left(times, w0)
        hi = bisect.bisect_right(times, w1)
        median = window_median_speed_mps(points[lo:hi], w0, w1)
        if median is not None and median < min_speed_mps:
            continue
        kept.append(sec)
    return kept


def _even_stride(items: list, cap: int) -> list:
    """At most ``cap`` items, strided evenly across the WHOLE list.

    Never exceeds the cap, always starts at the first item, and reaches into
    the final stride — so a capped day's evidence spans the drive instead of
    clustering in its opening minutes.
    """
    if cap <= 0:
        return []
    n = len(items)
    if n <= cap:
        return list(items)
    return [items[(i * n) // cap] for i in range(cap)]


def _find_sign_crop(detector, frame):
    """Padded crop of the highest-confidence sign region, or ``None``.

    Mirrors ``SignDetector.detect_candidates``: the sign band is
    ``roi_top_percent .. roi_bottom_percent`` of the frame height, the area
    bounds are ``min/max_sign_area_percent`` of the FRAME area in pixels, and
    the crop carries the same padding. Reusing ``_find_sign_regions`` keeps one
    detector definition instead of a second, drifting copy.
    """
    height, width = frame.shape[:2]
    frame_area = width * height
    if frame_area <= 0:
        return None
    y_start = int(height * detector.roi_top)
    y_end = int(height * detector.roi_bottom)
    if y_end <= y_start:
        return None

    regions = detector._find_sign_regions(
        frame[y_start:y_end, :],
        frame_area * detector.min_area,
        frame_area * detector.max_area,
    )
    if not regions:
        return None

    # ``_find_sign_regions`` returns confidence-descending, so [0] is the best.
    (x, y, w, h), _confidence = regions[0]
    top = max(0, y + y_start - _SIGN_CROP_PAD_PX)
    bottom = min(height, y + y_start + h + _SIGN_CROP_PAD_PX)
    left = max(0, x - _SIGN_CROP_PAD_PX)
    right = min(width, x + w + _SIGN_CROP_PAD_PX)
    crop = frame[top:bottom, left:right]
    return crop if getattr(crop, "size", 0) else None


def _episode_candidates(
    *,
    episode,
    episode_index: int,
    ep_master_start: float,
    duration: float,
    gps_entry,
    min_speed_mps: float,
) -> list[_Candidate]:
    """Sampleable seconds for one episode (moving-only when GPS is usable)."""
    points = _episode_points(gps_entry, duration)
    video_path = str(getattr(episode, "concat_path", "") or "")
    if points:
        times = [p[0] for p in points]
        seconds = _moving_seconds(points, times, duration, min_speed_mps)
    else:
        # GPS-less: keep every second. A day with no trace is exactly where
        # independent sign evidence matters most, so it is never skipped for
        # want of a speed reading.
        times = []
        seconds = list(range(max(0, int(duration))))

    out: list[_Candidate] = []
    for sec in seconds:
        t_local = float(sec)
        lat, lon = _nearest_fix(points, times, t_local)
        out.append(
            _Candidate(
                episode_index=episode_index,
                video_path=video_path,
                t_local=t_local,
                ep_master_start=ep_master_start,
                lat=lat,
                lon=lon,
            )
        )
    return out


def _hit_for_candidate(
    candidate: _Candidate,
    date: str,
    *,
    detector,
    grab_frame_fn,
    ocr_fn,
    parse_claims_fn,
) -> Optional[SignHit]:
    """One sampled second → a ``SignHit``, or ``None`` when it yields no claim."""
    frame = grab_frame_fn(candidate.video_path, candidate.t_local)
    if frame is None or getattr(frame, "size", 0) == 0:
        return None

    crop = _find_sign_crop(detector, frame)
    if crop is None:
        return None  # cheap filter — no sign region, so no OCR for this second

    result = ocr_fn(crop)
    if not result:
        return None
    text, confidence = result[0], result[1]
    if not text or not str(text).strip():
        return None

    claims = _claim_dicts(parse_claims_fn(str(text)))
    if not claims:
        # OCR noise that names no place is NOT evidence — discard it rather
        # than persisting unreadable text into state.
        return None

    return SignHit(
        episode_index=candidate.episode_index,
        t_local=candidate.t_local,
        ts_day=candidate.ep_master_start + candidate.t_local,
        date=date,
        raw_text=str(text),
        ocr_conf=float(confidence or 0.0),
        lat=candidate.lat,
        lon=candidate.lon,
        claims=claims,
    )


def collect_sign_hits(
    config,
    ingest_plan,
    episode_files: list,
    gps_per_episode: list,
    *,
    grab_frame_fn=None,
    ocr_fn=None,
    parse_claims_fn=None,
) -> dict[str, list[SignHit]]:
    """Read road signs out of a drive's raw footage, one bucket per calendar day.

    Args:
        config: the pipeline ``Config``. ``config.drive`` supplies
            ``sign_verify_max_samples_per_day`` / ``sign_verify_min_speed_mps``
            (both read defensively); ``config.sign_detection`` and
            ``config.ocr`` supply the detection / OCR thresholds — this pass
            adds no duplicate confidence knobs of its own.
        ingest_plan: the Phase A ingest plan (its ``episodes``).
        episode_files: the drive-wide ordered episode concat paths. An
            episode's position here is its GLOBAL index — the index a plan
            segment's ``source_file_index`` uses, and the one recorded on every
            hit.
        gps_per_episode: index-parallel ``(GPSData | None, gps_offset)``.
        grab_frame_fn: ``(video_path, t_local) -> BGR ndarray | None``.
        ocr_fn: ``(bgr_roi) -> (text, confidence) | None``.
        parse_claims_fn: ``(text) -> list[SignClaim]``.

    Returns:
        ``{date: [SignHit, ...]}`` sorted by ``ts_day``. EVERY grouped day is
        present, including days that yielded nothing — an empty list is the
        honest record "this day was read and had no legible sign", and it is
        what lets the Phase A back-fill be idempotent (a day that produced no
        evidence must not look un-processed and get re-decoded next resume).
    """
    from ..pipeline.day_grouper import group_episodes_by_day
    from .signs import SignDetector

    day_batches = group_episodes_by_day(ingest_plan)
    if not day_batches:
        return {}

    max_samples = int(
        getattr(config.drive, "sign_verify_max_samples_per_day", _DEFAULT_MAX_SAMPLES_PER_DAY)
        or _DEFAULT_MAX_SAMPLES_PER_DAY
    )
    min_speed_mps = float(
        getattr(config.drive, "sign_verify_min_speed_mps", _DEFAULT_MIN_SPEED_MPS) or 0.0
    )

    index_by_path = {str(path): i for i, path in enumerate(episode_files or [])}
    detector = SignDetector(config)
    grab = grab_frame_fn or _default_grab_frame
    ocr = ocr_fn or _LazyEasyOCR(config)
    parse = _LazyClaimParser(parse_claims_fn)

    hits_by_date: dict[str, list[SignHit]] = {}
    for batch in day_batches:
        date = getattr(batch, "date", None)
        if not date:
            continue

        candidates: list[_Candidate] = []
        day_cursor = 0.0
        for episode in getattr(batch, "episodes", None) or []:
            # Running cursor over the day's episodes in batch order — the same
            # computation the day master's own concat performs, so an
            # episode-local second maps to ``ep_master_start + t_local``.
            ep_master_start = day_cursor
            duration = float(getattr(episode, "total_duration", 0.0) or 0.0)
            day_cursor += duration

            episode_index = index_by_path.get(str(getattr(episode, "concat_path", "") or ""))
            if episode_index is None:
                logger.debug(
                    "sign pass: episode %r not in episode_files — skipped",
                    getattr(episode, "index", "?"),
                )
                continue
            gps_entry = (
                gps_per_episode[episode_index]
                if gps_per_episode and 0 <= episode_index < len(gps_per_episode)
                else None
            )
            try:
                candidates.extend(
                    _episode_candidates(
                        episode=episode,
                        episode_index=episode_index,
                        ep_master_start=ep_master_start,
                        duration=duration,
                        gps_entry=gps_entry,
                        min_speed_mps=min_speed_mps,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — one bad episode never kills the day
                logger.warning(
                    "sign pass: candidate scan failed for episode %d on %s: %s",
                    episode_index,
                    date,
                    exc,
                )

        sampled = _even_stride(candidates, max_samples)
        day_hits: list[SignHit] = []
        for candidate in sampled:
            try:
                hit = _hit_for_candidate(
                    candidate,
                    date,
                    detector=detector,
                    grab_frame_fn=grab,
                    ocr_fn=ocr,
                    parse_claims_fn=parse,
                )
            except Exception as exc:  # noqa: BLE001 — advisory: skip the second
                logger.debug(
                    "sign pass: sample skipped (episode %d +%.1fs): %s",
                    candidate.episode_index,
                    candidate.t_local,
                    exc,
                )
                continue
            if hit is not None:
                day_hits.append(hit)

        day_hits.sort(key=lambda h: h.ts_day)
        hits_by_date[date] = day_hits
        logger.info(
            "  Sign pass %s: %d hit(s) from %d sampled second(s) (%d candidate(s))",
            date,
            len(day_hits),
            len(sampled),
            len(candidates),
        )

    return hits_by_date
