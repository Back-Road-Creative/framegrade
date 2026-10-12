"""Per-day STOP mask — the structural veto that keeps parked/roadside footage
out of every planned surface (shorts, LFs).

Root cause: a published short once shipped extended parked/roadside footage. The
fix is robust-by-construction — a single per-day boolean array where ``True``
marks a second that is a STOP (vehicle not meaningfully moving). Downstream
planners consume the mask so they *cannot* select a masked second, rather than
relying on a late gate to catch a stopped clip after it was already picked.

Design choices (over-excluding a stop is SAFER than shipping one):

* GPS speed is authoritative and ADDITIVE. A sustained sub-``stop_speed_kmh``
  span is a stop even when optical motion is high (parked with passing
  traffic), and parked-GPS-drift phantom crawl (0.5–2 km/h) is still below the
  threshold, so it masks too.
* When a second's speed can't be measured (no GPS fixes in its window, or a
  GPS-less day) the mask FALLS BACK to optical motion energy — it never
  auto-passes an unmeasurable second.
* Only SUSTAINED runs mask: a single slow second at a traffic light must not
  veto a window. A run of ``candidate`` seconds must be at least
  ``min_stop_duration_s`` long to count.

Pure, deterministic, no I/O. Scores are 1 Hz (one row per second); a GPS point
is a ``(t_master, lat, lon)`` tuple in the same 0-based day-master seconds as
``day_scores``. Speed is always derived from coordinate deltas via
``gps_quality.window_median_speed_mps`` — GPS points carry no speed channel.

``build_gps_gap_mask`` is the shorts-only companion veto: on a day with GPS it
marks every second a short could not vote its own place over, so the shorts
planner cannot pick a clip that would have to borrow a place.
"""

from __future__ import annotations

import logging

import numpy as np

from .gps_quality import _in_window_coords, usable_fix_times, window_median_speed_mps

logger = logging.getLogger(__name__)

# Column index of per-second optical motion energy (mirror unified_scorer).
COL_MOTION_ENERGY = 2


def stop_mask_kwargs(config) -> dict:
    """The three ``build_stop_mask`` knobs read from ``config.stop_detection``.

    ``stop_detection`` is a TOP-LEVEL ``Config`` block (``config.py``), not a
    ``DriveConfig`` field. One reader for every mask builder — the day
    pipeline's per-day mask and the day-first drive plan's drive-wide mask —
    so the two can never drift. Falls back to ``config.drive.stop_detection``
    (duck-typed test configs), then to the builder's own defaults.
    """
    sd = getattr(config, "stop_detection", None) if config is not None else None
    if sd is None:
        drive = getattr(config, "drive", None) if config is not None else None
        sd = getattr(drive, "stop_detection", None) if drive is not None else None
    stop_speed_kmh, min_stop_dur, optical = 10.0, 5.0, 0.008
    if sd is not None:
        min_stop_dur = float(getattr(sd, "min_stop_duration", min_stop_dur))
        optical = float(getattr(sd, "motion_threshold", optical))
        gsc = getattr(sd, "gps_speed_corroboration", None)
        if gsc is not None:
            stop_speed_kmh = float(getattr(gsc, "stop_speed_kmh", stop_speed_kmh))
    return {
        "stop_speed_kmh": stop_speed_kmh,
        "min_stop_duration_s": min_stop_dur,
        "optical_motion_threshold": optical,
    }


def _sustained_runs(candidate: np.ndarray, min_len_s: float) -> np.ndarray:
    """Keep only maximal contiguous ``True`` runs at least ``min_len_s`` long.

    Each row is one second, so a run's length in seconds equals the number of
    consecutive ``True`` rows. A single slow second (run length 1) never
    survives; a run must reach ``min_len_s`` seconds to be masked.
    """
    n = len(candidate)
    mask = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if candidate[i]:
            j = i
            while j < n and candidate[j]:
                j += 1
            if (j - i) >= min_len_s:
                mask[i:j] = True
            i = j
        else:
            i += 1
    return mask


def build_stop_mask(
    day_scores: np.ndarray,
    day_gps_points,
    *,
    stop_speed_kmh: float = 10.0,
    min_stop_duration_s: float = 5.0,
    optical_motion_threshold: float = 0.008,
    speed_window_s: float = 5.0,
) -> np.ndarray:
    """Build a per-second STOP mask for one day's scored timeline.

    Args:
        day_scores: (N, >=3) score array, one row per second, timestamps
            rebased to 0. Column ``COL_MOTION_ENERGY`` is the optical fallback.
        day_gps_points: the day's GPS trace as ``[(t_master, lat, lon), ...]``
            in the SAME 0-based day-master seconds as ``day_scores``, or None
            for a GPS-less day.
        stop_speed_kmh: ground speed below which a second is "stopped".
        min_stop_duration_s: the minimum length (seconds) of a contiguous
            stopped run before it masks — shorter blips (a light) don't.
        optical_motion_threshold: motion-energy floor used when a second's GPS
            speed can't be measured.
        speed_window_s: width of the CENTERED window (seconds) over which a
            second's median GPS speed is measured.

    Returns:
        A boolean ``np.ndarray`` of length ``len(day_scores)`` — ``True`` marks
        a STOP second to EXCLUDE. Empty array when ``day_scores`` is empty.
    """
    n = int(len(day_scores)) if day_scores is not None else 0
    if n == 0:
        return np.zeros(0, dtype=bool)

    stop_speed_mps = stop_speed_kmh / 3.6
    have_gps = day_gps_points is not None and len(day_gps_points) >= 2
    motion = np.asarray(day_scores[:, COL_MOTION_ENERGY], dtype=np.float64)
    half = speed_window_s / 2.0

    candidate = np.zeros(n, dtype=bool)
    for i in range(n):
        speed = None
        if have_gps:
            w0 = max(0.0, i - half)
            w1 = min(float(n), i + half)
            speed = window_median_speed_mps(day_gps_points, w0, w1)
        if speed is not None:
            # GPS authoritative + additive: a sub-threshold span is a stop even
            # when optical motion is high (parked with passing traffic).
            candidate[i] = speed < stop_speed_mps
        else:
            # No measurable speed here — never auto-pass; fall back to optical.
            candidate[i] = float(motion[i]) < optical_motion_threshold

    return _sustained_runs(candidate, min_stop_duration_s)


# A short may name only a place its own clip's GPS votes (quorum: two fixes). A
# second inside a fix-to-fix gap longer than this — or inside a lead before the
# first fix / a tail after the last one longer than it — is vetoed for shorts.
# 7 s keeps every window of the planner's 15-s minimum holding at least two
# fixes, so every window left open can vote its own place.
GPS_MAX_FIX_GAP_S = 7.0


def build_gps_gap_mask(
    n: int, day_gps_points, *, max_gap_s: float = GPS_MAX_FIX_GAP_S
) -> np.ndarray:
    """Per-second shorts veto for one day: ``True`` marks a second no short may
    cover, because it sits in a GPS gap its clip could not vote a place over.

    A second's gap runs from the last usable fix before it (or the start of the
    day) to the first usable fix at or after it (or the end of the day); the
    second is vetoed when that gap exceeds ``max_gap_s``. A GPS-less day (no
    usable fix) vetoes nothing — its shorts keep the operator/county anchor.
    """
    mask = np.zeros(max(int(n), 0), dtype=bool)
    fixes = np.asarray(usable_fix_times(day_gps_points), dtype=np.float64)
    if mask.size == 0 or fixes.size == 0:
        return mask
    seconds = np.arange(mask.size, dtype=np.float64)
    nxt = np.searchsorted(fixes, seconds, side="left")
    next_t = np.where(nxt < fixes.size, fixes[np.minimum(nxt, fixes.size - 1)], float(mask.size))
    prev_t = np.where(nxt > 0, fixes[np.maximum(nxt - 1, 0)], 0.0)
    return (next_t - prev_t) > max_gap_s


# Reverse-geocode cadence for the county-line veto: one lookup per this many
# seconds of usable GPS. A sampling cadence (cost vs. resolution of the line),
# NOT a policy threshold — the veto's width comes from the planner's window.
COUNTY_MASK_SAMPLE_S = 30


def build_county_change_mask(
    n: int,
    day_gps_points,
    geocoder,
    *,
    sample_s: float = COUNTY_MASK_SAMPLE_S,
    half_window_s: float,
) -> np.ndarray:
    """Per-second shorts veto: ``True`` marks a second a short may not cover
    because its window could straddle a county line.

    Usable fixes are sampled every ``sample_s`` seconds and reverse-geocoded to
    ``(county, state)``; a sample that geocodes to nothing is skipped, never
    read as a change. Between two consecutive geocoded samples that differ, every
    second in ``[t_prev - half_window_s, t_next + half_window_s]`` (clipped to
    the day) is vetoed, so no window of length <= ``half_window_s`` can start
    inside it and cross the line. No geocoder, no usable fix or ``n == 0``
    vetoes nothing.
    """
    mask = np.zeros(max(int(n), 0), dtype=bool)
    if mask.size == 0 or geocoder is None:
        return mask
    points = sorted(_in_window_coords(day_gps_points, float("-inf"), float("inf")))
    prev = None  # (t, (county, state)) of the last geocoded sample
    next_t = float("-inf")
    for t, lat, lon in points:
        if t < next_t:
            continue
        next_t = t + float(sample_s)
        try:
            detailed = geocoder.reverse_geocode_detailed(lat, lon) or {}
        except Exception as exc:  # noqa: BLE001 — a failed lookup is a skipped sample
            logger.debug("county-line mask: reverse_geocode_detailed failed: %s", exc)
            continue
        key = (detailed.get("county"), detailed.get("state"))
        if not key[0] and not key[1]:
            continue
        if prev is not None and prev[1] != key:
            lo = max(int(np.floor(prev[0] - half_window_s)), 0)
            hi = min(int(np.ceil(t + half_window_s)), mask.size - 1)
            mask[lo : hi + 1] = True
        prev = (t, key)
    return mask


def mask_summary(mask: np.ndarray) -> dict:
    """Summarize a stop mask for logging / persistence."""
    total = int(mask.size)
    masked = int(np.count_nonzero(mask)) if total else 0
    return {
        "masked_seconds": masked,
        "total_seconds": total,
        "masked_fraction": (masked / total) if total else 0.0,
    }
