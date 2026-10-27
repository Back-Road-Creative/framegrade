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
from enum import IntEnum

import numpy as np

from .gps_quality import _in_window_coords, usable_fix_times, window_median_speed_mps
from .route_claims import GAP_S as _ROUTE_CLAIM_GAP_S

logger = logging.getLogger(__name__)

# Column index of per-second optical motion energy (mirror unified_scorer).
COL_MOTION_ENERGY = 2


class MotionState(IntEnum):
    """Per-second motion class (the values persist in ``stop_mask_<date>.npz``)."""

    MOVING = 0
    STOPPED = 1
    CREEPING = 2  # sustained slow GPS speed, kept on purpose (creep_policy="keep")
    UNCERTAIN = 3  # no usable GPS speed AND no usable optical reading


CREEP_POLICIES = ("remove", "keep")
DEFAULT_CREEP_MIN_SPEED_KMH = 2.0


def stop_mask_kwargs(config) -> dict:
    """The ``build_stop_mask`` knobs read from ``config.stop_detection``.

    ``stop_detection`` is a TOP-LEVEL ``Config`` block (``config.py``), not a
    ``DriveConfig`` field. One reader for every mask builder — the day
    pipeline's per-day mask and the day-first drive plan's drive-wide mask —
    so the two can never drift. Falls back to ``config.drive.stop_detection``
    (duck-typed test configs), then to the builder's own defaults.

    The creep and hysteresis knobs are emitted only when they differ from the
    builder's defaults, so a default config yields exactly the three base keys.
    """
    sd = getattr(config, "stop_detection", None) if config is not None else None
    if sd is None:
        drive = getattr(config, "drive", None) if config is not None else None
        sd = getattr(drive, "stop_detection", None) if drive is not None else None
    stop_speed_kmh, min_stop_dur, optical = 10.0, 5.0, 0.008
    extra: dict = {}
    if sd is not None:
        min_stop_dur = float(getattr(sd, "min_stop_duration", min_stop_dur))
        optical = float(getattr(sd, "motion_threshold", optical))
        gsc = getattr(sd, "gps_speed_corroboration", None)
        if gsc is not None:
            stop_speed_kmh = float(getattr(gsc, "stop_speed_kmh", stop_speed_kmh))
        policy = getattr(sd, "creep_policy", "remove")
        if policy != "remove":
            extra["creep_policy"] = policy
        creep_min = float(getattr(sd, "creep_min_speed_kmh", DEFAULT_CREEP_MIN_SPEED_KMH))
        if creep_min != DEFAULT_CREEP_MIN_SPEED_KMH:
            extra["creep_min_speed_kmh"] = creep_min
        hyst = getattr(sd, "hysteresis", None)
        if hyst is not None and getattr(hyst, "enabled", False) is True:
            extra["hysteresis_enabled"] = True
            extra["hysteresis_exit_multiplier"] = float(getattr(hyst, "exit_multiplier", 1.0))
    return {
        "stop_speed_kmh": stop_speed_kmh,
        "min_stop_duration_s": min_stop_dur,
        "optical_motion_threshold": optical,
        **extra,
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


def classify_motion_states(
    day_scores: np.ndarray,
    day_gps_points,
    *,
    stop_speed_kmh: float = 10.0,
    min_stop_duration_s: float = 5.0,
    optical_motion_threshold: float = 0.008,
    speed_window_s: float = 5.0,
    creep_policy: str = "remove",
    creep_min_speed_kmh: float = DEFAULT_CREEP_MIN_SPEED_KMH,
    hysteresis_enabled: bool = False,
    hysteresis_exit_multiplier: float = 1.0,
) -> np.ndarray:
    """Classify every second of one day into a :class:`MotionState` (int8).

    Args:
        day_scores: (N, >=3) score array, one row per second, timestamps
            rebased to 0. Column ``COL_MOTION_ENERGY`` is the optical fallback.
        day_gps_points: the day's GPS trace as ``[(t_master, lat, lon), ...]``
            in the SAME 0-based day-master seconds as ``day_scores``, or None
            for a GPS-less day.
        stop_speed_kmh: ground speed below which a second is "stopped".
        min_stop_duration_s: the minimum length (seconds) of a contiguous
            stopped run before it is STOPPED — shorter blips (a light) are not.
        optical_motion_threshold: motion-energy floor used when a second's GPS
            speed can't be measured.
        speed_window_s: width of the CENTERED window (seconds) over which a
            second's median GPS speed is measured.
        creep_policy: ``"remove"`` (default) treats sub-``stop_speed_kmh``
            driving as a stop. ``"keep"`` leaves seconds whose speed is in
            ``[creep_min_speed_kmh, stop_speed_kmh)`` as CREEPING (not a stop);
            only speed below ``creep_min_speed_kmh`` stays a stop.
        creep_min_speed_kmh: the floor of the creep band (provisional).
        hysteresis_enabled / hysteresis_exit_multiplier: once a second is a stop
            candidate, the run continues until the signal clears its threshold
            times the multiplier (same semantics as the legacy optical state
            machine in ``analysis/motion.py``). Off by default.

    Returns:
        ``int8`` array of length ``len(day_scores)``. STOPPED seconds are
        exactly the seconds the pre-states mask called a stop. UNCERTAIN marks
        a second with neither a GPS speed nor a finite optical reading; it is
        not a stop (as before), only visible.
    """
    if creep_policy not in CREEP_POLICIES:
        raise ValueError(f"creep_policy must be one of {CREEP_POLICIES}, got {creep_policy!r}")
    n = int(len(day_scores)) if day_scores is not None else 0
    if n == 0:
        return np.zeros(0, dtype=np.int8)

    keep_creep = creep_policy == "keep"
    # Speed below this is a stop candidate; the creep band sits above it.
    stop_floor_kmh = min(creep_min_speed_kmh, stop_speed_kmh) if keep_creep else stop_speed_kmh
    stop_floor_mps = stop_floor_kmh / 3.6
    stop_speed_mps = stop_speed_kmh / 3.6
    exit_mult = float(hysteresis_exit_multiplier)
    have_gps = day_gps_points is not None and len(day_gps_points) >= 2
    motion = np.asarray(day_scores[:, COL_MOTION_ENERGY], dtype=np.float64)
    half = speed_window_s / 2.0

    candidate = np.zeros(n, dtype=bool)
    raw = np.full(n, int(MotionState.MOVING), dtype=np.int8)
    in_stop = False
    for i in range(n):
        speed = None
        if have_gps:
            w0 = max(0.0, i - half)
            w1 = min(float(n), i + half)
            speed = window_median_speed_mps(day_gps_points, w0, w1)
        if speed is not None:
            # GPS authoritative + additive: a sub-threshold span is a stop even
            # when optical motion is high (parked with passing traffic).
            stopped = speed < stop_floor_mps
            if not stopped and in_stop and hysteresis_enabled:
                stopped = speed <= stop_floor_mps * exit_mult
            if not stopped and keep_creep and speed < stop_speed_mps:
                raw[i] = MotionState.CREEPING
        else:
            m = float(motion[i])
            if np.isnan(m):
                # No GPS speed and no optical reading: never a stop (as before),
                # but say so.
                stopped = False
                raw[i] = MotionState.UNCERTAIN
            else:
                # No measurable speed here — never auto-pass; fall back to optical.
                stopped = m < optical_motion_threshold
                if not stopped and in_stop and hysteresis_enabled:
                    stopped = m <= optical_motion_threshold * exit_mult
        candidate[i] = stopped
        in_stop = stopped

    sustained = _sustained_runs(candidate, min_stop_duration_s)
    raw[sustained] = MotionState.STOPPED
    return raw


def build_stop_mask(day_scores: np.ndarray, day_gps_points, **kwargs) -> np.ndarray:
    """Per-second STOP mask for one day: ``True`` marks a STOP second to EXCLUDE.

    The boolean projection of :func:`classify_motion_states` (same keyword
    arguments); CREEPING and UNCERTAIN seconds are never ``True``. Empty array
    when ``day_scores`` is empty.
    """
    return states_to_stop_mask(classify_motion_states(day_scores, day_gps_points, **kwargs))


def states_to_stop_mask(states: np.ndarray) -> np.ndarray:
    """``True`` exactly where ``states`` is :attr:`MotionState.STOPPED`."""
    return np.asarray(states) == int(MotionState.STOPPED)


# A short may name only a place its own clip's GPS votes (quorum: two fixes). A
# second inside a fix-to-fix gap longer than this — or inside a lead before the
# first fix / a tail after the last one longer than it — is vetoed for shorts.
# It IS the route-claim gap (``route_claims.GAP_S``, a fix-to-fix interval longer
# than that is a gap; exactly equal is not): a Short needs 100% GPS coverage, so
# a looser veto lets the planner pick a window the claim gate then refuses
# (a 6.0 s interval passed a 7 s veto). 5 s keeps
# every window of the planner's 15-s minimum holding at least two fixes just as
# 7 s did, so every window left open can vote its own place.
GPS_MAX_FIX_GAP_S = _ROUTE_CLAIM_GAP_S


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
