"""Upstream gate: prove the stop-removal stage actually cut the stops.

Runs on the CUT output BEFORE the irreversible stage-9 encode: re-runs the
motion scan on that output and reconciles its duration against the kept-segment
math, so a silently-failed cut (residual idle footage, wrong-duration stitch,
corrupt keep-segment list) hard-fails here — not hours later after the day
master is encoded and uploaded. Escape: ``--skip-stop-verify``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..config import Config

logger = logging.getLogger(__name__)

# Stitch duration tolerance (seconds). Stream-copy segment boundaries snap to
# keyframes and the crossfade re-encode introduces sub-frame drift, so the
# reconciled total is never bit-exact. Generous enough to absorb that, tight
# enough to catch a whole residual stop (≥ min_stop_duration) left in.
DEFAULT_DURATION_TOLERANCE_S = 3.0


class StopVerificationError(RuntimeError):
    """Stop-removal output failed verification. Greppable tag: ``STOP-VERIFY``."""


def _expected_output_duration(
    keep_segments: list[tuple[float, float]], crossfade_duration: float
) -> float:
    """Expected output duration, mirroring ``StopProcessor._cut_and_stitch``.

    Segments are stream-copied + concatenated → Σ(end−start); with
    ``crossfade_duration`` > 0 and N>1 segments each of the N−1 transitions
    overlaps, so subtract ``(N−1) * crossfade_duration``.
    """
    total = sum(end - start for start, end in keep_segments)
    if crossfade_duration > 0 and len(keep_segments) > 1:
        total -= (len(keep_segments) - 1) * crossfade_duration
    return total


def _validate_keep_segments(
    keep_segments: list[tuple[float, float]], source_duration: float
) -> None:
    """Check (c): keep-segments monotonic increasing and inside [0, source]."""
    prev_end = 0.0
    for i, seg in enumerate(keep_segments):
        start, end = seg
        if end <= start:
            raise StopVerificationError(
                f"STOP-VERIFY: keep_segments[{i}] is non-increasing "
                f"(start={start:.2f} >= end={end:.2f})"
            )
        if start < 0 or end > source_duration + DEFAULT_DURATION_TOLERANCE_S:
            raise StopVerificationError(
                f"STOP-VERIFY: keep_segments[{i}]=({start:.2f}, {end:.2f}) "
                f"falls outside [0, source_duration={source_duration:.2f}]"
            )
        if start < prev_end - DEFAULT_DURATION_TOLERANCE_S:
            raise StopVerificationError(
                f"STOP-VERIFY: keep_segments[{i}] start={start:.2f} overlaps the "
                f"prior segment end={prev_end:.2f} — segments must be ordered"
            )
        prev_end = end


def verify_stops_removed(
    output_path: Path,
    keep_segments: Optional[list[tuple[float, float]]],
    source_duration: float,
    cfg: Config,
    crossfade_duration: float = 0.0,
    _stop_processor=None,
) -> None:
    """Hard-fail (``StopVerificationError``) when the cut output still has stops.

    ``keep_segments`` is ``None`` for no-stops/legacy LFs → only check (a) runs.
    ``crossfade_duration`` (0 = hard cuts) feeds both the stitch math and the
    re-scan processor. ``_stop_processor`` is a test seam; production builds a
    fresh ``StopProcessor`` against ``output_path``. Checks:
        (a) re-scanning the cut output finds ZERO spans ≥ ``min_stop_duration``;
        (b) ``|output_duration − expected_total| > tolerance`` (stitch math);
        (c) keep_segments monotonic increasing and within [0, source_duration].
    """
    output_path = Path(output_path)
    min_stop = cfg.stop_detection.min_stop_duration

    # (c) Structural keep-segment validation (cheap; do it first).
    if keep_segments:
        _validate_keep_segments(keep_segments, source_duration)

    # (a) Residual-stop re-scan: re-run stop detection on the CUT output.
    if _stop_processor is None:
        from .stop_processor import StopProcessor

        _stop_processor = StopProcessor(cfg, crossfade_duration=crossfade_duration)
    residual = _stop_processor.detect_stops(output_path)
    qualifying = [s for s in residual if s.duration >= min_stop]
    if qualifying:
        worst = max(qualifying, key=lambda s: s.duration)
        raise StopVerificationError(
            f"STOP-VERIFY: cut output {output_path.name} still has "
            f"{len(qualifying)} residual stop(s) ≥ {min_stop:.1f}s — worst is "
            f"{worst.duration:.1f}s at {worst.start_time:.1f}s–{worst.end_time:.1f}s. "
            f"Stop removal did not take effect."
        )

    # (b) Duration reconciliation against the keep-segment stitch math.
    if keep_segments:
        output_duration = _stop_processor._get_duration(output_path)
        expected = _expected_output_duration(keep_segments, crossfade_duration)
        drift = abs(output_duration - expected)
        if drift > DEFAULT_DURATION_TOLERANCE_S:
            raise StopVerificationError(
                f"STOP-VERIFY: cut output duration {output_duration:.2f}s differs "
                f"from the expected stitched total {expected:.2f}s by {drift:.2f}s "
                f"(> {DEFAULT_DURATION_TOLERANCE_S:.1f}s tolerance) across "
                f"{len(keep_segments)} kept segment(s), crossfade={crossfade_duration}s."
            )

    logger.info("  STOP-VERIFY: %s clean — 0 residual stops ≥ %.1fs", output_path.name, min_stop)


def verify_retained_timeline_moving(
    keep_segments: Optional[list[tuple[float, float]]],
    source_duration: float,
    cfg: Config,
    stop_processor,
) -> None:
    """Independent GPS-speed check of the RETAINED timeline.

    The optical STOP-VERIFY re-scan reuses the same blind optical detector and
    is skipped entirely when detection reports zero stops. This check runs
    REGARDLESS of ``stops_found`` and uses the GPS trace (not optics): it
    hard-fails when a sustained parked GPS span (``>= min_stop_duration``)
    survives inside the footage about to be encoded — a stop the optical path
    missed. ``keep_segments`` None → the whole ``[0, source_duration]`` is
    retained (no cut happened). Strict no-op for GPS-less LFs and when GPS
    corroboration is disabled.

    Greppable tag: ``STOP-VERIFY(GPS)``.
    """
    if stop_processor is None or getattr(stop_processor, "gps_data", None) is None:
        return
    if not cfg.stop_detection.gps_speed_corroboration.enabled:
        return
    parked = stop_processor._gps_derived_stops()
    if not parked:
        return
    retained = list(keep_segments) if keep_segments else [(0.0, source_duration)]
    min_stop = cfg.stop_detection.min_stop_duration
    for stop in parked:
        survived = sum(
            max(0.0, min(stop.end_time, r_end) - max(stop.start_time, r_start))
            for r_start, r_end in retained
        )
        if survived >= min_stop:
            raise StopVerificationError(
                f"STOP-VERIFY(GPS): a parked GPS span {stop.start_time:.1f}s–"
                f"{stop.end_time:.1f}s survives {survived:.1f}s inside the retained "
                f"timeline (≥ {min_stop:.1f}s) — the optical detector missed it. "
                "Do NOT encode/upload this LF."
            )
    logger.info("  STOP-VERIFY(GPS): retained timeline clean — no parked span survives")
