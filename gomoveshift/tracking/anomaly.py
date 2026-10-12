"""Post-level dip/spike detection over already-collected short metrics — shadow only.

A caller-policy layer over ``seo_manager.analytics.anomaly_stats`` (a qualification kernel that
holds no thresholds of its own). This module supplies the shadow-policy thresholds and
nothing more: it never fetches, never writes, and nothing downstream reads its output to act.
Every finding and refusal line carries ``shadow=true``. Docs: ``docs/social-metrics-anomaly.md``.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional, Union

from seo_manager.analytics.anomaly_stats import (
    LOG1P,
    Feasible,
    Refusal,
    feasibility,
    robust_deviation,
    tail_rank_p,
)

from .decline import METRIC, platform_series

logger = logging.getLogger(__name__)

# Shadow policy; changing these is a new policy revision, never a runtime
# knob.
TOTAL_ALPHA = 0.05
REGISTERED_TESTS = 2
REFERENCE_CAP = 200
LOOKBACK_DAYS = 180
MIN_ABS_DEVIATION = 3.5
MIN_RELATIVE_EFFECT = 0.25

DIP = "dip"
SPIKE = "spike"


@dataclass(frozen=True)
class AnomalyFinding:  # one candidate dip/spike, shadow only
    platform: str
    video_id: str
    kind: str
    target: int
    ref_median: float
    deviation: float
    relative_effect: float
    p_value: float
    n: int
    allocated_alpha: float

    def log_line(self) -> str:
        return (
            f"POST_ANOMALY platform={self.platform} video={self.video_id} kind={self.kind} "
            f"metric={METRIC} target={self.target} ref_median={self.ref_median:.0f} "
            f"deviation={self.deviation:+.2f} relative={self.relative_effect:+.0%} "
            f"p={self.p_value:.4f} n={self.n} allocated_alpha={self.allocated_alpha} shadow=true"
        )


@dataclass(frozen=True)
class AnomalyRefusal:  # the kernel refused before any outcome was inspected
    platform: str
    reason: str
    n: int
    required: Optional[int]

    def log_line(self) -> str:
        required = self.required if self.required is not None else "-"
        return (
            f"POST_ANOMALY_REFUSED platform={self.platform} reason={self.reason} "
            f"n={self.n} required={required} shadow=true"
        )


def check_feasibility() -> Union[Feasible, Refusal]:
    """Can the current policy constants ever produce a confirmable result?"""
    return feasibility(TOTAL_ALPHA, REGISTERED_TESTS, REFERENCE_CAP)


def _references(points: List[tuple], target_posted) -> List[tuple]:
    """Earlier points within the trailing lookback, newest ``REFERENCE_CAP`` at most.

    ``points`` is oldest-first, so the newest are already at the tail — capping is a slice.
    """
    cutoff = target_posted - timedelta(days=LOOKBACK_DAYS)
    windowed = [p for p in points if cutoff <= p[0] < target_posted]
    return windowed[-REFERENCE_CAP:]


def detect_post_anomaly(
    records, platform: str, allocated_alpha: float
) -> Optional[Union[AnomalyFinding, AnomalyRefusal]]:
    """A candidate for the newest short on ``platform``, a typed refusal, or ``None``
    (no series, no references, or a non-candidate) — never a fabricated number."""
    points = platform_series(records, platform)
    if len(points) < 2:
        return None
    target_posted, target_value, target_video = points[-1]
    reference_points = _references(points[:-1], target_posted)
    if not reference_points:
        return None
    reference_values = [v for _, v, _ in reference_points]

    deviation_result = robust_deviation(reference_values, target_value, transform=LOG1P)
    if isinstance(deviation_result, Refusal):
        return AnomalyRefusal(platform, deviation_result.reason, len(reference_values), None)

    rank_result = tail_rank_p(reference_values, target_value, allocated_alpha=allocated_alpha)
    if isinstance(rank_result, Refusal):
        return AnomalyRefusal(
            platform, rank_result.reason, len(reference_values), rank_result.required_references
        )

    if rank_result.p_two_sided > allocated_alpha:
        return None
    if deviation_result.relative_effect is None:
        return None
    if abs(deviation_result.deviation) < MIN_ABS_DEVIATION:
        return None
    if abs(deviation_result.relative_effect) < MIN_RELATIVE_EFFECT:
        return None

    kind = DIP if deviation_result.deviation < 0 else SPIKE
    return AnomalyFinding(
        platform=platform,
        video_id=target_video,
        kind=kind,
        target=target_value,
        ref_median=statistics.median(reference_values),
        deviation=deviation_result.deviation,
        relative_effect=deviation_result.relative_effect,
        p_value=rank_result.p_two_sided,
        n=rank_result.n,
        allocated_alpha=allocated_alpha,
    )
