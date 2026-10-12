"""Platform-decline detection over already-collected short metrics.

Median day-7 views over the newest ``window`` shorts vs the ``window`` before, reported only when
``PerformanceValidator.compare_strategies`` also separates the two groups — a Welch two-sample t-test
whose real p-value goes in the log line, so a big dip that is only noise stays unreported. Docs:
``docs/social-metrics-decline.md``.
"""

from __future__ import annotations

import logging
import statistics
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from .velocity import first_hours_velocity

logger = logging.getLogger(__name__)

DAY7_DAYS = 7  #: day-7 views span a short's first 7 daily points; a shorter one is skipped
METRIC = "day7_views"
SIGNIFICANT = ("moderate", "strong")  #: SignificanceLevel values with p <= 0.05


@dataclass(frozen=True)
class DeclineFinding:  # one platform's decline, in the validator's own terms
    platform: str
    window: int
    delta_pct: float
    velocity: str
    p_value: float
    significance: str  # the validator's own SignificanceLevel value

    def log_line(self) -> str:
        return (
            f"PLATFORM_DECLINE platform={self.platform} window={self.window} "
            f"metric={METRIC} delta={self.delta_pct:+.0f}% p={self.p_value:.4f} "
            f"significance={self.significance} velocity={self.velocity}"
        )


def platforms_seen(records) -> List[str]:  # never a hardcoded platform list
    return sorted({p for r in records for s in r.shorts for p in s.metrics})


def platform_series(records, platform: str) -> List[Tuple[datetime, int, str]]:
    """``(posted_at, day7, video_id)`` oldest first; no ``posted_at``/7-day series, an
    unreported day inside the window, or a video_id already seen means skipped.

    A missing day inside the window is unreported, not a measured zero — mirrors
    ``velocity._sum_reported_views``: a partial sum would silently answer a smaller
    question while looking identical to a complete one. A video_id already present
    (the same content tracked twice, e.g. a project rename/backfill) is deduplicated
    so one short can't vote twice in a window; an empty video_id never dedupes.
    """
    by_video: dict = {}
    points: List[Tuple[datetime, int, str]] = []
    for record in records:
        for short in record.shorts:
            daily = getattr(short.metrics.get(platform), "daily", None) or []
            if len(daily) < DAY7_DAYS or not short.posted_at:
                continue
            try:
                posted = datetime.fromisoformat(short.posted_at)
            except (TypeError, ValueError):
                continue
            window = daily[:DAY7_DAYS]
            if any(p.views is None for p in window):
                continue
            day7 = sum(p.views for p in window)
            video_id = short.platform_ids.get(platform, "")
            point = (posted, day7, video_id)
            if video_id:
                by_video[video_id] = point
            else:
                points.append(point)
    return sorted(points + list(by_video.values()), key=lambda p: p[0])


def _compare(prior: List[int], recent: List[int], window: int):
    """``ValidationResult`` (Welch t-test, real p-value), ``None`` if the SEO
    manager isn't installed. ``min_sample_size`` is the window because the window
    IS the sample size; below it the validator only warns, and the ``2 * window``
    gate already refused thinner data. Its 100-entry history goes to scratch."""
    try:
        from seo_manager.analytics.performance_validator import PerformanceValidator
    except ImportError:
        logger.warning("seo_manager not installed — platform-decline detection skipped")
        return None
    scratch = Path(tempfile.gettempdir()) / "gms-trend"
    validator = PerformanceValidator(min_sample_size=window, state_dir=scratch)
    a, b = [{METRIC: v} for v in prior], [{METRIC: v} for v in recent]
    return validator.compare_strategies(a, b, METRIC, "day-7 views fell on this platform")


def detect_platform_decline(
    records, platform: str, window: int = 14, min_drop_pct: float = 20.0, store=None
) -> Optional[DeclineFinding]:
    """A ``DeclineFinding``, else ``None`` — no records, thin data, too small a
    dip and an insignificant one are all non-findings, not errors."""
    points = platform_series(records, platform)
    if len(points) < 2 * window:
        return None
    points = points[-2 * window :]
    prior_vals = [v for _, v, _ in points[:window]]
    recent_vals = [v for _, v, _ in points[window:]]
    prior = statistics.median(prior_vals)
    if prior <= 0:
        return None
    delta_pct = (statistics.median(recent_vals) - prior) / prior * 100.0
    if delta_pct > -min_drop_pct:
        return None
    res = _compare(prior_vals, recent_vals, window)
    sig = res.significance.value if res is not None else ""
    if sig not in SIGNIFICANT:
        return None

    def vel(half) -> Optional[float]:  # early signal: first-48h share of total views
        # ``None`` is "nothing was measured", which is NOT a measured zero: folding the two
        # together reported a collapse nobody observed. A real 0.0 still counts.
        got = [
            v for v in (first_hours_velocity(i, store=store) for _, _, i in half) if v is not None
        ]
        return statistics.fmean(got) if got else None

    now, was = vel(points[window:]), vel(points[:window])
    velocity = "unknown"  # stays unknown unless BOTH halves were actually measured
    if was is not None and now is not None and was > 0:
        velocity = "falling" if now < was * 0.9 else "rising" if now > was * 1.1 else "steady"
    return DeclineFinding(platform, window, delta_pct, velocity, res.p_value, sig)
