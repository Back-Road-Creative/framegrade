"""Early-performance velocity from the per-day analytics time series.

Requirement: first-48h velocity feeds the retitle/scheduling worklist.
"""

from __future__ import annotations

from typing import List, Optional

from .schema import DailyMetricPoint
from .store import TrackingStore


def _sum_reported_views(points: List[DailyMetricPoint]) -> Optional[int]:
    """Sum of ``views`` across ``points``, or None if any point is unreported.

    Mirrors ``PlatformMetrics.engagement_rate``: a ratio built from a
    partial sum would silently answer a different (smaller) question than
    one built from a complete window while looking identical to the
    caller, so an unreported day poisons the sum rather than scoring as 0.
    """
    total = 0
    for point in points:
        if point.views is None:
            return None
        total += point.views
    return total


def first_hours_velocity(
    video_id: str, hours: int = 48, store: Optional[TrackingStore] = None
) -> Optional[float]:
    """Fraction of a video's total views collected within its first `hours`.

    None when the video isn't found, has no daily time series yet, or the
    series has an unreported day inside the window (or, absent an aggregate
    ``views`` total, anywhere in the series) — see ``_sum_reported_views``.
    """
    store = store or TrackingStore()
    days_window = max(1, hours // 24)

    for record in store.load_all_records():
        for short in record.shorts:
            for platform, vid in short.platform_ids.items():
                if vid != video_id:
                    continue
                metrics = short.metrics.get(platform)
                if metrics is None or not metrics.daily:
                    return None
                early_views = _sum_reported_views(metrics.daily[:days_window])
                if early_views is None:
                    return None
                if metrics.views is not None:
                    total_views = metrics.views
                else:
                    total_views = _sum_reported_views(metrics.daily)
                    if total_views is None:
                        return None
                return early_views / total_views if total_views else 0.0

    return None
