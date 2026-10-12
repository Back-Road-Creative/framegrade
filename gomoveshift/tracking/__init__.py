"""GoMoveShift Effectiveness Tracking.

Collects social media performance data after shorts are published,
stores it alongside content metadata (hooks, posting times, weather),
and generates insight reports to improve future content decisions.
"""

from .collector import MetricsCollector
from .report import PerformanceReporter
from .schema import ContentMeta, PlatformMetrics, ShortRecord, TrackingRecord
from .slot_performance import (
    MetricRow,
    SlotRow,
    slots_ranked,
    upsert_audience_online,
    upsert_from_metrics,
)
from .store import TrackingStore

__all__ = [
    "MetricsCollector",
    "TrackingStore",
    "PerformanceReporter",
    "TrackingRecord",
    "ShortRecord",
    "ContentMeta",
    "PlatformMetrics",
    "MetricRow",
    "SlotRow",
    "upsert_from_metrics",
    "slots_ranked",
    "upsert_audience_online",
]
