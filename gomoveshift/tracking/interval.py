"""The first production consumer of ``observation.interval_between``.

``ShortRecord.observation_log`` already holds an append-only series of raw
``MetricObservation.to_dict()`` entries per platform; nothing before this
module turned two of them into a change. :func:`latest_interval` reads the
last two entries for one platform and rebuilds them, so a caller asking "how
did this move since the last check" gets an honest answer -- including a
refusal on either end, or a counter that moved backwards -- rather than a
raise or a fabricated zero.
"""

from __future__ import annotations

from typing import Optional

from .observation import IntervalDelta, MetricObservation, interval_between
from .schema import ShortRecord


def latest_interval(short: ShortRecord, platform: str) -> Optional[IntervalDelta]:
    """The change between the last two recorded observations for
    ``platform`` on ``short``, or ``None`` when there is no series, only one
    entry, or the stored entries can't be reconstructed into observations of
    the same post -- never a zero-filled delta standing in for "unknown".
    """
    series = short.observation_log.get(platform)
    if not series or len(series) < 2:
        return None
    try:
        earlier = MetricObservation.from_dict(series[-2])
        later = MetricObservation.from_dict(series[-1])
        return interval_between(earlier, later)
    except (ValueError, TypeError, KeyError):
        return None
