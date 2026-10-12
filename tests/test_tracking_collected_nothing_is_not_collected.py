"""'Collected N views' must not be how the collector reports that it
measured nothing.

``MetricsCollector.collect_for_short`` logged every successful call as
``"Collected %s metrics for short #%d: %d views"``. ``PlatformMetrics.views``
is ``Optional[int]`` and is ``None`` for "not reported by this platform" — the
schema's whole point. ``%d`` against ``None`` raises
``TypeError: %d format: a real number is required, not NoneType`` inside
``logging``, which swallows it: the operator gets a ``--- Logging error ---``
traceback on stderr and no usable line, exactly in the case where knowing
"nothing came back" matters most. Nothing propagates, so no caller notices.

A platform that reported nothing, a platform that is not configured, and a
platform that genuinely measured zero are three different states. Only the
third is a measurement.
"""

from __future__ import annotations

import logging

from gomoveshift.tracking.collector import MetricsCollector
from gomoveshift.tracking.observation import MetricObservation, ObservationStatus
from gomoveshift.tracking.schema import PlatformMetrics, ShortRecord


class _SilentCollector:
    """Available, answers, and reports nothing — the shape that broke the log."""

    def is_available(self):
        return True

    def collect_observation(self, video_id, published_at=None):
        return MetricObservation(
            platform="tiktok",
            status=ObservationStatus.NOT_RETURNED,
            post_id=video_id,
            post_format="short",
            event_at=published_at,
            fetched_at="2026-09-10T00:00:00+00:00",
            detail="queried; the platform reported no values for this post",
        )


class _MeasuredZeroCollector:
    """A real, measured zero. Different from silence, and still a measurement."""

    def is_available(self):
        return True

    def collect_observation(self, video_id, published_at=None):
        return MetricObservation(
            platform="tiktok",
            status=ObservationStatus.MEASURED,
            post_id=video_id,
            post_format="short",
            event_at=published_at,
            fetched_at="2026-09-10T00:00:00+00:00",
            cumulative=PlatformMetrics(views=0),
        )


def _collect(collector, caplog):
    mc = MetricsCollector()
    mc._collectors["tiktok"] = collector
    short = ShortRecord(index=3, platform_ids={"tiktok": "tt_123"})
    with caplog.at_level(logging.INFO, logger="gomoveshift.tracking.collector"):
        results = mc.collect_for_short(short)
    return results, caplog


def test_unreported_views_do_not_break_the_log(caplog) -> None:
    """The log line must render at all when views is None."""
    _, caplog = _collect(_SilentCollector(), caplog)

    # Every record must be formattable. record.getMessage() is what logging
    # itself calls, so this reproduces the real failure rather than imitating it.
    for record in caplog.records:
        record.getMessage()


def test_reporting_nothing_is_not_reported_as_collecting(caplog) -> None:
    _, caplog = _collect(_SilentCollector(), caplog)
    text = " ".join(r.getMessage() for r in caplog.records)

    assert "tiktok" in text
    # "Collected ... views" is a measurement claim. Nothing was measured.
    assert "Collected tiktok metrics" not in text
    assert "no views reported" in text.lower() or "reported no" in text.lower()


def test_a_measured_zero_is_still_a_measurement(caplog) -> None:
    """The fix must not flatten a real 0 into "reported nothing"."""
    results, caplog = _collect(_MeasuredZeroCollector(), caplog)
    text = " ".join(r.getMessage() for r in caplog.records)

    assert results["tiktok"].views == 0
    assert "0 views" in text
    assert "no views reported" not in text.lower()
