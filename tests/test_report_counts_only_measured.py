"""A short every platform REFUSED about is not a short with metrics.

``collect_all`` stamps ``collected_at`` on every attempt, refusals
included -- deliberately, because that field means "a collection was
attempted" and a run where every platform refused IS an attempt. It also
guards its own return value so the CLI's "Collected metrics for N short(s)"
line counts only shorts something MEASURED.

The report surface was not guarded, and this is the regression that opened:
``TrackingRecord.shorts_with_metrics`` selects on ``collected_at is not
None``, so a refusal-only short satisfies a property whose name
promises metrics, and the cross-project header counts it under "Total shorts
with metrics". That restates, a third time and one level further out, the
claim ``test_tracking_collected_nothing_is_not_collected`` refuses for the log
line and the collector refuses for the count.

The refusals are not discarded -- they are in ``observation_log``, and the
report says so explicitly rather than letting an absent number read as a
measured zero.
"""

from __future__ import annotations

from gomoveshift.tracking.observation import MetricObservation, ObservationStatus
from gomoveshift.tracking.report import PerformanceReporter
from gomoveshift.tracking.schema import PlatformMetrics, ShortRecord, TrackingRecord
from gomoveshift.tracking.store import append_observation


def _refusal_only_short(index: int = 1) -> ShortRecord:
    """A short a collection run reached and every platform refused about."""
    short = ShortRecord(index=index, platform_ids={"instagram": f"ig_{index}"})
    append_observation(
        _record_with(short),
        index,
        MetricObservation(
            platform="instagram",
            status=ObservationStatus.PERMISSION_DENIED,
            post_id=f"ig_{index}",
            post_format="reel",
            fetched_at="2026-09-15T00:00:00+00:00",
            detail="token is missing the read_insights scope",
        ),
    )
    # The collector stamps this on any attempt, refusal-only included.
    short.collected_at = "2026-09-15T00:00:00+00:00"
    return short


def _measured_short(index: int = 2) -> ShortRecord:
    short = ShortRecord(
        index=index,
        platform_ids={"youtube": f"yt_{index}"},
        metrics={"youtube": PlatformMetrics(views=100, likes=5)},
    )
    short.collected_at = "2026-09-15T00:00:00+00:00"
    return short


def _record_with(*shorts: ShortRecord) -> TrackingRecord:
    return TrackingRecord(project="proj", shorts=list(shorts))


def test_a_refusal_only_short_is_not_a_short_with_metrics() -> None:
    record = _record_with(_refusal_only_short(1), _measured_short(2))

    with_metrics = record.shorts_with_metrics

    assert [s.index for s in with_metrics] == [2]


def test_a_project_of_only_refusals_reports_nothing_measured() -> None:
    """Not "0 views across 3 shorts" -- nothing was measured, and a report
    that renders its sections over an empty aggregate would present that
    absence as a result."""
    record = _record_with(_refusal_only_short(1), _refusal_only_short(2))

    assert record.shorts_with_metrics == []


def test_the_report_says_refusals_were_recorded_rather_than_staying_silent(
    tmp_path,
) -> None:
    """Suppressing the inflated count is only half the fix. A run that reached
    every short and was refused by every platform must not read the same as a
    run that never happened."""

    class _Store:
        def load(self, project):
            return _record_with(_refusal_only_short(1), _refusal_only_short(2))

    text = PerformanceReporter(store=_Store()).generate_project_report("proj")

    assert "refus" in text.lower()
    # ...and it must not claim a measurement it does not have.
    assert "0 views" not in text.lower()
