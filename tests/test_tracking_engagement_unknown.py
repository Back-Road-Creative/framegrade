"""ShortRecord.total_engagement must be unknown, not a fabricated 0.0, when
no platform ever reported a view count.

Regression coverage: `total_engagement` folded `total_views` (already 0 for
"nobody reported") straight into a `0.0` rate, indistinguishable in a report
from a short that was genuinely seen and ignored. This must not touch the
deliberate fold-to-zero behaviour of totals (`total_views`,
`report._platform_breakdown`) — those stay exactly as they are.
"""

from __future__ import annotations

from gomoveshift.tracking.report import PerformanceReporter
from gomoveshift.tracking.schema import PlatformMetrics, ShortRecord
from gomoveshift.tracking.store import TrackingStore


def _short(index: int, **platforms: PlatformMetrics) -> ShortRecord:
    return ShortRecord(index=index, file=f"short{index}.mp4", metrics=platforms)


def test_total_engagement_is_unknown_when_nothing_was_reported():
    """Every platform failed to report views at all — the rate is unknown,
    never a fabricated 0.0."""
    short = _short(
        1,
        youtube=PlatformMetrics(),  # never reported anything
        tiktok=PlatformMetrics(),  # never reported anything
    )
    assert short.total_engagement is None


def test_total_engagement_with_no_platforms_at_all_is_unknown():
    short = _short(1)
    assert short.total_engagement is None


def test_total_engagement_keeps_genuine_measured_zero():
    """At least one platform actually reported (0 views) — this is a
    well-defined 0.0, not the missing-data case. Unchanged behaviour."""
    short = _short(
        1,
        youtube=PlatformMetrics(views=0, likes=0, comments=0, shares=0),
    )
    assert short.total_engagement == 0.0


def test_total_engagement_computes_when_something_was_reported():
    short = _short(
        1,
        youtube=PlatformMetrics(views=100, likes=5, comments=3, shares=2),
        tiktok=PlatformMetrics(),  # unreported, contributes nothing to the total
    )
    assert short.total_engagement == 0.10


def test_total_views_still_folds_unreported_to_zero():
    """The totals-vs-rates asymmetry: total_views is unchanged by this fix."""
    short = _short(1, youtube=PlatformMetrics(), tiktok=PlatformMetrics())
    assert short.total_views == 0


def test_report_renders_dash_for_unknown_and_averages_only_known():
    reporter = PerformanceReporter(store=TrackingStore())
    known_short = _short(
        1,
        youtube=PlatformMetrics(views=100, likes=10, comments=0, shares=0),
    )
    known_short.collected_at = "2026-01-01T00:00:00"
    unknown_short = _short(2, youtube=PlatformMetrics())
    unknown_short.collected_at = "2026-01-01T00:00:00"

    shorts = [known_short, unknown_short]

    summary = reporter._summary(shorts)
    assert "10.00%" in summary  # the one known rate is the average of one

    best_worst = reporter._best_worst(shorts)
    assert "—" in best_worst  # unknown short's engagement cell

    recs = reporter._recommendations(shorts)
    assert recs  # must not crash building recommendations


def test_report_average_does_not_crash_when_every_short_is_unknown():
    reporter = PerformanceReporter(store=TrackingStore())
    unknown_short = _short(1, youtube=PlatformMetrics())
    unknown_short.collected_at = "2026-01-01T00:00:00"

    shorts = [unknown_short]

    summary = reporter._summary(shorts)
    assert "—" in summary

    recs = reporter._recommendations(shorts)
    assert recs
