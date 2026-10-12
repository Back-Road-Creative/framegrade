"""An average must state what it averaged over.

``compare_views`` (observation.py) already refuses to blend view
counts across platforms because a YouTube view and a TikTok view are not the
same measurement. That refusal stopped at the module boundary:
``report._avg_engagement`` took ``statistics.mean`` over every short handed
to it regardless of which platforms it was posted to, silently averaging a
YouTube-only engagement rate with a TikTok-only one into one number the
report prints as a finding.

This propagates the same "don't blend unlike things" invariant to the
report's own aggregate: a mean over shorts posted to different platform
combinations must not compute a bare number.
"""

from __future__ import annotations

import pytest

from gomoveshift.tracking.report import PerformanceReporter, _avg_engagement
from gomoveshift.tracking.schema import PlatformMetrics, ShortRecord
from gomoveshift.tracking.store import TrackingStore


def _short(index: int, **platforms: PlatformMetrics) -> ShortRecord:
    return ShortRecord(index=index, file=f"short{index}.mp4", metrics=platforms)


def test_avg_engagement_declines_across_platform_combinations():
    """Two shorts with known rates from different platform sets must not
    silently average into one number."""
    youtube_only = _short(1, youtube=PlatformMetrics(views=100, likes=10, comments=0, shares=0))
    tiktok_only = _short(2, tiktok=PlatformMetrics(views=100, likes=5, comments=0, shares=0))

    basis = _avg_engagement([youtube_only, tiktok_only])

    assert basis.value is None
    assert basis.sample_size == 2
    assert basis.declined_reason  # says WHY, not just that it failed


def test_avg_engagement_computes_within_one_platform_combination():
    """Same platform set on every short is still safe to average."""
    first = _short(1, youtube=PlatformMetrics(views=100, likes=10, comments=0, shares=0))
    second = _short(2, youtube=PlatformMetrics(views=100, likes=20, comments=0, shares=0))

    basis = _avg_engagement([first, second])

    assert basis.value == pytest.approx(0.15)
    assert basis.sample_size == 2
    assert basis.platforms == frozenset({"youtube"})
    assert basis.declined_reason == ""


def test_avg_engagement_declined_reason_empty_state_is_distinct():
    """ "Nothing measured" and "not comparable" are different reasons, not the
    same blank state."""
    nothing_known = _short(1, youtube=PlatformMetrics())

    basis = _avg_engagement([nothing_known])

    assert basis.value is None
    assert basis.sample_size == 0
    assert "no shorts" in basis.declined_reason.lower()


def test_summary_report_does_not_print_a_blended_cross_platform_average():
    """The rendered report must not carry a fabricated blended percentage —
    it renders the unknown marker and states why."""
    reporter = PerformanceReporter(store=TrackingStore())
    youtube_only = _short(1, youtube=PlatformMetrics(views=100, likes=10, comments=0, shares=0))
    youtube_only.collected_at = "2026-01-01T00:00:00"
    tiktok_only = _short(2, tiktok=PlatformMetrics(views=100, likes=5, comments=0, shares=0))
    tiktok_only.collected_at = "2026-01-01T00:00:00"

    summary = reporter._summary([youtube_only, tiktok_only])

    assert "10.00%" not in summary  # youtube's own rate, not blended in alone
    assert "7.50%" not in summary  # the naive (wrong) blended mean
    assert "Avg engagement rate | — " in summary or "Avg engagement rate | —" in summary
    assert "platform" in summary.lower()  # the decline reason is surfaced
