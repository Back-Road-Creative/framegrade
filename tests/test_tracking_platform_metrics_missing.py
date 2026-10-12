"""PlatformMetrics must distinguish "not reported" from a measured 0.

Regression coverage for the dataclass-zero-as-measured-zero bug: every
PlatformMetrics measurement field defaulted to numeric 0, so a platform that
never reported a metric was indistinguishable from one that reported a real
zero, and the fabricated zero was what got persisted and averaged.
"""

from __future__ import annotations

import json

from gomoveshift.tracking.schema import PlatformMetrics


def test_missing_metric_round_trips_as_missing_not_zero():
    """A metric absent from the platform payload stays missing after a
    round trip through to_dict/from_dict/JSON — never coerced to 0."""
    metrics = PlatformMetrics(views=100)  # shares never reported
    restored = PlatformMetrics.from_dict(json.loads(json.dumps(metrics.to_dict())))

    assert restored.shares is None
    assert metrics.to_dict()["shares"] is None


def test_reported_zero_survives_as_zero_not_missing():
    """A platform that genuinely reported 0 shares must stay 0, and stay
    distinguishable from a platform that never reported shares at all."""
    metrics = PlatformMetrics(views=100, shares=0)
    restored = PlatformMetrics.from_dict(json.loads(json.dumps(metrics.to_dict())))

    assert restored.shares == 0
    assert restored.shares is not None


def test_engagement_rate_is_unknown_when_inputs_are_missing():
    """engagement_rate must not invent a rate from missing inputs. We
    return None ("unknown") rather than 0.0 — 0.0 would silently
    reintroduce the fabricated-zero bug this fix exists to remove."""
    metrics = PlatformMetrics(views=100)  # likes/comments/shares unreported
    assert metrics.engagement_rate is None


def test_engagement_rate_with_known_zero_views_is_zero_not_unknown():
    """A genuinely measured 0 views (not missing) still yields a defined
    0.0 engagement rate — this is not the missing-data case."""
    metrics = PlatformMetrics(views=0, likes=0, comments=0, shares=0)
    assert metrics.engagement_rate == 0.0


def test_engagement_rate_computes_when_all_inputs_known():
    metrics = PlatformMetrics(views=100, likes=5, comments=3, shares=2)
    assert metrics.engagement_rate == 0.10


def test_legacy_zero_payload_still_loads():
    """A payload persisted before this fix, carrying plain zeros, must
    still load without raising. We cannot retroactively know whether a
    legacy zero was measured or missing, so it loads as the literal value
    on disk (0), same as any other schema version."""
    legacy_payload = {
        "views": 0,
        "likes": 0,
        "comments": 0,
        "shares": 0,
        "watch_time_seconds": 0.0,
        "avg_view_duration_seconds": 0.0,
        "impressions": 0,
        "click_through_rate": 0.0,
    }
    restored = PlatformMetrics.from_dict(legacy_payload)

    assert restored.views == 0
    assert restored.shares == 0


def test_default_platform_metrics_is_all_missing():
    """The zero-arg constructor (used by every collector's error/no-data
    return path) now signals unknown, not a measured zero, for every
    field."""
    metrics = PlatformMetrics()
    assert metrics.views is None
    assert metrics.likes is None
    assert metrics.comments is None
    assert metrics.shares is None
    assert metrics.watch_time_seconds is None
    assert metrics.avg_view_duration_seconds is None
    assert metrics.impressions is None
    assert metrics.click_through_rate is None
