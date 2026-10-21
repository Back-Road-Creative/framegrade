"""A verified retained-route region becomes a Waypost long-form title fact."""

from __future__ import annotations

import pytest
from seo_manager.analytics import metric_source
from seo_manager.analytics.title_selection import EvidenceSnapshot, TemplateObservation

from gomoveshift.pipeline.route_claims import RouteWindow, measure_route
from gomoveshift.pipeline.title_decision import PreRenderTitleError, approve_retained_title

from test_final_text_record import _CarolinaGeocoder


def _route():
    fixes = [(float(t), 9.5 if t < 300 else 10.2, 20.0) for t in range(600)]
    return measure_route([RouteWindow("lf1", 0, 600, fixes)], _CarolinaGeocoder())


def test_county_fallback_is_a_waypost_region_decision_not_missing_facts():
    decision = approve_retained_title("Ashby County, SC", _route(), None)
    assert decision["chosen_title"] == "Ashby County, SC in 4K"
    assert decision["reason"] == "insufficient_evidence"
    assert decision["candidates"][0]["template_id"] == "region_only_lf"
    assert decision["candidates"][0]["claim_evidence"]


def test_waypost_region_variant_can_win_only_on_measured_title_lift():
    def observation(template, rate):
        return TemplateObservation(
            template_id=template,
            format="longform",
            traffic_source="search",
            metric_source=metric_source.YOUTUBE_THUMBNAIL_IMPRESSION_CTR,
            rate=rate,
            n=500,
            window_start="2026-03-01T00:00:00",
            window_end="2026-03-14T00:00:00",
        )

    snapshot = EvidenceSnapshot(
        as_of="2026-03-20T00:00:00",
        observations=(observation("region_only_lf", 0.04), observation("region_drive_lf", 0.06)),
    )
    decision = approve_retained_title("Ashby County, SC", _route(), None, snapshot=snapshot)
    assert decision["chosen_title"] == "Ashby County, SC Drive in 4K"
    assert decision["reason"] == "lift_over_baseline"


def test_wrong_county_still_refuses_before_waypost_decision():
    with pytest.raises(PreRenderTitleError, match="unsupported"):
        approve_retained_title("Other County, SC", _route(), None)
