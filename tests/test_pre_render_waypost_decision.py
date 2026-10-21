"""Waypost's long-form title is chosen and checked before an encode."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from seo_manager.analytics import metric_source
from seo_manager.analytics.title_selection import (
    EvidenceSnapshot,
    Place,
    TemplateObservation,
    TitleFacts,
)

from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator
from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    _approve_waypost_titles_before_encode,
)
from gomoveshift.pipeline.drive_state_model import PipelineState
from gomoveshift.pipeline.route_claims import RouteWindow, measure_route
from gomoveshift.pipeline.title_decision import (
    PreRenderTitleError,
    approve_retained_title,
)


class _Geo:
    def reverse_geocode_detailed(self, lat, lon):
        name = "Alder" if lat < 10 else "Birch" if lat < 11 else "Cedar"
        return {"town": name, "county": "Ash County", "state": "Virginia"}


def _route():
    fixes = [(float(t), 9.0 if t < 120 else 10.0 if t < 240 else 11.0, -81.0) for t in range(360)]
    return measure_route([RouteWindow("lf1", 0, 360, fixes)], _Geo())


def _cross_state_route():
    class CrossGeo(_Geo):
        def reverse_geocode_detailed(self, lat, lon):
            value = super().reverse_geocode_detailed(lat, lon)
            if lat >= 11:
                value["state"] = "West Virginia"
            return value

    fixes = [(float(t), 9.0 if t < 120 else 10.0 if t < 240 else 11.0, -81.0) for t in range(360)]
    return measure_route([RouteWindow("lf1", 0, 360, fixes)], CrossGeo())


def _snapshot():
    def obs(template, rate):
        return TemplateObservation(
            template_id=template,
            format="longform",
            traffic_source="search",
            metric_source=metric_source.YOUTUBE_THUMBNAIL_IMPRESSION_CTR,
            rate=rate,
            n=5000,
            window_start="2026-10-01T00:00:00+00:00",
            window_end="2026-10-10T00:00:00+00:00",
        )

    return EvidenceSnapshot(
        as_of="2026-10-14T00:00:00+00:00",
        observations=(obs("route", 0.04), obs("region_route", 0.07)),
    )


def test_seo_winner_omitting_material_place_yields_safe_baseline():
    label = "Alder and Birch and Cedar, VA"
    facts = TitleFacts(
        format="longform",
        baseline_title="Alder and Birch and Cedar, VA in 4K",
        origin=Place("Alder", "a"),
        destination=Place("Cedar", "c"),
        region=Place("Virginia", "v"),
        suffix=" in 4K",
    )
    decision = approve_retained_title(label, _route(), None, snapshot=_snapshot(), facts=facts)
    assert decision["chosen_title"] == facts.baseline_title
    assert decision["chosen_index"] == 0
    assert decision["reason"] == "retained_route_excluded_winner"
    assert decision["geo_rejected_index"] == 2


def test_false_single_town_baseline_refuses_before_encode():
    with pytest.raises(PreRenderTitleError, match="Birch"):
        approve_retained_title("Alder, VA", _route(), None)


@pytest.mark.parametrize("state", ["VA", "West Virginia"])
def test_title_region_must_cover_every_observed_state(state):
    with pytest.raises(PreRenderTitleError, match="outside"):
        approve_retained_title(f"Alder and Birch and Cedar, {state}", _cross_state_route(), None)


def test_metadata_uses_pre_render_decision_without_selecting_again():
    decision = approve_retained_title("Ash County, VA", _route(), None)
    with (
        patch.object(LongFormMetadataGenerator, "_init_seo_manager"),
        patch.object(LongFormMetadataGenerator, "_init_ledger"),
        patch(
            "gomoveshift.pipeline.title_decision.select_title",
            side_effect=AssertionError("late select"),
        ),
    ):
        gen = LongFormMetadataGenerator()
        gen._manager = None
        metadata = gen.generate("Ash County, VA", preselected_title_decision=decision)
    assert metadata["title"] == decision["chosen_title"]
    assert metadata["title_decision"] == decision


def test_changed_baseline_cannot_reuse_an_old_decision():
    decision = approve_retained_title("Ash County, VA", _route(), None)
    with (
        patch.object(LongFormMetadataGenerator, "_init_seo_manager"),
        patch.object(LongFormMetadataGenerator, "_init_ledger"),
    ):
        gen = LongFormMetadataGenerator()
        gen._manager = None
        with pytest.raises(PreRenderTitleError, match="baseline"):
            gen.generate("Elm County, VA", preselected_title_decision=decision)


def test_split_part_keeps_the_same_approved_title_in_metadata():
    decision = approve_retained_title("Ash County, VA", _route(), None, title_part=(1, 2))
    with (
        patch.object(LongFormMetadataGenerator, "_init_seo_manager"),
        patch.object(LongFormMetadataGenerator, "_init_ledger"),
    ):
        gen = LongFormMetadataGenerator()
        gen._manager = None
        metadata = gen.generate(
            "Ash County, VA", title_part=(1, 2), preselected_title_decision=decision
        )
    assert metadata["title"] == decision["chosen_title"]
    assert metadata["title"].endswith("Part 1 of 2")


def test_phase_bc_records_waypost_decision_from_retained_route():
    state = PipelineState()
    state.per_lf_titles = {1: "Ash County, VA"}
    points = [
        {"time": float(t), "lat": 9.0 if t < 120 else 10.0 if t < 240 else 11.0, "lon": -81.0}
        for t in range(360)
    ]
    _approve_waypost_titles_before_encode(
        state,
        {1: "Ash County, VA"},
        [SimpleNamespace(index=1, part_count=1)],
        {1: points},
        _Geo(),
        {1: _route()},
    )
    decision = state.per_lf_pre_render_title_decisions[1]
    assert decision["chosen_title"] == "Ash County, VA in 4K"
    assert decision["ost_text"] == "Ash County, VA"
    assert state.per_lf_location_evidence[1]["places"]
    restored = PipelineState.from_dict(state.to_dict())
    assert restored.per_lf_pre_render_title_decisions[1] == decision


def test_phase_bc_refuses_retained_gps_without_route_review():
    state = PipelineState()
    state.per_lf_titles = {1: "Ash County, VA"}
    with pytest.raises(PreRenderTitleError, match="no route evidence"):
        _approve_waypost_titles_before_encode(
            state,
            {1: "Ash County, VA"},
            [SimpleNamespace(index=1, part_count=1)],
            {1: [{"time": 1.0, "lat": 9.0, "lon": -81.0}]},
            _Geo(),
            {},
        )
