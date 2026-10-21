"""Description, tags and chapter titles name no road while road coverage
is unmeasured.

Road names come from untimed reverse-geocode votes and Overpass route matches;
neither is measured over the LF's retained time, so a road in published text
is an unbacked claim. Every road-emitting site takes
``road_coverage_measured`` (default False). Each test pairs a measured=True
control, which still produces the road, with the default, which does not.
Synthetic places only.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gomoveshift.pipeline.chapters.marker_merger import MarkerMerger
from gomoveshift.pipeline.longform_metadata import (
    LongFormMetadataGenerator,
    _seo_search_line,
)

ROAD = "US-21"
LABEL = f"Harrow to Pellmere via {ROAD}, SC"
PROFILE = {
    "towns_traversed": ["Harrow", "Bexley", "Pellmere"],
    "roads_traversed": [ROAD, "Fern Parkway"],
    "features": [],
    "states_traversed": ["South Carolina"],
    "start_town": "Harrow",
    "end_town": "Pellmere",
}


def _base_result() -> dict:
    return {
        "title": "Harrow to Pellmere, SC",
        "description": "POV drive.\n\n📍 Location: Harrow to Pellmere, SC\n",
        "tags": ["scenic drive"],
        "location": "Harrow to Pellmere, SC",
    }


class TestSeoSearchLine:
    def test_measured_control_restores_via_road(self):
        assert f"via {ROAD}" in _seo_search_line(LABEL, road_coverage_measured=True)

    def test_default_names_no_road_keeps_arc_and_region(self):
        out = _seo_search_line(LABEL)
        assert ROAD not in out and " via " not in out
        assert "Harrow to Pellmere" in out and "South Carolina" in out

    def test_default_ignores_named_routes(self):
        out = _seo_search_line("Harrow to Pellmere, SC", named_routes=[ROAD])
        assert ROAD not in out


class TestRouteProfileEnrichment:
    def test_measured_control_names_roads(self):
        out = LongFormMetadataGenerator._enrich_with_route_profile(
            _base_result(), dict(PROFILE), road_coverage_measured=True
        )
        assert f"Roads traveled: {ROAD}" in out["description"]
        assert ROAD in out["tags"]

    def test_default_names_no_road_in_description_tags_or_routes(self):
        out = LongFormMetadataGenerator._enrich_with_route_profile(_base_result(), dict(PROFILE))
        assert ROAD not in out["description"] and "Fern Parkway" not in out["description"]
        assert "Roads traveled" not in out["description"]
        assert not any(r in out["tags"] for r in (ROAD, "Fern Parkway"))
        assert not out.get("named_routes")
        # Places stay: towns are measured by the retained-route record.
        assert "Started in Harrow, ended in Pellmere" in out["description"]
        assert "Bexley" in out["tags"]


class TestFactualBody:
    def test_measured_control_names_route(self):
        body = LongFormMetadataGenerator._build_factual_body(
            "Harrow, SC", "10m", named_routes=[ROAD], road_coverage_measured=True
        )
        assert f"along {ROAD}" in body

    def test_default_names_no_route(self):
        body = LongFormMetadataGenerator._build_factual_body(
            "Harrow, SC", "10m", chapter_towns=["Harrow", "Bexley"], named_routes=[ROAD]
        )
        assert ROAD not in body and "Bexley" in body


@pytest.fixture
def generator():
    with patch.object(LongFormMetadataGenerator, "_init_seo_manager"):
        with patch.object(LongFormMetadataGenerator, "_init_ledger"):
            gen = LongFormMetadataGenerator()
            gen._seo_degraded = True
            gen._manager = None
            gen._ledger = None
            return gen


class TestGenerateEntryPoint:
    def _gen(self, generator, **kw):
        with patch("gomoveshift.pipeline.longform_metadata.ContentValidator"):
            # The label carries no via-road; roads arrive only through
            # the route profile, which is what this entry point must gate.
            return generator.generate(
                location="Harrow to Pellmere, SC", route_profile=dict(PROFILE), **kw
            )

    def test_measured_control_names_road(self, generator):
        out = self._gen(generator, road_coverage_measured=True)
        assert ROAD in out["description"] and ROAD in out["tags"]

    def test_default_names_no_road_anywhere(self, generator):
        out = self._gen(generator)
        assert ROAD not in out["description"] and "Fern Parkway" not in out["description"]
        assert not any(ROAD in t or "Fern Parkway" in t for t in out["tags"])


def _merge(merger: MarkerMerger) -> list[dict]:
    merger.route_waypoints = [(32.0, -80.0)]
    town = SimpleNamespace(city="Bexley", latitude=32.0, longitude=-80.0)
    return merger.merge_markers_with_chapters(
        waypoints_with_towns=[(200.0, town)],
        markers=None,
        video_duration=900.0,
        start_location="Harrow",
        end_location="Pellmere",
    )


class TestChapterTitles:
    def test_measured_control_suffixes_route(self):
        merger = MarkerMerger(named_routes=[ROAD], road_coverage_measured=True)
        assert any(c["title"] == f"Entering Bexley, {ROAD}" for c in _merge(merger))

    def test_default_names_no_route(self):
        chapters = _merge(MarkerMerger(named_routes=[ROAD]))
        assert any(c["title"] == "Entering Bexley" for c in chapters)
        assert not any(ROAD in c["title"] for c in chapters)
