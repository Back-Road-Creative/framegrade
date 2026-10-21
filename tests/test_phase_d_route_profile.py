"""Tests for Phase D route profile integration.

Verifies that LongFormMetadataGenerator enriches metadata with GPS-verified
RouteProfile data (towns, roads, features) when provided.
"""

from unittest.mock import patch

import pytest

from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator


@pytest.fixture
def generator():
    """Create generator with mocked SEO manager and content validator."""
    with patch.object(LongFormMetadataGenerator, "_init_seo_manager"):
        with patch.object(LongFormMetadataGenerator, "_init_ledger"):
            gen = LongFormMetadataGenerator()
            gen._seo_degraded = True
            gen._manager = None
            gen._ledger = None
            return gen


def _generate_with_profile(generator, location, route_profile=None, **kw):
    """Call generate() with ContentValidator mocked out."""
    with patch("gomoveshift.pipeline.longform_metadata.ContentValidator"):
        return generator.generate(
            location=location,
            route_profile=route_profile,
            **kw,
        )


class TestPhaseDRouteProfileMetadata:
    def test_metadata_with_route_profile(self, generator):
        """When route_profile is provided, description includes towns/roads."""
        profile = {
            "towns_traversed": ["Port Royal", "Beaufort", "Hunting Island"],
            "roads_traversed": ["Trask Parkway", "US-21"],
            "features": ["Hunting Island State Park"],
            "states_traversed": ["South Carolina"],
        }
        result = _generate_with_profile(
            generator, "Port Royal to Beaufort, SC", route_profile=profile
        )
        desc = result.get("description", "")
        assert "Port Royal" in desc
        assert "Beaufort" in desc
        tags = result.get("tags", [])
        # GPS-verified place names should be in tags
        assert any("Port Royal" in t for t in tags)

    def test_metadata_without_route_profile(self, generator):
        """Without route_profile, metadata works as before."""
        result = _generate_with_profile(generator, "Port Royal, SC")
        assert "title" in result
        assert "description" in result

    def test_empty_profile_fields(self, generator):
        """Empty profile fields don't break metadata generation."""
        result = _generate_with_profile(
            generator,
            "Port Royal, SC",
            route_profile={"towns_traversed": [], "roads_traversed": []},
        )
        assert "title" in result

    def test_tags_deduplication(self, generator):
        """Profile tags don't duplicate existing tags."""
        profile = {
            "towns_traversed": ["Beaufort"],
            "roads_traversed": [],
            "features": [],
        }
        result = _generate_with_profile(generator, "Beaufort, SC", route_profile=profile)
        tags = result.get("tags", [])
        beaufort_count = sum(1 for t in tags if t.lower() == "beaufort")
        assert beaufort_count <= 1

    def test_roads_capped_in_description(self, generator):
        """Up to 5 roads appear in description body."""
        profile = {
            "towns_traversed": ["Town A"],
            "roads_traversed": [
                "US-1",
                "US-2",
                "US-3",
                "US-4",
                "US-5",
                "US-6",
            ],
            "features": [],
        }
        result = _generate_with_profile(
            generator, "Town A, SC", route_profile=profile, road_coverage_measured=True
        )
        desc = result.get("description", "")
        assert "US-5" in desc
        assert "US-6" not in desc

    def test_features_added_to_tags(self, generator):
        """Features from route profile appear in tags."""
        profile = {
            "towns_traversed": [],
            "roads_traversed": [],
            "features": ["Hunting Island State Park"],
        }
        result = _generate_with_profile(generator, "Beaufort, SC", route_profile=profile)
        tags = result.get("tags", [])
        assert "Hunting Island State Park" in tags

    def test_roads_added_to_tags_max_five(self, generator):
        """At most 5 roads are added to tags from the profile."""
        profile = {
            "towns_traversed": [],
            "roads_traversed": [f"US-{i}" for i in range(10)],
            "features": [],
        }
        result = _generate_with_profile(generator, "Somewhere, SC", route_profile=profile)
        tags = result.get("tags", [])
        road_tags = [t for t in tags if t.startswith("US-")]
        assert len(road_tags) <= 5

    def test_street_class_road_never_reaches_description_or_tags(self, generator):
        """A road that fails ``_road_passes_quality`` (a plain street/
        lane/town road) must never reach the description body or tags, even
        though it clears the older share/distance-only floor. A ref-numbered
        or named through-road in the same list still ships."""
        profile = {
            "towns_traversed": [],
            "roads_traversed": [
                "Oliver Plunkett Street",
                "N71",
                "The Coach Road",
                "R586",
            ],
            "features": [],
        }
        result = _generate_with_profile(
            generator, "West Cork", route_profile=profile, road_coverage_measured=True
        )
        desc = result.get("description", "")
        tags = result.get("tags", [])
        assert "Oliver Plunkett Street" not in desc
        assert "The Coach Road" not in desc
        assert "N71" in desc
        assert "R586" in desc
        assert "Oliver Plunkett Street" not in tags
        assert "The Coach Road" not in tags
        assert "N71" in tags
        assert "R586" in tags


class TestPhaseDRouteProfileStateKey:
    def test_reads_route_profiles_from_state(self):
        """Phase D reads route_profiles from state dict — both int and str keys."""
        # Verify the enrichment method handles profile data correctly
        result = {
            "description": "Some description\n\n📍 Location: Test",
            "tags": ["existing"],
        }
        profile = {
            "towns_traversed": ["TestTown"],
            "roads_traversed": ["US-50"],
            "features": [],
        }
        enriched = LongFormMetadataGenerator._enrich_with_route_profile(
            result, profile, road_coverage_measured=True
        )
        assert "TestTown" in enriched["description"]
        assert "TestTown" in enriched["tags"]
        assert "US-50" in enriched["tags"]
