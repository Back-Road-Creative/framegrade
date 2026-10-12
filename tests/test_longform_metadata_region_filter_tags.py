"""TDD tests: region-filter the channel-history phrase-tag merge.

The SEO optimizer merges the channel's "top-performing tags" (analytics)
into ``metadata.tags`` with NO topical/region filter. Those historical tags
include location phrases like ``"County Kerry drive"`` stamped on past Irish
videos. Consumed wholesale by the phrase-tag merge in
``_generate_with_seo_manager``, they leak onto EVERY long-form's tag surface —
so once the channel's best performers are Irish, a USA LF ships Irish location
tags (and vice versa): a cross-region location leak.

The fix region-filters that merge against THIS LF's ``resolved.regions``:
a phrase carrying a county/region marker whose region isn't one of the LF's
resolved regions is dropped; generic phrases and in-region location phrases
survive.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from gomoveshift.models import ResolvedLocation
from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator


class _SeoManagerStubGenerator(LongFormMetadataGenerator):
    """SEO Manager path with controllable ``metadata.tags`` output."""

    seo_tags: list[str] = []

    def _init_seo_manager(self):  # pragma: no cover - simple override
        manager = MagicMock()
        manager.store.query_by_site.return_value = []

        def _make_metadata(*_args: Any, **_kwargs: Any) -> Any:
            meta = MagicMock()
            meta.title = "Drive Through The Place"
            meta.description = "A scenic drive."
            meta.tags = list(self.seo_tags)
            meta.keyword_density = 0.0
            meta.readability_score = 0.0
            meta.content_id = "stub-content-id"
            return meta

        manager.generate.side_effect = _make_metadata
        manager.save.return_value = "stub-event-id"
        self._manager = manager

        optimizer = MagicMock()
        optimizer.analyze_patterns.return_value = None
        optimizer.optimize_metadata.side_effect = _make_metadata
        self._optimizer = optimizer

        self._seo_degraded = False

    def _init_ledger(self):  # pragma: no cover - skip ledger IO in tests
        self._ledger = None
        self._ledger_path = None

    def _sync_youtube_analytics(self):  # pragma: no cover - skip network
        return None


@pytest.fixture
def seo_gen(tmp_path):
    return _SeoManagerStubGenerator(seo_storage_path=tmp_path)


# A USA LF (Nelson County, VA) whose optimizer-merged top-performing tags
# carry a cross-region Irish location phrase plus a generic and an in-region
# location phrase.
USA_LOCATION = "Nelson County, VA"
RECORDING_DATE = "2025-09-14"
USA_REGIONS = ["Nelson County, VA"]


def _generate_usa(seo_gen, tags: list[str]) -> dict:
    type(seo_gen).seo_tags = tags
    return seo_gen.generate(
        location=USA_LOCATION,
        video_duration=1800.0,
        project_name="t",
        recording_date=RECORDING_DATE,
        resolved_location=ResolvedLocation(regions=list(USA_REGIONS)),
    )


class TestRegionFilteredPhraseTags:
    def test_cross_region_location_tag_dropped(self, seo_gen):
        """An Irish location phrase leaked via top-performers must NOT ship
        on a USA LF; the LF's own USA location tags survive."""
        result = _generate_usa(
            seo_gen,
            ["County Kerry drive", "County Cork scenic drive"],
        )
        tags_lower = [t.lower() for t in result["tags"]]
        assert "county kerry drive" not in tags_lower
        assert "county cork scenic drive" not in tags_lower
        # THIS LF's own USA location tags (from _location_tags) survive.
        assert "nelson county drive" in tags_lower

    def test_generic_phrase_survives(self, seo_gen):
        """A non-location phrase carries no region marker and always survives,
        regardless of the LF's region."""
        result = _generate_usa(seo_gen, ["mountain pass drive"])
        assert "mountain pass drive" in result["tags"]

    def test_in_region_location_tag_survives(self, seo_gen):
        """A county phrase naming THIS LF's own region is not a cross-region
        leak and survives the merge (it is not produced by _location_tags)."""
        result = _generate_usa(seo_gen, ["Nelson County road trip"])
        tags_lower = [t.lower() for t in result["tags"]]
        assert "nelson county road trip" in tags_lower

    def test_cross_region_bare_town_road_tag_dropped(self, seo_gen):
        """Extend the filter beyond county/parish markers. A bare foreign
        TOWN or ROAD phrase carrying NO administrative marker must also drop —
        its proper noun ('kenmare', 'dingle') is not one of THIS LF's places."""
        result = _generate_usa(
            seo_gen,
            ["Kenmare loop drive", "Dingle coastal road", "Slea Head scenic route"],
        )
        tags_lower = [t.lower() for t in result["tags"]]
        assert "kenmare loop drive" not in tags_lower
        assert "dingle coastal road" not in tags_lower
        assert "slea head scenic route" not in tags_lower

    def test_own_route_town_phrase_survives(self, seo_gen):
        """A phrase naming one of THIS LF's own route towns (from
        ``ordered_place_sequence``) is in-region and survives WITHOUT any
        county/parish marker."""
        type(seo_gen).seo_tags = ["Faber scenic road trip", "Kenmare loop drive"]
        result = seo_gen.generate(
            location=USA_LOCATION,
            video_duration=1800.0,
            project_name="t",
            recording_date=RECORDING_DATE,
            resolved_location=ResolvedLocation(
                regions=list(USA_REGIONS),
                ordered_place_sequence=["Faber", "Massies Mill"],
            ),
        )
        tags_lower = [t.lower() for t in result["tags"]]
        assert "faber scenic road trip" in tags_lower  # own town → kept
        assert "kenmare loop drive" not in tags_lower  # foreign town → dropped
