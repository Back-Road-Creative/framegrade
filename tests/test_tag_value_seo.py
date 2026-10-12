"""Tag/hashtag value.

- Route facts: feed the SEO optimizer the LF's real route facts (place sequence,
  regions, distance) instead of fixed boilerplate, and CONSUME its returned
  multi-word analytics tags (single-word fragments stay dropped).
- Geo hashtags: the geo hashtags (location + country) must survive the 12-hashtag slice
  — they now lead instead of being appended after the 12 base hashtags.
- Seasonal tags: month + season tags derived from the recording date.
"""

from __future__ import annotations

from gomoveshift.models import ResolvedLocation
from gomoveshift.pipeline.longform_metadata import (
    GMS_BASE_TAGS,
    LongFormMetadataGenerator,
    _optimizer_body,
    _phrase_tags,
    _seasonal_tags,
)


class TestSeasonalTags:
    def test_autumn_month(self) -> None:
        assert _seasonal_tags("2025-09-14") == ["September", "autumn", "autumn drive"]

    def test_spring_month(self) -> None:
        assert _seasonal_tags("2026-05-31") == ["May", "spring", "spring drive"]

    def test_winter_month_january(self) -> None:
        assert _seasonal_tags("2026-01-02") == ["January", "winter", "winter drive"]

    def test_unparseable_date_yields_nothing(self) -> None:
        assert _seasonal_tags("May 2025") == []
        assert _seasonal_tags("") == []
        assert _seasonal_tags(None) == []

    def test_out_of_range_month_yields_nothing(self) -> None:
        assert _seasonal_tags("2025-13-01") == []
        assert _seasonal_tags("2025-00-01") == []


class TestOptimizerBody:
    def test_includes_place_sequence_and_distance(self) -> None:
        resolved = ResolvedLocation(ordered_place_sequence=["Killarney", "Kenmare", "Sneem"])
        body = _optimizer_body(
            "Killarney to Dingle, Kerry",
            "15 min",
            regions=[{"name": "County Kerry"}],
            distance_km=47.0,
            resolved=resolved,
        )
        assert "Killarney" in body and "Sneem" in body
        assert "County Kerry" in body
        assert "47 km" in body
        assert "4K drive" in body

    def test_degrades_without_gps(self) -> None:
        body = _optimizer_body("Somewhere", "10 min")
        # No place sequence / regions / distance — evergreen sentences only.
        assert "Somewhere" in body
        assert "passes through" not in body
        assert "scenic road trip" not in body


class TestPhraseTags:
    def test_keeps_multiword_drops_fragments(self) -> None:
        existing = {"scenic drive"}
        out = _phrase_tags(
            ["dingle", "toward", "mountain pass drive", "coastal cliff route"],
            existing,
        )
        assert out == ["mountain pass drive", "coastal cliff route"]

    def test_dedups_against_existing(self) -> None:
        existing = {"scenic drive"}
        out = _phrase_tags(["Scenic Drive", "new long phrase"], existing)
        assert out == ["new long phrase"]

    def test_none_and_non_strings_safe(self) -> None:
        assert _phrase_tags(None, set()) == []
        assert _phrase_tags([123, None, "two words"], set()) == ["two words"]


class _StubGen(LongFormMetadataGenerator):
    def _init_seo_manager(self):  # pragma: no cover - simple override
        self._manager = None
        self._optimizer = None
        self._seo_degraded = True

    def _init_ledger(self):  # pragma: no cover - skip ledger IO
        self._ledger = None
        self._ledger_path = None


class TestHashtagGeoSurvival:
    """Per-component geo hashtags lead so the [:12] slice keeps them."""

    def _desc(self, gen: _StubGen) -> str:
        return gen._build_gms_description(
            location="Killarney, County Kerry, Ireland",
            seo_description="POV drive through Killarney, County Kerry, Ireland",
            duration_str="15 min",
            publish_day="Sunday",
        )

    def test_geo_hashtags_lead(self, tmp_path) -> None:
        gen = _StubGen(seo_storage_path=tmp_path)
        hashtag_line = self._desc(gen).rstrip().split("\n")[-1]
        tags = hashtag_line.split()
        # One hashtag per location component (town, county, country), in order,
        # leading so the [:12] slice never strips the drive-specific geo tags.
        assert tags[0] == "#Killarney"
        assert tags[1] == "#CountyKerry"
        assert tags[2] == "#Ireland"

    def test_geo_hashtags_present_within_cap(self, tmp_path) -> None:
        gen = _StubGen(seo_storage_path=tmp_path)
        hashtag_line = self._desc(gen).rstrip().split("\n")[-1]
        tags = hashtag_line.split()
        assert len(tags) <= 12
        assert "#Killarney" in tags and "#Ireland" in tags


class TestSeasonalTagsInLocationTags:
    """Month/season tags reach the YouTube tag list via _location_tags."""

    def test_month_and_season_in_generated_tags(self, tmp_path) -> None:
        gen = _StubGen(seo_storage_path=tmp_path)
        result = gen.generate(
            location="Killarney, County Kerry",
            video_duration=1800.0,
            project_name="t",
            recording_date="2025-09-14",
        )
        assert "September" in result["tags"]
        assert "autumn" in result["tags"]
        assert "autumn drive" in result["tags"]
        # Evergreen base tags still lead.
        assert result["tags"][0] in GMS_BASE_TAGS
