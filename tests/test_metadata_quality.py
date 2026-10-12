"""Tests for YouTube metadata quality fixes.

These tests define expected behavior for metadata quality gates:
1. No publish date in description (clutters, ages poorly)
2. No duplicate chapter blocks in description
3. Generic hook detection and replacement
4. PipelineValidator._check_metadata_quality gate
5. Chapter embedding strips existing chapter block before appending
6. US business name filtering (publish phase)
"""

import pytest

from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator
from gomoveshift.pipeline.pipeline_validator import GateResult, PipelineValidator


class TestBuildGmsDescription:
    """Tests for _build_gms_description output quality."""

    @pytest.fixture
    def generator(self):
        """Create LongFormMetadataGenerator (SEO manager optional)."""
        return LongFormMetadataGenerator()

    def test_description_no_publish_date(self, generator):
        """Description should NOT include a 'Published:' line — it clutters and ages poorly."""
        result = generator._build_gms_description(
            location="Sheepshead Peninsula, County Cork, Ireland",
            seo_description="Experience the stunning coastal roads of West Cork.",
            duration_str="1:02:34",
            publish_day="Saturday",
            publish_date="2026-03-14",
        )
        assert "Published:" not in result, (
            f"Description should not contain 'Published:' line, got:\n{result}"
        )

    def test_no_duplicate_chapters_in_description(self, generator):
        """Description should have at most one CHAPTERS block after chapter embedding."""
        base_desc = generator._build_gms_description(
            location="Bantry, County Cork, Ireland",
            seo_description="Drive through the heart of West Cork.",
            duration_str="55:00",
            publish_day="Wednesday",
        )

        chapters_block = (
            "\n\n\U0001f4cd CHAPTERS\n"
            "0:00 Start - Bantry, County Cork\n"
            "12:30 Entering Durrus, County Cork\n"
            "45:00 Finish - Durrus, County Cork"
        )

        # Simulate the bug: chapters appended twice
        desc_with_dupes = base_desc + chapters_block + chapters_block

        # The dedup function should strip duplicates
        cleaned = generator._dedup_chapters(desc_with_dupes)
        count = cleaned.count("\U0001f4cd CHAPTERS")
        assert count == 1, f"Expected exactly 1 CHAPTERS block, found {count} in:\n{cleaned}"

    def test_generic_hook_replaced(self, generator):
        """Generic LLM filler hooks should be replaced with location-based POV hook."""
        result = generator._build_gms_description(
            location="Sheepshead Peninsula, County Cork, Ireland",
            seo_description="Dive into the latest developments of scenic Irish roads.",
            duration_str="1:02:34",
            publish_day="Saturday",
        )
        assert "Dive into the latest developments" not in result, (
            "Generic hook text should be replaced with driving-specific hook"
        )
        # Should contain location-based driving language instead
        assert "Sheepshead Peninsula" in result


class TestMetadataQualityGate:
    """Tests for PipelineValidator._check_metadata_quality gate."""

    @pytest.fixture
    def validator(self):
        return PipelineValidator()

    def test_metadata_quality_catches_duplicate_chapters(self, validator):
        """Validator should FAIL when description has duplicate chapter blocks."""
        metadata = {
            "title": "Bantry to Durrus \u2014 POV Driving West Cork, Ireland 4K",
            "description": (
                "Drive through the heart of West Cork.\n\n"
                "\U0001f4cd CHAPTERS\n"
                "0:00 Start - Bantry\n"
                "12:30 Entering Durrus\n\n"
                "\U0001f4cd CHAPTERS\n"
                "0:00 Start - Bantry\n"
                "12:30 Entering Durrus"
            ),
        }
        result = validator._check_metadata_quality(metadata)
        assert isinstance(result, GateResult)
        assert not result.passed, f"Should fail for duplicate chapters: {result.message}"

    def test_metadata_quality_catches_publish_date(self, validator):
        """Validator should FAIL when description contains 'Published:' line."""
        metadata = {
            "title": "Bantry to Durrus \u2014 POV Driving West Cork 4K",
            "description": (
                "Drive through the heart of West Cork.\n"
                "\U0001f4c5 Published: Saturday, March 14, 2026\n"
                "\U0001f4cd Location: Bantry, County Cork"
            ),
        }
        result = validator._check_metadata_quality(metadata)
        assert isinstance(result, GateResult)
        assert not result.passed, f"Should fail for publish date: {result.message}"

    def test_metadata_quality_catches_generic_filler(self, validator):
        """Validator should FAIL when description contains generic LLM filler."""
        metadata = {
            "title": "Bantry to Durrus \u2014 POV Driving West Cork 4K",
            "description": (
                "Dive into the latest developments of scenic Irish roads.\n"
                "\U0001f4cd Location: Bantry, County Cork"
            ),
        }
        result = validator._check_metadata_quality(metadata)
        assert isinstance(result, GateResult)
        assert not result.passed, f"Should fail for generic filler: {result.message}"

    def test_metadata_quality_catches_title_location_mismatch(self, validator):
        """Validator should FAIL when title locations don't match metadata location."""
        metadata = {
            "title": "Bantry to Kenmare \u2014 POV Driving Ireland 4K",
            "description": "POV drive through Sheepshead Peninsula to Durrus.",
            "location": "Sheepshead Peninsula to Durrus, County Cork, Ireland",
        }
        # Title says "Bantry to Kenmare" but location is "Sheepshead Peninsula to Durrus"
        # Neither "Sheepshead" nor "Durrus" appear in the title
        result = validator._check_metadata_quality(metadata)
        assert isinstance(result, GateResult)
        assert not result.passed, f"Should fail for location mismatch: {result.message}"

    def test_metadata_quality_catches_low_chapter_density(self, validator):
        """Validator should FAIL when too few chapters for video duration."""
        metadata = {
            "title": "Sheepshead Peninsula to Durrus | Ireland",
            "description": "POV drive through Sheepshead Peninsula.",
            "location": "Sheepshead Peninsula to Durrus, County Cork, Ireland",
            "chapters": [
                {"timestamp": "0:00", "title": "Start - Sheepshead"},
                {"timestamp": "43:15", "title": "Entering Kilcrohane, County Cork"},
                {"timestamp": "1:02:27", "title": "Finish - Durrus"},
            ],
            "video_duration": 3747.0,  # 62.5 min -> need 4+ intermediate
        }
        result = validator._check_metadata_quality(metadata)
        assert isinstance(result, GateResult)
        assert not result.passed, f"Should fail for low chapter density: {result.message}"
        assert "intermediate chapter" in result.message

    def test_metadata_quality_catches_publish_slot_in_description(self, validator):
        """Validator should FAIL when description has operational publish slot data."""
        metadata = {
            "title": "Sheepshead Peninsula to Durrus | Ireland",
            "description": (
                "POV drive through Sheepshead.\n"
                "NEXT PUBLISH SLOT: Saturday 2026-03-14 at 11:00:00 EST"
            ),
        }
        result = validator._check_metadata_quality(metadata)
        assert isinstance(result, GateResult)
        assert not result.passed, f"Should fail for publish slot: {result.message}"

    def test_metadata_quality_passes_clean_metadata(self, validator):
        """Validator should PASS for clean metadata with no quality issues."""
        metadata = {
            "title": "Sheepshead Peninsula to Durrus \u2014 POV Driving West Cork 4K",
            "description": (
                "POV drive along the stunning Sheepshead Peninsula coastal road "
                "to Durrus in West Cork, Ireland.\n\n"
                "\U0001f4cd Location: Sheepshead Peninsula, County Cork, Ireland\n"
                "\U0001f3ac Duration: 1:02:34\n\n"
                "\U0001f4cd CHAPTERS\n"
                "0:00 Start - Sheepshead Peninsula\n"
                "12:00 Entering Glanlough\n"
                "24:00 Entering Ahakista\n"
                "36:00 Entering Kilcrohane\n"
                "50:00 Entering Durrus\n"
                "1:02:00 Finish - Durrus"
            ),
            "location": "Sheepshead Peninsula to Durrus, County Cork, Ireland",
            "chapters": [
                {"timestamp": "0:00", "title": "Start - Sheepshead Peninsula"},
                {"timestamp": "12:00", "title": "Entering Glanlough"},
                {"timestamp": "24:00", "title": "Entering Ahakista"},
                {"timestamp": "36:00", "title": "Entering Kilcrohane"},
                {"timestamp": "50:00", "title": "Entering Durrus"},
                {"timestamp": "1:02:00", "title": "Finish - Durrus"},
            ],
            "video_duration": 3747.0,
        }
        result = validator._check_metadata_quality(metadata)
        assert isinstance(result, GateResult)
        assert result.passed, f"Should pass for clean metadata: {result.message}"


class TestChapterEmbedding:
    """Tests for chapter embedding in descriptions."""

    @pytest.fixture
    def generator(self):
        return LongFormMetadataGenerator()

    def test_chapter_embed_replaces_existing(self, generator):
        """When description has an existing chapter block, replace it, not append."""
        old_chapters = "\U0001f4cd CHAPTERS\n0:00 Start - Old Town\n10:00 Old Middle\n20:00 Old End"
        description = (
            f"A scenic drive through Ireland.\n\n{old_chapters}\n\n#POVDriving #ScenicDrive"
        )

        new_chapters = (
            "0:00 Start - Bantry, County Cork\n"
            "12:30 Entering Durrus, County Cork\n"
            "45:00 Finish - Durrus, County Cork"
        )

        result = generator._embed_chapters(description, new_chapters)

        # Old chapters should be gone
        assert "Old Town" not in result
        assert "Old Middle" not in result

        # New chapters should be present
        assert "Bantry, County Cork" in result
        assert "Entering Durrus" in result

        # Exactly one CHAPTERS block
        count = result.count("\U0001f4cd CHAPTERS")
        assert count == 1, f"Expected 1 CHAPTERS block, found {count}"

        # Hashtags should still be present
        assert "#POVDriving" in result


class TestChapterGeneratorFallback:
    """Tests for chapter generator locality/interval fallback."""

    def test_format_location_uses_locality_when_no_city(self):
        """Format should use locality when city is not available."""
        from gomoveshift.pipeline.chapters import ChapterGenerator

        gen = ChapterGenerator.__new__(ChapterGenerator)
        from gps_journey.core.models import Location

        loc = Location(
            latitude=51.56,
            longitude=-9.75,
            city=None,
            locality="Ahakista",
            county="County Cork",
            country="Ireland",
        )
        result = gen._format_location(loc)
        assert "Ahakista" in result
        assert "County Cork" in result

    def test_format_location_uses_city_first(self):
        """City should take priority over locality."""
        from gomoveshift.pipeline.chapters import ChapterGenerator

        gen = ChapterGenerator.__new__(ChapterGenerator)
        from gps_journey.core.models import Location

        loc = Location(
            latitude=51.63,
            longitude=-9.56,
            city="Kilcrohane",
            locality="Townland",
            county="County Cork",
            country="Ireland",
        )
        result = gen._format_location(loc)
        assert "Kilcrohane" in result
        assert "Townland" not in result


class TestRecordingDateFromGPS:
    """Recording date must use gps_date when available, not journey.start_date."""

    def test_extract_recording_date_uses_gps_date(self):
        """When gps_date is provided, recording date must match it."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator()
        result = orch._extract_recording_date(
            gpx_date="2025-09-13",
            ffprobe_creation_time=None,
            project_name="demoroute2_3840p",
        )
        assert result is not None
        assert result.day == 13
        assert result.month == 9
        assert result.year == 2025

    def test_gps_date_overrides_ffprobe(self):
        """gps_date must take priority over ffprobe creation_time."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator()
        result = orch._extract_recording_date(
            gpx_date="2025-09-13",
            ffprobe_creation_time="2025-09-03T11:12:19Z",
            project_name="demoroute2_3840p",
        )
        assert result is not None
        assert result.day == 13, f"Expected day 13 (gps_date), got {result.day}"

    def test_description_contains_correct_date(self):
        """Description must show the GPS date, not the GPX file start date."""
        gen = LongFormMetadataGenerator.__new__(LongFormMetadataGenerator)
        desc = gen._build_gms_description(
            location="County Kerry, Ireland",
            seo_description="POV drive.",
            duration_str="60m",
            publish_day="Saturday",
            recording_date="September 13, 2025",
        )
        # Long-form routes through the shared "Mon YYYY" formatter, so
        # the GPS-derived date surfaces as "Sep 2025" (not the raw input).
        assert "📅 Filmed: Sep 2025" in desc
        assert "September 03" not in desc


class TestHashtagsAfterChapters:
    """Hashtags must come AFTER chapters in YouTube descriptions."""

    def test_embed_chapters_places_before_hashtags(self):
        description = (
            "POV drive through Kerry.\n\n"
            "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n\n"
            "#POVDriving #ScenicDrive #CountyKerry"
        )
        chapter_text = "0:00 Start\n5:00 Entering Kenmare\n10:00 Finish"
        result = LongFormMetadataGenerator._embed_chapters(description, chapter_text)

        hashtag_pos = result.rfind("#POVDriving")
        chapter_pos = result.find("\U0001f4cd CHAPTERS")
        assert chapter_pos < hashtag_pos, (
            f"Chapters (pos {chapter_pos}) must come BEFORE hashtags (pos {hashtag_pos}).\n"
            f"Got:\n{result}"
        )

    def test_full_description_chapters_before_hashtags(self):
        """Full description template must have: content -> chapters -> hashtags."""
        gen = LongFormMetadataGenerator.__new__(LongFormMetadataGenerator)
        desc = gen._build_gms_description(
            location="County Kerry, Ireland",
            seo_description="POV drive through stunning Kerry roads.",
            duration_str="49m 14s",
            publish_day="Saturday",
        )
        chapter_text = "0:00 Start\n5:00 Entering Kenmare"
        final = LongFormMetadataGenerator._embed_chapters(desc, chapter_text)

        import re

        hashtag_match = re.search(r"#\w+Driving", final)
        chapter_match = re.search(r"\U0001f4cd CHAPTERS", final)
        assert hashtag_match and chapter_match
        assert chapter_match.start() < hashtag_match.start(), (
            "Chapters must appear before hashtags in final description"
        )


class TestChapterPOIFiltering:
    """Chapters must use real place names, not tourism businesses."""

    def test_tourism_pois_excluded(self):
        """Names like 'Sheep Dog Training' must not be chapter names."""
        from gomoveshift.pipeline.chapters import is_valid_chapter_placename

        bad_names = [
            "Sheep Dog Training",
            "Kerry Bog Village",
            "Beehive Cells",
            "Ring of Kerry Golf",
            "Skellig Chocolate",
        ]
        for name in bad_names:
            assert not is_valid_chapter_placename(name), (
                f"'{name}' should be filtered as a tourism/business POI"
            )

    def test_real_towns_pass(self):
        """Real town names must pass the filter."""
        from gomoveshift.pipeline.chapters import is_valid_chapter_placename

        good_names = [
            "Kenmare",
            "Sneem",
            "Caherdaniel",
            "Waterville",
            "Portmagee",
            "Cahersiveen",
            "Glenbeigh",
            "Glencar",
        ]
        for name in good_names:
            assert is_valid_chapter_placename(name), f"'{name}' should pass"


class TestUSBusinessNameFiltering:
    """US business names from Nominatim must be rejected as chapter names."""

    def test_us_business_names_rejected(self):
        """Common US businesses must not become chapter titles."""
        from gomoveshift.pipeline.chapters import is_valid_chapter_placename

        bad_names = [
            "Community Beer Garden",
            "Maynard's Ice Cream & Cafe",
            "CPM Federal Credit Union",
            "JJones Gallery",
            "Parce's Place Bar & Grill",
            "Beaufort Gas Station",
            "First National Bank",
            "Coastal Auto Parts",
            "Lowcountry Dental Clinic",
            "Island Liquor Store",
            "Dollar General",
        ]
        for name in bad_names:
            assert not is_valid_chapter_placename(name), (
                f"'{name}' should be filtered as a business name"
            )

    def test_real_us_towns_pass(self):
        """Real US town/place names must pass the filter."""
        from gomoveshift.pipeline.chapters import is_valid_chapter_placename

        good_names = [
            "Beaufort",
            "Port Royal",
            "Longwood",
            "Fort Fremont",
            "Hilton Head Island",
            "Charleston",
            "Savannah",
            "Bluffton",
            "Gardens Corner",
            "Lobeco",
            "Grays Hill",
            "Frogmore",
        ]
        for name in good_names:
            assert is_valid_chapter_placename(name), f"'{name}' should pass"

    def test_format_location_rejects_business(self):
        """_format_location must return '' for business names."""
        from gomoveshift.pipeline.chapters import ChapterGenerator
        from gps_journey.core.models import Location

        gen = ChapterGenerator.__new__(ChapterGenerator)
        loc = Location(
            latitude=32.43,
            longitude=-80.67,
            city="Community Beer Garden",
            locality=None,
            county="Beaufort County",
            country="United States",
        )
        result = gen._format_location(loc)
        assert result == "", f"Business name should return '', got: '{result}'"

    def test_validator_catches_business_in_title(self):
        """Pipeline validator must flag titles containing business names."""
        validator = PipelineValidator()
        metadata = {
            "title": "Community Beer Garden to Gardens Corner | United States",
            "description": (
                "POV drive through South Carolina.\n\n"
                "\U0001f4cd Location: South Carolina, United States"
            ),
            "location": "South Carolina, United States",
            "chapters": [
                {"timestamp": "0:00", "title": "Start - South Carolina"},
                {"timestamp": "10:00", "title": "Entering Beaufort, Beaufort County"},
                {"timestamp": "20:00", "title": "Finish - South Carolina"},
            ],
            "video_duration": 1200.0,
        }
        result = validator._check_metadata_quality(metadata)
        assert not result.passed, f"Should fail for business name in title: {result.message}"
        assert "business" in result.message.lower()


class TestChapterRegionCrossCheck:
    """The location-vs-chapter cross-check must COLLECT the chapter
    region even when the title carries a route suffix or a multi-word
    county — otherwise the only chapter<->location cross-check silently
    no-ops on exactly those drives."""

    def test_route_suffixed_multiword_county_regions_collected(self):
        from gomoveshift.pipeline.validation.content_gates import check_metadata_quality

        # Every chapter names a county absent from the episode location; once
        # the parser collects them, the majority-mismatch check must fire.
        metadata = {
            "title": "Scenic Drive through Manassas, Virginia 4K POV",
            "description": "A relaxing scenic drive through Manassas, Virginia.",
            "location": "Manassas, Virginia, United States",
            "chapters": [
                {"timestamp": "0:00", "title": "Start - Virginia"},
                {
                    "timestamp": "10:00",
                    "title": "Entering Manassas, Prince William County, Skyline Drive",
                },
                {
                    "timestamp": "20:00",
                    "title": "Entering Fairfax, Prince William County, Skyline Drive",
                },
                {"timestamp": "30:00", "title": "Finish - Virginia"},
            ],
            "video_duration": 300,  # < 600s so the chapter-density gate is inert
            "tags": ["Virginia", "Manassas"],
            "has_gps": True,
        }
        result = check_metadata_quality(metadata)
        assert "chapter regions don't match episode location" in result.message, (
            f"chapter-region check no-oped on a route-suffixed multi-word-county chapter set: {result.message}"
        )


class TestResolvedLocationCorroboration:
    """The title-location and chapter-region checks must consult the LF's own ``resolved_location`` before
    calling a title or a chapter region wrong.

    A settlement-anchored title ("Kingsport, TN") and a county-form location
    ("Sullivan County, Tennessee") share no keyword, yet both are correct —
    Kingsport is in Sullivan County, and the label_quality gate actively asks
    for a settlement anchor. Likewise a multi-county drive names counties in
    its chapters that its single episode label cannot contain. In both cases
    the pipeline already computed the answer: ``ordered_place_sequence`` says
    where the trace went, ``regions`` says which counties it sampled.
    """

    _TAGS = ["POV driving", "drive", "road trip", "4K driving"]

    def _metadata(self, **over):
        base = {
            "title": "Kingsport, TN in 4K",
            "description": "POV drive through northeast Tennessee.",
            "location": "Sullivan County, Tennessee",
            "video_duration": 300,  # < 600s so the chapter-density gate is inert
            "tags": list(self._TAGS),
            "has_gps": True,
            "resolved_location": {
                "dominant_county": "Sullivan County",
                "dominant_state": "Tennessee",
                "regions": [
                    "Unicoi County, Tennessee",
                    "Washington County, Tennessee",
                    "Sullivan County, Tennessee",
                ],
                "ordered_place_sequence": ["Unicoi", "Johnson City", "Kingsport"],
            },
        }
        base.update(over)
        return base

    def test_settlement_title_passes_when_place_is_in_the_trace(self):
        """'Kingsport, TN' + 'Sullivan County, Tennessee' must PASS.

        Kingsport is in the LF's own ordered_place_sequence, so the title is
        corroborated even though it shares no word with the county label.
        """
        from gomoveshift.pipeline.validation.content_gates import check_metadata_quality

        result = check_metadata_quality(self._metadata())
        assert result.passed, f"settlement-anchored title rejected: {result.message}"

    def test_title_naming_a_place_off_the_trace_still_fails(self):
        """A title naming somewhere the drive never went must still FAIL."""
        from gomoveshift.pipeline.validation.content_gates import check_metadata_quality

        result = check_metadata_quality(self._metadata(title="Richmond, VA in 4K"))
        assert "no location keywords" in result.message, (
            f"title-location check lost its teeth on an off-trace title: {result.message}"
        )

    def test_legacy_metadata_without_resolved_location_stays_strict(self):
        """No resolved_location -> the original keyword check, unchanged."""
        from gomoveshift.pipeline.validation.content_gates import check_metadata_quality

        meta = self._metadata()
        del meta["resolved_location"]
        result = check_metadata_quality(meta)
        assert "no location keywords" in result.message, (
            f"legacy strict path weakened: {result.message}"
        )

    def test_chapter_region_the_drive_actually_sampled_is_not_a_mismatch(self):
        """A chapter naming a sampled county must not count as a mismatch.

        A long-form video that crosses from Wythe into Pulaski names Pulaski in
        its chapters; Pulaski is in ``regions``, so the drive genuinely went
        there and the episode label is not wrong.
        """
        from gomoveshift.pipeline.validation.content_gates import check_metadata_quality

        result = check_metadata_quality(
            self._metadata(
                title="Cline, VA in 4K",
                location="Wythe County, Virginia",
                resolved_location={
                    "dominant_county": "Wythe County",
                    "dominant_state": "Virginia",
                    "regions": [
                        "Wythe County, Virginia",
                        "Pulaski County, Virginia",
                        "Montgomery County, Virginia",
                    ],
                    "ordered_place_sequence": ["Wythe County", "Cline", "Draper"],
                },
                chapters=[
                    {"timestamp": "0:00", "title": "Start - Wythe County, Virginia"},
                    {"timestamp": "10:09", "title": "Entering Draper, Pulaski County"},
                    {"timestamp": "14:17", "title": "Finish - Wythe County, Virginia"},
                ],
            )
        )
        assert result.passed, f"sampled-county chapter called a mismatch: {result.message}"

    def test_chapter_region_the_drive_never_sampled_still_fails(self):
        """A chapter naming a county absent from regions must still FAIL."""
        from gomoveshift.pipeline.validation.content_gates import check_metadata_quality

        result = check_metadata_quality(
            self._metadata(
                title="Cline, VA in 4K",
                location="Wythe County, Virginia",
                resolved_location={
                    "dominant_county": "Wythe County",
                    "dominant_state": "Virginia",
                    "regions": ["Wythe County, Virginia"],
                    "ordered_place_sequence": ["Wythe County", "Cline"],
                },
                chapters=[
                    {"timestamp": "0:00", "title": "Start - Wythe County, Virginia"},
                    {"timestamp": "10:09", "title": "Entering Draper, Pulaski County"},
                ],
            )
        )
        assert "chapter regions don't match episode location" in result.message, (
            f"chapter-region check lost its teeth on an unsampled county: {result.message}"
        )


class TestSeriesDetection:
    """Multi-part video series must be detected from filenames."""

    def test_extract_series_number(self):
        from gomoveshift.pipeline.orchestrator import extract_series_info

        assert extract_series_info("demoroute1_3840p") == ("demoroute", 1)
        assert extract_series_info("demoroute2_3840p") == ("demoroute", 2)
        assert extract_series_info("demoroute4_3840p") == ("demoroute", 4)

    def test_non_series_returns_none(self):
        from gomoveshift.pipeline.orchestrator import extract_series_info

        base, num = extract_series_info("20250101example_3840p")
        assert num is None

    def test_series_title_includes_part(self):
        from gomoveshift.pipeline.orchestrator import make_series_title

        title = make_series_title(
            named_route="Ring of Kerry",
            part_number=2,
            total_parts=4,
            resolution_label="4K",
        )
        assert "Part 2" in title
        assert "Ring of Kerry" in title


class TestMetadataQualityScoreEnforcement:
    """The SEO structural score must GATE (fail) below-floor metadata before the
    publish artifact is written, not merely pretty-printed."""

    def test_below_floor_rejected(self):
        from gomoveshift.pipeline.stages.metadata import _quality_score_error

        assert _quality_score_error({"quality_score": 0.3}) is not None

    def test_at_floor_passes(self):
        from gomoveshift.pipeline.stages.metadata import _quality_score_error

        assert _quality_score_error({"quality_score": 0.5}) is None

    def test_above_floor_passes(self):
        from gomoveshift.pipeline.stages.metadata import _quality_score_error

        assert _quality_score_error({"quality_score": 0.8}) is None

    def test_none_or_absent_score_not_gated(self):
        """Degraded-SEO path (quality_score=None) is gated separately."""
        from gomoveshift.pipeline.stages.metadata import _quality_score_error

        assert _quality_score_error({"quality_score": None}) is None
        assert _quality_score_error({}) is None


class TestRouteFormHashtagsAndRegion:
    """Route-form labels must not smash the 'A to B via Road' arc into one
    hashtag or a garbled region noun, and single-component labels must not emit
    a meaningless '#Travel' filler."""

    def test_route_form_hashtags_are_per_component(self):
        gen = LongFormMetadataGenerator.__new__(LongFormMetadataGenerator)
        desc = gen._build_gms_description(
            location="Killarney to Dingle via N86, County Kerry",
            seo_description="Stunning scenic roads await.",  # subjective -> triggers rewrite
            duration_str="49m",
            publish_day="Saturday",
        )
        assert "#KillarneytoDingle" not in desc
        assert "#Killarney" in desc
        assert "#Dingle" in desc
        assert "#CountyKerry" in desc
        assert "#Travel" not in desc

    def test_route_form_region_noun_is_the_region_not_the_arc(self):
        gen = LongFormMetadataGenerator.__new__(LongFormMetadataGenerator)
        desc = gen._build_gms_description(
            location="Killarney to Dingle via N86, County Kerry",
            seo_description="Stunning scenic roads await.",
            duration_str="49m",
            publish_day="Saturday",
        )
        assert "through Killarney to Dingle via N86" not in desc
        assert "through County Kerry" in desc

    def test_single_component_label_no_travel_filler(self):
        gen = LongFormMetadataGenerator.__new__(LongFormMetadataGenerator)
        desc = gen._build_gms_description(
            location="County Tipperary",
            seo_description="Stunning drive.",
            duration_str="30m",
            publish_day="Sunday",
        )
        assert "#Travel" not in desc
        assert "#CountyTipperary" in desc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
