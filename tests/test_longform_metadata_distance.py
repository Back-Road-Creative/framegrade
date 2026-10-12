"""Per-LF distance line in long-form descriptions.

Country-aware unit formatting:
- US drives → ``🛣️ Distance: 29 mi`` (mi only)
- non-US drives → ``🛣️ Distance: 47 km (29 mi)``
- unknown country → defaults to ``km (mi)``
- None distance → line omitted entirely
"""

import pytest

from gomoveshift.pipeline.longform_metadata import _format_distance_line


class TestFormatDistanceLineCountryDetection:
    def test_usa_drive_uses_mi_only(self):
        line = _format_distance_line(47.0, "Blue Ridge Parkway, Virginia, USA")
        assert line == "🛣️ Distance: 29 mi\n"

    def test_ireland_drive_uses_km_with_mi_parenthetical(self):
        line = _format_distance_line(47.0, "Dingle Peninsula, County Kerry, Ireland")
        assert line == "🛣️ Distance: 47 km (29 mi)\n"

    def test_unknown_country_defaults_to_km_with_mi(self):
        # Single-component location — no country derivable. Default to international form.
        line = _format_distance_line(47.0, "Some Mountain Pass")
        assert line == "🛣️ Distance: 47 km (29 mi)\n"

    def test_series_suffix_stripped_before_country_detection(self):
        """Series suffix must not leak into country detection."""
        line = _format_distance_line(47.0, "Blue Ridge Parkway, Virginia, USA — Part 2")
        assert line == "🛣️ Distance: 29 mi\n"

    def test_us_lowercase_alias_detected(self):
        line = _format_distance_line(47.0, "Some Road, California, us")
        assert "mi\n" in line
        assert "km" not in line

    def test_united_states_alias_detected(self):
        line = _format_distance_line(47.0, "Some Road, Pennsylvania, United States")
        assert "mi\n" in line
        assert "km" not in line

    def test_county_state_label_no_country_detected_as_us(self):
        """Dominant-county labels are ``"County, State"`` with NO country
        component (gps_enrichment.derive_per_lf_locations).
        A trailing US state name must still resolve to US units."""
        line = _format_distance_line(47.0, "Beaufort County, South Carolina")
        assert line == "🛣️ Distance: 29 mi\n"

    def test_state_abbr_label_no_country_detected_as_us(self):
        """The OST/title cascade emits ``"... , SC"`` (2-letter abbr).
        A trailing US state abbreviation must resolve to US units too."""
        line = _format_distance_line(47.0, "Port Royal to Lady's Island, SC")
        assert line == "🛣️ Distance: 29 mi\n"

    def test_non_us_state_like_last_component_stays_metric(self):
        """A 2-letter token that is not a US state abbr (e.g. a country
        code) must NOT be mistaken for a US state."""
        line = _format_distance_line(47.0, "Some Town, IE")
        assert line == "🛣️ Distance: 47 km (29 mi)\n"


class TestFormatDistanceLineOmission:
    def test_none_distance_returns_empty_string(self):
        assert _format_distance_line(None, "Anywhere, Ireland") == ""

    def test_zero_distance_returns_empty_string(self):
        """A zero-distance LF is a GPS-less or single-point recording — surfacing
        "0 km" would be more misleading than omitting the line."""
        assert _format_distance_line(0.0, "Anywhere, Ireland") == ""

    def test_negative_distance_returns_empty_string(self):
        """Defensive — should never happen but tolerate it."""
        assert _format_distance_line(-1.0, "Anywhere, Ireland") == ""


class TestFormatDistanceLineRounding:
    def test_miles_rounded_to_integer(self):
        # 47 km = 29.2045 mi → 29 mi
        line = _format_distance_line(47.0, "Dingle, Ireland")
        assert "(29 mi)" in line

    def test_km_rounded_to_integer(self):
        # 47.6 km → 48 km
        line = _format_distance_line(47.6, "Dingle, Ireland")
        assert "48 km" in line

    def test_thousands_comma_for_large_distances(self):
        """Rare but possible — long multi-day drive segment."""
        line = _format_distance_line(1500.0, "Cross-Country Loop, Ireland")
        assert "1,500 km" in line
        assert "(932 mi)" in line


class TestBuildGmsDescriptionThreadsDistance:
    """End-to-end: ``_build_gms_description`` inserts the line between 🎬 Final cut and 📅 Filmed."""

    @pytest.fixture
    def generator(self):
        from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator

        return LongFormMetadataGenerator.__new__(LongFormMetadataGenerator)

    def test_distance_line_appears_in_description_ireland(self, generator):
        description = generator._build_gms_description(
            location="Dingle Peninsula, County Kerry, Ireland",
            seo_description="POV drive through Dingle.",
            duration_str="32 min",
            publish_day="Wednesday",
            distance_km=47.0,
        )
        assert "🛣️ Distance: 47 km (29 mi)" in description

    def test_distance_line_appears_in_description_usa(self, generator):
        description = generator._build_gms_description(
            location="Blue Ridge Parkway, Virginia, USA",
            seo_description="POV drive through Blue Ridge Parkway.",
            duration_str="32 min",
            publish_day="Wednesday",
            distance_km=47.0,
        )
        assert "🛣️ Distance: 29 mi" in description
        # No km parenthetical on US drives
        assert "km" not in description.split("🛣️ Distance: 29 mi")[1].split("\n")[0]

    def test_distance_line_omitted_when_none(self, generator):
        description = generator._build_gms_description(
            location="Dingle Peninsula, County Kerry, Ireland",
            seo_description="POV drive through Dingle.",
            duration_str="32 min",
            publish_day="Wednesday",
            distance_km=None,
        )
        assert "🛣️" not in description
        assert "Distance:" not in description

    def test_distance_line_ordering_between_final_cut_and_filmed(self, generator):
        description = generator._build_gms_description(
            location="Dingle Peninsula, County Kerry, Ireland",
            seo_description="POV drive through Dingle.",
            duration_str="32 min",
            publish_day="Wednesday",
            recording_date="2026-02-21",
            distance_km=47.0,
            source_duration=3600.0,
            video_duration=1920.0,
        )
        # Order: 🎬 Final cut → 🛣️ Distance → 📅 Filmed
        final_cut_pos = description.index("🎬 Final cut:")
        distance_pos = description.index("🛣️ Distance:")
        filmed_pos = description.index("📅 Filmed:")
        assert final_cut_pos < distance_pos < filmed_pos
