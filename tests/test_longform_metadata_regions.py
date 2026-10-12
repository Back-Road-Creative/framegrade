"""Per-LF regions line in long-form descriptions.

Single ``🗺️ Through:`` label across countries (Ireland and USA both use the same prefix). Behaviour:

- ``regions`` falsy (None / []) → line omitted entirely.
- one or more regions → ``🗺️ Through: <name1>, <name2>, ...``.
- beyond _REGIONS_DESCRIPTION_CAP regions (12; it was lifted from
  4) → first N joined, then ``" … +N more"``.
"""

import pytest

from gomoveshift.pipeline.longform_metadata import (
    _derive_country,
    _format_regions_line,
)


def _r(name: str, country: str = "Ireland", admin_level: str = "county") -> dict:
    return {"name": name, "country": country, "admin_level": admin_level}


# ────────────────────────────────────────────────────────────────────
# _derive_country
# ────────────────────────────────────────────────────────────────────


class TestDeriveCountry:
    def test_ireland_three_parts(self):
        assert _derive_country("Dingle Peninsula, County Kerry, Ireland") == "ireland"

    def test_usa_three_parts(self):
        assert _derive_country("Blue Ridge Parkway, Virginia, USA") == "usa"

    def test_series_suffix_stripped(self):
        assert _derive_country("Blue Ridge Parkway, Virginia, USA — Part 2") == "usa"

    def test_empty_returns_empty_string(self):
        assert _derive_country("") == ""
        assert _derive_country(None) == ""

    def test_single_component_returns_self_lowercased(self):
        # No comma → only one part → it IS the "country" component.
        assert _derive_country("Ireland") == "ireland"


# ────────────────────────────────────────────────────────────────────
# _format_regions_line — basic rendering
# ────────────────────────────────────────────────────────────────────


class TestFormatRegionsLineBasic:
    def test_single_region_ireland(self):
        line = _format_regions_line([_r("County Kerry")], "Dingle, County Kerry, Ireland")
        assert line == "🗺️ Through: County Kerry\n"

    def test_multi_region_ireland(self):
        regions = [_r("County Kerry"), _r("County Cork")]
        line = _format_regions_line(regions, "Cork-Kerry Route, Ireland")
        assert line == "🗺️ Through: County Kerry, County Cork\n"

    def test_usa_state(self):
        regions = [_r("Virginia", country="USA", admin_level="state")]
        line = _format_regions_line(regions, "Blue Ridge Parkway, Virginia, USA")
        assert line == "🗺️ Through: Virginia\n"

    def test_usa_multi_state(self):
        regions = [
            _r("Virginia", country="USA", admin_level="state"),
            _r("North Carolina", country="USA", admin_level="state"),
        ]
        line = _format_regions_line(regions, "Blue Ridge Parkway, USA")
        assert line == "🗺️ Through: Virginia, North Carolina\n"


# ────────────────────────────────────────────────────────────────────
# _format_regions_line — empty / None / suppression
# ────────────────────────────────────────────────────────────────────


class TestFormatRegionsLineSuppression:
    def test_none_returns_empty_string(self):
        assert _format_regions_line(None, "Anywhere, Ireland") == ""

    def test_empty_list_returns_empty_string(self):
        assert _format_regions_line([], "Anywhere, Ireland") == ""

    def test_entries_without_name_are_filtered(self):
        """A region dict missing ``name`` should not produce a stray comma."""
        regions = [{"name": None}, _r("County Kerry"), {"name": ""}]
        line = _format_regions_line(regions, "Anywhere, Ireland")
        assert line == "🗺️ Through: County Kerry\n"

    def test_all_entries_nameless_returns_empty_string(self):
        regions = [{"name": None}, {"name": ""}]
        assert _format_regions_line(regions, "Anywhere, Ireland") == ""


# ────────────────────────────────────────────────────────────────────
# _format_regions_line — cap at 12 with "+N more" beyond
# ────────────────────────────────────────────────────────────────────


class TestFormatRegionsLineCap:
    def test_exactly_four_renders_all(self):
        regions = [_r(f"County {i}") for i in ["Kerry", "Cork", "Limerick", "Clare"]]
        line = _format_regions_line(regions, "Ireland")
        assert line == "🗺️ Through: County Kerry, County Cork, County Limerick, County Clare\n"
        assert "more" not in line

    def test_five_renders_all_after_cap_lift(self):
        # Cap lifted 4 → 12, so 5 regions now list in full (the old
        # 4-cap dropped the long-tail names this line's SEO value lives in).
        regions = [
            _r("County Kerry"),
            _r("County Cork"),
            _r("County Limerick"),
            _r("County Clare"),
            _r("County Tipperary"),
        ]
        line = _format_regions_line(regions, "Ireland")
        assert (
            line == "🗺️ Through: County Kerry, County Cork, County Limerick, "
            "County Clare, County Tipperary\n"
        )
        assert "more" not in line

    def test_seven_renders_all_after_cap_lift(self):
        regions = [_r(f"County {n}") for n in "ABCDEFG"]
        line = _format_regions_line(regions, "Ireland")
        assert "more" not in line
        assert "County A" in line and "County G" in line

    def test_beyond_twelve_still_elides(self):
        # The ceiling stays only as a wall-of-text backstop for a pathological
        # trace: 15 regions → first 12 + "… +3 more".
        regions = [_r(f"County {i}") for i in range(15)]
        line = _format_regions_line(regions, "Ireland")
        assert "… +3 more" in line
        assert "County 11" in line and "County 12" not in line


# ────────────────────────────────────────────────────────────────────
# End-to-end via _build_gms_description: ordering + threading
# ────────────────────────────────────────────────────────────────────


class TestBuildGmsDescriptionThreadsRegions:
    @pytest.fixture
    def generator(self):
        from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator

        return LongFormMetadataGenerator.__new__(LongFormMetadataGenerator)

    def test_regions_line_appears_ireland(self, generator):
        description = generator._build_gms_description(
            location="Dingle Peninsula, County Kerry, Ireland",
            seo_description="POV drive through Dingle.",
            duration_str="32 min",
            publish_day="Wednesday",
            regions=[_r("County Kerry"), _r("County Cork")],
        )
        assert "🗺️ Through: County Kerry, County Cork" in description

    def test_regions_line_appears_usa(self, generator):
        description = generator._build_gms_description(
            location="Blue Ridge Parkway, Virginia, USA",
            seo_description="POV drive through Blue Ridge.",
            duration_str="32 min",
            publish_day="Wednesday",
            regions=[
                _r("Virginia", country="USA", admin_level="state"),
                _r("North Carolina", country="USA", admin_level="state"),
            ],
        )
        assert "🗺️ Through: Virginia, North Carolina" in description

    def test_regions_line_omitted_when_none(self, generator):
        description = generator._build_gms_description(
            location="Dingle Peninsula, County Kerry, Ireland",
            seo_description="POV drive through Dingle.",
            duration_str="32 min",
            publish_day="Wednesday",
            regions=None,
        )
        assert "🗺️" not in description
        assert "Through:" not in description

    def test_regions_line_omitted_when_empty(self, generator):
        description = generator._build_gms_description(
            location="Dingle Peninsula, County Kerry, Ireland",
            seo_description="POV drive through Dingle.",
            duration_str="32 min",
            publish_day="Wednesday",
            regions=[],
        )
        assert "🗺️" not in description

    def test_ordering_distance_regions_filmed(self, generator):
        description = generator._build_gms_description(
            location="Dingle Peninsula, County Kerry, Ireland",
            seo_description="POV drive through Dingle.",
            duration_str="32 min",
            publish_day="Wednesday",
            recording_date="2026-02-21",
            distance_km=47.0,
            elevation_gain_m=340.0,
            regions=[_r("County Kerry"), _r("County Cork")],
            source_duration=3600.0,
            video_duration=1920.0,
        )
        # Order: 🎬 Final cut → 🛣️ Distance → 🗺️ Through → 📅 Filmed.
        # Elevation lines dropped — never rendered even when
        # elevation_gain_m is supplied.
        final_cut_pos = description.index("🎬 Final cut:")
        distance_pos = description.index("🛣️ Distance:")
        regions_pos = description.index("🗺️ Through:")
        filmed_pos = description.index("📅 Filmed:")
        assert final_cut_pos < distance_pos < regions_pos < filmed_pos
        assert "⛰️ Elevation gain:" not in description
        assert "📈 Elevation:" not in description


# ────────────────────────────────────────────────────────────────────
# _format_regions_line — never restate the LF's own label
# ────────────────────────────────────────────────────────────────────
#
# "Through" means the regions the route PASSED THROUGH; the LF's own
# region carries no information there. It is also a third byte-identical
# occurrence of the label already in the hook and the Location line,
# which breaches the description-redundancy 2x location budget — the
# description-redundancy failure on a multi-county LF
# ("Sullivan County, Tennessee" 3x).


class TestFormatRegionsLineSelfExclusion:
    def test_line_omitted_when_the_only_region_is_the_lf_label(self):
        line = _format_regions_line(
            [_r("Sullivan County, Tennessee", country="USA")],
            "Sullivan County, Tennessee",
        )
        assert line == ""

    def test_line_lists_only_the_other_regions(self):
        regions = [
            _r("Sullivan County, Tennessee", country="USA"),
            _r("Washington County, Virginia", country="USA"),
        ]
        line = _format_regions_line(regions, "Sullivan County, Tennessee")
        assert line == "🗺️ Through: Washington County, Virginia\n"

    def test_match_is_exact_never_substring(self):
        # Load-bearing: a town-form LF label legitimately CONTAINS its
        # county, and that line must still render. A substring filter
        # would delete it.
        line = _format_regions_line([_r("County Kerry")], "Dingle, County Kerry, Ireland")
        assert line == "🗺️ Through: County Kerry\n"
