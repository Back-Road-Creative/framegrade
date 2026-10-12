"""Description hook = GPS nouns in the first two lines.

YouTube renders only ~the first two lines of a description above the "…more"
fold — the SEO-critical zone. Line 1 stays the trim-banded hook (POV drive /
Highlights from); line 2 is a searchable SEO line that front-loads the
evergreen "4K drive" term + the region noun + the GPS-verified town arc
+ the via-road. The existing subjective/generic hook quality gate is preserved.
"""

from __future__ import annotations

from gomoveshift.models import ResolvedLocation
from gomoveshift.pipeline.longform_metadata import (
    LongFormMetadataGenerator,
    _seo_arc,
    _seo_search_line,
)


class TestSeoSearchLineShape:
    def test_non_us_arc_region_and_road(self) -> None:
        out = _seo_search_line("Killarney to Dingle via N86, Kerry")
        assert out == "Kerry 4K drive — Killarney to Dingle via N86.\n"

    def test_us_full_state_name_front_loaded(self) -> None:
        out = _seo_search_line("Asheville to Boone via US-221, NC")
        # US drives front-load the FULL state name for SEO value, not the abbr.
        assert out == "North Carolina 4K drive — Asheville to Boone via US-221.\n"

    def test_bare_region_label(self) -> None:
        out = _seo_search_line("Colleton County, SC")
        # Single region, no arc → still surfaces the evergreen term + region.
        assert out == "4K drive through South Carolina.\n"

    def test_bare_arc_no_region(self) -> None:
        out = _seo_search_line("Killarney to Dingle")
        assert out == "Killarney to Dingle — 4K drive.\n"

    def test_leading_verb_and_series_suffix_stripped(self) -> None:
        out = _seo_search_line("Driving Killarney to Dingle via N86, Kerry — Part 2")
        assert out == "Kerry 4K drive — Killarney to Dingle via N86.\n"

    def test_empty_location_omits_line(self) -> None:
        assert _seo_search_line("") == ""
        assert _seo_search_line(None) == ""  # type: ignore[arg-type]

    def test_named_routes_fallback_when_label_has_no_via(self) -> None:
        out = _seo_search_line("Killarney to Dingle, Kerry", named_routes=["N86"])
        assert out == "Kerry 4K drive — Killarney to Dingle via N86.\n"

    def test_road_dropped_when_absent(self) -> None:
        out = _seo_search_line("Killarney to Dingle, Kerry")
        assert out == "Kerry 4K drive — Killarney to Dingle.\n"


class TestSeoArc:
    def test_label_arc_preferred(self) -> None:
        assert _seo_arc("Killarney to Dingle") == "Killarney to Dingle"

    def test_same_endpoint_guarded(self) -> None:
        assert _seo_arc("Dingle to Dingle") == ""

    def test_resolver_sequence_endpoints_fallback(self) -> None:
        resolved = ResolvedLocation(ordered_place_sequence=["Killarney", "Kenmare", "Sneem"])
        # Bare region label carries no arc → endpoints of the vote-derived seq.
        assert _seo_arc("Kerry", resolved) == "Killarney to Sneem"

    def test_no_arc_no_sequence(self) -> None:
        assert _seo_arc("Kerry") == ""
        assert _seo_arc("Kerry", ResolvedLocation()) == ""


class TestResolverConsumed:
    def test_resolver_drives_us_full_state(self) -> None:
        resolved = ResolvedLocation(dominant_state="North Carolina", is_us=True, country_code="US")
        out = _seo_search_line("Asheville to Boone via US-221, NC", resolved)
        assert out.startswith("North Carolina 4K drive — ")

    def test_degraded_resolver_does_not_flip_non_us(self) -> None:
        out = _seo_search_line("Killarney to Dingle via N86, Kerry", ResolvedLocation())
        assert out == "Kerry 4K drive — Killarney to Dingle via N86.\n"

    def test_bare_region_uses_resolver_sequence_for_arc(self) -> None:
        resolved = ResolvedLocation(ordered_place_sequence=["Killarney", "Kenmare", "Sneem"])
        out = _seo_search_line("Kerry", resolved)
        assert out == "Kerry 4K drive — Killarney to Sneem.\n"


class TestDescriptionIntegration:
    def _gen(self) -> LongFormMetadataGenerator:
        return LongFormMetadataGenerator()

    def test_seo_line_is_line_two_above_the_fold(self) -> None:
        gen = self._gen()
        desc = gen._build_gms_description(
            location="Killarney to Dingle via N86, Kerry",
            seo_description="POV drive through Killarney to Dingle via N86, Kerry",
            duration_str="15 min",
            publish_day="Sunday",
        )
        lines = desc.split("\n")
        # Line 1 = the hook; line 2 = the SEO search line.
        assert "Killarney" in lines[0]
        assert lines[1] == "Kerry 4K drive — Killarney to Dingle via N86."

    def test_evergreen_term_above_the_fold(self) -> None:
        gen = self._gen()
        desc = gen._build_gms_description(
            location="Killarney to Dingle via N86, Kerry",
            seo_description="POV drive through Killarney to Dingle via N86, Kerry",
            duration_str="15 min",
            publish_day="Sunday",
        )
        head = "\n".join(desc.split("\n")[:2]).lower()
        assert "4k drive" in head
        assert "kerry" in head

    def test_trim_band_hook_preserved_on_line_one(self) -> None:
        gen = self._gen()
        desc = gen._build_gms_description(
            location="Killarney to Dingle via N86, Kerry",
            seo_description="Join me on a relaxing drive!",
            duration_str="15 min",
            publish_day="Sunday",
            trim_ratio=0.3,
            source_duration=3600.0,
            video_duration=900.0,
        )
        # The banded "Highlights from" framing survives on line 1 (the SEO line must
        # not clobber the trim-honesty hook).
        assert desc.split("\n")[0].startswith("Highlights from")
        assert desc.split("\n")[1] == "Kerry 4K drive — Killarney to Dingle via N86."

    def test_empty_location_data_preserves_back_compat(self) -> None:
        gen = self._gen()
        desc = gen._build_gms_description(
            location="",
            seo_description="POV drive through the hills",
            duration_str="15 min",
            publish_day="Sunday",
        )
        # No usable location → line 2 omitted; hook then a blank line.
        lines = desc.split("\n")
        assert lines[1] == ""
