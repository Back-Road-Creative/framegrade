"""Tests for social_generator hashtag/region/series-suffix fixes.

Construction-level fixes:
- Location hashtags split by comma (no smashed concatenation, no Part N).
- Region tags derived from country (no hardcoded "road trip USA").
- Series suffix " — Part N" stripped at the title→tag boundary.
"""

from __future__ import annotations

import pytest

from gomoveshift.models import ResolvedLocation
from gomoveshift.pipeline.social_generator import (
    SocialGenerator,
    _location_hashtags,
    _region_tags,
    _strip_series_suffix,
)


# ─── location hashtags ────────────────────────────────────────────


class TestLocationHashtags:
    def test_dingle_three_components_no_smash_no_part_suffix(self):
        result = _location_hashtags("Dingle Peninsula, County Kerry, Ireland — Part 1")
        assert result == ["#DinglePeninsula", "#CountyKerry", "#Ireland"]
        # Critically, the smashed form is absent
        assert "#DinglePeninsulaCountyKerryIrelandPart1" not in result
        for tag in result:
            assert "Part" not in tag

    def test_no_comma_single_component(self):
        result = _location_hashtags("Dingle")
        assert result == ["#Dingle"]

    def test_strips_punctuation_inside_components(self):
        # apostrophes / hyphens get stripped per the YouTube/IG rule
        result = _location_hashtags("St. John's, Newfoundland")
        assert result == ["#StJohns", "#Newfoundland"]

    def test_handles_empty_components(self):
        # Trailing comma should not produce an empty hashtag like "#"
        result = _location_hashtags("Dingle, , Ireland")
        assert "#" not in result
        assert all(len(t) > 1 for t in result)

    def test_strips_part_suffix_with_em_dash(self):
        result = _location_hashtags("Asheville, North Carolina, USA — Part 3")
        assert result == ["#Asheville", "#NorthCarolina", "#USA"]


# ─── region tags ──────────────────────────────────────────────────


class TestRegionTags:
    def test_ireland_never_emits_road_trip_usa(self):
        result = _region_tags("Dingle Peninsula, County Kerry, Ireland — Part 1")
        assert "Ireland road trip" in result
        assert "Ireland drive" in result
        assert "Ireland POV drive" in result  # factual replacement for the dropped "scenic" variant
        assert "Ireland scenic drive" not in result
        assert "scenic ireland" not in result
        assert "road trip USA" not in result

    def test_usa_correctly_emits_usa_road_trip(self):
        result = _region_tags("Asheville, North Carolina, USA")
        assert "USA road trip" in result
        assert "USA drive" in result
        assert "USA POV drive" in result  # factual replacement for the dropped "scenic" variant
        assert "USA scenic drive" not in result
        assert "scenic usa" not in result

    def test_strips_part_suffix_before_country_extraction(self):
        # If we forgot to strip series suffix, "Ireland — Part 1" would be the country
        result = _region_tags("Foo, Bar, Ireland — Part 5")
        assert "Ireland road trip" in result
        for tag in result:
            assert "Part" not in tag

    def test_county_state_label_no_country_emits_usa_tags(self):
        # Dominant-county labels are "County, State" with NO country component.
        # A trailing US state must map to USA so the country slot never
        # holds a state name.
        result = _region_tags("Beaufort County, South Carolina")
        assert "USA road trip" in result
        assert "USA drive" in result
        assert "USA scenic drive" not in result
        assert "scenic usa" not in result
        assert "South Carolina road trip" not in result

    def test_state_abbr_label_no_country_emits_usa_tags(self):
        result = _region_tags("Port Royal to Lady's Island, SC")
        assert "USA road trip" in result
        assert "SC road trip" not in result

    def test_non_us_two_letter_last_component_unchanged(self):
        result = _region_tags("Some Town, IE")
        assert "IE road trip" in result
        assert "USA road trip" not in result

    def test_verbose_arc_label_without_country_emits_no_region_variants(self):
        # GPS-less "X to Y via Z" label has NO comma → parts[-1] is the whole
        # 46-char arc, so "{arc} road trip"/"{arc} drive"
        # blew the YouTube 500-char tag budget (Metricool import rejected it).
        # A real region noun is short; skip the variants when it isn't one.
        result = _region_tags("Kilgarvan to Ballylickey via the Famine Tunnel")
        assert result == []


# ─── region tags consume the ResolvedLocation model ─────────────


class TestRegionTagsConsumeResolved:
    def test_resolved_us_emits_usa_regardless_of_label_string(self):
        # The display label parses to a bare county with no country; the
        # resolver authoritatively says US. Tags follow the model.
        resolved = ResolvedLocation(
            dominant_county="Beaufort County",
            dominant_state="South Carolina",
            country_code="US",
            is_us=True,
        )
        result = _region_tags("Beaufort County, South Carolina", resolved=resolved)
        assert "USA road trip" in result
        assert "USA drive" in result
        assert "USA scenic drive" not in result
        assert "scenic usa" not in result
        assert "South Carolina road trip" not in result

    def test_resolved_us_overrides_misparseable_label(self):
        # A label whose last component is NOT a recognised state would, under
        # the pure string-parse path, mint a wrong-region tag. The resolver's
        # is_us flag (the same signal units/playlist read) keeps it on USA.
        resolved = ResolvedLocation(is_us=True, country_code="US")
        result = _region_tags("Lake Lure to Bat Cave, Henderson", resolved=resolved)
        assert "USA road trip" in result
        assert "Henderson road trip" not in result

    def test_resolved_non_us_uses_geocoded_country(self):
        # A non-US drive carries its GEOCODED country in the model, so the
        # region tags name the COUNTRY ("Ireland") off the same signal that
        # routes the playlist — not the label's last comma-component, which on a
        # real production label is the county/province ("Kerry"), never Ireland.
        resolved = ResolvedLocation(
            dominant_county="County Kerry",
            dominant_state="Munster",
            country_code="Ireland",
            is_us=False,
        )
        result = _region_tags("Killarney to Dingle via N86, Kerry", resolved=resolved)
        assert "Ireland road trip" in result
        assert "Kerry road trip" not in result
        assert "USA road trip" not in result

    def test_empty_resolved_falls_back_to_string_parse(self):
        # A resolver with no usable data (no geocoder / no points) must not
        # suppress the back-compat string-parse path.
        result = _region_tags("Asheville, North Carolina, USA", resolved=ResolvedLocation())
        assert "USA road trip" in result

    def test_no_resolved_arg_is_back_compat(self):
        # Callers without a resolved location keep the original behaviour.
        result = _region_tags("Asheville, North Carolina, USA")
        assert "USA road trip" in result


# ─── strip series suffix ──────────────────────────────────────────


class TestStripSeriesSuffix:
    def test_strips_em_dash_part_n(self):
        assert (
            _strip_series_suffix("Dingle Peninsula, County Kerry, Ireland — Part 1")
            == "Dingle Peninsula, County Kerry, Ireland"
        )

    def test_no_suffix_passthrough(self):
        assert (
            _strip_series_suffix("Asheville, North Carolina, USA")
            == "Asheville, North Carolina, USA"
        )

    def test_double_digit_part(self):
        assert _strip_series_suffix("Foo, Bar — Part 12") == "Foo, Bar"

    def test_only_strips_trailing_part(self):
        # If "Part" appears mid-string, it should NOT be stripped.
        assert _strip_series_suffix("Some Part of Town, Ireland") == "Some Part of Town, Ireland"


# ─── Integrated behaviour through SocialGenerator ───────────────────────


@pytest.fixture
def dingle_generator():
    return SocialGenerator(
        location="Dingle Peninsula, County Kerry, Ireland — Part 1",
        project_name="example-2025-09-01",
        series_part=1,
        recording_date="2025-09-01",
    )


class TestComposedOutputs:
    def test_caption_hashtags_split_per_component(self, dingle_generator):
        post = dingle_generator.generate_post(short_index=1)
        caption = post.caption
        # Smashed form must be absent
        assert "#DinglePeninsulaCountyKerryIrelandPart1" not in caption
        # Per-component hashtags should appear (at least Dingle Peninsula
        # which is the primary location component for short captions)
        assert "#DinglePeninsula" in caption

    def test_first_comment_uses_split_hashtags(self, dingle_generator):
        post = dingle_generator.generate_post(short_index=1)
        comment = post.first_comment
        assert "#DinglePeninsulaCountyKerryIrelandPart1" not in comment

    def test_tags_contain_ireland_not_part_suffix(self, dingle_generator):
        post = dingle_generator.generate_post(short_index=1)
        tag_list = [t.strip() for t in post.youtube_tags.split(",")]
        assert "Ireland" in tag_list
        assert "Ireland — Part 1" not in tag_list
        for t in tag_list:
            assert "— Part" not in t
            assert "Part 1" not in t

    def test_tags_contain_country_road_trip_not_hardcoded_usa(self, dingle_generator):
        post = dingle_generator.generate_post(short_index=1)
        tag_list = [t.strip() for t in post.youtube_tags.split(",")]
        assert "Ireland road trip" in tag_list
        assert "road trip USA" not in tag_list

    def test_usa_drive_still_emits_usa_road_trip(self):
        gen = SocialGenerator(
            location="Asheville, North Carolina, USA",
            project_name="asheville-test",
            series_part=1,
            recording_date="2025-09-01",
        )
        post = gen.generate_post(short_index=1)
        tag_list = [t.strip() for t in post.youtube_tags.split(",")]
        assert "USA road trip" in tag_list


# ─── state abbreviation tag uses word boundaries + longest-first ────


class TestStateAbbrevTag:
    """State abbrev tag must match whole state names, longest-first, so a
    state name that is a substring of another (Virginia ⊂ West Virginia,
    Carolina/Dakota families) never mis-tags."""

    @staticmethod
    def _abbrevs(location: str) -> list[str]:
        gen = SocialGenerator(location=location, project_name="region-test")
        tags = [t.strip() for t in gen._build_tags(None).split(",")]
        return [t for t in tags if t in {"VA", "WV", "NC", "SC", "ND", "SD"}]

    def test_west_virginia_tags_wv_not_va(self):
        abbrevs = self._abbrevs("Charleston, West Virginia")
        assert "WV" in abbrevs
        assert "VA" not in abbrevs

    def test_virginia_still_tags_va(self):
        abbrevs = self._abbrevs("Richmond, Virginia")
        assert "VA" in abbrevs
        assert "WV" not in abbrevs

    def test_north_carolina_tags_nc_not_sc(self):
        abbrevs = self._abbrevs("Asheville, North Carolina")
        assert abbrevs == ["NC"]

    def test_north_dakota_tags_nd_only(self):
        abbrevs = self._abbrevs("Bismarck, North Dakota")
        assert abbrevs == ["ND"]

    def test_substring_not_a_whole_word_is_not_tagged(self):
        # "Virginiatown" contains "Virginia" as a substring but not as a token.
        abbrevs = self._abbrevs("Virginiatown, Ontario")
        assert abbrevs == []


# ─── GPS-less drive-arc never leaks into casual social surfaces ────


class TestArcSafeLocationShort:
    """A GPS-less LF titled with an "A to B via {road}" drive arc must not leak
    the WHOLE arc into ``location_short`` / captions / titles / hashtags. Before
    the fix, ``parts[0]`` kept the entire arc (the state is ``parts[1]``), so
    every ``{location}`` template read "Rolling Through Fancy Gap to Hillsville
    via I-77 🚗" and the smashed hashtag was ``#FancyGaptoHillsvilleviaI77``.
    The casual surfaces anchor on a single caption-safe place — the destination.
    """

    _ARC = "Fancy Gap to Hillsville via I-77, VA"

    def test_location_short_is_arc_free_anchor(self):
        gen = SocialGenerator(location=self._ARC, project_name="arc_test")
        assert gen.location_short == "Hillsville"
        assert " to " not in gen.location_short
        assert " via " not in gen.location_short

    def test_arc_never_reaches_title_or_caption_surfaces(self):
        gen = SocialGenerator(location=self._ARC, project_name="arc_test")
        post = gen.generate_post(short_index=1)
        for surface in (post.youtube_title, post.tiktok_title, post.caption):
            assert " to " not in surface, surface
            assert " via " not in surface, surface
            assert "I-77" not in surface, surface

    def test_smashed_arc_hashtag_absent(self):
        result = _location_hashtags(self._ARC)
        assert "#Hillsville" in result
        assert "#FancyGaptoHillsvilleviaI77" not in result

    def test_description_and_tags_keep_full_route(self):
        # The rule is scoped to the BURNED-caption surfaces (caption/title/hooks +
        # smashed hashtag). The long-form youtube_description and the tag list
        # still name the whole origin→destination route — the origin place is
        # not dropped there.
        gen = SocialGenerator(location=self._ARC, project_name="arc_test")
        post = gen.generate_post(short_index=1)
        assert "Fancy Gap" in post.youtube_description
        assert "Fancy Gap" in post.youtube_tags

    def test_plain_single_place_unchanged(self):
        # A normal label carries no arc — location_short / hashtags untouched.
        gen = SocialGenerator(location="County Kerry, Ireland", project_name="arc_test")
        assert gen.location_short == "County Kerry"
        assert _location_hashtags("County Kerry, Ireland") == ["#CountyKerry", "#Ireland"]
