"""Tests for route_context — chapter parsing and title generation.

Covers both Irish ("County X") and US ("X County") county formats.
"""

from gomoveshift.pipeline.route_context import extract_route_context


class TestExtractRouteContextUSCounty:
    """US county format: 'Entering Beaufort, Beaufort County'."""

    def test_us_county_format_extracts_towns(self):
        chapters = [
            {"timestamp": "0:00", "title": "Start - South Carolina, United States"},
            {"timestamp": "17:00", "title": "Entering Longwood, Beaufort County"},
            {"timestamp": "23:00", "title": "Entering Fort Fremont, Beaufort County"},
            {"timestamp": "26:44", "title": "Finish - South Carolina, United States"},
        ]
        ctx = extract_route_context("South Carolina, United States", chapters)
        assert ctx.gps_towns == ["Longwood", "Fort Fremont"]
        assert ctx.county == "Beaufort County"

    def test_us_county_format_smart_start_end(self):
        chapters = [
            {"timestamp": "0:00", "title": "Start - South Carolina"},
            {"timestamp": "3:30", "title": "Entering Beaufort, Beaufort County"},
            {"timestamp": "23:25", "title": "Entering Port Royal, Beaufort County"},
            {"timestamp": "48:13", "title": "Finish - South Carolina"},
        ]
        ctx = extract_route_context("South Carolina, United States", chapters)
        assert ctx.smart_start == "Beaufort"
        assert ctx.smart_end == "Port Royal"

    def test_us_county_default_title_uses_towns(self):
        chapters = [
            {"timestamp": "0:00", "title": "Start - South Carolina"},
            {"timestamp": "3:30", "title": "Entering Beaufort, Beaufort County"},
            {"timestamp": "33:52", "title": "Entering Port Royal, Beaufort County"},
            {"timestamp": "48:13", "title": "Finish - South Carolina"},
        ]
        ctx = extract_route_context("South Carolina, United States", chapters)
        title = ctx.make_default_title()
        assert "Beaufort" in title
        assert "Port Royal" in title


class TestExtractRouteContextIrishCounty:
    """Irish county format: 'Entering Kilrush, County Clare' — must still work."""

    def test_irish_county_format_extracts_towns(self):
        chapters = [
            {"timestamp": "0:00", "title": "Start - County Clare, Ireland"},
            {"timestamp": "5:00", "title": "Entering Kilrush, County Clare"},
            {"timestamp": "15:00", "title": "Entering Ennistymon, County Clare"},
            {"timestamp": "30:00", "title": "Finish - County Clare, Ireland"},
        ]
        ctx = extract_route_context("County Clare, Ireland", chapters)
        assert ctx.gps_towns == ["Kilrush", "Ennistymon"]
        assert ctx.county == "County Clare"

    def test_irish_county_default_title(self):
        chapters = [
            {"timestamp": "0:00", "title": "Start - County Clare, Ireland"},
            {"timestamp": "5:00", "title": "Entering Kilrush, County Clare"},
            {"timestamp": "30:00", "title": "Entering Lahinch, County Clare"},
            {"timestamp": "45:00", "title": "Finish - County Clare, Ireland"},
        ]
        ctx = extract_route_context("County Clare, Ireland", chapters)
        title = ctx.make_default_title()
        assert "Kilrush" in title
        assert "Lahinch" in title


class TestExtractRouteContextMultiWordAndSuffix:
    """'Entering …, County' parsers tolerate multi-word counties and a
    trailing named-route suffix."""

    def test_multiword_us_county_extracts_towns(self):
        chapters = [
            {"timestamp": "0:00", "title": "Start - Virginia"},
            {"timestamp": "10:00", "title": "Entering Manassas, Prince William County"},
            {"timestamp": "20:00", "title": "Entering Fairfax, Prince William County"},
            {"timestamp": "30:00", "title": "Finish - Virginia"},
        ]
        ctx = extract_route_context("Virginia, United States", chapters)
        assert ctx.gps_towns == ["Manassas", "Fairfax"]
        assert ctx.county == "Prince William County"

    def test_route_suffixed_county_extracts_towns(self):
        # Chapter titles can carry a trailing named-route suffix (", Slea Head Drive").
        chapters = [
            {"timestamp": "0:00", "title": "Start - County Kerry"},
            {"timestamp": "10:00", "title": "Entering Dingle, County Kerry, Slea Head Drive"},
            {"timestamp": "20:00", "title": "Entering Ventry, County Kerry, Slea Head Drive"},
        ]
        ctx = extract_route_context("County Kerry, Ireland", chapters)
        assert ctx.gps_towns == ["Dingle", "Ventry"]
        assert ctx.county == "County Kerry"

    def test_singleword_county_forms_still_parse(self):
        """Regression: existing single-word Irish + US county forms still parse."""
        for title, town, county in (
            ("Entering Killorglin, County Kerry", "Killorglin", "County Kerry"),
            ("Entering Charleston, Charleston County", "Charleston", "Charleston County"),
        ):
            ctx = extract_route_context("somewhere", [{"timestamp": "0:00", "title": title}])
            assert ctx.gps_towns == [town]
            assert ctx.county == county

    def test_non_county_entering_line_stays_unmatched(self):
        """A plain 'Entering …' line with no county shape must NOT be parsed
        as a town (non-match must stay a non-match)."""
        chapters = [{"timestamp": "0:00", "title": "Entering Dingle, Ireland"}]
        ctx = extract_route_context("Kerry, Ireland", chapters)
        assert ctx.gps_towns == []


class TestExtractRouteContextCountrySlot:
    """A county label in the last location segment must not fill the
    country slot (renders a spurious '| County Kerry' country noun)."""

    def test_irish_county_not_treated_as_country(self):
        ctx = extract_route_context("Kenmare, County Kerry", [])
        assert ctx.country != "County Kerry"
        assert ctx.country is None
        # The county is still captured in its own slot.
        assert ctx.county == "County Kerry"

    def test_us_county_not_treated_as_country(self):
        ctx = extract_route_context("Beaufort, Beaufort County", [])
        assert ctx.country != "Beaufort County"
        assert ctx.country is None

    def test_real_country_still_kept(self):
        """Regression: a genuine country segment is unaffected."""
        ctx = extract_route_context("Beara Peninsula, County Cork, Ireland", [])
        assert ctx.country == "Ireland"


class TestSmartEndpointsAllRejected:
    """When every GPS town is validator-rejected (obscure/business), the
    smart endpoints must fall back to None — NOT re-admit the raw first/last
    town that was just rejected — so make_default_title takes its
    location-string branch instead of burning a business name."""

    def test_all_towns_rejected_smart_start_returns_none(self):
        chapters = [
            {"timestamp": "0:00", "title": "Entering Sally's Cafe, Beaufort County"},
            {"timestamp": "10:00", "title": "Entering Mike's Grill, Beaufort County"},
        ]
        ctx = extract_route_context("South Carolina, United States", chapters)
        # Both towns are collected by the parser...
        assert ctx.gps_towns == ["Sally's Cafe", "Mike's Grill"]
        # ...but every one is rejected, so smart_start must be None, not the
        # raw first town it just filtered out.
        assert ctx.smart_start is None

    def test_all_towns_rejected_smart_end_returns_none(self):
        chapters = [
            {"timestamp": "0:00", "title": "Entering Sally's Cafe, Beaufort County"},
            {"timestamp": "10:00", "title": "Entering Mike's Grill, Beaufort County"},
        ]
        ctx = extract_route_context("South Carolina, United States", chapters)
        assert ctx.smart_end is None

    def test_rejected_business_name_absent_from_title(self):
        chapters = [
            {"timestamp": "0:00", "title": "Entering Sally's Cafe, Beaufort County"},
            {"timestamp": "10:00", "title": "Entering Mike's Grill, Beaufort County"},
        ]
        ctx = extract_route_context("South Carolina, United States", chapters)
        title = ctx.make_default_title()
        assert "Sally's Cafe" not in title
        assert "Mike's Grill" not in title
        assert "South Carolina" in title


class TestExtractRouteContextCountyEquivalents:
    """Parish (LA), Borough / Census Area (AK) and independent City (VA)
    are county-equivalents and must parse exactly like 'County' — otherwise the
    whole chapter line fails to match, leaving gps_towns empty and county None
    for entire Louisiana / Alaska / Virginia drives."""

    def test_louisiana_parish_extracts_towns_and_county(self):
        chapters = [
            {"timestamp": "0:00", "title": "Start - Louisiana"},
            {"timestamp": "10:00", "title": "Entering New Iberia, Iberia Parish"},
            {"timestamp": "20:00", "title": "Entering Jeanerette, Iberia Parish"},
            {"timestamp": "30:00", "title": "Finish - Louisiana"},
        ]
        ctx = extract_route_context("Louisiana, United States", chapters)
        assert ctx.gps_towns == ["New Iberia", "Jeanerette"]
        assert ctx.county == "Iberia Parish"

    def test_alaska_borough_extracts_county(self):
        chapters = [
            {"timestamp": "0:00", "title": "Entering Palmer, Matanuska-Susitna Borough"},
            {"timestamp": "10:00", "title": "Entering Wasilla, Matanuska-Susitna Borough"},
        ]
        ctx = extract_route_context("Alaska, United States", chapters)
        assert ctx.gps_towns == ["Palmer", "Wasilla"]
        assert ctx.county == "Matanuska-Susitna Borough"

    def test_alaska_census_area_extracts_county(self):
        chapters = [
            {"timestamp": "0:00", "title": "Entering Nome, Nome Census Area"},
            {"timestamp": "10:00", "title": "Entering Teller, Nome Census Area"},
        ]
        ctx = extract_route_context("Alaska, United States", chapters)
        assert ctx.gps_towns == ["Nome", "Teller"]
        assert ctx.county == "Nome Census Area"

    def test_virginia_independent_city_extracts_county(self):
        chapters = [
            {"timestamp": "0:00", "title": "Entering Richmond, Richmond City"},
            {"timestamp": "10:00", "title": "Entering Manchester, Richmond City"},
        ]
        ctx = extract_route_context("Virginia, United States", chapters)
        assert ctx.gps_towns == ["Richmond", "Manchester"]
        assert ctx.county == "Richmond City"

    def test_bare_city_word_not_matched_as_county(self):
        """Over-match guard: a bare 'City' (no preceding name) must NOT parse as
        a county-equivalent — that word alone is not an administrative unit."""
        ctx = extract_route_context(
            "somewhere", [{"timestamp": "0:00", "title": "Entering Foo, City"}]
        )
        assert ctx.gps_towns == []


class TestMakeDefaultTitle:
    """Title generation edge cases."""

    def test_single_town_fallback(self):
        chapters = [
            {"timestamp": "0:00", "title": "Start - Cork"},
            {"timestamp": "10:00", "title": "Entering Cobh, County Cork"},
            {"timestamp": "20:00", "title": "Finish - Cork"},
        ]
        ctx = extract_route_context("Cork, Ireland", chapters)
        title = ctx.make_default_title()
        # Single town — should still produce a valid title
        assert len(title) >= 10
        assert "Cobh" in title or "Cork" in title

    def test_title_within_length_bounds(self):
        chapters = [
            {"timestamp": "0:00", "title": "Start - SC"},
            {"timestamp": "5:00", "title": "Entering Beaufort, Beaufort County"},
            {"timestamp": "35:00", "title": "Entering Port Royal, Beaufort County"},
            {"timestamp": "48:00", "title": "Finish - SC"},
        ]
        ctx = extract_route_context("South Carolina, United States", chapters)
        title = ctx.make_default_title()
        assert len(title) <= 100
