"""Shared region/units/playlist derivation helper."""

from __future__ import annotations

import pytest

from gomoveshift.models import ResolvedLocation
from gomoveshift.pipeline.region_resolver import (
    derive_region,
    derive_short_title_region,
    is_us_region,
)


@pytest.mark.parametrize(
    "location,expected",
    [
        ("Dingle Peninsula, County Kerry, Ireland", "ireland"),
        ("Blue Ridge Parkway, North Carolina, USA", "usa"),
        ("Some Road, California, u.s.", "usa"),  # alias
        ("Beaufort County, South Carolina", "usa"),  # trailing US state
        ("Port Royal to Lady's Island, SC", "usa"),  # trailing US abbr
        ("Some Town, IE", "ie"),  # non-US 2-letter
        ("Blue Ridge Parkway, Virginia, USA — Part 2", "usa"),  # series suffix
        ("", ""),
        (None, ""),
        ("Some Mountain Pass", "some mountain pass"),  # single component
        # A trailing "(road)" disambiguator must not poison the state
        # token — was 'south carolina (blocker run road)'.
        ("Colleton County, South Carolina (Blocker Run Road)", "usa"),
        # Same class: Irish "County <X>" tail (labels carry no
        # trailing country) — was 'county kerry'.
        ("Kenmare, County Kerry", "ireland"),
        ("County Kerry", "ireland"),  # bare Irish county label
        ("Colleton County", "usa"),  # bare US "<X> County" tail
        ("Beaufort, SC", "usa"),  # regression: trailing US abbr
        # bare Irish county tail (no "County" prefix) — was leaking to the
        # "All Drives" fallback because the bare token round-tripped as-is.
        ("Tralee to Killarney, Kerry", "ireland"),
        ("Gap of Dunloe, Kerry", "ireland"),
        ("Kilgarvan to Ballingeary, Cork", "ireland"),
        ("Dingle, Kerry", "ireland"),
        ("Somewhere, Bavaria", "bavaria"),  # non-county tail still unchanged
    ],
)
def test_derive_region_label_fallback(location, expected):
    assert derive_region(location) == expected


def test_derive_region_bare_irish_county_full_token_match_only():
    # "Kerryman Road" must not spuriously match on a "kerry" substring.
    assert derive_region("Somewhere, Kerryman Road") == "kerryman road"


@pytest.mark.parametrize("country_code", ["IE", "ie"])
def test_derive_region_resolved_ireland_country_code(country_code):
    resolved = ResolvedLocation(country_code=country_code)
    assert derive_region("Some Town", resolved=resolved) == "ireland"
    assert is_us_region("Some Town", resolved=resolved) is False


def test_resolved_model_is_authoritative():
    # is_us wins over a non-US label; country_code over a US label; empty shell
    # falls back to the label (must not mask it).
    assert derive_region("Somewhere, Ireland", resolved=ResolvedLocation(is_us=True)) == "usa"
    assert derive_region("Mislabelled, USA", resolved=ResolvedLocation(country_code="de")) == "de"
    assert derive_region("Blue Ridge Parkway, NC", resolved=ResolvedLocation()) == "usa"
    assert is_us_region("Dingle Peninsula, County Kerry, Ireland") is False
    assert is_us_region("Somewhere, Ireland", resolved=ResolvedLocation(is_us=True)) is True


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        # Ireland: country word, flag, "County <Irish county>" (the Irish word order).
        ("Would You Drive This Road? — County Kerry, Ireland", "ireland"),
        ("Rolling Through Lough Inagh | Galway, Ireland 🇮🇪", "ireland"),
        ("Scenic Drive Through County Limerick", "ireland"),
        ("Relaxing Irish Drive 🍀 County Clare Coastal", "ireland"),
        ("Guess Where We're Going 🗺️ County Clare", "ireland"),
        ("Drive Through Northern Ireland", "ireland"),
        # Ireland: a bare county token (no US signal) and the seed place gazetteer.
        ("Boyle - Roscommon", "ireland"),
        ("Slow Drive Through West Cork", "ireland"),
        ("Do You Know This Part of Clare?", "ireland"),
        ("Dingle to Conor Pass — Raw Dashcam", "ireland"),
        ("Beaufort to Annascaul POV Drive", "ireland"),
        ("POV Road Trip Through Creeslough", "ireland"),
        ("Road Trip Toward Mt Errigal", "ireland"),
        ("Relaxing POV Drive Around Lough Veagh", "ireland"),
        ("The Emerald Isle's Hidden Gem: The Sperrin AONB", "ireland"),
        # USA: "<X> County" suffix form, state names, USPS codes, country words.
        ("Colleton County — Raw Dashcam", "usa"),
        ("Beaufort County — Nothing but Road", "usa"),
        ("Adams County Vibes 🚗", "usa"),
        ("Sullivan County, Tennessee", "usa"),
        ("Roads POV — Madison County, North Carolina", "usa"),
        ("Windows Down — Wythe County, Virginia 🚗", "usa"),
        ("Into the Blue Ridge | Scenic Drive Toward Highlands, NC", "usa"),
        ("Fort Garland | Colorado Road Trip", "usa"),
        ("Tennessee travels", "usa"),
        ("Upper Mesa Falls #scenicdrive #waterfall #idaho", "usa"),
        ("River Stillness Near Needmore Road | Rippling Water in the NC Mountains", "usa"),
        ("Driving the Back Roads 🇺🇸", "usa"),
        # A US state beside a bare Irish-county name is the American place.
        ("Clare, Michigan Road Trip", "usa"),
        # No signal, or a contradictory one: never guessed.
        ("Tail of the Dragon POV Drive | 318 Curves in 11 Miles", None),
        ("Rolling Down the Roads", None),
        ("Dirty water", None),
        ("Slow Down", None),
        ("County Road 12 Drive", None),
        ("Ireland to Tennessee, the Long Way", None),
        ("", None),
        # "<X> County" followed by Down/Kerry etc. is ambiguous, never Ireland.
        ("Wythe County Down the Back Roads", None),
        ("Sullivan County Cork Road", None),
        ("Colleton county down south", None),
        ("Wythe County Down the Road, Virginia", "usa"),
        ("Emerald Isle Sunset Drive", None),
        # The Irish word order still wins when a preposition leads, or alone.
        ("Road Trip Through County Kerry", "ireland"),
        ("Windows Down in County Down", "ireland"),
        ("County Kerry Vibes", "ireland"),
        # Names that are also US towns/counties in the channel's own area are
        # not Irish evidence on their own (Dublin VA, Tyrone/Limerick PA).
        ("Pulaski to Dublin — Raw Dashcam", None),
        ("Behind the Wheel — Pulaski to Dublin 🚗", None),
        ("Christiansburg to Dublin", None),
        ("Dublin to Pulaski via Route 11", None),
        ("Driving Through Tyrone", None),
        ("Limerick Back Roads", None),
        ("Waterford Sunset Drive", None),
        # "<Irish county> County" is the US word order (Clare/Antrim MI): ambiguous.
        ("Clare County Drive", None),
        ("Antrim County Back Roads", None),
        ("Roscommon County Vibes", None),
        ("Wexford County Drive", None),
        ("Clare County, Michigan", "usa"),
        # Dublin / Tyrone stay Irish through the explicit forms.
        ("Slow Drive Through County Dublin", "ireland"),
        ("Dublin, Ireland", "ireland"),
        # Mixed-case / all-caps US county names are not the Irish word order.
        ("McDowell County Drive", "usa"),
        ("DeKalb County Back Roads", "usa"),
        ("McDowell County Down the Mountain", None),
        ("WYTHE County Down", None),
        ("McMinn County Kerry Road", None),
    ],
)
def test_derive_short_title_region(title, expected):
    assert derive_short_title_region(title) == expected


@pytest.mark.parametrize(
    "title",
    [
        "Colleton County Vibes 🚗",
        "Beaufort County — 4K",
        "Adams County in 4K 🚗",
        "Driving Gettysburg in 4K 🚗, PA",
        "Sullivan County — Raw Dashcam",
        "Transylvania County — No Commentary",
        "Wytheville to Fort Chiswell via I-77, VA",
        "Johnson City, Tennessee Drive",
        "Unicoi County, Tennessee POV",
        "Dashcam Diaries — Derry Township",
        "Pulaski to Dublin via I-81",
        "Cumberland Township Back Roads",
        "McDowell County Drive",
        "DeKalb County, NC Back Roads",
        "McDowell County Down the Mountain, NC",
    ],
)
def test_derive_short_title_region_never_flips_american_titles_to_ireland(title):
    assert derive_short_title_region(title) == "usa"
