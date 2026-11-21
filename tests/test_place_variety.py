"""A Short's place text varies when its clip measured another place (unit C2).

Near-zero-view Shorts line up with the same place named Short after Short. The
chooser draws only from places the Short's final-text review accepts (a town held
for every retained second, a county the clip voted) plus the clip's measured road,
for the description line only: the review refuses a road named in a title.
"""

from __future__ import annotations

import pytest

from gomoveshift.pipeline import longform_metadata as lm
from gomoveshift.pipeline.social_generator import (
    REASON_DISTINCT,
    REASON_NO_ALTERNATIVE,
    REASON_VARIED,
    SocialGenerator,
    TitleFacts,
    choose_place_body,
    measured_places,
    town_evidence,
)
from gomoveshift.pipeline.title_decision import short_place_facts

LABEL = "Polk County, NC"
REGIONS = ["Polk County, North Carolina"]
DATE = "2026-02-24"


@pytest.fixture(autouse=True)
def _no_published_titles(monkeypatch):
    monkeypatch.setattr(lm, "load_published_titles", lambda: set())


def _route(towns=(("Tryon", 38.0),), retained=38.0):
    return {
        "retained_seconds": retained,
        "containment_basis": "retained_gps",
        "places": [
            {"name": n, "evidence_id": f"ev-{n}", "required": True, "retained_seconds": s}
            for n, s in towns
        ],
    }


def _facts(ordinals, date=DATE):
    return [TitleFacts(date=date, ordinal=k, secs=38, place_ok=True) for k in ordinals]


def _gen():
    return SocialGenerator(
        location=LABEL, project_name="drive", series_part=1, spooled_texts={}, experiments=()
    )


def _run(roads, routes, ordinals=(1, 2), previous_places=None, date=DATE):
    n = len(ordinals)
    return _gen().generate_all(
        num_shorts=n,
        per_short_locations=[LABEL] * n,
        per_short_regions=[REGIONS] * n,
        per_short_roads=roads,
        per_short_route_records=routes,
        per_short_title_facts=_facts(ordinals, date),
        previous_places=previous_places,
    )


# --- the pure chooser -------------------------------------------------


def test_chooser_keeps_today_when_it_is_not_a_repeat():
    c = choose_place_body(["Polk County", "Tryon"], ["Saluda"])
    assert (c.body, c.reason) == ("Polk County", REASON_DISTINCT)


def test_chooser_takes_the_first_other_candidate_when_today_repeats():
    c = choose_place_body(
        ["Polk County", "Tryon", "US-64"], ["polk  county"], kinds=["today", "town", "road"]
    )
    assert (c.body, c.reason, c.kind) == ("Tryon", REASON_VARIED, "town")


def test_chooser_skips_a_candidate_that_also_repeats():
    c = choose_place_body(["Polk County", "Tryon", "US-64"], ["Polk County", "Tryon"])
    assert c.body == "US-64" and c.reason == REASON_VARIED


def test_chooser_with_nothing_else_keeps_today_and_says_so():
    c = choose_place_body(["Polk County"], ["Polk County"])
    assert (c.body, c.reason) == ("Polk County", REASON_NO_ALTERNATIVE)


def test_measured_places_are_town_road_county_and_only_review_accepted_towns():
    route = _route(towns=(("Tryon", 38.0), ("Saluda", 10.0)))  # Saluda: part of the clip
    got = measured_places(REGIONS, ["US-64", "US-64"], route)
    assert got == [("town", "Tryon"), ("road", "US-64"), ("county", "Polk County")]
    assert town_evidence(route) == {"Tryon": "ev-Tryon"}


# --- the composer -----------------------------------------------------


def test_second_short_with_the_same_top_place_varies_its_title_to_a_measured_town():
    posts = _run([[], []], [None, _route()])
    assert posts[0].youtube_title.endswith("Polk County")
    assert "Tryon" in posts[1].youtube_title and "Polk County" not in posts[1].youtube_title
    # the town is the title's first choice: the Short's most specific measured place
    assert posts[1].place_variety["title"]["reason"] == REASON_DISTINCT
    # the review accepts it: a town the clip's route held for every retained second
    assert short_place_facts(
        posts[1].youtube_title,
        "Tryon",
        REGIONS,
        segment_id="s",
        towns=town_evidence(_route()),
    )


def test_second_short_with_a_measured_road_names_the_road_in_its_description():
    posts = _run([[], ["US-64"]], [None, None])
    assert "US-64" not in posts[0].youtube_description
    assert "US-64" in posts[1].youtube_description
    assert posts[1].place_variety["description"]["reason"] == REASON_VARIED
    assert posts[1].place_variety["description"]["kind"] == "road"
    # a road is never put in a title: the review refuses it
    assert "US-64" not in posts[1].youtube_title


def test_no_alternative_keeps_todays_text_and_records_it():
    posts = _run([[], []], [None, None])
    assert "Polk County" in posts[0].youtube_title and "Polk County" in posts[1].youtube_title
    assert posts[1].place_variety["title"]["reason"] == REASON_NO_ALTERNATIVE
    assert posts[1].place_variety["description"]["reason"] == REASON_NO_ALTERNATIVE
    assert posts[1].place_variety["title"]["body"] == "Polk County"
    assert posts[1].to_dict()["place_variety"] == posts[1].place_variety


def test_the_previous_days_last_short_counts_as_previous():
    prev = {"title": "Polk County", "description": "Polk County"}
    posts = _run([[]], [_route()], ordinals=(1,), previous_places=prev, date="2026-02-25")
    assert "Tryon" in posts[0].youtube_title
    assert posts[0].place_variety["title"]["reason"] == REASON_DISTINCT
    # the first Short of a project with nothing before it names the town as well
    first = _run([[]], [_route()], ordinals=(1,))
    assert first[0].youtube_title == "Tryon"


def test_a_town_the_review_would_refuse_is_never_chosen():
    # held for part of the clip only, and not required: the review does not accept it
    partial = _route(towns=(("Saluda", 12.0),))
    unneeded = _route()
    unneeded["places"][0]["required"] = False
    for route in (partial, unneeded):
        posts = _run([[], []], [None, route])
        assert "Saluda" not in posts[1].youtube_title and "Tryon" not in posts[1].youtube_title
        assert posts[1].place_variety["title"]["reason"] == REASON_NO_ALTERNATIVE


def test_the_description_varies_in_step_with_the_title():
    posts = _run([[], []], [None, _route()])
    assert "Polk County" in posts[0].youtube_description
    assert "Tryon" in posts[1].youtube_description
    assert posts[1].place_variety["description"]["body"] == "Tryon"
    assert "#PolkCounty" in posts[1].youtube_description  # region tag stays pinned
