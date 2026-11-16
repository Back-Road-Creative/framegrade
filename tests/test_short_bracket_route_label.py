"""A GPS-less bracket-route drive's Shorts are titled with the backed label itself.

2026-11-16, nctopa-2026-03-05-rec0063: ``_arc_safe_anchor`` collapsed the backed
label "Virginia to Pennsylvania" to its destination, so 21 Shorts named
"Pennsylvania", an endpoint the record never measured, and the Short review refused
each one (``description_no_clip_place``). A Short whose text is exactly a valid
bracket-route label keeps it whole; every other "A to B" arc still collapses and
every other direction framing is still refused.
"""

from __future__ import annotations

import pytest

from gomoveshift.pipeline import longform_metadata as lm
from gomoveshift.pipeline.social_generator import (
    SocialGenerator,
    TitleFacts,
    _arc_safe_anchor,
    _location_hashtags,
)

LABEL = "Virginia to Pennsylvania"
RECORD = {"label": LABEL, "level": "states", "gap_h": 17.5}


@pytest.fixture(autouse=True)
def _no_published_titles(monkeypatch):
    monkeypatch.setattr(lm, "load_published_titles", lambda: set())


def _gen(location=LABEL, bracket_route=RECORD):
    return SocialGenerator(
        location=location, project_name="rec0063", series_part=1, bracket_route=bracket_route
    )


def _posts(gen, label=LABEL, n=3):
    facts = [
        TitleFacts(date="2026-03-05", ordinal=k, secs=38, place_ok=True) for k in range(1, n + 1)
    ]
    return gen.generate_all(
        num_shorts=n,
        per_short_locations=[label] * n,
        per_short_regions=[[label]] * n,
        per_short_title_facts=facts,
    )


def test_backed_bracket_label_titles_the_short_with_the_whole_label():
    posts = _posts(_gen())
    assert posts
    for post in posts:
        assert "stretch: " in post.youtube_title
        assert LABEL in post.youtube_title
        assert post.sampled_regions == [LABEL]


def test_backed_bracket_short_carries_the_record_for_the_review():
    assert all(p.bracket_route == RECORD for p in _posts(_gen()))


def test_title_never_names_the_destination_alone():
    for post in _posts(_gen()):
        stripped = post.youtube_title.replace(LABEL, "")
        assert "Pennsylvania" not in stripped


def test_non_bracket_arc_still_collapses_to_its_destination():
    assert _arc_safe_anchor("Fancy Gap to Hillsville via I-77") == "Hillsville"
    gen = _gen(location="Fancy Gap to Hillsville", bracket_route=None)
    assert gen.location_short == "Hillsville"
    assert gen.location_short_full == "Fancy Gap to Hillsville"


def test_label_not_matching_the_record_still_collapses():
    gen = _gen(location="Virginia to Maryland", bracket_route=RECORD)
    assert gen.location_short == "Maryland"


def test_invalid_bracket_record_does_not_lift_the_collapse():
    over_cap = {**RECORD, "gap_h": 99}
    assert _gen(bracket_route=over_cap).location_short == "Pennsylvania"


def test_direction_framing_other_than_the_backed_label_is_still_refused():
    from gomoveshift.pipeline.social_generator import HandTitle

    gen = _gen()
    gen._check_hand_text_place(1, HandTitle(title=f"Mar 5: {LABEL}", caption=""))
    for text in ("Heading to Pennsylvania", f"{LABEL} and on to Gettysburg"):
        with pytest.raises(ValueError, match="direction framing"):
            gen._check_hand_text_place(1, HandTitle(title=text, caption=""))


def test_without_a_record_the_label_is_still_refused_as_direction_framing():
    from gomoveshift.pipeline.social_generator import HandTitle

    with pytest.raises(ValueError, match="direction framing"):
        _gen(bracket_route=None)._check_hand_text_place(
            1, HandTitle(title=f"Mar 5: {LABEL}", caption="")
        )


def test_backed_bracket_label_emits_no_location_hashtag():
    # JP 2026-11-16: drop the location hashtag; "#VirginiatoPennsylvania" reads badly.
    assert _location_hashtags(LABEL, RECORD) == []
    assert _location_hashtags("Pennsylvania") == ["#Pennsylvania"]
    assert _location_hashtags(LABEL, None) == ["#Pennsylvania"]


def test_no_text_field_of_a_backed_bracket_short_carries_a_place_hashtag():
    for post in _posts(_gen()):
        for name, value in vars(post).items():
            if isinstance(value, str):
                assert "#Virginiato" not in value, name
                assert "#Pennsylvania" not in value, name
