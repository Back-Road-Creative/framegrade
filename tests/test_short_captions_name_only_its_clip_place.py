"""A short's title/caption may name ONLY the place its own clip was filmed."""

import re

import pytest
import yaml

from gomoveshift.pipeline import longform_metadata as lm
from gomoveshift.pipeline.social_generator import TITLE_FRAME_TEMPLATES, SocialGenerator


@pytest.fixture(autouse=True)
def _no_published_titles(monkeypatch):
    monkeypatch.setattr(lm, "load_published_titles", lambda: set())


def _titles(tmp_path, entry):
    path = tmp_path / "titles.yaml"
    path.write_text(yaml.safe_dump({1: entry}))
    return path


def _parent(**kw):
    from gomoveshift.models import ResolvedLocation

    return SocialGenerator(
        location="County Kerry, Munster",
        project_name="p",
        series_part=1,
        terrain_type="coastal",
        road_name="N70",
        roads=["N70"],
        nearest_poi="Derrynane House",
        county="County Kerry",
        resolved_location=ResolvedLocation(destination="Gap of Dunloe", country_code="IE"),
        **kw,
    )


def _surfaces(post):
    return " | ".join(
        [
            post.youtube_title,
            post.tiktok_title,
            post.caption,
            *post.captions.values(),
            *post.youtube_tags,
            post.first_comment or "",
            post.youtube_description or "",
        ]
    ).lower()


def test_gps_bearing_same_label_short_drops_parent_facts():
    posts = _parent().generate_all(
        num_shorts=1,
        per_short_locations=["County Kerry, Munster"],
        per_short_regions=[["County Kerry, Munster"]],
    )
    text = _surfaces(posts[0])
    for token in ("gap of dunloe", "derrynane", "n70", "coastal"):
        assert token not in text, token


def test_gps_less_same_label_short_keeps_parent_facts():
    posts = _parent().generate_all(num_shorts=1, per_short_locations=["County Kerry, Munster"])
    text = _surfaces(posts[0])
    assert any(t in text for t in ("gap of dunloe", "derrynane", "n70", "coastal"))


@pytest.mark.parametrize(
    "field,text",
    [
        ("title", "Back roads into Gettysburg, PA"),
        ("title", "Last miles to Goleen, West Cork"),
        ("caption", "Heading to Goleen"),
    ],
)
def test_direction_framed_hand_text_refused(tmp_path, field, text):
    entry = {"title": "Slow miles, no talking"}
    entry[field] = text
    gen = SocialGenerator(
        location="County Kerry, Munster",
        project_name="p",
        series_part=1,
        titles_file=_titles(tmp_path, entry),
    )
    with pytest.raises(ValueError, match="filmed") as exc:
        gen.generate_post(1)
    assert "short_index 1" in str(exc.value)
    assert text in str(exc.value)


@pytest.mark.parametrize("title", ["Slow miles, no talking", "Back roads of Adams County"])
def test_place_free_or_own_place_hand_title_passes(tmp_path, title):
    gen = SocialGenerator(
        location="Adams County, PA",
        project_name="p",
        series_part=1,
        titles_file=_titles(tmp_path, {"title": title}),
    )
    assert gen.generate_post(1).youtube_title == title


def _variant_gen(tmp_path, title):
    parent = SocialGenerator(
        location="Gettysburg Battlefield, PA",
        project_name="p",
        series_part=1,
        titles_file=_titles(tmp_path, {"title": title}),
    )
    return parent, parent._per_short_variant("Cashtown, PA")


def test_variant_refuses_parent_place_token(tmp_path):
    _, variant = _variant_gen(tmp_path, "Gettysburg back roads")
    with pytest.raises(ValueError) as exc:
        variant.generate_post(1)
    assert "Gettysburg" in str(exc.value) and "Cashtown" in str(exc.value)


def test_variant_refuses_parent_place_token_in_caption(tmp_path):
    path = _titles(tmp_path, {"title": "Quiet miles", "caption": "Near Gettysburg tonight."})
    parent = SocialGenerator(
        location="Gettysburg Battlefield, PA",
        project_name="p",
        series_part=1,
        titles_file=path,
    )
    with pytest.raises(ValueError, match="Gettysburg"):
        parent._per_short_variant("Cashtown, PA").generate_post(1)


def test_variant_allows_own_place_and_parent_is_unaffected(tmp_path):
    parent, variant = _variant_gen(tmp_path, "Cashtown back roads")
    assert variant.generate_post(1).youtube_title == "Cashtown back roads"
    p2 = SocialGenerator(
        location="Gettysburg Battlefield, PA",
        project_name="p",
        series_part=1,
        titles_file=_titles(tmp_path, {"title": "Gettysburg back roads"}),
    )
    assert p2.generate_post(1).youtube_title == "Gettysburg back roads"


def test_no_frame_template_uses_direction_framing():
    pat = re.compile(r"\b(?:to|into|toward|towards)\s+\{", re.I)
    assert len(TITLE_FRAME_TEMPLATES) == 6
    assert [t for t in TITLE_FRAME_TEMPLATES if pat.search(t)] == []


# ── region tags come from the short's own GPS, never the parent's country ───


def _region_tags_of(parent, location, regions):
    """The region phrases the variant derives. Asserted on ``_region_tags``
    itself: the Short tag rule drops non-template tags (region phrases included) from a
    Short's published tags, so the published list no longer shows them."""
    from gomoveshift.pipeline.social_generator import _region_tags

    variant = parent._per_short_variant(location, list(regions) or None)
    tags = _region_tags(variant.location, variant.resolved_location, variant._own_regions)
    return " ".join(tags).lower()


def test_variant_region_tags_follow_the_shorts_own_state_not_the_parents_country():
    """Parent LF resolved in Ireland; this short's own GPS is in Virginia."""
    tags = _region_tags_of(_parent(), "Wythe County, Virginia", ["Wythe County, Virginia"])
    assert "usa road trip" in tags
    assert "ie road trip" not in tags and "ireland" not in tags


def test_variant_in_a_us_parent_does_not_inherit_usa_for_a_stateless_short():
    """Parent LF resolved in the US; this short's own regions carry no US state."""
    from gomoveshift.models import ResolvedLocation

    parent = SocialGenerator(
        location="Wythe County, Virginia",
        project_name="p",
        series_part=1,
        resolved_location=ResolvedLocation(is_us=True, country_code="US"),
    )
    tags = _region_tags_of(parent, "County Cork", ["County Cork"])
    assert "usa" not in tags


def test_gps_less_variant_without_regions_keeps_the_parents_verdict():
    """Audit-run legacy path (no own regions): unchanged."""
    tags = _region_tags_of(_parent(), "Somewhere Else, Munster", [])
    assert "ie road trip" in tags
