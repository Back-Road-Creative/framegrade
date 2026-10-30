"""Captions are composed from independently rotated parts, never drawn from a
fixed template pool (U3): unique first words per batch and across projects,
rotating generic hashtags, no spam-shaped openers."""

from __future__ import annotations

from collections import Counter

import pytest

from gomoveshift.pipeline.social_generator import (
    GENERIC_HASHTAGS,
    LOCATION_WITHHELD,
    SocialGenerator,
)
from gomoveshift.publishing.anti_spam_gate import _normalize_words

GENERIC = {t.lower() for t in GENERIC_HASHTAGS}


def _gen(project="u3_proj", location="County Cork, Ireland", **kw):
    return SocialGenerator(
        location=location,
        project_name=project,
        series_part=1,
        road_name="R572",
        county="County Cork",
        **kw,
    )


def _captions(gen, platform, n=12):
    # Captions are composed per short independent of the 5-title capacity ceiling.
    return [gen._build_platform_captions(None, i)[platform] for i in range(1, n + 1)]


def _prefix(text, n=6):
    return tuple(_normalize_words(text)[:n])


@pytest.mark.parametrize("platform", ["facebook", "instagram", "tiktok"])
def test_first_six_words_unique_across_twelve_shorts(platform):
    caps = _captions(_gen(), platform)
    prefixes = [_prefix(c) for c in caps]
    assert len(set(prefixes)) == 12, caps


@pytest.mark.parametrize("platform", ["facebook", "instagram", "tiktok"])
def test_two_projects_same_location_differ(platform):
    a = _captions(_gen("proj_alpha"), platform)
    b = _captions(_gen("proj_bravo"), platform)
    assert a != b
    assert sum(x == y for x, y in zip(a, b, strict=True)) <= 2


def test_deterministic_per_project_and_short():
    assert _captions(_gen(), "facebook") == _captions(_gen(), "facebook")


@pytest.mark.parametrize("platform", ["facebook", "instagram", "tiktok"])
def test_spam_shaped_openers_are_gone(platform):
    for cap in _captions(_gen(), platform):
        low = cap.lower()
        assert "somewhere in" not in low
        assert "ever been through" not in low


@pytest.mark.parametrize(
    ("platform", "location"),
    [("facebook", "Killarney"), ("instagram", "County Cork, Ireland")],
)
def test_generic_hashtag_set_rotates_across_captions(platform, location):
    # A region-bearing Facebook caption's 2-tag limit is filled by the
    # mandatory place+region tags, so FB rotation shows on a regionless place.
    sets = []
    for cap in _captions(_gen(location=location), platform):
        tags = frozenset(t.lower() for t in cap.split() if t.startswith("#"))
        sets.append(tags & GENERIC)
    assert len(set(sets)) > 1, sets
    assert Counter(sets).most_common(1)[0][1] < 12


def test_place_and_region_tags_stay_mandatory_on_every_facebook_caption():
    gen = _gen()
    mandatory = [t.lower() for t in gen._mandatory_location_tags()]
    assert mandatory
    for cap in _captions(gen, "facebook"):
        low = cap.lower()
        assert all(t in low for t in mandatory), cap


def test_interactive_caption_withholds_place_and_asks_something(tmp_path):
    import yaml

    titles = tmp_path / "titles.yaml"
    titles.write_text(yaml.safe_dump({1: {"title": "Somewhere scenic", "interactive": True}}))
    gen = _gen(titles_file=titles)
    post = gen.generate_post(short_index=1)
    assert post.interactive
    fb = post.captions["facebook"]
    assert gen.social_location not in fb
    assert LOCATION_WITHHELD in fb.lower()
    assert "?" in fb


def test_detail_comes_only_from_known_data():
    gen = _gen()
    gen.recording_date = "2026-08-01"
    post = gen.generate_post(short_index=3)
    assert "2026-08-01" in post.captions["instagram"]
