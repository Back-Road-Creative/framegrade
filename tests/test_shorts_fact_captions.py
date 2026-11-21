"""A Short's platform captions state measured facts and nothing else.

The owner's rule: captions stay factual, with no generic filler. A caption is the
place the clip measured, a terrain clause only when the terrain is classified
(coastal, mountain, winding), and, on an interactive Short, a guess prompt in
place of the withheld place. The "Filmed <Month YYYY>." sentence is added by the
plan builder from the recording date, and the hashtags are unchanged.
"""

from __future__ import annotations

import re

import pytest

from gomoveshift.pipeline import longform_metadata as lm
from gomoveshift.pipeline import social_generator as sg
from gomoveshift.pipeline.social_generator import (
    _PLATFORM_CAPTION_TEMPLATES,
    CAPTION_DETAIL_TEMPLATES,
    INTERACTIVE_PROMPT_TEMPLATES,
    SocialGenerator,
    TitleFacts,
    caption_template_id,
)

LABEL = "Polk County, NC"
REGIONS = ["Polk County, North Carolina"]
PLATFORMS = ("youtube", "facebook", "instagram", "tiktok")
FILLER = re.compile(
    r"mostly|just roads|the whole way|nothing but|taking in|drive with me|ride along|"
    r"worth the detour|would you|come along|next stop|hello from|open road|cruising|"
    r"wheel time|on the move|rolling|heading out",
    re.I,
)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(lm, "load_published_titles", lambda: set())
    monkeypatch.setattr(SocialGenerator, "_meta_ranked", lambda self, platform, order: order)
    monkeypatch.setattr(sg, "shipped_title_snapshot", lambda: None)


def _posts(terrain=None, n=4, **kw):
    gen = SocialGenerator(
        location=LABEL,
        project_name="drive",
        series_part=1,
        spooled_texts={},
        experiments=(),
        terrain_type=terrain,
        recording_date="2025-09-14",
        **kw,
    )
    return gen.generate_all(
        num_shorts=n,
        per_short_locations=[LABEL] * n,
        per_short_regions=[REGIONS] * n,
        per_short_title_facts=[TitleFacts("2025-09-14", k, 38, True) for k in range(1, n + 1)],
    )


def _sentences(caption: str) -> "list[str]":
    """The caption's sentences, without hashtags, the date line and the weather emoji."""
    lines = [ln for ln in caption.split("\n") if ln.strip()]
    text = " ".join(ln for ln in lines if not ln.startswith(("#", "\U0001f4f9", "Filmed")))
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s.,'?!-]", "", text)
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+|(?<=[a-z])\s*$", text) if s.strip()]


def test_the_old_generic_opener_and_detail_pools_are_gone():
    shapes = [t for pool in _PLATFORM_CAPTION_TEMPLATES.values() for t in pool]
    shapes += CAPTION_DETAIL_TEMPLATES
    assert not FILLER.search(" ".join(shapes))
    assert not any("Mostly" in t or "Just" in t for t in shapes)
    assert all("{location}" in t for pool in _PLATFORM_CAPTION_TEMPLATES.values() for t in pool)


def test_no_platform_shares_a_caption_shape():
    pools = [set(p) for p in _PLATFORM_CAPTION_TEMPLATES.values()]
    assert sum(len(p) for p in pools) == len(set().union(*pools))


def test_unknown_terrain_captions_carry_no_terrain_clause():
    for terrain in (None, "countryside"):
        for post in _posts(terrain):
            for platform in PLATFORMS:
                caption = post.captions[platform]
                assert not FILLER.search(caption), (platform, caption)
                assert "roads" not in caption.lower(), (platform, caption)


@pytest.mark.parametrize(
    "terrain,clause",
    [("coastal", "Coastal roads."), ("mountain", "Mountain roads."), ("winding", "Winding roads.")],
)
def test_a_classified_terrain_is_stated_once_as_a_fact(terrain, clause):
    gen = SocialGenerator(
        location=LABEL, project_name="drive", series_part=1, spooled_texts={}, terrain_type=terrain
    )
    caption = gen._build_platform_captions(None, 1)["youtube"]
    assert clause in caption and caption.count("roads") == 1
    assert not FILLER.search(caption)


def test_every_caption_fact_is_in_the_record():
    # The drive path's per-Short clone carries no terrain (a clip's own GPS never
    # classified one), so its captions are the place alone.
    for post in _posts("coastal"):
        for platform in PLATFORMS:
            sentences = _sentences(post.captions[platform])
            place = [s for s in sentences if "Polk County" in s]
            assert len(place) == 1, (platform, sentences)
            # the place is the record's own measured place, nothing else is named
            assert set(re.findall(r"[A-Z][a-z]{3,}", place[0])) <= {
                "Polk",
                "County",
                "Passing",
                "Through",
                "Location",
                "Clip",
                "Footage",
                "Here",
                "This",
            }
            others = [s for s in sentences if s is not place[0]]
            assert others == [], (platform, others)


def test_the_caption_names_the_place_in_every_non_interactive_platform():
    for post in _posts():
        for platform in PLATFORMS:
            assert LABEL.split(",")[0] in post.captions[platform], platform


def test_interactive_caption_is_a_guess_prompt_and_withholds_the_place():
    gen = SocialGenerator(
        location=LABEL,
        project_name="drive",
        series_part=1,
        spooled_texts={},
        terrain_type="coastal",
    )
    picked: dict = {}
    captions = gen._build_platform_captions(None, 2, interactive=True, picked=picked)
    prompts = {p.strip() for p in INTERACTIVE_PROMPT_TEMPLATES}
    for platform in PLATFORMS:
        text = captions[platform]
        assert "Polk" not in text and "this route" not in text.lower()
        assert any(p in text for p in prompts), (platform, text)
        assert "roads" not in text.lower()
        assert picked[platform] in {caption_template_id(p) for p in INTERACTIVE_PROMPT_TEMPLATES}


def test_caption_template_id_is_one_stable_id_per_shape():
    first = _posts()[0].caption_templates
    again = _posts()[0].caption_templates
    assert first == again
    for platform, tid in first.items():
        by_id = {caption_template_id(t): t for t in _PLATFORM_CAPTION_TEMPLATES[platform]}
        assert tid in by_id and len(by_id) == len(_PLATFORM_CAPTION_TEMPLATES[platform])
    assert caption_template_id("Passing through {location}.") == caption_template_id(
        "Passing through {location}."
    )


#: Words and emoji that claim a camera, a vehicle, a viewpoint, the weather or a time of day:
#: things the pipeline does not measure, so no caption shape may say them.
UNBACKED = re.compile(
    r"dashcam|dash cam|camera|gopro|insta360|\bpov\b|driver|wheel|windshield|front[- ]seat|"
    r"by car|\bcar\b|sunny|rain|cloud|overcast|foggy|snow|sunrise|sunset|morning|evening|"
    r"night|golden hour|[\u2600-\u26ff\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]",
    re.I,
)


def test_no_caption_shape_claims_what_the_pipeline_does_not_measure():
    from gomoveshift.pipeline import social_generator as sg

    for platform, pool in sg._PLATFORM_CAPTION_TEMPLATES.items():
        for template in pool:
            # the pin icon is a map marker for the place, not a weather or vehicle claim
            assert not UNBACKED.search(template.replace("\U0001f4cd", "")), (platform, template)
    for post in _posts("coastal"):
        for platform in PLATFORMS:
            text = post.captions[platform].replace("\U0001f4cd", "").replace("\U0001f4f9", "")
            body = text.split("\n\n#")[0].split("#")[0]
            assert not UNBACKED.search(body), (platform, body)
