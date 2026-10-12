"""Every Short names its place AND its state/county on every platform.

The place tag and the region tag (the last location component) are
mandatory on Facebook / Instagram / TikTok captions and in the YouTube
Shorts description; platform tag limits and length cuts never drop them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from gomoveshift.pipeline.social_generator import SocialGenerator


def _fixtures():
    sibling = Path(__file__).resolve().parent / "test_publishing_plan_builder.py"
    spec = importlib.util.spec_from_file_location("_gms_pb_fixtures_state", sibling)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._build, mod._post


_build, _post = _fixtures()

CASES = [
    ("Unicoi, Tennessee", "#Unicoi", "#Tennessee"),
    ("Crookhaven, County Cork", "#Crookhaven", "#CountyCork"),
    ("Dingle, County Kerry, Ireland", "#Dingle", "#Ireland"),
]


def _gen(location: str, **kw) -> SocialGenerator:
    return SocialGenerator(location=location, project_name="state_tags", series_part=1, **kw)


def _surfaces(gen: SocialGenerator, short_index: int):
    """Captions + YouTube description, skipping title composition (which
    needs grounded facts to reach short 6)."""
    caps = gen._build_platform_captions(None, short_index)
    return caps, gen._build_youtube_description(short_index)


def _has(text: str, tag: str) -> bool:
    return (
        tag.lower() in text.lower().split()
        or tag.lower() in text.lower().replace("\n", " ").split()
    )


@pytest.mark.parametrize("location,place,region", CASES)
@pytest.mark.parametrize("short_index", range(1, 7))
def test_place_and_region_on_every_platform(location, place, region, short_index):
    caps, yt = _surfaces(_gen(location), short_index)
    for platform in ("facebook", "instagram", "tiktok"):
        text = caps[platform]
        assert _has(text, place), (platform, text)
        assert _has(text, region), (platform, text)
    assert _has(yt, place), yt
    assert _has(yt, region), yt
    assert len(yt) <= 200


@pytest.mark.parametrize("short_index", range(1, 7))
def test_youtube_description_orders_region_then_place_last(short_index):
    desc = _gen("Unicoi, Tennessee")._build_youtube_description(short_index)
    tags = [w for w in desc.split() if w.startswith("#")]
    assert tags[-2:] == ["#Tennessee", "#Unicoi"], desc
    assert tags[0] == "#Shorts"


def test_output_is_deterministic():
    a = _gen("Unicoi, Tennessee").generate_post(short_index=3)
    b = _gen("Unicoi, Tennessee").generate_post(short_index=3)
    assert a.captions == b.captions
    assert a.youtube_description == b.youtube_description


def test_hashtag_order_still_varies_between_shorts():
    orders = {
        tuple(
            w
            for w in _gen("Dingle, County Kerry, Ireland")
            ._build_platform_captions(None, i)["instagram"]
            .split()
            if w.startswith("#")
        )
        for i in range(1, 7)
    }
    assert len(orders) > 1


def test_tiktok_length_cut_drops_generic_tags_not_the_region():
    gen = _gen("Unicoi, Tennessee")
    gen.social_location = "Unicoi, Tennessee"
    post = gen.generate_post(short_index=1)
    tt = post.captions["tiktok"]
    assert len(tt) <= 150
    assert "#Tennessee" in tt and "#Unicoi" in tt


def test_long_base_is_shortened_on_a_word_boundary_before_a_tag_is_cut():
    gen = _gen("Unicoi, Tennessee")
    caption = "word " * 60
    post = gen._build_platform_captions(None, 1, hand_caption=caption.strip())
    tt = post["tiktok"]
    assert len(tt) <= 150
    assert "#Tennessee" in tt and "#Unicoi" in tt


def test_interactive_still_withholds_place_and_its_tags(tmp_path):
    import yaml

    f = tmp_path / "titles.yaml"
    f.write_text(yaml.safe_dump({1: {"title": "Somewhere", "interactive": True}}))
    post = _gen("Unicoi, Tennessee", titles_file=f).generate_post(short_index=1)
    for platform in ("facebook", "instagram", "tiktok"):
        assert "#Unicoi" not in post.captions[platform]
        assert "#Tennessee" not in post.captions[platform]
    assert "#Tennessee" not in post.youtube_description.split("\n\n")[0]


def test_gate_raises_when_a_caption_lacks_the_region(monkeypatch):
    gen = _gen("Unicoi, Tennessee")
    monkeypatch.setattr(
        SocialGenerator, "_platform_hashtags", lambda self, p, i, inter, limit: ["#Unicoi"]
    )
    with pytest.raises(ValueError, match="region"):
        gen.generate_post(short_index=1)


def test_single_component_location_is_not_gated():
    post = _gen("Unicoi").generate_post(short_index=1)
    assert post.captions["facebook"]


def test_plan_builder_youtube_title_carries_place_and_region():
    desc = _gen("Unicoi, Tennessee").generate_post(short_index=1).youtube_description
    record = _build(
        posts=[_post(youtube_title="Mar 4, 1st stretch: Unicoi", youtube_description=desc)]
    ).shorts[0]
    title = record.payloads.youtube.title
    assert title.startswith("Mar 4, 1st stretch: Unicoi #Unicoi #Tennessee"), title
