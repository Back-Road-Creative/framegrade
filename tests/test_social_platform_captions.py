"""TDD for per-platform captions with a mandatory fact slot.

See docs/pipeline/135-shorts-titles-and-interactive-family.md
"Per-platform captions".
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from gomoveshift.pipeline.social_generator import SocialGenerator, SocialPost


def _load_sibling_fixtures():
    """``tests/`` is not a package — load the plan_builder test's `_build`/
    `_post` fixtures by path instead of ``from tests.xxx import``."""
    sibling = Path(__file__).resolve().parent / "test_publishing_plan_builder.py"
    spec = importlib.util.spec_from_file_location("_gms_plan_builder_fixtures", sibling)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._build, mod._post


_build, _post = _load_sibling_fixtures()


def _generator(**kwargs):
    return SocialGenerator(
        location="County Cork, Ireland",
        project_name="a2_platform_captions",
        series_part=1,
        road_name="R572",
        county="County Cork",
        **kwargs,
    )


class TestPlatformDistinctness:
    def test_youtube_facebook_instagram_tiktok_all_differ(self):
        post = _generator().generate_post(short_index=1)
        texts = {post.captions[p] for p in ("youtube", "facebook", "instagram", "tiktok")}
        assert len(texts) == 4, post.captions

    def test_place_present_in_every_composed_caption(self):
        gen = _generator()
        post = gen.generate_post(short_index=1)
        for platform in ("youtube", "facebook", "instagram", "tiktok"):
            assert gen.social_location in post.captions[platform], (
                platform,
                post.captions[platform],
            )

    def test_interactive_withholds_place_from_every_caption(self, tmp_path):
        import yaml

        titles_file = tmp_path / "titles.yaml"
        titles_file.write_text(
            yaml.safe_dump({1: {"title": "Somewhere scenic", "interactive": True}})
        )
        gen = _generator(titles_file=titles_file)
        post = gen.generate_post(short_index=1)
        assert post.interactive is True
        for platform in ("youtube", "facebook", "instagram", "tiktok"):
            assert gen.social_location not in post.captions[platform], (
                platform,
                post.captions[platform],
            )


class TestHashtagCounts:
    def test_youtube_has_no_hashtag_wall(self):
        post = _generator().generate_post(short_index=1)
        assert post.captions["youtube"].count("#") <= 2

    def test_facebook_at_most_two_hashtags(self):
        post = _generator().generate_post(short_index=1)
        assert post.captions["facebook"].count("#") <= 2

    def test_instagram_at_most_eight_hashtags(self):
        post = _generator().generate_post(short_index=1)
        assert post.captions["instagram"].count("#") <= 8

    def test_tiktok_at_most_four_hashtags_and_150_chars(self):
        post = _generator().generate_post(short_index=1)
        assert post.captions["tiktok"].count("#") <= 4
        assert len(post.captions["tiktok"]) <= 150


class TestHandCaption:
    def test_hand_caption_becomes_youtube_caption_verbatim(self, tmp_path):
        import yaml

        titles_file = tmp_path / "titles.yaml"
        titles_file.write_text(
            yaml.safe_dump({1: {"title": "A hand title", "caption": "A hand-authored caption."}})
        )
        gen = _generator(titles_file=titles_file)
        post = gen.generate_post(short_index=1)
        assert post.captions["youtube"] == "A hand-authored caption."
        # Other platforms compose it as the opening sentence, still distinct.
        assert post.captions["facebook"].startswith("A hand-authored caption.")
        assert post.captions["instagram"].startswith("A hand-authored caption.")
        assert post.captions["tiktok"].startswith("A hand-authored caption.")
        texts = {post.captions[p] for p in ("youtube", "facebook", "instagram", "tiktok")}
        assert len(texts) == 4


class TestToDictAndDeterminism:
    def test_to_dict_includes_captions(self):
        post = _generator().generate_post(short_index=1)
        d = post.to_dict()
        assert d["captions"] == post.captions
        assert d["caption"] == post.captions["instagram"]

    def test_legacy_dict_without_captions_still_builds(self):
        post = SocialPost(
            short_index=1,
            caption="legacy caption",
            youtube_title="Legacy Title",
            youtube_tags="tag1,tag2",
            youtube_category="19",
            tiktok_title="Legacy TT",
            pinterest_board="board",
            first_comment="#legacy",
        )
        assert post.captions == {}
        d = post.to_dict()
        assert d["captions"] == {}
        assert d["caption"] == "legacy caption"

    def test_deterministic_per_project_and_short_index(self):
        post_a = _generator().generate_post(short_index=2)
        post_b = _generator().generate_post(short_index=2)
        assert post_a.captions == post_b.captions


class TestPlanBuilderMapsPerPlatform:
    def test_plan_builder_reads_each_platforms_own_caption(self):
        post = _post(
            caption="legacy fallback text",
            captions={
                "youtube": "YT caption",
                "facebook": "FB caption",
                "instagram": "IG caption",
                "tiktok": "TT caption",
            },
        )
        record = _build(posts=[post]).shorts[0]
        assert record.payloads.instagram.caption.startswith("IG caption")
        assert record.payloads.facebook.description.startswith("FB caption")
        assert record.payloads.tiktok.caption.startswith("TT caption")

    def test_plan_builder_falls_back_to_legacy_caption_without_captions_dict(self):
        post = _post(caption="only the legacy column")
        record = _build(posts=[post]).shorts[0]
        assert record.payloads.instagram.caption.startswith("only the legacy column")
        assert record.payloads.facebook.description.startswith("only the legacy column")
        assert record.payloads.tiktok.caption.startswith("only the legacy column")
