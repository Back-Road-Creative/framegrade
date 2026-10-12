"""TDD for the interactive ("where is this?") shorts family.

See docs/pipeline/135-shorts-titles-and-interactive-family.md "Interactive family, resolved".
Covers what test_social_quality.py / test_shorts_yt_description_category.py /
test_phase_d.py / test_social_generator_safety.py don't get inline twins for
(diff-cap): the place-absence/presence contract, the validator carve-out, the
hand-title leak guard, and the burned-label guard.
"""

from __future__ import annotations

import pytest
import yaml

from gomoveshift.pipeline.content_validator import ContentValidator
from gomoveshift.pipeline.social_generator import (
    InteractiveShortHasBurnedLabel,
    SocialGenerator,
)


def _write_titles_file(tmp_path, mapping):
    path = tmp_path / "titles.yaml"
    path.write_text(yaml.safe_dump(mapping))
    return path


def _generator(**kwargs):
    return SocialGenerator(
        location="County Kerry, Ireland",
        project_name="a3_interactive",
        series_part=1,
        county="County Kerry",
        **kwargs,
    )


class TestPlaceAbsentFromDiscoverySurfaces:
    def test_title_tags_captions_never_name_the_place(self, tmp_path):
        titles_file = _write_titles_file(
            tmp_path, {1: {"title": "Can you guess this road?", "interactive": True}}
        )
        gen = _generator(titles_file=titles_file)
        post = gen.generate_post(1)

        place_strings = [
            gen.location_short,
            gen.location_short_full,
            gen.social_location,
            gen.county,
        ]
        surfaces = [
            post.youtube_title,
            post.tiktok_title,
            post.youtube_tags,
            post.caption,
            *post.captions.values(),
        ]
        for surface in surfaces:
            for place in place_strings:
                assert place.lower() not in surface.lower(), (place, surface)

    def test_place_present_in_description_last_line(self, tmp_path):
        titles_file = _write_titles_file(
            tmp_path, {1: {"title": "Can you guess this road?", "interactive": True}}
        )
        gen = _generator(titles_file=titles_file)
        post = gen.generate_post(1)
        last_line = post.youtube_description.strip().splitlines()[-1]
        assert last_line == f"\U0001f4cd {gen.social_location_full}"

    def test_place_named_first_in_first_comment(self, tmp_path):
        titles_file = _write_titles_file(
            tmp_path, {1: {"title": "Can you guess this road?", "interactive": True}}
        )
        gen = _generator(titles_file=titles_file)
        post = gen.generate_post(1)
        first_line = post.first_comment.strip().splitlines()[0]
        assert first_line == f"\U0001f4cd {gen.social_location_full}"

    def test_non_interactive_hand_title_unaffected(self, tmp_path):
        titles_file = _write_titles_file(tmp_path, {1: {"title": "A drive through Kerry"}})
        gen = _generator(titles_file=titles_file)
        post = gen.generate_post(1)
        assert post.interactive is False
        assert gen.county.lower() in post.youtube_tags.lower() or gen.location_short.lower() in (
            post.youtube_tags.lower()
        )


class TestHandTitleLeakGuard:
    def test_hand_title_naming_the_place_raises(self, tmp_path):
        titles_file = _write_titles_file(
            tmp_path, {1: {"title": "Driving through County Kerry today", "interactive": True}}
        )
        gen = _generator(titles_file=titles_file)
        with pytest.raises(ValueError, match="leaks the place"):
            gen.generate_post(1)

    def test_hand_caption_naming_the_place_raises(self, tmp_path):
        titles_file = _write_titles_file(
            tmp_path,
            {
                1: {
                    "title": "Guess this road",
                    "caption": "A quiet drive through County Kerry.",
                    "interactive": True,
                }
            },
        )
        gen = _generator(titles_file=titles_file)
        with pytest.raises(ValueError, match="leaks the place"):
            gen.generate_post(1)


class TestBurnedLabelGuard:
    def test_interactive_short_with_burned_label_raises(self, tmp_path):
        titles_file = _write_titles_file(
            tmp_path, {1: {"title": "Guess this road", "interactive": True}}
        )
        gen = _generator(titles_file=titles_file)
        with pytest.raises(InteractiveShortHasBurnedLabel):
            gen.generate_all(num_shorts=1, per_short_locations=["County Kerry, Ireland"])

    def test_interactive_short_with_empty_label_is_fine(self, tmp_path):
        titles_file = _write_titles_file(
            tmp_path, {1: {"title": "Guess this road", "interactive": True}}
        )
        gen = _generator(titles_file=titles_file)
        posts = gen.generate_all(num_shorts=1, per_short_locations=[""])
        assert posts[0].interactive is True

    def test_non_interactive_short_with_burned_label_is_unaffected(self, tmp_path):
        titles_file = _write_titles_file(tmp_path, {1: {"title": "A drive through Kerry"}})
        gen = _generator(titles_file=titles_file)
        posts = gen.generate_all(num_shorts=1, per_short_locations=["County Kerry, Ireland"])
        assert posts[0].interactive is False


class TestValidatorCarveOut:
    def test_interactive_skips_self_contradiction_pattern(self):
        validator = ContentValidator()
        result = validator.validate("Where is this? \U0001f914", interactive=True)
        assert result.valid, result.violations

    def test_non_interactive_still_blocks_self_contradiction_pattern(self):
        validator = ContentValidator()
        result = validator.validate("Where is this? \U0001f914", interactive=False)
        assert not result.valid

    def test_interactive_still_blocks_every_other_rule(self):
        validator = ContentValidator()
        result = validator.validate("turn up the volume", interactive=True)
        assert not result.valid

    def test_enforce_threads_interactive_through_validate_all(self):
        validator = ContentValidator()
        # Should not raise — the self-contradiction pattern is the only
        # thing present, and interactive=True carve-out swallows it.
        validator.enforce(titles=["Where is this?"], interactive=True)


class TestToDictCarriesInteractive:
    def test_to_dict_includes_interactive_field(self, tmp_path):
        titles_file = _write_titles_file(
            tmp_path, {1: {"title": "Guess this road", "interactive": True}}
        )
        gen = _generator(titles_file=titles_file)
        post = gen.generate_post(1)
        d = post.to_dict()
        assert d["interactive"] is True

    def test_to_dict_false_for_non_interactive(self, tmp_path):
        titles_file = _write_titles_file(tmp_path, {1: {"title": "A drive through Kerry"}})
        gen = _generator(titles_file=titles_file)
        post = gen.generate_post(1)
        assert post.to_dict()["interactive"] is False
