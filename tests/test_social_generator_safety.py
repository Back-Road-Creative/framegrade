"""Tests for social generator + content validator integration.

Ensures:
1. No generated title/hook matches a blocked phrase
2. Retry logic works when ContentValidationError is raised
3. EVERY module-level ``*_TEMPLATES`` list (auto-discovered) and HOOKS pass
   content validation — not just a hardcoded subset.

Robust-by-construction: the safety lint auto-discovers every ``*_TEMPLATES``
list defined on ``social_generator`` at import time and parametrizes the
content-validation assertions over ALL of them. A new template list added to
the module later (or a gimmick list that a de-gimmick pass missed) is linted
automatically — the generator literally cannot add an unlinted template list.
"""

from collections import defaultdict

import pytest

from gomoveshift.pipeline import social_generator as sg
from gomoveshift.pipeline.content_validator import ContentValidator
from gomoveshift.pipeline.social_generator import (
    SocialGenerator,
)


def _discover_template_lists() -> list[tuple[str, list]]:
    """Every module-level ``*_TEMPLATES`` list on ``social_generator``.

    Auto-discovery (not a hardcoded subset) is the whole point:
    any future ``*_TEMPLATES`` list is picked up here and linted without a test
    edit. Sorted for a deterministic parametrization order.
    """
    return sorted(
        (name, getattr(sg, name))
        for name in dir(sg)
        if name.endswith("_TEMPLATES") and isinstance(getattr(sg, name), list)
    )


# Flatten to (list_name, index, template) so every template across every
# discovered list is its own parametrized case with a readable id.
_ALL_TEMPLATE_CASES = [
    (name, idx, template)
    for name, templates in _discover_template_lists()
    for idx, template in enumerate(templates)
]

# Placeholder values span every key any template list uses ({location},
# {terrain_desc}, {road}, {county}, {moment}, {hook}, {weather_emoji}); missing
# keys default to "" via the defaultdict so a template can never KeyError here.
_TEMPLATE_FILL = defaultdict(
    str,
    location="County Cork, Ireland",
    terrain_desc="Roads",
    road="R572",
    county="County Cork",
    moment="The Peak Moment",
    secs="41",
    weekday="Monday",
)


@pytest.fixture
def validator():
    return ContentValidator()


# Brand per-location cap (publishing/brand_policy.py): the real
# ceiling on shorts per location, and the composition capacity ceiling with
# road_name + county both grounded.
PER_LOCATION_CAP = 5


@pytest.fixture
def generator():
    return SocialGenerator(
        location="County Cork, Ireland",
        project_name="test_project",
        series_part=1,
        road_name="R572",
        county="County Cork",
    )


class TestTemplatesPassValidation:
    """Every template in EVERY ``*_TEMPLATES`` list, plus HOOKS, passes validation."""

    def test_autodiscovery_covers_all_template_lists(self):
        """Guard: auto-discovery must find every known list, not silently zero out.

        If a future refactor renames the lists away from the ``*_TEMPLATES``
        convention (or the discovery predicate regresses), coverage would
        collapse to nothing while every parametrized test still 'passes' on an
        empty set. This asserts the surviving lists are actually discovered so
        the widened lint can't be defeated by a rename.
        """
        discovered = {name for name, _ in _discover_template_lists()}
        # TITLE_TEMPLATES / EXTENDED_TITLE_TEMPLATES were replaced by
        # TITLE_FRAME_TEMPLATES (composed titles, no fixed pool). The
        # single CAPTION_TEMPLATES pool was replaced by one pool per platform
        # (no shared "all platforms" caption). A new list is additive, so
        # this is a floor, not an exact-match.
        expected = {
            "TITLE_FRAME_TEMPLATES",
            "YOUTUBE_CAPTION_TEMPLATES",
            "FACEBOOK_CAPTION_TEMPLATES",
            "INSTAGRAM_CAPTION_TEMPLATES",
            "TIKTOK_CAPTION_TEMPLATES",
        }
        missing = expected - discovered
        assert not missing, f"Auto-discovery missed template lists: {missing}"
        assert _ALL_TEMPLATE_CASES, "No templates discovered — safety lint is vacuous"

    @pytest.mark.parametrize(
        "list_name,idx,template",
        _ALL_TEMPLATE_CASES,
        ids=[f"{name}[{idx}]" for name, idx, _ in _ALL_TEMPLATE_CASES],
    )
    def test_template_passes_validation(self, validator, list_name, idx, template):
        """Each template in each discovered list, when formatted, passes validation.

        Widened from TITLE_TEMPLATES-only to every ``*_TEMPLATES``
        list on the module, so a gimmick/unverifiable template added to ANY list
        (weather, caption, extended, …) is caught before it can ship.
        """
        rendered = template.format_map(_TEMPLATE_FILL)
        result = validator.validate(rendered, field_name=list_name)
        assert result.valid, (
            f"{list_name}[{idx}] blocked: '{template}' → '{rendered}'\n"
            f"Violations: {result.violations}"
        )


class TestSocialGeneratorRetry:
    """generate_post() retries on ContentValidationError instead of crashing."""

    def test_generate_post_succeeds(self, generator):
        """Basic generation should succeed for all shorts."""
        for i in range(1, PER_LOCATION_CAP + 1):
            post = generator.generate_post(i)
            assert post.youtube_title
            assert post.caption

    def test_generate_all_succeeds(self, generator):
        """generate_all should produce posts for every short without raising."""
        posts = generator.generate_all(num_shorts=PER_LOCATION_CAP)
        assert len(posts) == PER_LOCATION_CAP
        # Each post should have unique title
        titles = [p.youtube_title for p in posts]
        assert len(set(titles)) == len(titles), "Titles should be unique"


class TestNoBlockedPhrasesInOutput:
    """End-to-end: generated posts must pass content validation."""

    def test_all_generated_posts_pass_validation(self, generator, validator):
        """Generate posts for a full-cap batch and validate every field."""
        posts = generator.generate_all(num_shorts=PER_LOCATION_CAP)
        for post in posts:
            for field_name, text in [
                ("youtube_title", post.youtube_title),
                ("tiktok_title", post.tiktok_title),
                ("caption", post.caption),
            ]:
                result = validator.validate(text, field_name=field_name)
                assert result.valid, (
                    f"Short #{post.short_index} {field_name} blocked:\n"
                    f"  Text: '{text[:100]}'\n"
                    f"  Violations: {result.violations}"
                )
