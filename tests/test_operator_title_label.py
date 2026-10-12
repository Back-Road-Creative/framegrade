"""An operator ``--per-lf-location`` TITLE must not leak into place nouns.

``--per-lf-location '1=To Clonakilty, County Cork'`` is an OST-burn title, not
a place noun. ``parse_lf_label`` splits it into ``display`` (verbatim, for the
OST burn / video title) and ``place`` (leading title word stripped) so every
other noun slot — description hook, shorts titles, tags, hashtags,
``social_location`` — reads ``place``, never ``display``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from gomoveshift.pipeline._series_utils import is_title_prefixed, parse_lf_label
from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator, _location_tags
from gomoveshift.pipeline.social_generator import SocialGenerator, _location_hashtags
from gomoveshift.pipeline.validation.content_gates import (
    _GRAMMAR_LEADING_PREP_RE,
    _label_quality_diagnosis,
    find_grammar_issues,
)

# The full curated leading-word set (union of every word either the old
# label-quality regex or the old grammar regex recognized) -- the single
# list every gate must agree with via `is_title_prefixed`.
CURATED_TITLE_PREFIXES = [
    "To",
    "Through",
    "Around",
    "Toward",
    "Towards",
    "Up around",
    "The road to",
]


class TestParseLfLabel:
    def test_leading_to_stripped_into_place(self) -> None:
        parsed = parse_lf_label("To Clonakilty, County Cork")
        assert parsed.display == "To Clonakilty, County Cork"
        assert parsed.place == "Clonakilty, County Cork"
        assert parsed.region == "County Cork"

    @pytest.mark.parametrize(
        "prefix",
        ["Through", "Up around", "Around", "The road to", "Toward", "Towards", "to"],
    )
    def test_every_curated_prefix_stripped(self, prefix: str) -> None:
        parsed = parse_lf_label(f"{prefix} Mizen Head, County Cork")
        assert parsed.place == "Mizen Head, County Cork"

    def test_plain_label_is_unchanged(self) -> None:
        parsed = parse_lf_label("Clonakilty, County Cork")
        assert parsed.display == parsed.place == "Clonakilty, County Cork"
        assert parsed.region == "County Cork"


class TestIsTitlePrefixedSingleSource:
    """The curated leading-word list lives in one place (``_TITLE_PREFIX_RE``
    via ``is_title_prefixed``); the label-quality gate and the grammar gate
    must never disagree with it about what counts as a title-prefixed
    label -- that drift is exactly what let the same prefix-word bug
    resurface across both gates."""

    @pytest.mark.parametrize("prefix", CURATED_TITLE_PREFIXES)
    def test_helper_recognizes_every_curated_word(self, prefix: str) -> None:
        assert is_title_prefixed(f"{prefix} Goleen, County Cork")

    def test_helper_rejects_plain_label(self) -> None:
        assert not is_title_prefixed("Goleen, County Cork")

    @pytest.mark.parametrize("prefix", CURATED_TITLE_PREFIXES)
    def test_label_quality_gate_agrees_with_helper(self, prefix: str) -> None:
        metadata = {
            "location": f"{prefix} Goleen, County Cork",
            "resolved_location": {"ordered_place_sequence": ["Goleen", "County Cork"]},
        }
        diagnosis = _label_quality_diagnosis(metadata)
        assert diagnosis is not None
        kind, _settlements = diagnosis
        assert kind == "title"

    @pytest.mark.parametrize("prefix", CURATED_TITLE_PREFIXES)
    def test_grammar_gate_agrees_with_helper(self, prefix: str) -> None:
        issues = find_grammar_issues(f"{prefix} Goleen, County Cork")
        assert any("bare preposition" in issue for issue in issues)

    def test_grammar_leading_prep_re_stays_importable(self) -> None:
        """Deprecated compat alias for ``cli.parser._validate_per_lf_location_args``,
        which imports ``_GRAMMAR_LEADING_PREP_RE`` from ``content_gates`` directly.
        Remove once that call site switches to ``is_title_prefixed``."""
        assert _GRAMMAR_LEADING_PREP_RE.match("To Goleen, County Cork")
        assert not _GRAMMAR_LEADING_PREP_RE.match("Goleen, County Cork")


@pytest.fixture
def generator():
    with patch.object(LongFormMetadataGenerator, "_init_seo_manager"):
        with patch.object(LongFormMetadataGenerator, "_init_ledger"):
            gen = LongFormMetadataGenerator()
            gen._seo_degraded = True
            return gen


class TestHookReadsPlaceNotDisplay:
    def test_hook_drops_operator_title_word(self, generator) -> None:
        desc = generator._build_gms_description(
            location="To Clonakilty, County Cork",
            seo_description="POV drive.",
            duration_str="12m 50s",
            publish_day="Sunday",
            trim_ratio=0.9,
        )
        assert "through Clonakilty, County Cork" in desc
        assert "To Clonakilty" not in desc

    def test_plain_label_hook_unchanged(self, generator) -> None:
        desc = generator._build_gms_description(
            location="Clonakilty, County Cork",
            seo_description="POV drive.",
            duration_str="12m 50s",
            publish_day="Sunday",
            trim_ratio=0.9,
        )
        assert "through Clonakilty, County Cork" in desc


class TestLocationTagsReadPlace:
    def test_tags_never_carry_operator_title_word(self) -> None:
        tags = _location_tags("To Clonakilty, County Cork", recording_date=None)
        assert "Clonakilty" in tags
        assert not any(t.startswith("To ") for t in tags)

    def test_plain_label_tags_unchanged(self) -> None:
        tags = _location_tags("Clonakilty, County Cork", recording_date=None)
        assert "Clonakilty" in tags


class TestShortsTitleAndHashtagReadPlace:
    def test_shorts_title_drops_operator_title_word(self) -> None:
        gen = SocialGenerator(location="To Clonakilty, County Cork", project_name="proj")
        assert gen.location_short == "Clonakilty"
        post = gen.generate_post(short_index=1)
        assert "To Clonakilty" not in post.youtube_title
        assert "To Clonakilty" not in post.caption

    def test_no_smashed_to_hashtag(self) -> None:
        assert "#ToClonakilty" not in _location_hashtags("To Clonakilty, County Cork")
        assert "#Clonakilty" in _location_hashtags("To Clonakilty, County Cork")

    def test_social_location_drops_operator_title_word(self) -> None:
        gen = SocialGenerator(location="To Clonakilty, County Cork", project_name="proj")
        assert gen.social_location == "Clonakilty"

    def test_plain_label_location_short_unchanged(self) -> None:
        gen = SocialGenerator(location="Clonakilty, County Cork", project_name="proj")
        assert gen.location_short == "Clonakilty"

    def test_shorts_description_drops_operator_title_word(self) -> None:
        """``location_short_full`` (feeds ``social_location_full`` ->
        the shorts ``youtube_description``) kept the WHOLE first
        component verbatim, so an operator title like "To Clonakilty"
        leaked as "POV driving through To Clonakilty" — a leading verb
        read as a place noun. Only the leading curated title word is
        stripped; a mid-string arc ("A to B") is untouched."""
        gen = SocialGenerator(location="To Clonakilty, County Cork", project_name="proj")
        assert gen.location_short_full == "Clonakilty"
        post = gen.generate_post(short_index=1)
        assert "To Clonakilty" not in post.youtube_description
