"""TDD tests for title uniqueness, description hashtags and recording-date format.

Title uniqueness breaks when series_part=None across invocations
Description hashtags hardcoded instead of from tags list
Recording date format not normalized (YYYY-MM-DD -> Mon YYYY)
"""

import re
from unittest.mock import patch

import pytest

from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator
from gomoveshift.pipeline.social_generator import SocialGenerator


# ── Title uniqueness with series_part=None ─────────────────────


class TestTitleUniqueness:
    """series_part=None must not produce duplicate titles across invocations."""

    def test_different_series_parts_produce_different_titles(self):
        """Two generators with same project but different series_part get different titles."""
        # road_name + county widen the applicable-frame set to 5, giving the
        # per-series shuffle enough permutations that the two series parts'
        # frame orders reliably diverge.
        gen1 = SocialGenerator(
            location="County Kerry, Ireland",
            project_name="kerry_coastal_drive",
            series_part=1,
            road_name="R569",
            county="County Kerry",
        )
        gen2 = SocialGenerator(
            location="County Kerry, Ireland",
            project_name="kerry_coastal_drive",
            series_part=2,
            road_name="R569",
            county="County Kerry",
        )
        posts1 = gen1.generate_all(num_shorts=5)
        posts2 = gen2.generate_all(num_shorts=5)
        titles1 = [p.youtube_title for p in posts1]
        titles2 = [p.youtube_title for p in posts2]
        # The title ORDER should differ between series parts
        matching = sum(1 for a, b in zip(titles1, titles2) if a == b)
        assert matching < len(titles1), (
            f"All {len(titles1)} titles identical between series_part=1 and series_part=2"
        )

    def test_series_part_none_raises_when_batch_generated(self):
        """generate_all() with series_part=None must raise ValueError.

        When series_part is None, title uniqueness across invocations
        cannot be guaranteed, so we require it to be explicitly set.
        """
        gen = SocialGenerator(
            location="County Kerry, Ireland",
            project_name="kerry_coastal_drive",
            series_part=None,
        )
        with pytest.raises(ValueError, match="series_part"):
            gen.generate_all(num_shorts=5)

    def test_series_part_zero_is_valid(self):
        """series_part=0 is a valid value (distinct from None)."""
        gen = SocialGenerator(
            location="County Kerry, Ireland",
            project_name="kerry_coastal_drive",
            series_part=0,
        )
        posts = gen.generate_all(num_shorts=3)
        assert len(posts) == 3


# ── Description hashtags from tags list ─────────────────────────


class TestDescriptionHashtags:
    """Description hashtags must be derived from the tags list, not hardcoded."""

    @pytest.fixture
    def generator(self):
        """Create generator with mocked SEO manager."""
        with patch.object(LongFormMetadataGenerator, "_init_seo_manager"):
            with patch.object(LongFormMetadataGenerator, "_init_ledger"):
                gen = LongFormMetadataGenerator()
                gen._seo_degraded = True
                return gen

    def test_description_hashtags_are_from_tags(self, generator):
        """Hashtags in description must be derived from GMS_BASE_TAGS."""
        from gomoveshift.pipeline.longform_metadata import GMS_BASE_TAGS

        desc = generator._build_gms_description(
            location="County Kerry, Ireland",
            seo_description="POV drive through Kerry.",
            duration_str="1h 23m",
            publish_day="Saturday",
        )
        # Extract hashtags from the description
        hashtag_line = ""
        for line in desc.split("\n"):
            if line.strip().startswith("#"):
                hashtag_line = line.strip()
                break

        assert hashtag_line, "No hashtag line found in description"

        # Extract individual hashtags
        hashtags_in_desc = re.findall(r"#(\w+)", hashtag_line)
        assert len(hashtags_in_desc) >= 5, (
            f"Expected at least 5 hashtags, got {len(hashtags_in_desc)}"
        )

        # Build expected hashtag forms from tags
        # Tags like "POV driving" -> "POVDriving", "scenic drive" -> "ScenicDrive"
        def tag_to_hashtag(tag):
            return "".join(word.capitalize() for word in tag.split())

        valid_hashtag_sources = {tag_to_hashtag(t) for t in GMS_BASE_TAGS}
        # Location and country are also valid
        valid_hashtag_sources.add("CountyKerry")
        valid_hashtag_sources.add("Ireland")

        # All hashtags must be derived from known sources (tags, location, country)
        for ht in hashtags_in_desc:
            assert ht in valid_hashtag_sources, (
                f"Hashtag #{ht} not derived from tags list. "
                f"Valid sources: {sorted(valid_hashtag_sources)}"
            )

    def test_no_hardcoded_relaxing_hashtag(self, generator):
        """#Relaxing must not appear unless 'relaxing drive' is in GMS_BASE_TAGS."""
        from gomoveshift.pipeline.longform_metadata import GMS_BASE_TAGS

        has_relaxing_tag = any("relaxing" in t.lower() for t in GMS_BASE_TAGS)

        desc = generator._build_gms_description(
            location="County Kerry, Ireland",
            seo_description="POV drive.",
            duration_str="30m",
            publish_day="Wednesday",
        )
        hashtags_in_desc = re.findall(r"#(\w+)", desc)
        hashtag_names_lower = [h.lower() for h in hashtags_in_desc]

        if not has_relaxing_tag:
            assert "relaxing" not in hashtag_names_lower, (
                "#Relaxing in description but 'relaxing' not in GMS_BASE_TAGS"
            )


# ── Recording date format normalization ─────────────────────────


class TestDateFormatNormalization:
    """format_recording_date() normalizes YYYY-MM-DD to Mon YYYY."""

    def test_iso_date_to_month_year(self):
        """YYYY-MM-DD -> 'Mon YYYY'."""
        from gomoveshift.pipeline.social_generator import format_recording_date

        assert format_recording_date("2025-09-15") == "Sep 2025"

    def test_already_formatted_passthrough(self):
        """Already-formatted 'Mon YYYY' passes through unchanged."""
        from gomoveshift.pipeline.social_generator import format_recording_date

        assert format_recording_date("Sep 2025") == "Sep 2025"

    def test_empty_string_passthrough(self):
        """Empty string returns empty string."""
        from gomoveshift.pipeline.social_generator import format_recording_date

        assert format_recording_date("") == ""

    def test_none_returns_empty(self):
        """None returns empty string."""
        from gomoveshift.pipeline.social_generator import format_recording_date

        assert format_recording_date(None) == ""

    def test_full_date_format(self):
        """Full date like 'September 10, 2025' -> 'Sep 2025'."""
        from gomoveshift.pipeline.social_generator import format_recording_date

        assert format_recording_date("September 10, 2025") == "Sep 2025"

    def test_social_generator_normalizes_recording_date(self):
        """SocialGenerator auto-normalizes YYYY-MM-DD recording_date."""
        gen = SocialGenerator(
            location="County Kerry, Ireland",
            project_name="kerry_01",
            recording_date="2025-09-15",
            series_part=1,
        )
        post = gen.generate_post(1)
        assert "📹 Sep 2025" in post.caption
        assert "2025-09-15" not in post.caption
