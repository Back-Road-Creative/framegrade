"""Tests for social post quality — diversity, truncation, tags, hashtags.

TDD: Tests written BEFORE implementation fixes.
"""

from collections import Counter

import pytest

from gomoveshift.pipeline.content_validator import ContentValidator
from gomoveshift.pipeline.social_generator import (
    FACEBOOK_CAPTION_TEMPLATES,
    INSTAGRAM_CAPTION_TEMPLATES,
    TIKTOK_CAPTION_TEMPLATES,
    TITLE_FRAME_TEMPLATES,
    YOUTUBE_CAPTION_TEMPLATES,
    SocialGenerator,
    TitleCapacityExceeded,
)

_ALL_PLATFORM_CAPTION_TEMPLATES = {
    "youtube": YOUTUBE_CAPTION_TEMPLATES,
    "facebook": FACEBOOK_CAPTION_TEMPLATES,
    "instagram": INSTAGRAM_CAPTION_TEMPLATES,
    "tiktok": TIKTOK_CAPTION_TEMPLATES,
}

# Brand per-location cap (publishing/brand_policy.py): the real
# ceiling on shorts per location. Batch tests below size to this, not an
# arbitrary round number.
PER_LOCATION_CAP = 5


@pytest.fixture
def generator():
    # road_name + county widen the applicable-frame set to PER_LOCATION_CAP
    # (5) distinct composed titles — the same fact-availability a real
    # drive with road/county metadata has.
    return SocialGenerator(
        location="Sheepshead Peninsula to Durrus, County Cork, Ireland",
        project_name="sheepshead_durrus_01",
        recording_date="Sep 2025",
        series_part=1,
        road_name="R572",
        county="County Cork",
    )


class TestCaptionDiversity:
    """Captions should be unique across all shorts in a batch."""

    def test_no_duplicate_captions_at_cap(self, generator):
        """A batch at the per-location cap should have that many unique captions."""
        posts = generator.generate_all(num_shorts=PER_LOCATION_CAP)
        captions = [p.caption for p in posts]
        assert len(set(captions)) == PER_LOCATION_CAP, (
            f"Expected {PER_LOCATION_CAP} unique captions, got {len(set(captions))}"
        )


class TestCaptionTruncation:
    """No caption should be truncated with '...' — templates must fit."""

    def test_no_truncated_captions(self, generator):
        """No caption should end with '...' from truncation."""
        posts = generator.generate_all(num_shorts=PER_LOCATION_CAP)
        for p in posts:
            assert not p.caption.endswith("..."), (
                f"Short #{p.short_index} caption truncated: ...{p.caption[-50:]}"
            )

    def test_all_captions_under_280_chars(self, generator):
        """All captions must fit within 280-char Twitter limit."""
        posts = generator.generate_all(num_shorts=PER_LOCATION_CAP)
        for p in posts:
            assert len(p.caption) <= 280, (
                f"Short #{p.short_index} caption is {len(p.caption)} chars (max 280)"
            )


class TestTagDiversity:
    """Tags and first comments should vary across shorts."""

    def test_tags_vary_across_shorts(self, generator):
        """Tags should not be identical across all shorts."""
        posts = generator.generate_all(num_shorts=PER_LOCATION_CAP)
        tag_sets = [p.youtube_tags for p in posts]
        unique_tags = len(set(tag_sets))
        # At least 3 unique tag sets across a full-cap batch
        assert unique_tags >= 3, f"Expected at least 3 unique tag sets, got {unique_tags}"

    def test_first_comment_varies_across_shorts(self, generator):
        """First comment hashtags should vary across shorts."""
        posts = generator.generate_all(num_shorts=PER_LOCATION_CAP)
        comments = [p.first_comment for p in posts]
        unique_comments = len(set(comments))
        # At least 3 unique first comments across a full-cap batch
        assert unique_comments >= 3, (
            f"Expected at least 3 unique first comments, got {unique_comments}"
        )

    def test_tags_include_location(self, generator):
        """Every tag set should include location-related tags."""
        posts = generator.generate_all(num_shorts=5)
        for p in posts:
            assert "Sheepshead" in p.youtube_tags, (
                f"Short #{p.short_index} tags missing location: {p.youtube_tags}"
            )


def _leading_words(text: str, n: int = 3) -> str:
    return " ".join(text.split()[:n]).lower()


class TestTitleDiversity:
    """Titles should all be unique within a batch, and never near-duplicates."""

    def test_all_titles_unique(self, generator):
        """A batch at the per-location cap must produce that many unique titles."""
        posts = generator.generate_all(num_shorts=PER_LOCATION_CAP)
        titles = [p.youtube_title for p in posts]
        assert len(set(titles)) == len(titles), (
            f"Expected {len(titles)} unique titles, got {len(set(titles))}"
        )

    def test_no_shared_three_word_prefix(self, generator):
        """No two composed titles in a batch may share a 3-word prefix
        (case-insensitive) — that shape is the templated-spam signal the
        anti-spam gate refuses at enqueue."""
        posts = generator.generate_all(num_shorts=PER_LOCATION_CAP)
        prefixes = [_leading_words(p.youtube_title) for p in posts]
        assert len(set(prefixes)) == len(prefixes), (
            f"Two or more composed titles share a 3-word prefix: {prefixes}"
        )

    def test_location_and_terrain_alone_compose_at_least_three_distinct_titles(self):
        """With only the always-available slots (location, terrain), composition
        must still produce >=3 distinct titles with no shared 3-word prefix."""
        gen = SocialGenerator(
            location="County Kerry, Ireland",
            project_name="kerry_bare",
            series_part=1,
        )
        posts = gen.generate_all(num_shorts=3)
        titles = [p.youtube_title for p in posts]
        assert len(set(titles)) == 3, f"Expected 3 distinct titles, got {titles}"
        prefixes = [_leading_words(t) for t in titles]
        assert len(set(prefixes)) == 3, f"Shared 3-word prefix among: {titles}"

    def test_batch_over_capacity_raises_title_capacity_exceeded(self, generator):
        """A batch asking for more distinct titles than the grounded facts
        can compose must raise, naming the offending index — never emit a
        near-duplicate to pad the batch out."""
        with pytest.raises(TitleCapacityExceeded) as exc_info:
            generator.generate_all(num_shorts=PER_LOCATION_CAP + 1)
        message = str(exc_info.value)
        assert str(PER_LOCATION_CAP + 1) in message


class TestTemplatePoolSize:
    """Template pools must be large enough to avoid repetition."""

    def test_caption_opener_pools_cover_a_twelve_short_batch(self):
        """Each platform's opener pool holds >=12 entries, none repeated, and
        every opener carries the mandatory {location} slot: a caption is an
        opener composed with a detail/prompt part (never a whole fixed
        template), and 12 distinct openers keep a 12-short batch from
        repeating one (the anti-spam gate's 6-word-prefix rule)."""
        for platform, templates in _ALL_PLATFORM_CAPTION_TEMPLATES.items():
            assert len(templates) >= 12, f"{platform}: only {len(templates)} openers"
            assert len(set(templates)) == len(templates), f"{platform}: duplicate opener"
            assert all("{location}" in t for t in templates), platform

    def test_title_frames_capped_at_six(self):
        """Composed titles draw from at most 6 grammar frames — a frame
        is sentence shape only, never a fixed pool needing a size floor."""
        assert 1 <= len(TITLE_FRAME_TEMPLATES) <= 6, (
            f"Expected 1-6 title frames, got {len(TITLE_FRAME_TEMPLATES)}"
        )

    def test_every_title_frame_has_at_least_two_slots(self):
        """Every frame must reference >=2 distinct fact placeholders — a
        frame is never a complete title without slot data."""
        from gomoveshift.pipeline.social_generator import _frame_fields

        for tmpl in TITLE_FRAME_TEMPLATES:
            fields = _frame_fields(tmpl)
            assert len(fields) >= 2, f"Frame {tmpl!r} has <2 slots: {fields}"


class TestNoLinkInBio:
    """YouTube has no bio link — "link in bio" is meaningless in Shorts captions."""

    def test_no_caption_template_references_link_in_bio(self):
        """No caption template may contain "link in bio" (case-insensitive)."""
        for platform, templates in _ALL_PLATFORM_CAPTION_TEMPLATES.items():
            for i, tmpl in enumerate(templates):
                assert "link in bio" not in tmpl.lower(), (
                    f"{platform} caption template #{i} references a bio link: {tmpl!r}"
                )

    def test_no_generated_caption_references_link_in_bio(self, generator):
        """A batch of generated captions never says "link in bio"."""
        posts = generator.generate_all(num_shorts=PER_LOCATION_CAP)
        for p in posts:
            assert "link in bio" not in p.caption.lower(), (
                f"Short #{p.short_index} caption references a bio link: {p.caption!r}"
            )


class TestCrossEpisodeTitleDiversity:
    """Title templates must not repeat excessively across 8 episodes."""

    def test_no_excessive_repeats_across_8_episodes(self):
        """No title template shape should appear >3x across 8 episodes x 7 shorts.

        With deterministic per-episode shuffling, occasional 2-3x repeats are
        expected; >3x indicates the pool is too small or shuffling is broken.
        """
        locations = [
            ("Port Royal to Lady's Island, South Carolina", "coastal", "US-21", "Beaufort County"),
            ("Hunting Island to Beaufort, South Carolina", "coastal", "US-21", "Beaufort County"),
            ("Port Royal to Burton, South Carolina", "coastal", "Ribaut Road", "Beaufort County"),
            ("Burton to Yemassee, South Carolina", "coastal", "SC-128", "Hampton County"),
            ("Yemassee to Adams Run, South Carolina", "coastal", "US-17", "Charleston County"),
            ("Beaufort to Hollywood, South Carolina", "coastal", "SC-162", "Charleston County"),
            ("Highlands to Brevard, North Carolina", "winding", "US-64", "Transylvania County"),
            ("Highlands to Franklin, North Carolina", "mountain", "US-64", "Macon County"),
        ]
        all_shapes = []
        for ep_idx, (loc, terrain, road, county) in enumerate(locations, start=1):
            gen = SocialGenerator(
                location=loc,
                project_name="sc-batch",
                series_part=ep_idx,
                terrain_type=terrain,
                road_name=road,
                county=county,
            )
            for i in range(1, PER_LOCATION_CAP + 1):
                title = gen.generate_post(short_index=i).youtube_title
                # Normalize: replace location-specific text with placeholder
                shape = title.replace(loc.split(",")[0], "LOC")
                all_shapes.append(shape)
        counts = Counter(all_shapes)
        repeated = {t: c for t, c in counts.items() if c > 3}
        assert not repeated, f"Title templates used >3x across 8 episodes: {repeated}"

    def test_high_uniqueness_ratio_across_episodes(self):
        """At least 70% of title shapes across 8 episodes must be unique."""
        locations = [
            ("Port Royal to Lady's Island, South Carolina", "coastal", "US-21", "Beaufort County"),
            ("Hunting Island to Beaufort, South Carolina", "coastal", "US-21", "Beaufort County"),
            ("Port Royal to Burton, South Carolina", "coastal", "Ribaut Road", "Beaufort County"),
            ("Burton to Yemassee, South Carolina", "coastal", "SC-128", "Hampton County"),
            ("Yemassee to Adams Run, South Carolina", "coastal", "US-17", "Charleston County"),
            ("Beaufort to Hollywood, South Carolina", "coastal", "SC-162", "Charleston County"),
            ("Highlands to Brevard, North Carolina", "winding", "US-64", "Transylvania County"),
            ("Highlands to Franklin, North Carolina", "mountain", "US-64", "Macon County"),
        ]
        all_shapes = []
        for ep_idx, (loc, terrain, road, county) in enumerate(locations, start=1):
            gen = SocialGenerator(
                location=loc,
                project_name="sc-batch",
                series_part=ep_idx,
                terrain_type=terrain,
                road_name=road,
                county=county,
            )
            for i in range(1, PER_LOCATION_CAP + 1):
                title = gen.generate_post(short_index=i).youtube_title
                shape = title.replace(loc.split(",")[0], "LOC")
                all_shapes.append(shape)
        unique_ratio = len(set(all_shapes)) / len(all_shapes)
        assert unique_ratio >= 0.70, (
            f"Only {unique_ratio:.0%} unique shapes ({len(set(all_shapes))}/{len(all_shapes)}) — need 70%+"
        )


class TestContentSafety:
    """All templates must pass content validation."""

    def test_all_caption_templates_pass_validator(self):
        """Every platform caption template must pass content validation."""
        from collections import defaultdict

        validator = ContentValidator()
        fmap = defaultdict(
            str, location="County Kerry, Ireland", terrain_desc="roads", weather_emoji="🚗"
        )
        for platform, templates in _ALL_PLATFORM_CAPTION_TEMPLATES.items():
            for i, tmpl in enumerate(templates):
                filled = tmpl.format_map(fmap).strip()
                result = validator.validate(filled)
                assert result.valid, (
                    f"{platform} caption template {i} fails: {result.violations}\n"
                    f"Template: {tmpl[:80]}"
                )

    def test_all_title_frames_pass_validator(self):
        """Every title frame, filled, must pass content validation."""
        from collections import defaultdict

        validator = ContentValidator()
        fmap = defaultdict(
            str,
            location="County Kerry, Ireland",
            terrain_desc="Roads",
            road="R572",
            county="County Kerry",
            moment="The Peak Moment",
        )
        for tmpl in TITLE_FRAME_TEMPLATES:
            filled = tmpl.format_map(fmap)
            result = validator.validate(filled)
            assert result.valid, f"Title fails: '{filled}' — {result.violations}"


class TestSeriesAwareSeeding:
    """Series part number must influence deterministic seed."""

    def test_series_part_changes_caption_order(self):
        """Same location + different series_part = different caption order."""
        gen1 = SocialGenerator(
            location="County Kerry, Ireland",
            project_name="kerry_part1",
            series_part=1,
        )
        gen2 = SocialGenerator(
            location="County Kerry, Ireland",
            project_name="kerry_part2",
            series_part=2,
        )
        posts1 = gen1.generate_all(num_shorts=3)
        posts2 = gen2.generate_all(num_shorts=3)

        captions1 = [p.caption for p in posts1]
        captions2 = [p.caption for p in posts2]

        matching = sum(1 for a, b in zip(captions1, captions2) if a == b)
        assert matching <= 1, f"{matching}/3 captions identical between Part 1 and Part 2"


class TestPlatformSpecific:
    """Platform-specific title and hashtag optimization."""

    def test_tiktok_titles_differ_from_youtube(self, generator):
        """TikTok titles should NOT just be truncated YouTube titles."""
        posts = generator.generate_all(num_shorts=5)
        for p in posts:
            # TikTok title should be a distinct, shorter title
            assert p.tiktok_title != p.youtube_title[:90], (
                f"Short #{p.short_index}: TikTok title is just truncated YouTube title"
            )

    def test_tiktok_titles_under_90_chars(self, generator):
        """TikTok titles must be under 90 characters."""
        posts = generator.generate_all(num_shorts=PER_LOCATION_CAP)
        for p in posts:
            assert len(p.tiktok_title) <= 90

    def test_instagram_first_comment_max_15_hashtags(self, generator):
        """Instagram first comment should have at most 15 hashtags."""
        posts = generator.generate_all(num_shorts=PER_LOCATION_CAP)
        for p in posts:
            hashtag_count = p.first_comment.count("#")
            assert hashtag_count <= 15, f"Short #{p.short_index}: {hashtag_count} hashtags (max 15)"


class TestTerrainAwareTemplates:
    """Terrain classification must drive road terminology in social posts."""

    def test_coastal_terrain_no_back_roads_in_titles(self):
        """Coastal terrain must never produce 'back road' in any title."""
        gen = SocialGenerator(
            location="South Carolina, United States",
            project_name="sc_lowcountry_01",
            terrain_type="coastal",
            series_part=1,
        )
        posts = gen.generate_all(num_shorts=3)
        for p in posts:
            combined = f"{p.youtube_title} {p.tiktok_title}"
            assert "back road" not in combined.lower(), (
                f"Short #{p.short_index}: '{combined}' contains 'back road' for coastal terrain"
            )

    def test_countryside_terrain_uses_neutral_roads(self):
        """Countryside terrain must NOT claim 'back roads' — GPS can't prove road type."""
        gen = SocialGenerator(
            location="County Tipperary, Ireland",
            project_name="tipperary_01",
            terrain_type="countryside",
            series_part=1,
        )
        posts = gen.generate_all(num_shorts=3)
        for p in posts:
            combined = f"{p.youtube_title} {p.tiktok_title}"
            assert "back road" not in combined.lower(), (
                f"Short #{p.short_index}: '{combined}' claims 'back road' — "
                f"GPS data cannot determine road type for countryside terrain"
            )

    def test_no_terrain_uses_neutral_descriptors(self):
        """When terrain is None, no terrain-specific terms should appear."""
        gen = SocialGenerator(
            location="County Kerry, Ireland",
            project_name="kerry_01",
            terrain_type=None,
            series_part=1,
        )
        posts = gen.generate_all(num_shorts=3)
        for p in posts:
            combined = f"{p.youtube_title} {p.tiktok_title}"
            lower = combined.lower()
            assert "coastal road" not in lower
            assert "mountain road" not in lower
            assert "winding road" not in lower
            assert "back road" not in lower, (
                f"Short #{p.short_index}: '{combined}' claims 'back road' without terrain data"
            )

    def test_coastal_tags_include_coastal_terms(self):
        """Coastal terrain should produce coastal-specific tags."""
        gen = SocialGenerator(
            location="Beaufort, South Carolina, USA",
            project_name="sc_coastal_01",
            terrain_type="coastal",
            series_part=1,
        )
        posts = gen.generate_all(num_shorts=3)
        all_tags = " ".join(p.youtube_tags for p in posts)
        assert "coastal" in all_tags.lower(), (
            f"Coastal terrain should have coastal tags, got: {all_tags}"
        )
        assert "back roads" not in all_tags.lower(), (
            "Coastal terrain should not include 'back roads' tag"
        )

    def test_mountain_terrain_templates(self):
        """Mountain terrain should produce 'mountain roads' in at least one title."""
        gen = SocialGenerator(
            location="County Kerry, Ireland",
            project_name="healy_pass_01",
            terrain_type="mountain",
            series_part=1,
        )
        posts = gen.generate_all(num_shorts=3)
        all_text = " ".join(f"{p.youtube_title} {p.tiktok_title}" for p in posts)
        assert "mountain road" in all_text.lower(), (
            "Mountain terrain should produce 'mountain roads' in at least one "
            "title across a bare-slot batch"
        )

    def test_per_short_variant_drops_parent_lf_terrain(self):
        """A short cut ELSEWHERE on the drive (its own resolved location
        differs from the parent LF) must NOT inherit the LF-wide terrain vote.
        An inland far-corner clip of a COASTAL-classified LF may not carry
        #CoastalDrive / coastal tags / 'coastal roads' — an unverified claim."""
        gen = SocialGenerator(
            location="Beaufort, South Carolina, USA",  # coastal parent LF
            project_name="sc_coastal_01",
            terrain_type="coastal",
            series_part=1,
        )
        posts = gen.generate_all(
            num_shorts=1,
            per_short_locations=["Walterboro, South Carolina, USA"],  # inland corner
        )
        p = posts[0]
        surface = " ".join(
            [
                p.youtube_title,
                p.tiktok_title,
                p.caption,
                p.youtube_tags,
                p.first_comment,
                p.youtube_description,
            ]
        ).lower()
        assert "coastal" not in surface, f"inland short inherited coastal terrain: {surface}"
        assert "#coastaldrive" not in surface
        assert "seaside" not in surface

    def test_same_location_short_keeps_lf_terrain(self):
        """Contrast: a short whose own location MATCHES the parent LF keeps the
        LF terrain — the fix only strips terrain when the location differs."""
        gen = SocialGenerator(
            location="Beaufort, South Carolina, USA",
            project_name="sc_coastal_02",
            terrain_type="coastal",
            series_part=1,
        )
        posts = gen.generate_all(
            num_shorts=3,
            per_short_locations=["Beaufort, South Carolina, USA"] * 3,
        )
        all_tags = " ".join(p.youtube_tags for p in posts).lower()
        assert "coastal" in all_tags, "same-location short must keep the LF coastal tags"

    def test_per_short_variant_drops_parent_road_poi_county(self):
        """A short cut ELSEWHERE on the drive must not inherit
        the parent LF's road_name / roads / nearest_poi / county. Inheriting
        them shipped the parent's road in {road} titles and 'POV driving near
        {poi}' descriptions, plus county/road tags the CSV county gate misses."""
        gen = SocialGenerator(
            location="Beaufort, South Carolina, USA",  # parent LF
            project_name="sc_drive_01",
            series_part=1,
            road_name="Sea Island Parkway",
            roads=["Sea Island Parkway", "US-21"],
            nearest_poi="Hunting Island State Park",
            county="Beaufort County",
        )
        posts = gen.generate_all(
            num_shorts=1,
            per_short_locations=["Walterboro, South Carolina, USA"],  # inland corner
        )
        p = posts[0]
        surface = " ".join(
            [
                p.youtube_title,
                p.tiktok_title,
                p.caption,
                p.youtube_tags,
                p.first_comment,
                p.youtube_description,
            ]
        ).lower()
        assert "sea island parkway" not in surface, f"parent road leaked: {surface}"
        assert "us-21" not in surface, f"parent road leaked: {surface}"
        assert "hunting island state park" not in surface, f"parent POI leaked: {surface}"
        assert "beaufort county" not in surface, f"parent county leaked: {surface}"


class TestGlobalIndexCSVIntegration:
    """Global indices must match CSV schedule slot_lookup keys."""

    def test_global_indices_match_schedule_slots(self):
        """Multi-LF posts with offsets must match global schedule slots."""
        from gomoveshift.pipeline.scheduler import ReleaseSchedule, ScheduledSlot

        # Simulate 2 LFs, each with 3 shorts → global schedule has slots 1-6
        slots = [
            ScheduledSlot(
                short_index=i,
                date="2026-04-02",
                time="08:00:00",
                datetime_est="2026-04-02T08:00:00-05:00",
                day_of_week="Wed",
            )
            for i in range(1, 7)
        ]
        schedule = ReleaseSchedule(
            project_name="test",
            location="Ireland",
            long_form_publish="",
            window_start="",
            window_end="",
            num_shorts=6,
            num_slots=6,
            slots=slots,
        )

        # LF1: 3 shorts, offset=0 → indices 1,2,3
        gen1 = SocialGenerator(location="Ireland", project_name="lf01", series_part=1)
        posts_lf1 = gen1.generate_all(num_shorts=3, index_offset=0)

        # LF2: 3 shorts, offset=3 → indices 4,5,6
        gen2 = SocialGenerator(location="Ireland", project_name="lf02", series_part=2)
        posts_lf2 = gen2.generate_all(num_shorts=3, index_offset=3)

        all_posts = posts_lf1 + posts_lf2

        # Build slot lookup (same as csv_generator does)
        slot_lookup = {slot.short_index: slot for slot in schedule.slots}

        # Every post must find its matching slot
        for post in all_posts:
            assert post.short_index in slot_lookup, (
                f"Post index {post.short_index} not found in schedule slots "
                f"{list(slot_lookup.keys())}"
            )


class TestGlobalIndexOffset:
    """index_offset in generate_all() remaps short_index for multi-LF drives."""

    def test_offset_zero_starts_at_one(self, generator):
        """Default offset=0: short indices start at 1."""
        posts = generator.generate_all(num_shorts=3, index_offset=0)
        indices = [p.short_index for p in posts]
        assert indices == [1, 2, 3]

    def test_offset_three_starts_at_four(self, generator):
        """offset=3: short indices start at 4."""
        posts = generator.generate_all(num_shorts=3, index_offset=3)
        indices = [p.short_index for p in posts]
        assert indices == [4, 5, 6]

    def test_two_lfs_produce_global_indices(self):
        """Simulating 2 LFs with 3 shorts each — indices should be 1-6."""
        gen1 = SocialGenerator(
            location="County Kerry, Ireland",
            project_name="drive_lf01",
            series_part=1,
        )
        gen2 = SocialGenerator(
            location="County Kerry, Ireland",
            project_name="drive_lf02",
            series_part=2,
        )
        lf1_posts = gen1.generate_all(num_shorts=3, index_offset=0)
        lf2_posts = gen2.generate_all(num_shorts=3, index_offset=3)

        all_indices = [p.short_index for p in lf1_posts + lf2_posts]
        assert all_indices == [1, 2, 3, 4, 5, 6]

    def test_offset_does_not_change_content_quality(self, generator):
        """Offset should not affect content diversity — all posts still unique."""
        posts = generator.generate_all(num_shorts=5, index_offset=10)
        captions = [p.caption for p in posts]
        assert len(set(captions)) == 5, "All captions should be unique with offset"

    def test_default_offset_is_zero(self, generator):
        """Calling without index_offset should behave like offset=0."""
        posts_default = generator.generate_all(num_shorts=3)
        posts_explicit = generator.generate_all(num_shorts=3, index_offset=0)
        assert [p.short_index for p in posts_default] == [p.short_index for p in posts_explicit]


class TestCaptionHashtags:
    """Captions should end with 3-5 niche hashtags for platform discovery."""

    def test_caption_contains_hashtags(self, generator):
        """Every caption must contain at least 3 hashtags."""
        posts = generator.generate_all(num_shorts=PER_LOCATION_CAP)
        for p in posts:
            count = p.caption.count("#")
            assert count >= 3, f"Short #{p.short_index}: caption has {count} hashtags (min 3)"

    def test_caption_hashtags_at_end(self, generator):
        """Hashtags should appear at the end of the caption, not the middle."""
        posts = generator.generate_all(num_shorts=5)
        for p in posts:
            lines = p.caption.strip().split("\n")
            last_line = lines[-1].strip()
            assert last_line.startswith("#"), (
                f"Short #{p.short_index}: last line is not hashtags: '{last_line}'"
            )

    def test_caption_max_8_hashtags(self, generator):
        """The Instagram caption (post.caption) has at most 8 hashtags —
        not the full 15 of first_comment."""
        posts = generator.generate_all(num_shorts=PER_LOCATION_CAP)
        for p in posts:
            count = p.caption.count("#")
            assert count <= 8, f"Short #{p.short_index}: caption has {count} hashtags (max 8)"

    def test_caption_still_under_280_chars(self, generator):
        """Captions with hashtags must still fit 280-char limit."""
        posts = generator.generate_all(num_shorts=PER_LOCATION_CAP)
        for p in posts:
            assert len(p.caption) <= 280, (
                f"Short #{p.short_index}: caption is {len(p.caption)} chars"
            )


class TestTagSEOCompliance:
    """Tags must follow YouTube SEO best practices (dedup, 500-char limit)."""

    def test_no_case_insensitive_duplicate_tags(self, generator):
        """No two tags should be identical when compared case-insensitively."""
        posts = generator.generate_all(num_shorts=PER_LOCATION_CAP)
        for p in posts:
            tags = [t.strip() for t in p.youtube_tags.split(",")]
            lower_tags = [t.lower() for t in tags]
            assert len(lower_tags) == len(set(lower_tags)), (
                f"Short #{p.short_index}: duplicate tags: {tags}"
            )

    def test_tag_string_under_500_chars(self, generator):
        """Total tag string must be under 500 chars (YouTube API limit)."""
        posts = generator.generate_all(num_shorts=PER_LOCATION_CAP)
        for p in posts:
            assert len(p.youtube_tags) <= 500, (
                f"Short #{p.short_index}: tags are {len(p.youtube_tags)} chars"
            )

    def test_tag_count_5_to_20(self, generator):
        """Tag count should be between 5 and 20 (YouTube cap)."""
        posts = generator.generate_all(num_shorts=PER_LOCATION_CAP)
        for p in posts:
            count = len([t for t in p.youtube_tags.split(",") if t.strip()])
            assert 5 <= count <= 20, f"Short #{p.short_index}: {count} tags (want 5-20)"


class TestPOIInDescription:
    """YouTube description should include nearest POI when available."""

    def test_description_contains_poi_name(self):
        """When nearest_poi is set, description should contain it."""
        gen = SocialGenerator(
            location="Port Royal to Lady's Island, South Carolina",
            nearest_poi="Hunting Island State Park",
            series_part=1,
        )
        posts = gen.generate_all(num_shorts=3)
        for p in posts:
            assert "Hunting Island State Park" in p.youtube_description

    def test_description_without_poi_still_works(self):
        """Without nearest_poi, description uses location as before."""
        gen = SocialGenerator(
            location="Port Royal to Lady's Island, South Carolina",
            series_part=1,
        )
        posts = gen.generate_all(num_shorts=3)
        for p in posts:
            assert "Port Royal" in p.youtube_description
            assert len(p.youtube_description) <= 200

    def test_long_poi_falls_back_to_location(self):
        """If POI + location exceeds 200 chars, fall back to location only."""
        gen = SocialGenerator(
            location="Sheepshead Peninsula to Durrus, County Cork, Ireland",
            nearest_poi="Some Extremely Long Point of Interest Name That Would Exceed Two Hundred Characters Easily",
            series_part=1,
        )
        posts = gen.generate_all(num_shorts=3)
        for p in posts:
            assert len(p.youtube_description) <= 200

    def test_poi_added_as_tag(self):
        """When nearest_poi is set, it should appear in tags."""
        gen = SocialGenerator(
            location="Port Royal, South Carolina",
            nearest_poi="Hunting Island State Park",
            series_part=1,
        )
        posts = gen.generate_all(num_shorts=1)
        assert "Hunting Island State Park" in posts[0].youtube_tags


class TestRoadsInTags:
    """Road names should appear in YouTube tags."""

    def test_roads_included_in_tags(self):
        """When roads list is provided, road names appear in tags."""
        gen = SocialGenerator(
            location="Port Royal, South Carolina",
            roads=["US-21", "Sea Island Parkway"],
            series_part=1,
        )
        posts = gen.generate_all(num_shorts=1)
        tags = posts[0].youtube_tags
        assert "US-21" in tags
        assert "Sea Island Parkway" in tags

    def test_roads_not_duplicated_with_road_name(self):
        """If road_name is also in roads list, allow standalone + one compound tag."""
        gen = SocialGenerator(
            location="Highlands, North Carolina",
            road_name="Blue Ridge Parkway",
            roads=["US-64", "Blue Ridge Parkway", "NC-28"],
            series_part=1,
        )
        posts = gen.generate_all(num_shorts=1)
        tags_lower = posts[0].youtube_tags.lower().split(",")
        brp_count = sum(1 for t in tags_lower if "blue ridge parkway" in t)
        assert brp_count <= 2

    def test_roads_fallback_sets_road_name(self):
        """When roads provided but road_name is None, first road becomes road_name."""
        gen = SocialGenerator(
            location="Port Royal, South Carolina",
            roads=["US-21", "Sea Island Parkway"],
            series_part=1,
        )
        assert gen.road_name == "US-21"

    def test_tags_still_under_500_chars(self):
        """Even with roads, total tags must be <= 500 chars."""
        gen = SocialGenerator(
            location="Port Royal, South Carolina",
            roads=["US-21", "Sea Island Parkway", "SC-802", "Ribaut Road"],
            series_part=1,
        )
        posts = gen.generate_all(num_shorts=4)
        for p in posts:
            assert len(p.youtube_tags) <= 500


class TestRecordingDateFromDatetime:
    """recording_date passed as datetime must be auto-converted to string."""

    def test_recording_date_from_datetime(self):
        from datetime import datetime as dt

        gen = SocialGenerator(
            location="Port Royal to Lady's Island, South Carolina",
            project_name="test",
            series_part=1,
            recording_date=dt(2026, 2, 20),
        )
        post = gen.generate_post(short_index=1)
        assert "Feb" in post.caption, f"Expected recording date in caption, got: {post.caption}"

    def test_recording_date_string_still_works(self):
        gen = SocialGenerator(
            location="Port Royal to Lady's Island, South Carolina",
            project_name="test",
            series_part=1,
            recording_date="Feb 2026",
        )
        post = gen.generate_post(short_index=1)
        assert "Feb" in post.caption, f"Expected recording date in caption, got: {post.caption}"


class TestGeneratePostEnforcesContentValidation:
    """generate_post() must raise ContentValidationError on blocked titles."""

    def test_blocked_title_raises_content_validation_error(self):
        from unittest.mock import patch

        from gomoveshift.pipeline.content_validator import ContentValidationError

        gen = SocialGenerator(
            location="Test, Ireland",
            project_name="test",
            series_part=1,
        )
        with patch.object(
            gen, "_pick_title", return_value="You Won't Believe This Road in Ireland"
        ):
            with pytest.raises(ContentValidationError):
                gen.generate_post(short_index=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
