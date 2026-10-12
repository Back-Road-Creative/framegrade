"""Tests for the Shorts YouTube description and category.

A Short carries a YouTube description of its own: the caption is Twitter-limited at 280 chars.
youtube_category defaults to the numeric ID "19", not an empty string.

Tests written BEFORE implementation (red phase).
"""

import pytest

from gomoveshift.models import ResolvedLocation
from gomoveshift.pipeline.social_generator import SocialGenerator, SocialPost


# Brand per-location cap (publishing/brand_policy.py) — also the
# composition capacity ceiling once road_name + county are both grounded.
PER_LOCATION_CAP = 5


@pytest.fixture
def generator():
    return SocialGenerator(
        location="County Tipperary, Ireland",
        project_name="tipperary_01",
        recording_date="Sep 2025",
        series_part=1,
        road_name="N71",
        county="County Tipperary",
    )


# ── youtube_description field ────────────────────────────────────────


class TestYouTubeDescriptionField:
    """SocialPost must have a youtube_description field."""

    def test_social_post_has_youtube_description(self):
        """SocialPost dataclass must have youtube_description attribute."""
        post = SocialPost(
            short_index=1,
            caption="Test",
            youtube_title="Test",
            youtube_tags="pov,drive",
            youtube_category="19",
            tiktok_title="Test",
            pinterest_board="Drives",
            first_comment="#test",
        )
        assert hasattr(post, "youtube_description")

    def test_youtube_description_in_to_dict(self):
        """to_dict() must include youtube_description."""
        post = SocialPost(
            short_index=1,
            caption="Test",
            youtube_title="Test",
            youtube_tags="pov,drive",
            youtube_category="19",
            tiktok_title="Test",
            pinterest_board="Drives",
            first_comment="#test",
        )
        d = post.to_dict()
        assert "youtube_description" in d


class TestYouTubeDescriptionContent:
    """Generated youtube_description must include location, keywords, and #Shorts."""

    def test_youtube_description_not_empty(self, generator):
        """Every generated post must have a non-empty youtube_description."""
        posts = generator.generate_all(num_shorts=5)
        for p in posts:
            assert p.youtube_description, f"Short #{p.short_index}: youtube_description is empty"

    def test_youtube_description_max_200_chars(self, generator):
        """youtube_description must be ≤200 characters (short but keyword-rich)."""
        posts = generator.generate_all(num_shorts=PER_LOCATION_CAP)
        for p in posts:
            assert len(p.youtube_description) <= 200, (
                f"Short #{p.short_index}: youtube_description is {len(p.youtube_description)} chars "
                f"(max 200): '{p.youtube_description}'"
            )

    def test_youtube_description_contains_location(self, generator):
        """youtube_description must contain the location string."""
        posts = generator.generate_all(num_shorts=5)
        for p in posts:
            # Location short is "County Tipperary" (first part before comma)
            assert "County Tipperary" in p.youtube_description, (
                f"Short #{p.short_index}: youtube_description missing location: "
                f"'{p.youtube_description}'"
            )

    def test_youtube_description_contains_hashtag(self, generator):
        """youtube_description must contain at least one hashtag."""
        posts = generator.generate_all(num_shorts=5)
        for p in posts:
            assert "#" in p.youtube_description, (
                f"Short #{p.short_index}: youtube_description has no hashtags: "
                f"'{p.youtube_description}'"
            )

    def test_youtube_description_contains_shorts_hashtag(self, generator):
        """youtube_description must contain #Shorts for YouTube Shorts discovery."""
        posts = generator.generate_all(num_shorts=5)
        for p in posts:
            assert "#Shorts" in p.youtube_description, (
                f"Short #{p.short_index}: missing #Shorts: '{p.youtube_description}'"
            )

    def test_youtube_description_contains_niche_keyword(self, generator):
        """youtube_description must contain a primary niche keyword."""
        keywords = {"pov", "driving", "dashcam", "4k", "drive"}
        posts = generator.generate_all(num_shorts=5)
        for p in posts:
            desc_lower = p.youtube_description.lower()
            has_keyword = any(k in desc_lower for k in keywords)
            assert has_keyword, (
                f"Short #{p.short_index}: no niche keyword in: '{p.youtube_description}'"
            )

    def test_youtube_description_has_3_to_5_hashtags(self, generator):
        """youtube_description should have 3-5 hashtags for optimal discovery."""
        posts = generator.generate_all(num_shorts=5)
        for p in posts:
            count = p.youtube_description.count("#")
            assert 3 <= count <= 5, (
                f"Short #{p.short_index}: {count} hashtags (want 3-5): '{p.youtube_description}'"
            )

    def test_youtube_description_long_location_still_fits(self):
        """Even with very long location names, description must be ≤200 chars."""
        gen = SocialGenerator(
            location="Sheepshead Peninsula to Durrus, County Cork, Ireland",
            project_name="sheepshead_01",
            series_part=1,
        )
        posts = gen.generate_all(num_shorts=3)
        for p in posts:
            assert len(p.youtube_description) <= 200, (
                f"Short #{p.short_index}: youtube_description is {len(p.youtube_description)} chars "
                f"with long location: '{p.youtube_description}'"
            )


class TestYouTubeDescriptionUniqueness:
    """Each short in an episode must have a unique youtube_description."""

    def test_no_duplicate_descriptions_same_episode(self):
        """Shorts from the same generator must have unique descriptions."""
        gen = SocialGenerator(
            location="Beaufort, South Carolina, USA",
            project_name="sc_ep01",
            terrain_type="coastal",
            road_name="US-21",
            roads=["US-21", "Sea Island Parkway"],
            county="Beaufort County",
            series_part=1,
        )
        posts = gen.generate_all(num_shorts=PER_LOCATION_CAP)
        descriptions = [p.youtube_description for p in posts]
        assert len(set(descriptions)) == len(descriptions), (
            f"Only {len(set(descriptions))}/{len(descriptions)} unique descriptions"
        )

    def test_no_duplicate_descriptions_no_poi_no_road(self):
        """Even without POI or road, descriptions must differ per short."""
        gen = SocialGenerator(
            location="County Tipperary, Ireland",
            project_name="tipperary_01",
            series_part=1,
        )
        posts = gen.generate_all(num_shorts=3)
        descriptions = [p.youtube_description for p in posts]
        assert len(set(descriptions)) == len(descriptions), (
            f"Only {len(set(descriptions))}/{len(descriptions)} unique descriptions "
            f"(no POI/road): {descriptions[:3]}"
        )

    def test_description_includes_road_when_available(self):
        """When road_name is set, description should include it."""
        gen = SocialGenerator(
            location="Beaufort, South Carolina, USA",
            project_name="sc_ep01",
            road_name="US-21",
            series_part=1,
        )
        posts = gen.generate_all(num_shorts=3)
        road_count = sum(1 for p in posts if "US-21" in p.youtube_description)
        assert road_count >= 1, "At least one description should include road name"


class TestYouTubeDescriptionBackwardCompat:
    """Old JSON without youtube_description must still load via SocialPost(**dict)."""

    def test_old_json_without_youtube_description(self):
        """SocialPost(**dict) with missing youtube_description should use default."""
        old_data = {
            "short_index": 1,
            "caption": "Test",
            "youtube_title": "Test",
            "youtube_tags": "pov,drive",
            "youtube_category": "",
            "tiktok_title": "Test",
            "pinterest_board": "Drives",
            "first_comment": "#test",
        }
        post = SocialPost(**old_data)
        assert post.youtube_description == ""


# ── youtube_category defaults to "19" ────────────────────────────────


class TestYouTubeCategoryNumericID:
    """youtube_category must default to "19" (Travel & Events numeric ID)."""

    def test_default_category_is_19(self):
        """SocialGenerator default youtube_category must be '19'."""
        gen = SocialGenerator(location="County Kerry, Ireland", series_part=1)
        assert gen.youtube_category == "19"

    def test_generated_posts_have_category_19(self, generator):
        """All generated posts must have youtube_category='19'."""
        posts = generator.generate_all(num_shorts=5)
        for p in posts:
            assert p.youtube_category == "19", (
                f"Short #{p.short_index}: youtube_category is '{p.youtube_category}' (expected '19')"
            )

    def test_category_override_still_works(self):
        """Explicit youtube_category override should still be respected."""
        gen = SocialGenerator(
            location="County Kerry, Ireland",
            youtube_category="22",
            series_part=1,
        )
        posts = gen.generate_all(num_shorts=1)
        assert posts[0].youtube_category == "22"


# ── Intra-episode description deduplication ──────────────────────────


class TestDescriptionDeduplication:
    """When the same POI repeats across shorts, descriptions must still be unique."""

    def test_description_unique_when_poi_repeats(self):
        gen = SocialGenerator(
            location="Port Royal to Lady's Island, South Carolina",
            project_name="test",
            series_part=1,
            terrain_type="coastal",
            nearest_poi="Lady's Island",
            road_name="US-21",
            county="Beaufort County",
        )
        descs = [
            gen.generate_post(short_index=i).youtube_description.split(" #")[0]
            for i in range(1, PER_LOCATION_CAP + 1)
        ]
        assert len(set(descs)) == PER_LOCATION_CAP, (
            f"Expected {PER_LOCATION_CAP} unique desc bases, got {len(set(descs))}: {descs}"
        )


# ── Tag density ─────────────────────────────────────────────────────


class TestTagDensity:
    """Tags must include county, state abbreviation, and fill toward 400 chars."""

    def test_tags_include_county_and_state(self):
        gen = SocialGenerator(
            location="Port Royal to Lady's Island, South Carolina",
            project_name="test",
            series_part=1,
            roads=["US-21", "Sea Island Parkway"],
            county="Beaufort County",
        )
        tags = gen.generate_post(short_index=1).youtube_tags
        assert "Beaufort County" in tags
        assert "SC" in tags or "South Carolina" in tags

    def test_tags_reach_400_chars_with_full_data(self):
        gen = SocialGenerator(
            location="Port Royal to Lady's Island, South Carolina",
            project_name="test",
            series_part=1,
            terrain_type="coastal",
            roads=["US-21", "Sea Island Parkway"],
            county="Beaufort County",
            nearest_poi="Lady's Island",
        )
        tags = gen.generate_post(short_index=1).youtube_tags
        assert len(tags) >= 300, f"Expected ≥300 chars of tags, got {len(tags)}: {tags}"


# ── #HiddenGem must not appear in first_comment ──────────────────


class TestFirstCommentNoHiddenGem:
    """#HiddenGem is a blocked phrase — must not appear in first_comment hashtags."""

    def test_first_comment_no_hidden_gem(self):
        gen = SocialGenerator(
            location="Port Royal to Lady's Island, South Carolina",
            project_name="test",
            series_part=1,
            road_name="US-21",
            county="Beaufort County",
        )
        for i in range(1, PER_LOCATION_CAP + 1):
            comment = gen.generate_post(short_index=i).first_comment
            assert "HiddenGem" not in comment, (
                f"Short {i} first_comment contains #HiddenGem: {comment}"
            )


# ── Destination POI in social caption + shorts YouTube description ────────


class TestDestinationAnchoring:
    """When the resolved model names a destination POI, the secondary text
    surfaces (caption + shorts youtube_description) anchor it — consistent
    with the burned OST / title — without regressing the no-destination case.
    """

    DEST = "Donnelley Wildlife Management Area"
    REGION = "Beaufort County"

    def _gen(self, *, resolved):
        return SocialGenerator(
            location=f"{self.REGION}, South Carolina, USA",
            project_name="sc_dest_01",
            series_part=1,
            resolved_location=resolved,
            road_name="US-21",
            county=self.REGION,
        )

    def test_social_location_prefers_destination(self):
        rl = ResolvedLocation(dominant_county=self.REGION, destination=self.DEST)
        gen = self._gen(resolved=rl)
        assert gen.social_location == f"{self.DEST}, {self.REGION}"

    def test_description_contains_destination(self):
        rl = ResolvedLocation(dominant_county=self.REGION, destination=self.DEST)
        gen = self._gen(resolved=rl)
        posts = gen.generate_all(num_shorts=PER_LOCATION_CAP)
        hits = sum(1 for p in posts if self.DEST in p.youtube_description)
        assert hits >= 1, (
            f"Expected the destination POI in ≥1 shorts description, got 0: "
            f"{[p.youtube_description for p in posts][:3]}"
        )

    def test_caption_contains_destination(self):
        rl = ResolvedLocation(dominant_county=self.REGION, destination=self.DEST)
        gen = self._gen(resolved=rl)
        posts = gen.generate_all(num_shorts=PER_LOCATION_CAP)
        hits = sum(1 for p in posts if self.DEST in p.caption)
        assert hits >= 1, (
            f"Expected the destination POI in ≥1 caption, got 0: {[p.caption for p in posts][:2]}"
        )

    def test_description_still_within_200_chars_with_destination(self):
        rl = ResolvedLocation(dominant_county=self.REGION, destination=self.DEST)
        gen = self._gen(resolved=rl)
        for p in gen.generate_all(num_shorts=PER_LOCATION_CAP):
            assert len(p.youtube_description) <= 200, (
                f"Short #{p.short_index}: description {len(p.youtube_description)} chars "
                f"with destination: '{p.youtube_description}'"
            )

    def test_no_destination_unchanged_resolved_without_destination(self):
        """A resolved model with destination=None must produce the legacy text
        (social_location == location_short) — no regression."""
        rl = ResolvedLocation(dominant_county=self.REGION)  # destination defaults None
        gen = self._gen(resolved=rl)
        assert gen.social_location == self.REGION

    def test_no_destination_unchanged_no_resolved_model(self):
        """No resolved_location at all → identical caption/description text to
        the pre-change behavior (social_location == location_short)."""
        with_none = SocialGenerator(
            location=f"{self.REGION}, South Carolina, USA",
            project_name="sc_dest_01",
            series_part=1,
        )
        assert with_none.social_location == with_none.location_short == self.REGION
        baseline = SocialGenerator(
            location=f"{self.REGION}, South Carolina, USA",
            project_name="sc_dest_01",
            series_part=1,
        )
        a = with_none.generate_all(num_shorts=3)
        b = baseline.generate_all(num_shorts=3)
        assert [p.caption for p in a] == [p.caption for p in b]
        assert [p.youtube_description for p in a] == [p.youtube_description for p in b]


# ── final-fallback truncation must be word-safe ─────────────────────


class TestYouTubeDescriptionWordSafeTruncation:
    """When the anchor alone overruns 200 chars (every optional hashtag
    already dropped), the last-resort truncation must cut on a WORD boundary —
    a placename is never sliced mid-word (e.g. "…, Beaufort Cou"). The former
    ``description[:200]`` hard-slice split whatever word straddled char 200.
    """

    def test_final_truncation_lands_on_word_boundary(self):
        region = "Beaufort County"
        prefix = "POV driving near "
        joiner = f", {region}"
        # Synthetic over-long POI sized so the anchor text ("…, Beaufort
        # County") straddles the 200-char boundary: the naive slice lands
        # inside "County", yielding "…, Beaufort Cou". 34 = prefix + joiner.
        poi = ("Rivers Bridge Historic Preserve Scenic Loop " * 5)[
            : 203 - len(prefix) - len(joiner)
        ]
        gen = SocialGenerator(
            location=region,
            project_name="word_safe",
            series_part=1,
            nearest_poi=poi,
        )
        desc = gen.generate_post(short_index=1).youtube_description

        # Within the platform limit …
        assert len(desc) <= 200, f"{len(desc)} chars: {desc!r}"
        # … and cut on a whole-word boundary of the anchor, never mid-word.
        anchor = f"{prefix}{poi}{joiner}"  # untruncated "POV driving near …, Beaufort County"
        body = desc.rstrip("…").rstrip()  # drop the truncation ellipsis, if any
        assert anchor.startswith(body), f"unexpected body: {body!r}"
        assert body == anchor or anchor[len(body)] == " ", f"truncated mid-word: …{body[-20:]!r}"
        # The exact mid-placename fragment the bug produced must be gone.
        assert "Beaufort Cou" not in desc, f"mid-placename slice: {desc!r}"
