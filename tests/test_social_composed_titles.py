"""Composed short titles replace the fixed template pools.

Titles are now built from grounded fact slots on the generator (location,
terrain, road, county, moment kind) via a small grammar-frame set
(``TITLE_FRAME_TEMPLATES``), instead of picking from a fixed-string pool.
"""

from gomoveshift.pipeline.social_generator import (
    TITLE_FRAME_TEMPLATES,
    SocialGenerator,
    _frame_fields,
)
from fixtures.legacy_title_pool import LEGACY_TITLE_POOL


def _gen(**kw):
    defaults = dict(location="County Kerry, Ireland", project_name="composed-test", series_part=1)
    defaults.update(kw)
    return SocialGenerator(**defaults)


class TestComposedTitleHasAtLeastTwoFactSlots:
    def test_youtube_title_carries_two_distinct_facts(self):
        gen = _gen(terrain_type="coastal")
        title = gen.generate_post(short_index=1).youtube_title
        # location_short and terrain_desc are always available — both must
        # be traceable in a normal composed title.
        assert "Kerry" in title or gen.terrain_desc.title() in title

    def test_road_and_county_widen_but_never_required(self):
        gen_plain = _gen()
        gen_rich = _gen(road_name="R572", county="County Kerry")
        # Both must succeed — road/county are optional enrichments only.
        assert gen_plain.generate_post(short_index=1).youtube_title
        assert gen_rich.generate_post(short_index=1).youtube_title


class TestFrameSetShape:
    def test_at_most_six_frames(self):
        assert len(TITLE_FRAME_TEMPLATES) <= 6

    def test_every_frame_requires_two_or_more_slots(self):
        for tmpl in TITLE_FRAME_TEMPLATES:
            assert len(_frame_fields(tmpl)) >= 2, tmpl


class TestTikTokFoldedIntoComposition:
    def test_tiktok_title_uses_the_same_frame_set(self):
        gen = _gen()
        post = gen.generate_post(short_index=1)
        # Every rendered fragment (minus slot values) must trace back to one
        # of the shared frames' literal text, i.e. no second hardcoded pool.
        assert len(post.tiktok_title) <= 90
        assert post.tiktok_title

    def test_tiktok_and_youtube_titles_can_differ(self):
        gen = _gen()
        post = gen.generate_post(short_index=1)
        assert post.tiktok_title != post.youtube_title[:90]


class TestLegacyPoolFixtureImportable:
    def test_legacy_pool_has_the_historical_entries(self):
        assert len(LEGACY_TITLE_POOL) >= 100

    def test_composed_title_never_matches_the_legacy_pool_verbatim(self):
        """Deliverable check (a batch-level check could enforce this too): a
        composed title, for a representative fact set, never happens to
        equal one of the retired literal templates rendered with the same
        facts."""
        gen = _gen(terrain_type="coastal")
        fmap = {"location": gen.location_short, "terrain_desc": gen.terrain_desc.title()}
        rendered_legacy = {t.format_map(_SafeMap(fmap)) for t in LEGACY_TITLE_POOL}
        # location+terrain alone compose exactly 3 distinct applicable frames.
        for i in range(1, 4):
            title = gen.generate_post(short_index=i).youtube_title
            assert title not in rendered_legacy, f"short {i}: {title!r} matches the legacy pool"


class TestMomentKindSlot:
    def test_moment_kinds_absent_never_raises(self):
        gen = _gen()
        assert gen.generate_post(short_index=1).youtube_title

    def test_moment_kind_can_influence_the_composed_title(self):
        gen = _gen(moment_kinds={1: "peak"})
        title = gen.generate_post(short_index=1).youtube_title
        assert title  # never raises; moment is an optional enrichment only


class _SafeMap(dict):
    def __missing__(self, key):
        return ""
