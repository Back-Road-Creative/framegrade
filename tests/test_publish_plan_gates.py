"""Publish-plan gates: shorts region, timing and grammar invariants,
moved out of ``pipeline/csv_generator.py`` into ``publishing/plan_gates.py``
so the publish-plan sidecar owns them directly rather than borrowing them
from the retired Metricool CSV writer.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.publishing import plan_gates


def _post(
    short_index=1,
    *,
    sampled_regions=None,
    youtube_title="Kenmare to Sneem POV Drive",
    caption="Rolling through Kerry today",
    youtube_tags="road trip",
    tiktok_title="tt",
    youtube_description="",
    captions=None,
):
    return SimpleNamespace(
        short_index=short_index,
        caption=caption,
        youtube_title=youtube_title,
        youtube_tags=youtube_tags,
        tiktok_title=tiktok_title,
        youtube_description=youtube_description,
        captions=captions or {},
        sampled_regions=sampled_regions or [],
    )


def _slot(short_index=1, date="2026-07-21", time="17:00:00"):
    return SimpleNamespace(short_index=short_index, date=date, time=time)


class TestAssertShortsNotScheduledBeforePublish:
    def test_short_before_publish_raises(self):
        shorts_by_lf = {1: [_post(1)]}
        slots = {1: _slot(1, "2026-07-21", "14:00:00")}
        publish_at = {1: "2026-07-21T20:00:00+00:00"}  # 16:00 EDT, short posts 14:00 EDT
        with pytest.raises(AssertionError, match="PUBLISH-PLAN-SHORT-BEFORE-PUBLISH"):
            plan_gates.assert_shorts_not_scheduled_before_publish(shorts_by_lf, slots, publish_at)

    def test_short_after_publish_passes(self):
        shorts_by_lf = {1: [_post(1)]}
        slots = {1: _slot(1, "2026-07-21", "17:00:00")}
        publish_at = {1: "2026-07-21T20:00:00+00:00"}  # 16:00 EDT, short posts 17:00 EDT
        plan_gates.assert_shorts_not_scheduled_before_publish(shorts_by_lf, slots, publish_at)

    def test_missing_publish_at_is_noop(self):
        shorts_by_lf = {1: [_post(1)]}
        slots = {1: _slot(1, "2026-07-20", "14:00:00")}
        plan_gates.assert_shorts_not_scheduled_before_publish(shorts_by_lf, slots, {1: None})


class TestAssertShortsMetadataInRegion:
    def test_cross_region_tag_raises(self):
        post = _post(
            1,
            sampled_regions=["County Kerry, Munster"],
            youtube_tags="County Cork,road trip",
        )
        with pytest.raises(ValueError, match="(?i)region") as exc:
            plan_gates.assert_shorts_metadata_in_region({1: [post]})
        assert "tag" in str(exc.value)
        assert "PUBLISH-PLAN-CROSS-REGION" in str(exc.value)

    def test_cross_region_platform_caption_raises(self):
        """A per-platform caption (not the legacy shared ``caption``) carrying
        the confirmed foreign county is also reported -- captions['youtube']
        is skipped, every other platform is checked."""
        post = _post(
            1,
            sampled_regions=["County Kerry, Munster"],
            youtube_tags="County Cork,road trip",
            captions={"instagram": "Driving through County Cork today"},
        )
        with pytest.raises(ValueError, match="(?i)region") as exc:
            plan_gates.assert_shorts_metadata_in_region({1: [post]})
        assert "caption:instagram" in str(exc.value)

    def test_in_region_short_passes(self):
        post = _post(1, sampled_regions=["County Kerry, Munster"], youtube_tags="County Kerry")
        plan_gates.assert_shorts_metadata_in_region({1: [post]})

    def test_no_regions_is_noop(self):
        post = _post(1, sampled_regions=[], youtube_tags="County Cork")
        plan_gates.assert_shorts_metadata_in_region({1: [post]})


class TestAssertPlanTextGrammar:
    def test_title_splice_raises(self):
        post = _post(1, youtube_title="Slow Drive Through To Clonakilty")
        with pytest.raises(AssertionError, match="(?i)grammar"):
            plan_gates.assert_plan_text_grammar({1: [post]})

    def test_hashtag_splice_raises(self):
        post = _post(1, caption="Rolling through Clonakilty #ToClonakilty")
        with pytest.raises(AssertionError, match="PUBLISH-PLAN-GRAMMAR"):
            plan_gates.assert_plan_text_grammar({1: [post]})

    def test_clean_post_passes(self):
        post = _post(1)
        plan_gates.assert_plan_text_grammar({1: [post]})

    def test_youtube_caption_key_is_skipped(self):
        """A grammar splice living only in captions['youtube'] must not fire --
        the YouTube payload has no caption surface, only title/description."""
        post = _post(1, captions={"youtube": "drive through To Clonakilty"})
        plan_gates.assert_plan_text_grammar({1: [post]})

    def test_youtube_description_splice_raises(self):
        post = _post(1, youtube_description="drive through To Clonakilty")
        with pytest.raises(AssertionError, match="PUBLISH-PLAN-GRAMMAR"):
            plan_gates.assert_plan_text_grammar({1: [post]})


class TestStructuralInvariants:
    """The structural test: the moved gates carry no relative lazy
    imports and no reference to the retired Metricool CSV deliverable."""

    def test_plan_gates_has_no_lazy_relative_imports(self):
        source = Path(inspect.getfile(plan_gates)).read_text()
        for line in source.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("from ."), (
                f"plan_gates.py must use absolute imports, found: {stripped!r}"
            )

    def test_plan_gates_never_mentions_metricool(self):
        source = Path(inspect.getfile(plan_gates)).read_text()
        assert "metricool" not in source.lower()

    def test_plan_builder_imports_plan_gates(self):
        from gomoveshift.publishing import plan_builder

        source = Path(inspect.getfile(plan_builder)).read_text()
        assert "gomoveshift.publishing.plan_gates" in source
