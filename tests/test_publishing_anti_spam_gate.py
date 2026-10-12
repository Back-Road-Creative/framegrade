"""Pure-function tests for ``publishing.anti_spam_gate.check_batch``."""

from __future__ import annotations

import hashlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from gomoveshift.publishing.anti_spam_gate import (
    LEGACY_TITLE_POOL,
    check_batch,
    shared_prefix_groups,
    title_prefix_key,
)
from gomoveshift.publishing.models import (
    FacebookPayload,
    InstagramPayload,
    MediaFingerprint,
    PlatformPayloads,
    RouteSelection,
    ScheduleWindow,
    ShortPublishRecord,
    TikTokPayload,
    YouTubePayload,
)

SCHEDULED = datetime(2026, 9, 1, 19, 0, tzinfo=timezone.utc)

_MEDIA_PATH = Path(tempfile.mkdtemp(prefix="gms-anti-spam-gate-test-")) / "short.mp4"
_MEDIA_PATH.write_bytes(b"gms anti-spam gate test media fixture bytes")
_REAL_MEDIA = MediaFingerprint(
    str(_MEDIA_PATH),
    _MEDIA_PATH.stat().st_size,
    2.0,
    1080,
    1920,
    "h264",
    hashlib.sha256(_MEDIA_PATH.read_bytes()).hexdigest(),
)


def _record(
    short_index=0,
    *,
    youtube=None,
    instagram=None,
    facebook=None,
    tiktok=None,
    sampled_regions=("kerry",),
):
    return ShortPublishRecord(
        project_name="proj",
        lf_index=0,
        short_index=short_index,
        media=_REAL_MEDIA,
        schedule=ScheduleWindow(SCHEDULED, "America/Chicago"),
        recording_date="2026-08-01",
        sampled_regions=sampled_regions,
        parent_lf_video_id="vid",
        parent_lf_publish_at=SCHEDULED,
        payloads=PlatformPayloads(
            youtube=youtube, instagram=instagram, facebook=facebook, tiktok=tiktok
        ),
        consent_hash="c",
        reviewed_plan_hash="HASH1",
        routes=RouteSelection(selected=("youtube",)),
    )


def _yt(title, description="a real description of this clip"):
    return YouTubePayload(title=title, description=description)


# -- empty batch -------------------------------------------------------


def test_empty_batch_has_no_violations():
    assert check_batch([], per_location_cap=None) == []


# -- rule 1: legacy-title -------------------------------------------------


def test_legacy_title_is_a_violation():
    legacy_text = next(iter(LEGACY_TITLE_POOL))
    record = _record(youtube=_yt(legacy_text))

    violations = check_batch([record], per_location_cap=None)

    assert len(violations) == 1
    assert violations[0].rule == "legacy-title"
    assert violations[0].index == 0
    assert violations[0].platform == "youtube"


def test_car_cam_template_stays_in_the_retired_pool():
    """``Car Cam: <place>`` collided with the classifier's cam-title rule.

    No live generator emits the template (it is only a retired-pool literal), so
    the classifier fix is what keeps those shorts out of Mountains Cam Archive.
    Deleting the entry would not stop anything being generated; it would drop the
    gate's literal-leak guard and library-hygiene's series naming for the
    already-published Car Cam shorts.
    """
    from gomoveshift.publishing.anti_spam_gate import LEGACY_TITLE_POOL_RAW

    assert "Car Cam: {location} 🚗" in LEGACY_TITLE_POOL_RAW
    record = _record(youtube=_yt("Car Cam: {location} 🚗"))

    violations = check_batch([record], per_location_cap=None)

    assert any(v.rule == "legacy-title" for v in violations)


def test_legacy_title_matches_case_and_whitespace_insensitively():
    legacy_text = next(iter(LEGACY_TITLE_POOL))
    record = _record(youtube=_yt(f"  {legacy_text.upper()}  "))

    violations = check_batch([record], per_location_cap=None)

    assert any(v.rule == "legacy-title" for v in violations)


def test_composed_title_not_in_pool_is_not_a_violation():
    record = _record(youtube=_yt("Kerry's coast road unwinds past three river bends"))

    violations = check_batch([record], per_location_cap=None)

    assert not any(v.rule == "legacy-title" for v in violations)


# -- rule 2: shared-prefix -------------------------------------------------


def test_two_titles_sharing_a_three_word_prefix_are_both_flagged():
    a = _record(0, youtube=_yt("Driving the coast road toward Exampleton"))
    b = _record(1, youtube=_yt("Driving the coast highway near Tralee"))

    violations = check_batch([a, b], per_location_cap=None)

    prefix_violations = [v for v in violations if v.rule == "shared-prefix"]
    assert {v.index for v in prefix_violations} == {0, 1}


def test_shared_prefix_normalises_case_and_punctuation():
    a = _record(0, youtube=_yt("Driving, the Coast! road toward Exampleton"))
    b = _record(1, youtube=_yt("driving the coast — highway near Tralee"))

    violations = check_batch([a, b], per_location_cap=None)

    assert {v.index for v in violations if v.rule == "shared-prefix"} == {0, 1}


def test_titles_with_different_prefixes_are_not_flagged():
    a = _record(0, youtube=_yt("Driving the coast road toward Exampleton"))
    b = _record(1, youtube=_yt("Climbing the mountain pass near Killarney"))

    violations = check_batch([a, b], per_location_cap=None)

    assert not any(v.rule == "shared-prefix" for v in violations)


def test_short_titles_under_three_words_are_never_prefix_flagged():
    a = _record(0, youtube=_yt("Kerry drive"))
    b = _record(1, youtube=_yt("Kerry drive"))

    violations = check_batch([a, b], per_location_cap=None)

    assert not any(v.rule == "shared-prefix" for v in violations)


# -- rule 3: caption-collision -------------------------------------------------


def test_identical_caption_across_platforms_is_flagged():
    record = _record(
        youtube=_yt("A grounded, specific title here", description="Same words every time"),
        instagram=InstagramPayload(caption="Same words every time"),
    )

    violations = check_batch([record], per_location_cap=None)

    collisions = [v for v in violations if v.rule == "caption-collision"]
    assert len(collisions) == 1
    assert collisions[0].index == 0
    assert collisions[0].platform == "instagram"


def test_different_captions_across_platforms_are_not_flagged():
    record = _record(
        youtube=_yt("A grounded, specific title here", description="youtube-style description"),
        instagram=InstagramPayload(caption="instagram-style caption"),
        facebook=FacebookPayload(description="facebook-style description"),
        tiktok=TikTokPayload(title="tt", privacy_level="SELF_ONLY", caption="tiktok caption"),
    )

    violations = check_batch([record], per_location_cap=None)

    assert not any(v.rule == "caption-collision" for v in violations)


def test_empty_texts_are_never_compared_for_collision():
    record = _record(
        youtube=_yt("A grounded, specific title here", description=""),
        instagram=InstagramPayload(caption=""),
    )

    violations = check_batch([record], per_location_cap=None)

    assert not any(v.rule == "caption-collision" for v in violations)


# -- rule 4: location-cap -------------------------------------------------


def test_location_cap_none_never_flags():
    records = [_record(i, sampled_regions=("kerry",)) for i in range(10)]

    violations = check_batch(records, per_location_cap=None)

    assert not any(v.rule == "location-cap" for v in violations)


def test_location_over_cap_flags_the_excess_indices():
    records = [_record(i, sampled_regions=("kerry",)) for i in range(4)]

    violations = check_batch(records, per_location_cap=2)

    cap_violations = [v for v in violations if v.rule == "location-cap"]
    assert {v.index for v in cap_violations} == {2, 3}
    assert all(v.platform is None for v in cap_violations)


def test_location_at_or_under_cap_is_not_flagged():
    records = [_record(i, sampled_regions=("kerry",)) for i in range(2)]

    violations = check_batch(records, per_location_cap=2)

    assert not any(v.rule == "location-cap" for v in violations)


def test_records_with_no_sampled_regions_are_excluded_from_location_cap():
    records = [_record(i, sampled_regions=()) for i in range(5)]

    violations = check_batch(records, per_location_cap=2)

    assert not any(v.rule == "location-cap" for v in violations)


def test_fixture_pool_is_the_gate_pool_not_a_copy():
    """The test fixture re-exports the gate's raw pool.

    One source of truth: ``anti_spam_gate.LEGACY_TITLE_POOL_RAW`` owns the 100
    retired template strings; ``tests/fixtures/legacy_title_pool.py`` is a
    thin handle over it, never a second literal that can drift.
    """
    from fixtures.legacy_title_pool import LEGACY_TITLE_POOL as FIXTURE_POOL
    from gomoveshift.publishing.anti_spam_gate import LEGACY_TITLE_POOL_RAW

    assert FIXTURE_POOL == list(LEGACY_TITLE_POOL_RAW)
    assert len(LEGACY_TITLE_POOL_RAW) == 100
    assert LEGACY_TITLE_POOL == frozenset(t.strip().lower() for t in LEGACY_TITLE_POOL_RAW)


def test_legacy_title_is_still_refused_under_the_builders_title_hashtags():
    # plan_builder appends the description's hashtags to the YouTube title;
    # a legacy pool title must not slip past the pool by wearing them.
    legacy_text = next(iter(LEGACY_TITLE_POOL))
    record = _record(youtube=_yt(f"{legacy_text} #CountyKerry #Shorts"))
    violations = check_batch([record], per_location_cap=None)
    assert any(v.rule == "legacy-title" for v in violations)


def test_shared_prefix_reads_the_title_not_its_hashtags():
    # Two-word titles are below the prefix rule's 3-word floor; the hashtags
    # the builder appends must not lift them over it.
    a = _record(0, youtube=_yt("Kenmare Bay #Shorts", "first clip text"))
    b = _record(1, youtube=_yt("Kenmare Bay #Shorts", "second clip text"))
    violations = check_batch([a, b], per_location_cap=None)
    assert not any(v.rule == "shared-prefix" for v in violations)


def test_title_prefix_key_strips_hashtags_and_short_titles():
    assert title_prefix_key("Sep 14, 3rd stretch: Wythe County #Shorts #Drive") == (
        "sep",
        "14",
        "3rd",
    )
    assert title_prefix_key("Two words #Shorts #Drive #More") is None
    assert title_prefix_key("") is None
    assert shared_prefix_groups(["A b c one", "", "a B c two", "Other title entirely", "A b"]) == {
        ("a", "b", "c"): [0, 2]
    }
