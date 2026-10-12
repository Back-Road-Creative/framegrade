"""TDD for ``build_publish_plan``: payloads follow the mapping doc (no
Twitter truncation); the CSV path's validation gates are reused."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gomoveshift.publishing.models import MediaFingerprint
from gomoveshift.publishing.plan import compute_plan_hash
from gomoveshift.publishing.plan_builder import build_publish_plan


def _media(short_index: int = 1, sha: str = "a" * 64) -> MediaFingerprint:
    return MediaFingerprint(
        path=f"/output/lf1/short_{short_index:03d}.mp4",
        size_bytes=12_345,
        duration_seconds=42.5,
        width=1080,
        height=1920,
        codec="h264",
        sha256=sha,
    )


def _post(short_index=1, *, caption="Rolling through County Kerry today", **overrides):
    fields = {
        "short_index": short_index,
        "caption": caption,
        "youtube_title": f"County Kerry Drive #{short_index}",
        "youtube_category": "24",
        "youtube_tags": "County Kerry,road trip",
        "youtube_description": "A short drive through the hills.",
        "tiktok_title": f"tt {short_index}",
        "pinterest_board": "drives",
        "first_comment": f"#tag{short_index}",
        "sampled_regions": ["County Kerry, Munster"],
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _slot(short_index=1, date="2026-04-01", time="16:00:00"):
    return SimpleNamespace(short_index=short_index, date=date, time=time)


def _build(*, posts=None, parent_publish_at="2026-04-01T18:00:00+00:00", **overrides):
    posts = posts or [_post()]
    kwargs = dict(
        project_name="proj",
        shorts_by_lf={1: posts},
        slots_by_short_index={p.short_index: _slot(p.short_index) for p in posts},
        media_by_short_index={p.short_index: _media(p.short_index) for p in posts},
        parent_lf_video_id_by_short_index={p.short_index: "vidparent" for p in posts},
        parent_recording_date_by_short_index={p.short_index: "2026-04-01" for p in posts},
        parent_publish_at_by_short_index={p.short_index: parent_publish_at for p in posts},
    )
    kwargs.update(overrides)
    return build_publish_plan(**kwargs)


def test_payloads_match_mapping_doc_for_representative_short():
    record = _build().shorts[0]
    assert record.payloads.youtube.title == "County Kerry Drive #1"
    assert record.payloads.youtube.category == "24"
    assert record.payloads.youtube.tags == ("County Kerry", "road trip")
    assert record.payloads.tiktok.title == "tt 1"
    assert record.payloads.tiktok.privacy_level == "SELF_ONLY"
    expected_caption = "Rolling through County Kerry today. Filmed April 2026."
    assert record.payloads.instagram.caption == expected_caption
    assert record.payloads.facebook.description == expected_caption
    # Parent link lands only in the YouTube description, per the mapping doc.
    assert "https://youtu.be/vidparent" in record.payloads.youtube.description
    assert "youtu.be" not in record.payloads.instagram.caption
    assert "youtu.be" not in record.payloads.facebook.description


def test_long_caption_is_not_twitter_truncated():
    long_caption = "Driving the coast road. " * 20  # well over 280 chars
    assert len(long_caption) > 280
    record = _build(posts=[_post(caption=long_caption)]).shorts[0]
    assert record.payloads.instagram.caption.startswith(long_caption.rstrip())
    assert "…" not in record.payloads.instagram.caption
    assert len(record.payloads.instagram.caption) > 280
    assert len(record.payloads.facebook.description) > 280


@pytest.mark.parametrize(
    "exc_type, match, overrides",
    [
        (ValueError, "NO-SLOT", {"slots_by_short_index": {}}),
        (ValueError, "NO-MEDIA", {"media_by_short_index": {}}),
        (
            AssertionError,
            "SHORT-BEFORE-PUBLISH",
            {"parent_publish_at": "2026-04-01T22:00:00+00:00"},
        ),
        (ValueError, "(?i)region", {"posts": [_post(youtube_tags="County Cork,road trip")]}),
        (
            ValueError,
            "NO-PARENT-ID",
            {"posts": [_post(1), _post(2)], "parent_lf_video_id_by_short_index": {1: "vidparent"}},
        ),
    ],
    ids=["missing_slot", "missing_media", "before_publish", "foreign_region", "partial_parent_id"],
)
def test_unbuildable_plan_raises(exc_type, match, overrides):
    # before-publish: parent publishes 18:00 EDT, after the default 16:00 EDT slot.
    with pytest.raises(exc_type, match=match):
        _build(**overrides)


def test_plan_hash_stable_across_identical_builds():
    assert compute_plan_hash(_build()) == compute_plan_hash(_build())


@pytest.mark.parametrize(
    "overrides",
    [
        {"posts": [_post(caption="A totally different caption")]},
        {"slots_by_short_index": {1: _slot(time="17:00:00")}},
        {"media_by_short_index": {1: _media(sha="b" * 64)}},
        {"route_selected": ("youtube", "tiktok")},
    ],
    ids=["caption", "schedule", "media_sha", "routes"],
)
def test_plan_hash_changes_with_field(overrides):
    assert compute_plan_hash(_build()) != compute_plan_hash(_build(**overrides))


def test_tiktok_privacy_level_defaults_to_self_only_when_the_caller_omits_it():
    # Robust by construction: a caller that forgets the argument gets the
    # private audience, never a public post. The first audited TikTok run is
    # a canary, and a canary that publishes to everyone is not a canary.
    assert _build().shorts[0].payloads.tiktok.privacy_level == "SELF_ONLY"


def test_tiktok_privacy_level_is_taken_from_the_caller():
    record = _build(tiktok_privacy_level="PUBLIC_TO_EVERYONE").shorts[0]
    assert record.payloads.tiktok.privacy_level == "PUBLIC_TO_EVERYONE"


def test_unknown_tiktok_privacy_level_is_refused_before_any_record_is_built():
    # TikTok answers a bad privacy_level at /video/init/ time, which is after
    # the plan is written, reviewed and approved. Refuse it here instead.
    with pytest.raises(ValueError, match="privacy_level"):
        _build(tiktok_privacy_level="PUBLIC")


def test_tiktok_carries_the_same_caption_facebook_and_instagram_get():
    """TikTok used to carry only `tiktok_title` -- a few words, no hashtags,
    no filmed-date -- while FB/IG got the full caption from the same local.
    A numbered short would then publish a short label such as "Road trip"
    where Facebook gets a full paragraph. The caption is `post_info.title` on Direct
    Post and the text a human pastes on the inbox path; either way it is the
    caption that belongs here."""
    record = _build().shorts[0]
    expected_caption = "Rolling through County Kerry today. Filmed April 2026."
    assert record.payloads.tiktok.caption == expected_caption
    assert record.payloads.tiktok.caption == record.payloads.facebook.description


def test_tiktok_title_survives_beside_the_caption():
    """The short title still feeds the console list and the OST uniqueness
    check (`ost_validator.validate_uniqueness` reads `tiktok_title`), so
    adding the caption must not displace it."""
    record = _build().shorts[0]
    assert record.payloads.tiktok.title == "tt 1"


# -- hashtags where a viewer sees them ----------------------------------------
# A caption shaped "...\n\n#POVDriving #PA Filmed March 2030." has the date
# glued onto the hashtag line, and a YouTube title with no hashtag shows none
# where the Shorts player looks: the hashtags sit in the description, which
# the player folds away.


def test_filmed_date_closes_the_sentence_and_the_hashtags_close_the_caption():
    captions = {
        "facebook": "Worth the detour?\n\n#POVDriving #PA",
        "instagram": "From behind the wheel 🚗\n\n#PA #POVDriving #RoadTrip",
        "tiktok": "Just driving through Example Town\n\n#POVDriving #PA",
    }
    record = _build(posts=[_post(captions=captions)]).shorts[0]
    assert record.payloads.facebook.description == (
        "Worth the detour? Filmed April 2026.\n\n#POVDriving #PA"
    )
    assert record.payloads.instagram.caption == (
        "From behind the wheel 🚗 Filmed April 2026.\n\n#PA #POVDriving #RoadTrip"
    )
    assert record.payloads.tiktok.caption == (
        "Just driving through Example Town. Filmed April 2026.\n\n#POVDriving #PA"
    )


def test_a_number_sign_that_is_not_a_hashtag_stays_in_the_sentence():
    record = _build(posts=[_post(caption="Stop #2")]).shorts[0]
    assert record.payloads.facebook.description == "Stop #2. Filmed April 2026."


# -- a sentence that never ended ------------------------------------------
# A caption built "Just driving through Example Town Filmed March 2030." runs
# the date straight on into a sentence with no full stop.


def test_a_sentence_ending_on_a_word_gets_its_full_stop_before_the_date():
    captions = {
        "facebook": "Example Town — roads\n\n#POVDriving #RoadTrip",
        "instagram": "Just driving through Example Town",
        "tiktok": "Mile marker (17)",
    }
    record = _build(posts=[_post(captions=captions)]).shorts[0]
    assert record.payloads.facebook.description == (
        "Example Town — roads. Filmed April 2026.\n\n#POVDriving #RoadTrip"
    )
    assert record.payloads.instagram.caption == (
        "Just driving through Example Town. Filmed April 2026."
    )
    assert record.payloads.tiktok.caption == "Mile marker (17). Filmed April 2026."


def test_a_sentence_that_already_ended_gets_no_second_stop():
    captions = {
        "facebook": "Made it.",
        "instagram": "Wait for it…",
        "tiktok": "Which way!",
    }
    record = _build(posts=[_post(captions=captions)]).shorts[0]
    assert record.payloads.facebook.description == "Made it. Filmed April 2026."
    assert record.payloads.instagram.caption == "Wait for it… Filmed April 2026."
    assert record.payloads.tiktok.caption == "Which way! Filmed April 2026."


def test_youtube_short_title_carries_its_description_hashtags_most_specific_first():
    post = _post(
        youtube_title="Back roads of County Kerry",
        youtube_description="POV driving through County Kerry Pt.1 #Shorts #POVDriving #CountyKerry",
    )
    record = _build(posts=[post]).shorts[0]
    assert record.payloads.youtube.title == (
        "Back roads of County Kerry #CountyKerry #POVDriving #Shorts"
    )


def test_youtube_title_takes_at_most_three_hashtags():
    post = _post(
        youtube_title="Back roads of County Kerry",
        youtube_description="Kerry Pt.1 #Shorts #POVDriving #CoastalDrive #CountyKerry",
    )
    record = _build(posts=[post]).shorts[0]
    assert record.payloads.youtube.title == (
        "Back roads of County Kerry #CountyKerry #CoastalDrive #POVDriving"
    )


def test_youtube_title_hashtags_stop_before_the_100_char_limit():
    base = "The long slow coast road around the whole of the Iveragh peninsula, County Kerry"
    post = _post(
        youtube_title=base,
        youtube_description="Kerry Pt.1 #Shorts #POVDriving #CountyKerry",
    )
    title = _build(posts=[post]).shorts[0].payloads.youtube.title
    assert title == f"{base} #CountyKerry"
    assert len(title) <= 100


def test_youtube_title_never_repeats_a_hashtag_it_already_carries():
    post = _post(
        youtube_title="Kerry back roads #CountyKerry",
        youtube_description="Kerry Pt.1 #Shorts #countykerry",
    )
    record = _build(posts=[post]).shorts[0]
    assert record.payloads.youtube.title == "Kerry back roads #CountyKerry #Shorts"
