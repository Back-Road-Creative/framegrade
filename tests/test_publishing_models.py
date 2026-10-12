"""Tests for gomoveshift.publishing immutable models + deterministic plan hashing.

Mirrors compute_gate_cd_hash's exclusion discipline: covers every
consent-relevant field, excludes wall-clock/generation fields.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest

from gomoveshift.publishing.models import (
    FacebookPayload,
    InstagramPayload,
    MediaFingerprint,
    PayloadLimitError,
    PlatformPayloads,
    PublishPlan,
    RouteSelection,
    ScheduleWindow,
    ShortPublishRecord,
    TikTokPayload,
    YouTubePayload,
    validate_payloads,
)
from gomoveshift.pipeline.encoding_constants import PAYLOAD_LIMITS
from gomoveshift.publishing.plan import compute_plan_hash


def _media(sha: str = "a" * 64) -> MediaFingerprint:
    return MediaFingerprint(
        path="/output/lf1/short_003.mp4",
        size_bytes=12_345,
        duration_seconds=42.5,
        width=1080,
        height=1920,
        codec="h264",
        sha256=sha,
    )


def _payloads(caption: str = "Golden hour on the coast road") -> PlatformPayloads:
    return PlatformPayloads(
        youtube=YouTubePayload(title="Golden Hour Coast Drive", description="desc", category="24"),
        instagram=InstagramPayload(caption=caption),
        facebook=FacebookPayload(description="desc"),
        tiktok=TikTokPayload(title="Golden Hour Coast Drive", privacy_level="PUBLIC_TO_EVERYONE"),
    )


def _record(
    *,
    generated_at: datetime | None = None,
    media_sha: str = "a" * 64,
    caption: str = "Golden hour on the coast road",
    schedule_hour: int = 14,
    routes: tuple[str, ...] = ("youtube", "instagram"),
    consent_hash: str = "c" * 64,
) -> ShortPublishRecord:
    return ShortPublishRecord(
        project_name="example-2026-08",
        lf_index=1,
        short_index=3,
        media=_media(sha=media_sha),
        schedule=ScheduleWindow(
            scheduled_at_utc=datetime(2026, 8, 24, schedule_hour, 0, tzinfo=timezone.utc),
            display_timezone="America/New_York",
        ),
        recording_date="2026-08-20",
        sampled_regions=("County Kerry",),
        parent_lf_video_id="yt_abc123",
        parent_lf_publish_at=datetime(2026, 8, 24, 14, 0, tzinfo=timezone.utc),
        payloads=_payloads(caption=caption),
        consent_hash=consent_hash,
        reviewed_plan_hash="d" * 64,
        routes=RouteSelection(selected=routes),
        generated_at=generated_at or datetime(2026, 8, 23, 9, 0, tzinfo=timezone.utc),
    )


def _plan(**kwargs) -> PublishPlan:
    return PublishPlan(shorts=(_record(**kwargs),))


def test_identical_construction_yields_same_hash():
    assert compute_plan_hash(_plan()) == compute_plan_hash(_plan())


def test_generation_timestamp_does_not_change_hash():
    h1 = compute_plan_hash(_plan(generated_at=datetime(2026, 8, 23, 9, 0, tzinfo=timezone.utc)))
    h2 = compute_plan_hash(_plan(generated_at=datetime(2026, 8, 23, 23, 59, tzinfo=timezone.utc)))
    assert h1 == h2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"media_sha": "b" * 64},
        {"caption": "Different caption entirely"},
        {"schedule_hour": 18},
        {"routes": ("youtube", "tiktok")},
        {"consent_hash": "e" * 64},
    ],
    ids=["media_sha", "caption", "schedule", "route_selection", "consent_hash"],
)
def test_field_change_changes_hash(kwargs):
    assert compute_plan_hash(_plan()) != compute_plan_hash(_plan(**kwargs))


def test_short_publish_record_rejects_mutation():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _record().lf_index = 99


def test_media_fingerprint_rejects_mutation():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _media().sha256 = "z" * 64


def test_publish_plan_rejects_mutation():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _plan().shorts = ()


def test_empty_route_set_is_rejected():
    with pytest.raises(ValueError, match="route"):
        RouteSelection(selected=())


def test_unknown_platform_is_rejected():
    with pytest.raises(ValueError, match="platform"):
        RouteSelection(selected=("youtube", "mastodon"))


# --- validate_payloads -------------------------------------------------


def test_validate_payloads_accepts_text_at_the_limit():
    limit = PAYLOAD_LIMITS["youtube"].title_max_len
    payloads = PlatformPayloads(youtube=YouTubePayload(title="x" * limit, description="d"))
    validate_payloads(payloads)  # must not raise


def test_validate_payloads_refuses_youtube_title_one_over_the_limit():
    limit = PAYLOAD_LIMITS["youtube"].title_max_len
    payloads = PlatformPayloads(youtube=YouTubePayload(title="x" * (limit + 1), description="d"))
    with pytest.raises(PayloadLimitError, match="youtube") as exc:
        validate_payloads(payloads)
    assert "title" in str(exc.value)
    assert str(limit + 1) in str(exc.value)
    assert str(limit) in str(exc.value)


def test_validate_payloads_refuses_youtube_description_one_over_the_limit():
    limit = PAYLOAD_LIMITS["youtube"].description_max_len
    payloads = PlatformPayloads(youtube=YouTubePayload(title="t", description="d" * (limit + 1)))
    with pytest.raises(PayloadLimitError, match="description"):
        validate_payloads(payloads)


def test_validate_payloads_refuses_youtube_tags_one_over_the_limit():
    limit = PAYLOAD_LIMITS["youtube"].tags_total_max_len
    payloads = PlatformPayloads(
        youtube=YouTubePayload(title="t", description="d", tags=("x" * (limit + 1),))
    )
    with pytest.raises(PayloadLimitError, match="tags"):
        validate_payloads(payloads)


def test_validate_payloads_refuses_instagram_caption_one_over_the_limit():
    limit = PAYLOAD_LIMITS["instagram"].caption_max_len
    payloads = PlatformPayloads(instagram=InstagramPayload(caption="c" * (limit + 1)))
    with pytest.raises(PayloadLimitError, match="instagram") as exc:
        validate_payloads(payloads)
    assert "caption" in str(exc.value)


def test_validate_payloads_refuses_facebook_description_one_over_the_limit():
    limit = PAYLOAD_LIMITS["facebook"].description_max_len
    payloads = PlatformPayloads(facebook=FacebookPayload(description="d" * (limit + 1)))
    with pytest.raises(PayloadLimitError, match="facebook"):
        validate_payloads(payloads)


def test_validate_payloads_refuses_tiktok_caption_one_over_the_limit():
    limit = PAYLOAD_LIMITS["tiktok"].caption_max_len
    payloads = PlatformPayloads(
        tiktok=TikTokPayload(title="t", privacy_level="SELF_ONLY", caption="c" * (limit + 1))
    )
    with pytest.raises(PayloadLimitError, match="tiktok") as exc:
        validate_payloads(payloads)
    assert "caption" in str(exc.value)


def test_validate_payloads_accepts_all_platforms_absent():
    validate_payloads(PlatformPayloads())  # must not raise
