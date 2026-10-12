"""Immutable publish-plan models. See ``docs/social-publishing-plan-contract.md``."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

KNOWN_PLATFORMS = frozenset({"youtube", "instagram", "facebook", "tiktok"})

# The publishable formats this package can express. ``None`` on a record's
# ``format`` field means "not stated" and is a DISTINCT state from any of
# these -- see PublishRecord.format and plan_io's own, separate rule for
# sidecar files written before this field existed.
FORMAT_SHORT = "short"
FORMAT_PHOTO = "photo"
FORMAT_CAROUSEL = "carousel"
FORMAT_LONG_FORM = "long_form"
KNOWN_FORMATS = frozenset({FORMAT_SHORT, FORMAT_PHOTO, FORMAT_CAROUSEL, FORMAT_LONG_FORM})

# TikTok's documented audience values for ``privacy_level`` (Content Posting
# API, /v2/post/publish/creator_info/query/). An account only ever OFFERS a
# subset of these at a time — creator_info["privacy_level_options"] is the
# live authority, checked in tiktok_adapter.assert_public_routing_allowed.
# This set exists to refuse a typo ("PUBLIC") before a plan is ever built.
TIKTOK_PRIVACY_LEVELS = frozenset(
    {"PUBLIC_TO_EVERYONE", "MUTUAL_FOLLOW_FRIENDS", "FOLLOWER_OF_CREATOR", "SELF_ONLY"}
)


@dataclass(frozen=True, slots=True)
class MediaFingerprint:  # identity of one rendered/exported media item on local canonical storage
    path: str
    size_bytes: int
    duration_seconds: float
    width: int
    height: int
    codec: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ScheduleWindow:  # scheduled instant (UTC) + display timezone
    scheduled_at_utc: datetime
    display_timezone: str
    #: The platforms whose own best-time window moved them OFF
    #: ``scheduled_at_utc``, keyed by platform name. Empty is the normal
    #: case and means "every platform publishes at the slot the scheduler
    #: gave this short" -- a platform is listed only when a
    #: per-platform best-time window actually resolved a different hour for it, so the map never restates the slot.
    #: ``enqueue_from_plan`` reads it per delivery row; nothing else does.
    scheduled_at_utc_by_platform: dict[str, datetime] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class YouTubePayload:
    title: str
    description: str
    tags: tuple[str, ...] = ()
    category: str = ""


@dataclass(frozen=True, slots=True)
class InstagramPayload:
    caption: str


@dataclass(frozen=True, slots=True)
class FacebookPayload:
    description: str


@dataclass(frozen=True, slots=True)
class TikTokPayload:
    #: Short label -- the console list and `ost_validator` read this.
    title: str
    privacy_level: str
    #: The full caption: hashtags and filmed-date, same text FB/IG carry.
    #: TikTok has ONE text field (`post_info.title` on Direct Post; whatever
    #: the human pastes on the inbox path), and this is what belongs in it.
    #: Defaults to "" so older sidecars without it still load --
    #: those files are never regenerated, and their `reviewed_plan_hash`
    #: must not shift underneath a queued delivery.
    caption: str = ""
    disclose_branded_content: bool = False
    disclose_your_brand: bool = False


@dataclass(frozen=True, slots=True)
class PlatformPayloads:  # per-platform payloads; absent = None, no shared caption
    youtube: YouTubePayload | None = None
    instagram: InstagramPayload | None = None
    facebook: FacebookPayload | None = None
    tiktok: TikTokPayload | None = None


class PayloadLimitError(ValueError):
    """A populated platform payload's text field exceeds its documented
    limit -- names the platform, the field, the actual length, and the
    limit so an operator can fix it without re-deriving what tripped."""


def validate_payloads(payloads: PlatformPayloads) -> None:
    """Hard refusal when any populated platform payload's text field
    exceeds its documented limit. Limits live in
    ``pipeline.encoding_constants.PAYLOAD_LIMITS``, next to the media
    rendition limits -- one home for per-platform ceilings. Checks
    every populated platform, not just the first violation, so a plan with
    several over-limit fields is diagnosed in one pass.

    Import is function-local: ``pipeline`` imports from ``publishing``
    elsewhere in the package (e.g. ``phase_d.py``), so a module-level
    import here risks a circular import at package load time.
    """
    from ..pipeline.encoding_constants import PAYLOAD_LIMITS

    if payloads.youtube is not None:
        limits = PAYLOAD_LIMITS["youtube"]
        yt = payloads.youtube
        if len(yt.title) > limits.title_max_len:
            raise PayloadLimitError(
                f"youtube title: {len(yt.title)} chars > {limits.title_max_len} limit"
            )
        if len(yt.description) > limits.description_max_len:
            raise PayloadLimitError(
                f"youtube description: {len(yt.description)} chars > "
                f"{limits.description_max_len} limit"
            )
        tags_len = sum(len(t) for t in yt.tags)
        if tags_len > limits.tags_total_max_len:
            raise PayloadLimitError(
                f"youtube tags: combined {tags_len} chars > {limits.tags_total_max_len} limit"
            )

    if payloads.instagram is not None:
        limits = PAYLOAD_LIMITS["instagram"]
        caption = payloads.instagram.caption
        if len(caption) > limits.caption_max_len:
            raise PayloadLimitError(
                f"instagram caption: {len(caption)} chars > {limits.caption_max_len} limit"
            )

    if payloads.facebook is not None:
        limits = PAYLOAD_LIMITS["facebook"]
        description = payloads.facebook.description
        if len(description) > limits.description_max_len:
            raise PayloadLimitError(
                f"facebook description: {len(description)} chars > "
                f"{limits.description_max_len} limit"
            )

    if payloads.tiktok is not None:
        limits = PAYLOAD_LIMITS["tiktok"]
        caption = payloads.tiktok.caption
        if len(caption) > limits.caption_max_len:
            raise PayloadLimitError(
                f"tiktok caption: {len(caption)} chars > {limits.caption_max_len} limit"
            )


@dataclass(frozen=True, slots=True)
class RouteSelection:
    """Selected routes (non-empty subset of KNOWN_PLATFORMS)."""

    selected: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.selected:
            raise ValueError("RouteSelection.selected must name at least one route")
        unknown = set(self.selected) - KNOWN_PLATFORMS
        if unknown:
            raise ValueError(
                f"RouteSelection.selected names unknown platform(s): {sorted(unknown)}"
            )


@dataclass(frozen=True, slots=True)
class PublishRecord:  # one immutable publish record for a single publishable unit
    project_name: str
    lf_index: int
    short_index: int
    #: The record's one media item for every format except "carousel" (whose
    #: extra items live in ``additional_media``). Kept singular, not a tuple,
    #: because ``dispatch_queue.py`` and the adapters read ``record.media``
    #: directly for the short path.
    media: MediaFingerprint
    schedule: ScheduleWindow
    recording_date: str
    sampled_regions: tuple[str, ...]
    parent_lf_video_id: str
    parent_lf_publish_at: datetime
    payloads: PlatformPayloads
    consent_hash: str
    reviewed_plan_hash: str
    routes: RouteSelection
    generated_at: datetime = field(default_factory=datetime.utcnow)
    #: Explicit publishable format -- one of KNOWN_FORMATS, or ``None`` for
    #: "not stated". ``None`` must never be READ as FORMAT_SHORT by any
    #: consumer -- that silent default is the defect this field exists to
    #: kill. A builder that means "short" says so explicitly (see
    #: plan_builder.build_publish_plan). plan_io documents a separate,
    #: deliberate rule for sidecar files written before this field existed.
    format: Optional[str] = None
    #: Media items beyond ``media`` -- non-empty ONLY when format is
    #: FORMAT_CAROUSEL. This is what makes a many-item carousel expressible
    #: without a second record type: a short/photo/long-form record's single
    #: item is ``media`` itself, and __post_init__ refuses any other shape.
    additional_media: "tuple[MediaFingerprint, ...]" = ()

    def __post_init__(self) -> None:
        if self.format is not None and self.format not in KNOWN_FORMATS:
            raise ValueError(
                f"PublishRecord.format must be one of {sorted(KNOWN_FORMATS)} or "
                f"None (not stated), got {self.format!r}"
            )
        if self.format == FORMAT_CAROUSEL:
            if not self.additional_media:
                raise ValueError(
                    "PublishRecord.additional_media must be non-empty for "
                    f"format={FORMAT_CAROUSEL!r} -- a carousel needs at least two "
                    "media items"
                )
        elif self.additional_media:
            raise ValueError(
                "PublishRecord.additional_media is only valid for "
                f"format={FORMAT_CAROUSEL!r}; got format={self.format!r} with "
                f"{len(self.additional_media)} extra media item(s)"
            )


#: Back-compat alias -- dispatch.py, handlers.py and every adapter import
#: this name and read ``.media`` as a single MediaFingerprint; keep it valid.
ShortPublishRecord = PublishRecord


@dataclass(frozen=True, slots=True)
class PublishPlan:  # the full set of per-record publish records for a run, any format
    shorts: tuple[PublishRecord, ...] = ()
