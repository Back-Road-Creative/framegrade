"""Publish-plan builder — see docs/social-publishing-plan-builder.md.
Called from ``drive_phases.phase_d``; touches no other pipeline module."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from gomoveshift.pipeline.encoding_constants import PAYLOAD_LIMITS
from gomoveshift.pipeline.schedule_constants import EST
from gomoveshift.pipeline.text_format import (
    close_sentence,
    format_recording_date_pretty,
    hashtags_in,
    split_trailing_hashtags,
)
from gomoveshift.publishing.models import (
    FORMAT_SHORT,
    TIKTOK_PRIVACY_LEVELS,
    FacebookPayload,
    InstagramPayload,
    MediaFingerprint,
    PlatformPayloads,
    PublishPlan,
    RouteSelection,
    ScheduleWindow,
    ShortPublishRecord,
    TikTokPayload,
    YouTubePayload,
)
from gomoveshift.publishing.plan_gates import (
    assert_plan_text_grammar,
    assert_shorts_metadata_in_region,
    assert_shorts_not_scheduled_before_publish,
)

#: Hashtags a YouTube Short title carries at most -- the player shows the
#: title, and past three a title reads as a tag wall.
_TITLE_HASHTAG_MAX = 3


def _with_filmed(text: str, filmed: str) -> str:
    """The caption with ``filmed`` as the sentence after the caption's own and
    the hashtags closing the caption -- never the date glued onto the end of
    the hashtag line ("#POVDriving #XX Filmed March 2026."), and never run on into a
    sentence that had no full stop ("Just driving through Example Town Filmed
    March 2026.")."""
    body, hashtags = split_trailing_hashtags(text)
    sentence = f"{close_sentence(body)} {filmed}" if body else filmed
    return f"{sentence}\n\n{hashtags}" if hashtags else sentence


def _title_with_hashtags(title: str, description: str) -> str:
    """The YouTube title plus the description's own hashtags, most specific
    first (the generator writes them generic-to-specific: ``#Shorts``,
    ``#POVDriving``, terrain, place), at most ``_TITLE_HASHTAG_MAX`` in all,
    stopping at the first that would pass the title limit. The Shorts player
    shows the title and folds the description away, so hashtags left only in
    the description are hashtags no viewer sees. Taking them from the
    description keeps one hashtag choice per short -- an interactive short's
    description withholds the place tag, so its title does too."""
    if not title:
        return title
    max_len = PAYLOAD_LIMITS["youtube"].title_max_len
    have = {tag.lower() for tag in hashtags_in(title)}
    for tag in reversed(hashtags_in(description)):
        if len(have) >= _TITLE_HASHTAG_MAX:
            break
        if tag.lower() in have:
            continue
        if len(title) + 1 + len(tag) > max_len:
            break
        title = f"{title} {tag}"
        have.add(tag.lower())
    return title


def _est_to_utc(date_s: str, time_s: str) -> datetime:
    # Same EST-local -> UTC conversion as plan_gates.assert_shorts_not_scheduled_before_publish.
    time_s = time_s or "00:00:00"
    for tfmt in ("%H:%M:%S", "%H:%M"):
        try:
            naive = datetime.strptime(f"{date_s} {time_s}", "%Y-%m-%d " + tfmt)
        except ValueError:
            continue
        return naive.replace(tzinfo=EST).astimezone(timezone.utc)
    raise ValueError(f"unparseable slot date/time: {date_s!r} {time_s!r}")


def _slot_datetime_utc(slot) -> datetime:
    return _est_to_utc(
        (getattr(slot, "date", "") or "").strip(), (getattr(slot, "time", "") or "").strip()
    )


def _positions_within_day(slots_by_short_index: "Dict[int, object]") -> "Dict[int, int]":
    """Each short's ordinal within its own publish DAY, ranked by slot time.

    The scheduler fills a day's shorts from the region's window in order, so
    the short at position ``i`` is the one that took ``window[i]``. That is
    what makes a per-platform window assignable back onto the same shorts:
    Per-platform best-time scheduling reorders the window (its best hour moves to the front), and position
    ``i`` then names that platform's hour for the same short.
    """
    by_date: "Dict[str, list]" = defaultdict(list)
    for short_idx, slot in slots_by_short_index.items():
        date_s = (getattr(slot, "date", "") or "").strip()
        time_s = (getattr(slot, "time", "") or "").strip()
        by_date[date_s].append((time_s, short_idx))
    positions: "Dict[int, int]" = {}
    for entries in by_date.values():
        for position, (_, short_idx) in enumerate(sorted(entries)):
            positions[short_idx] = position
    return positions


def _per_platform_schedule(
    slot, position: int, shorts_times_by_platform: "Optional[Mapping[str, Sequence[str]]]"
) -> "Dict[str, datetime]":
    """The platforms whose own window puts this short somewhere ELSE.

    A platform whose resolved hour equals the slot's is left out: recording
    it would restate the slot in every record for no reader, and
    ``enqueue_from_plan`` already falls back to ``scheduled_at_utc``.
    """
    if not shorts_times_by_platform:
        return {}
    date_s = (getattr(slot, "date", "") or "").strip()
    base = _slot_datetime_utc(slot)
    moved: "Dict[str, datetime]" = {}
    for platform, window in shorts_times_by_platform.items():
        if position >= len(window):
            # This platform's window is shorter than the day's short count --
            # nothing to move it to, so it keeps the slot.
            continue
        when = _est_to_utc(date_s, (window[position] or "").strip())
        if when != base:
            moved[platform] = when
    return moved


def build_publish_plan(
    *,
    project_name: str,
    shorts_by_lf: "Dict[int, List]",
    slots_by_short_index: "Dict[int, object]",
    media_by_short_index: "Dict[int, MediaFingerprint]",
    parent_lf_video_id_by_short_index: "Dict[int, str]",
    parent_recording_date_by_short_index: "Dict[int, str]",
    parent_publish_at_by_short_index: "Dict[int, str]",
    route_selected: "Tuple[str, ...]" = ("youtube", "instagram", "facebook", "tiktok"),
    tiktok_privacy_level: str = "SELF_ONLY",
    shorts_times_by_platform: "Optional[Mapping[str, Sequence[str]]]" = None,
    consent_hash: str = "",
    reviewed_plan_hash: str = "",
    display_timezone: str = "America/New_York",
) -> PublishPlan:
    if tiktok_privacy_level not in TIKTOK_PRIVACY_LEVELS:
        raise ValueError(
            f"tiktok privacy_level must be one of {sorted(TIKTOK_PRIVACY_LEVELS)}, "
            f"got {tiktok_privacy_level!r}"
        )
    assert_shorts_metadata_in_region(shorts_by_lf)
    assert_plan_text_grammar(shorts_by_lf)
    if parent_publish_at_by_short_index:
        assert_shorts_not_scheduled_before_publish(
            shorts_by_lf, slots_by_short_index, parent_publish_at_by_short_index
        )
    routes = RouteSelection(selected=route_selected)
    positions = _positions_within_day(slots_by_short_index)
    records = []
    for lf_idx in sorted(shorts_by_lf):
        for post in shorts_by_lf[lf_idx]:
            short_idx = post.short_index
            slot = slots_by_short_index.get(short_idx)
            if slot is None:
                raise ValueError(f"PUBLISH-PLAN-NO-SLOT: short_index {short_idx} has no slot")
            media = media_by_short_index.get(short_idx)
            if media is None or not getattr(media, "sha256", ""):
                raise ValueError(f"PUBLISH-PLAN-NO-MEDIA: short_index {short_idx} has no sha256")
            recording_date = parent_recording_date_by_short_index.get(short_idx)
            if not isinstance(recording_date, str) or not recording_date.strip():
                raise ValueError(f"recording date missing for short_index {short_idx}")
            # Same recording-date suffix as the CSV path, MINUS Twitter-length
            # truncation (only needed for the retired X/Twitter shared column).
            filmed = f"Filmed {format_recording_date_pretty(recording_date)}."
            # Each platform reads its OWN composed caption
            # (``post.captions``), falling back to the legacy shared
            # ``post.caption`` column when the dict lacks the key (an old
            # social_posts.json sidecar written before per-platform captions, or a caller that
            # never populated ``captions``).
            post_captions = getattr(post, "captions", None) or {}
            legacy_caption = (getattr(post, "caption", "") or "").rstrip()

            def _platform_caption(platform: str) -> str:
                text = (post_captions.get(platform) or legacy_caption or "").rstrip()
                return _with_filmed(text, filmed)

            instagram_caption = _platform_caption("instagram")
            facebook_caption = _platform_caption("facebook")
            tiktok_caption = _platform_caption("tiktok")
            parent_id = parent_lf_video_id_by_short_index.get(short_idx, "")
            parent_publish_iso = parent_publish_at_by_short_index.get(short_idx)
            if not parent_id or not parent_publish_iso:
                raise ValueError(f"PUBLISH-PLAN-NO-PARENT-ID: short_index {short_idx} has none")
            parent_link = f"https://youtu.be/{parent_id}"
            yt_description = (getattr(post, "youtube_description", "") or "").strip()
            tags = tuple(
                t.strip() for t in (getattr(post, "youtube_tags", "") or "").split(",") if t.strip()
            )
            yt_desc = f"{yt_description}\n\n{parent_link}" if yt_description else parent_link
            payloads = PlatformPayloads(
                youtube=YouTubePayload(
                    title=_title_with_hashtags(
                        getattr(post, "youtube_title", "") or "", yt_description
                    ),
                    description=yt_desc,
                    tags=tags,
                    category=getattr(post, "youtube_category", "") or "",
                ),
                instagram=InstagramPayload(caption=instagram_caption),
                facebook=FacebookPayload(description=facebook_caption),
                tiktok=TikTokPayload(
                    title=getattr(post, "tiktok_title", "") or "",
                    caption=tiktok_caption,
                    privacy_level=tiktok_privacy_level,
                ),
            )
            parent_publish_at = datetime.fromisoformat(parent_publish_iso).astimezone(timezone.utc)
            records.append(
                ShortPublishRecord(
                    project_name=project_name,
                    lf_index=lf_idx,
                    short_index=short_idx,
                    media=media,
                    schedule=ScheduleWindow(
                        scheduled_at_utc=_slot_datetime_utc(slot),
                        display_timezone=display_timezone,
                        scheduled_at_utc_by_platform=_per_platform_schedule(
                            slot, positions[short_idx], shorts_times_by_platform
                        ),
                    ),
                    recording_date=recording_date,
                    sampled_regions=tuple(getattr(post, "sampled_regions", None) or ()),
                    parent_lf_video_id=parent_id,
                    parent_lf_publish_at=parent_publish_at,
                    payloads=payloads,
                    consent_hash=consent_hash,
                    reviewed_plan_hash=reviewed_plan_hash,
                    routes=routes,
                    # This builder only ever produces shorts -- say so
                    # explicitly rather than leaving format unstated.
                    format=FORMAT_SHORT,
                )
            )
    return PublishPlan(shorts=tuple(records))
