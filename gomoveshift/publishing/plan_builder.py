"""Publish-plan builder — see docs/social-publishing-plan-builder.md.
Called from ``drive_phases.phase_d``; touches no other pipeline module."""

from __future__ import annotations

import dataclasses
import logging
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
from gomoveshift.pipeline.title_decision import require_title_decision
from gomoveshift.publishing.asset_plan import review_hash
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

logger = logging.getLogger(__name__)

#: Hashtags a YouTube Short title carries at most -- the player shows the
#: title, and past three a title reads as a tag wall.
_TITLE_HASHTAG_MAX = 3


#: The sentence a Short's Facebook and Instagram captions end on once its
#: parent long-form is public. "Long version", not "full drive": a heavy-trim
#: parent is a highlights cut (``csv_generator._first_comment_lead``).
LONG_VERSION_POINTER = "The long version is on our YouTube channel."


def _with_filmed(text: str, filmed: str) -> str:
    """The caption with ``filmed`` as the sentence after the caption's own and
    the hashtags closing the caption -- never the date glued onto the end of
    the hashtag line ("#POVDriving #XX Filmed March 2026."), and never run on into a
    sentence that had no full stop ("Just driving through Example Town Filmed
    March 2026.")."""
    body, hashtags = split_trailing_hashtags(text)
    sentence = f"{close_sentence(body)} {filmed}" if body else filmed
    return f"{sentence}\n\n{hashtags}" if hashtags else sentence


def _title_with_hashtags(title: str, description: str, regions: Sequence[str] = ()) -> str:
    """The YouTube title plus the description's own hashtags, most specific
    first (the generator writes them generic-to-specific: ``#Shorts``,
    ``#POVDriving``, terrain, place), at most ``_TITLE_HASHTAG_MAX`` in all,
    stopping at the first that would pass the title limit. The Shorts player
    shows the title and folds the description away, so hashtags left only in
    the description are hashtags no viewer sees. Taking them from the
    description keeps one hashtag choice per short -- an interactive short's
    description withholds the place tag, so its title does too.

    With region votes (a GPS-bearing Short) a tag the title review cannot back is
    skipped (``title_hashtag_backed``): a description may name the country, a title
    whose clip votes only the county may not."""
    if not title:
        return title
    from .short_title_review import title_hashtag_backed

    max_len = PAYLOAD_LIMITS["youtube"].title_max_len
    have = {tag.lower() for tag in hashtags_in(title)}
    for tag in reversed(hashtags_in(description)):
        if len(have) >= _TITLE_HASHTAG_MAX:
            break
        if tag.lower() in have or (regions and not title_hashtag_backed(tag, regions)):
            continue
        if len(title) + 1 + len(tag) > max_len:
            break
        title = f"{title} {tag}"
        have.add(tag.lower())
    return title


def _repair_short_tags(payloads: PlatformPayloads, regions, roads) -> PlatformPayloads:
    """``payloads`` with the YouTube tags limited to what the final-text review can
    check (``reviewable_short_tags``, the filter ``sync-plan-tags`` applies to a queued
    Short): a Short whose only fault is an unreviewable tag drops it and passes."""
    from gomoveshift.pipeline.final_text_record import reviewable_short_tags

    yt = payloads.youtube
    if yt is None or not regions:  # a GPS-less Short has no review to satisfy
        return payloads
    tags = tuple(reviewable_short_tags(yt.tags, regions, roads))
    if tags == tuple(yt.tags):
        return payloads
    return dataclasses.replace(payloads, youtube=dataclasses.replace(yt, tags=tags))


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


def build_short_payloads(
    post,
    *,
    recording_date: str,
    parent_id: str,
    tiktok_privacy_level: str,
    short_at,
    parent_at,
) -> PlatformPayloads:
    """The four platform payloads of one Short from its generated ``post``.

    One place for the plan's text assembly (filmed suffix, parent link, title
    hashtags), shared by :func:`build_publish_plan` and the queued-Short text sync.
    The YouTube description ends with the parent link only when the parent is
    public by the Short's slot ``short_at`` (``parent_is_public_at``)."""
    from .first_comment import parent_is_public_at

    # Same recording-date suffix as the CSV path, MINUS Twitter-length
    # truncation (only needed for the retired X/Twitter shared column).
    filmed = f"Filmed {format_recording_date_pretty(recording_date)}."
    # Each platform reads its OWN composed caption (``post.captions``), falling back
    # to the legacy shared ``post.caption`` column when the dict lacks the key (an old
    # social_posts.json sidecar written before per-platform captions, or a caller that
    # never populated ``captions``).
    post_captions = getattr(post, "captions", None) or {}
    legacy_caption = (getattr(post, "caption", "") or "").rstrip()

    parent_link = (
        f"https://youtu.be/{parent_id}" if parent_is_public_at(short_at, parent_at) else ""
    )
    # Facebook and Instagram carry no link (Instagram captions don't click and
    # the first comment already holds it), so they name where the long version
    # is -- only once the parent is public, so it never points at nothing.
    pointer = f" {LONG_VERSION_POINTER}" if parent_link else ""

    def _platform_caption(platform: str, points_to_parent: bool = False) -> str:
        text = (post_captions.get(platform) or legacy_caption or "").rstrip()
        return _with_filmed(text, filmed + (pointer if points_to_parent else ""))

    yt_description = (getattr(post, "youtube_description", "") or "").strip()
    tags = tuple(
        t.strip() for t in (getattr(post, "youtube_tags", "") or "").split(",") if t.strip()
    )
    yt_desc = "\n\n".join(part for part in (yt_description, parent_link) if part)
    return PlatformPayloads(
        youtube=YouTubePayload(
            title=_title_with_hashtags(
                getattr(post, "youtube_title", "") or "",
                yt_description,
                tuple(getattr(post, "sampled_regions", None) or ()),
            ),
            description=yt_desc,
            tags=tags,
            category=getattr(post, "youtube_category", "") or "",
        ),
        instagram=InstagramPayload(caption=_platform_caption("instagram", True)),
        facebook=FacebookPayload(description=_platform_caption("facebook", True)),
        tiktok=TikTokPayload(
            title=getattr(post, "tiktok_title", "") or "",
            caption=_platform_caption("tiktok"),
            privacy_level=tiktok_privacy_level,
        ),
    )


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
            # U5: the title the payload is built from must be Waypost's
            # decided title; the hashtag append below is the only change after.
            require_title_decision(
                getattr(post, "youtube_title", "") or "",
                {"title_decision": getattr(post, "title_decision", None)},
            )
            slot = slots_by_short_index.get(short_idx)
            if slot is None:
                raise ValueError(f"PUBLISH-PLAN-NO-SLOT: short_index {short_idx} has no slot")
            media = media_by_short_index.get(short_idx)
            if media is None or not getattr(media, "sha256", ""):
                raise ValueError(f"PUBLISH-PLAN-NO-MEDIA: short_index {short_idx} has no sha256")
            recording_date = parent_recording_date_by_short_index.get(short_idx)
            if not isinstance(recording_date, str) or not recording_date.strip():
                raise ValueError(f"recording date missing for short_index {short_idx}")
            parent_id = parent_lf_video_id_by_short_index.get(short_idx, "")
            parent_publish_iso = parent_publish_at_by_short_index.get(short_idx)
            if not parent_id or not parent_publish_iso:
                raise ValueError(f"PUBLISH-PLAN-NO-PARENT-ID: short_index {short_idx} has none")
            parent_publish_at = datetime.fromisoformat(parent_publish_iso).astimezone(timezone.utc)
            payloads = build_short_payloads(
                post,
                recording_date=recording_date,
                parent_id=parent_id,
                tiktok_privacy_level=tiktok_privacy_level,
                short_at=_slot_datetime_utc(slot),
                parent_at=parent_publish_at,
            )
            payloads = _repair_short_tags(
                payloads,
                tuple(getattr(post, "sampled_regions", None) or ()),
                tuple(getattr(post, "sampled_roads", None) or ()),
            )
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
                    sampled_roads=tuple(getattr(post, "sampled_roads", None) or ()),
                    route_record=getattr(post, "route_record", None) or None,
                    bracket_route=getattr(post, "bracket_route", None) or None,
                    experiment_arms=dict(getattr(post, "experiment_arms", None) or {}),
                    caption_templates=dict(getattr(post, "caption_templates", None) or {}),
                    parent_lf_video_id=parent_id,
                    parent_lf_publish_at=parent_publish_at,
                    payloads=payloads,
                    consent_hash=consent_hash,
                    # Never left empty: the Short review needs an approval marker, so
                    # a caller with none gets the hash of the text this record sends.
                    reviewed_plan_hash=reviewed_plan_hash
                    or review_hash(
                        payloads.youtube.title,
                        payloads.youtube.description,
                        payloads.youtube.tags,
                        media.sha256,
                    ),
                    routes=routes,
                    # This builder only ever produces shorts -- say so
                    # explicitly rather than leaving format unstated.
                    format=FORMAT_SHORT,
                )
            )
    _warn_refused(records)
    return PublishPlan(shorts=tuple(records))


def _warn_refused(records: "List[ShortPublishRecord]") -> None:
    """Log every Short the final-text review still refuses after the tag repair. The
    record stays in the plan: ``enqueue_from_plan`` leaves it out of the queue and
    raises the incident, so the plan's other Shorts keep their slots."""
    from gomoveshift.publishing.youtube_adapter import short_review_refusal

    for record in records:
        if record.payloads.youtube is None:
            continue
        reason = short_review_refusal(record)
        if reason:
            logger.warning(
                "publish plan: short %s #%s will be refused at enqueue: %s",
                record.project_name,
                record.short_index,
                reason,
            )
