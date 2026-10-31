"""Rehydrate a Phase D publish-plan sidecar back into ``PublishPlan`` models.

The inverse of ``phase_d._serialize_publish_plan``. That writer is
``json.dumps(asdict(plan), sort_keys=True, indent=2,
default=_json_datetime_default)`` with each record's ``generated_at``
popped, so this reader must undo exactly three transforms:

1. ``asdict`` flattens every nested dataclass to a plain dict.
2. ``asdict`` turns each ``tuple`` field into a list; the models declare
   ``tuple[...]`` and ``RouteSelection`` is frozen, so they must come back
   as tuples.
3. ``_json_datetime_default`` renders ``datetime`` via ``.isoformat()``.

``generated_at`` is NOT in the file — the writer strips it so the gate C/D
hash stays byte-stable — so a rehydrated record takes the model default. It
is a wall-clock stamp of when the plan was BUILT and nothing reads it back;
treating its absence as an error would reject every valid sidecar.

The sidecar is untrusted JSON on disk: every failure raises
:class:`PublishPlanFormatError` naming the file path and the offending
field, never a bare ``KeyError``/``TypeError``/``ValueError``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Union

from gomoveshift.pipeline.text_format import hashtags_in

from .models import (
    FORMAT_SHORT,
    KNOWN_FORMATS,
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
from .plan_gates import assert_plan_text_grammar, assert_shorts_metadata_in_region


class PublishPlanFormatError(ValueError):
    """Sidecar JSON is malformed or shaped wrong. Message names the path."""


def _fail(path: Path, what: str) -> "PublishPlanFormatError":
    return PublishPlanFormatError(f"publish-plan sidecar {path}: {what}")


def _obj(raw: Any, key: str, path: Path, ctx: str) -> Dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise _fail(path, f"{key!r} must be an object, got {type(value).__name__} ({ctx})")
    return value


def _dt(raw: Any, key: str, path: Path, ctx: str) -> datetime:
    value = raw.get(key)
    if not isinstance(value, str):
        raise _fail(path, f"{key!r} must be an ISO-8601 string ({ctx})")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise _fail(path, f"{key!r} is not a valid ISO-8601 timestamp: {exc} ({ctx})") from exc


def _str_tuple(raw: Any, key: str, path: Path, ctx: str) -> tuple[str, ...]:
    value = raw.get(key)
    # asdict() wrote a tuple field out as a JSON array; it must come back a
    # tuple because the models are frozen with tuple[...] fields.
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise _fail(path, f"{key!r} must be a list of strings ({ctx})")
    return tuple(value)


def _str_tuple_or_empty(raw: Any, key: str, path: Path, ctx: str) -> tuple[str, ...]:
    """``_str_tuple`` for a key plans written before it existed may lack."""
    if key not in raw:
        return ()
    return _str_tuple(raw, key, path, ctx)


def _media(media_raw: Dict[str, Any], path: Path, ctx: str) -> MediaFingerprint:
    return MediaFingerprint(
        path=media_raw["path"],
        size_bytes=media_raw["size_bytes"],
        duration_seconds=media_raw["duration_seconds"],
        width=media_raw["width"],
        height=media_raw["height"],
        codec=media_raw["codec"],
        sha256=media_raw["sha256"],
    )


def _additional_media(raw: Dict[str, Any], path: Path, ctx: str) -> "tuple[MediaFingerprint, ...]":
    """Carousel's extra items. Absent (every sidecar before carousels
    existed) is tolerant-absent, same as ``scheduled_at_utc_by_platform`` --
    forward-compat additions default to empty, never refuse an old file."""
    value = raw.get("additional_media") or []
    if not isinstance(value, list):
        raise _fail(path, f"'additional_media' must be a list ({ctx})")
    items = []
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            raise _fail(path, f"additional_media[{i}] must be an object ({ctx})")
        items.append(_media(item, path, ctx))
    return tuple(items)


def _format(raw: Dict[str, Any], path: Path, ctx: str) -> Optional[str]:
    """Distinguishes three states, deliberately, not two:

    1. The ``format`` KEY is entirely absent -- a sidecar written before this
       field existed. This package only ever produced shorts before then, so
       an absent key is unambiguous and loads as FORMAT_SHORT.
    2. The key is present with an explicit ``null`` -- a record whose builder
       left format genuinely unstated (or asdict() round-tripping a record
       that never got one). Loads as ``None``. Collapsing this into "short"
       is exactly the silent default this field exists to prevent.
    3. The key names a real format string -- loads as that string, or raises
       if it names something outside KNOWN_FORMATS.
    """
    if "format" not in raw:
        return FORMAT_SHORT
    value = raw.get("format")
    if value is None:
        return None
    if not isinstance(value, str) or value not in KNOWN_FORMATS:
        raise _fail(path, f"'format' must be one of {sorted(KNOWN_FORMATS)} or null ({ctx})")
    return value


def _experiment_arms(raw: Dict[str, Any], path: Path, ctx: str) -> Dict[str, str]:
    """``experiment_arms``, absent-tolerant: sidecars written before caption
    experiments existed carry none."""
    value = raw.get("experiment_arms") or {}
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    ):
        raise _fail(path, f"'experiment_arms' must map experiment names to arm ids ({ctx})")
    return dict(value)


def _payloads(raw: Dict[str, Any], path: Path, ctx: str) -> PlatformPayloads:
    """Absent OR null means 'this platform has no payload' — ``asdict``
    writes ``None`` for an unset payload, so both shapes reach us."""

    def sub(key: str) -> Optional[Dict[str, Any]]:
        value = raw.get(key)
        if value is None:
            return None
        if not isinstance(value, dict):
            raise _fail(path, f"payloads.{key} must be an object or null ({ctx})")
        return value

    try:
        yt = sub("youtube")
        ig = sub("instagram")
        fb = sub("facebook")
        tt = sub("tiktok")
        return PlatformPayloads(
            youtube=(
                YouTubePayload(
                    title=yt["title"],
                    description=yt["description"],
                    tags=tuple(yt.get("tags") or ()),
                    category=yt.get("category", ""),
                )
                if yt
                else None
            ),
            instagram=InstagramPayload(caption=ig["caption"]) if ig else None,
            facebook=FacebookPayload(description=fb["description"]) if fb else None,
            tiktok=(
                TikTokPayload(
                    title=tt["title"],
                    # .get, not []: older sidecars have
                    # no caption key and must keep loading.
                    caption=tt.get("caption", ""),
                    privacy_level=tt["privacy_level"],
                    disclose_branded_content=bool(tt.get("disclose_branded_content", False)),
                    disclose_your_brand=bool(tt.get("disclose_your_brand", False)),
                )
                if tt
                else None
            ),
        )
    except KeyError as exc:
        raise _fail(path, f"payloads is missing required field {exc} ({ctx})") from exc
    except TypeError as exc:
        raise _fail(path, f"payloads has a wrong-typed field: {exc} ({ctx})") from exc


def _platform_schedule(schedule_raw: Dict[str, Any], path: Path, ctx: str) -> Dict[str, datetime]:
    """``schedule.scheduled_at_utc_by_platform``, absent-tolerant.

    A malformed entry is refused rather than dropped: a delivery silently
    falling back to the shared slot is the bug this map exists to prevent.
    """
    raw = schedule_raw.get("scheduled_at_utc_by_platform") or {}
    if not isinstance(raw, dict):
        raise _fail(path, f"schedule.scheduled_at_utc_by_platform is not an object ({ctx})")
    return {platform: _dt(raw, platform, path, ctx) for platform in sorted(raw)}


def _record(raw: Any, path: Path, index: int) -> ShortPublishRecord:
    if not isinstance(raw, dict):
        raise _fail(path, f"shorts[{index}] is not an object")
    ctx = f"shorts[{index}]"
    media_raw = _obj(raw, "media", path, ctx)
    schedule_raw = _obj(raw, "schedule", path, ctx)
    routes_raw = _obj(raw, "routes", path, ctx)
    try:
        return ShortPublishRecord(
            project_name=raw["project_name"],
            lf_index=raw["lf_index"],
            short_index=raw["short_index"],
            media=_media(media_raw, path, ctx),
            schedule=ScheduleWindow(
                scheduled_at_utc=_dt(schedule_raw, "scheduled_at_utc", path, ctx),
                display_timezone=schedule_raw["display_timezone"],
                # .get, not []: sidecars written before the per-platform map have no
                # per-platform map, and those files are never regenerated.
                scheduled_at_utc_by_platform=_platform_schedule(schedule_raw, path, ctx),
            ),
            recording_date=raw["recording_date"],
            sampled_regions=_str_tuple(raw, "sampled_regions", path, ctx),
            sampled_roads=_str_tuple_or_empty(raw, "sampled_roads", path, ctx),
            route_record=raw.get("route_record") or None,
            parent_lf_video_id=raw["parent_lf_video_id"],
            parent_lf_publish_at=_dt(raw, "parent_lf_publish_at", path, ctx),
            payloads=_payloads(_obj(raw, "payloads", path, ctx), path, ctx),
            consent_hash=raw["consent_hash"],
            reviewed_plan_hash=raw["reviewed_plan_hash"],
            # RouteSelection.__post_init__ validates the platform names, so
            # a corrupt sidecar is rejected by the model itself rather than
            # reaching the queue. A legacy sidecar's ``routes.rollback`` key
            # (metricool retired) is simply discarded here, never read.
            routes=RouteSelection(
                selected=_str_tuple(routes_raw, "selected", path, ctx),
            ),
            # generated_at is deliberately absent from the file; the model
            # default applies. See the module docstring.
            format=_format(raw, path, ctx),
            additional_media=_additional_media(raw, path, ctx),
            experiment_arms=_experiment_arms(raw, path, ctx),
        )
    except KeyError as exc:
        raise _fail(path, f"missing required field {exc} ({ctx})") from exc
    except ValueError as exc:
        # Includes RouteSelection.__post_init__'s own validation errors.
        if isinstance(exc, PublishPlanFormatError):
            raise
        raise _fail(path, f"{exc} ({ctx})") from exc


def load_publish_plan(sidecar_path: Union[str, Path]) -> PublishPlan:
    """Read a ``<project>_publish_plan.json`` sidecar back into a
    :class:`PublishPlan`. Raises :class:`PublishPlanFormatError` on any
    malformed input, always naming the file."""
    path = Path(sidecar_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise _fail(path, f"unreadable: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise _fail(path, f"not valid JSON: {exc}") from exc
    if isinstance(raw, dict) and (
        raw.get("kind") == "publish_preview" or raw.get("not_for_publish")
    ):
        raise _fail(
            path,
            "is a publish_preview (not_for_publish): a review-only payload preview "
            "with placeholder parent ids, never a plan",
        )
    if not isinstance(raw, dict) or not isinstance(raw.get("shorts"), list):
        raise _fail(path, "missing a 'shorts' list")
    plan = PublishPlan(shorts=tuple(_record(r, path, i) for i, r in enumerate(raw["shorts"])))
    _assert_consistent(plan, path)
    return plan


def _gate_view(record: ShortPublishRecord) -> SimpleNamespace:
    """A loaded record in the ``SocialPost`` shape the build-time plan gates
    read, so the loader reuses them instead of re-implementing them."""
    p = record.payloads
    captions = {}
    if p.instagram is not None:
        captions["instagram"] = p.instagram.caption
    if p.facebook is not None:
        captions["facebook"] = p.facebook.description
    if p.tiktok is not None:
        captions["tiktok"] = p.tiktok.caption
    return SimpleNamespace(
        short_index=record.short_index,
        sampled_regions=list(record.sampled_regions),
        youtube_title=p.youtube.title if p.youtube else "",
        youtube_description=p.youtube.description if p.youtube else "",
        youtube_tags=",".join(p.youtube.tags) if p.youtube else "",
        tiktok_title=p.tiktok.title if p.tiktok else "",
        caption="",
        captions=captions,
    )


def _assert_consistent(plan: PublishPlan, path: Path) -> None:
    """GMS-17c: the upstream gate before enqueue and dispatch. A plan edited
    or written apart from ``build_publish_plan`` is refused when its text no
    longer agrees with itself or with the short's own evidence:

    - every hashtag in a YouTube title appears in that short's description
      (the builder takes title hashtags from the description);
    - no surface names a county outside the clip's sampled regions
      (``assert_shorts_metadata_in_region``);
    - no surface carries a preposition splice (``assert_plan_text_grammar``).
    """
    by_lf: Dict[int, list] = {}
    for record in plan.shorts:
        yt = record.payloads.youtube
        if yt is not None:
            described = {tag.lower() for tag in hashtags_in(yt.description)}
            missing = [t for t in hashtags_in(yt.title) if t.lower() not in described]
            if missing:
                raise _fail(
                    path,
                    f"short #{record.short_index}: title hashtag(s) {missing} absent from "
                    "its description; title and description were edited apart",
                )
        by_lf.setdefault(record.lf_index, []).append(_gate_view(record))
    try:
        assert_shorts_metadata_in_region(by_lf)
        assert_plan_text_grammar(by_lf)
    except (ValueError, AssertionError) as exc:
        raise _fail(path, str(exc)) from exc
