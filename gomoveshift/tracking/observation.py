"""Observation-level honesty for collected analytics.

``schema.PlatformMetrics`` already keeps "not reported" (``None``) apart from
a measured 0, field by field. This module raises that same distinction to the
level of the whole observation, where four things were being conflated:

* **Why** there are no numbers. A collector that answers with an empty result
  gives the caller one word for "no credentials", "that id was never
  queryable", "the post is gone or private", "the token lacks the scope",
  "we were rate limited" and "we asked and the platform had nothing". Those
  are six different facts about the world and only the last is about the post.
* **Cumulative vs interval.** A lifetime view counter and a per-day change are
  different quantities; :func:`interval_between` derives the second from two
  of the first rather than letting one field mean both.
* **Which moment.** When the post went up, when we asked, and how stale the
  platform says its own answer is are three timestamps. Collapsed, a fresh
  zero and a two-day-stale zero read identically.
* **What it is bound to.** An observation names the account and the real
  public post id it describes — never an upload handle or a plan-local index.

It also refuses to average view counts whose definitions differ
(:func:`compare_views`): a YouTube view, a TikTok view and an Instagram play
are not the same measurement, and a blended number invents a quantity no
platform reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from .schema import PlatformMetrics

#: Upload handles (``v_inbox_file~v2.<digits>``) identify a file we handed a
#: platform, not a post the public can see. They must never bind an
#: observation — see ``collector._looks_like_upload_handle``.
_UPLOAD_HANDLE_PREFIX = "v_inbox_file~"


class ObservationStatus:
    """Why an observation does or does not carry numbers."""

    #: The platform answered about this post. Only this status may carry metrics.
    MEASURED = "measured"
    #: No credentials for this platform — nothing was asked.
    NOT_CONFIGURED = "not_configured"
    #: The id is not a queryable public post id (an upload handle).
    UNQUERYABLE_ID = "unqueryable_id"
    #: Queried successfully; the platform did not return this post. Deleted,
    #: private, or not on the account the token speaks for.
    NOT_RETURNED = "not_returned"
    #: The token lacks the scope/permission the metric needs.
    PERMISSION_DENIED = "permission_denied"
    #: The platform throttled us; the post's numbers are unknown, not zero.
    RATE_LIMITED = "rate_limited"
    #: Any other transport or API failure.
    PROVIDER_ERROR = "provider_error"
    #: Only for interval deltas: the counter moved backwards between reads.
    COUNTER_RESET = "counter_reset"

    #: Every status that is NOT a measurement of the post.
    UNANSWERED = frozenset(
        {
            NOT_CONFIGURED,
            UNQUERYABLE_ID,
            NOT_RETURNED,
            PERMISSION_DENIED,
            RATE_LIMITED,
            PROVIDER_ERROR,
        }
    )
    ALL = UNANSWERED | {MEASURED, COUNTER_RESET}


#: What each platform's "view" actually counts. Kept as prose because the
#: point is that they are NOT interchangeable numbers.
VIEW_DEFINITIONS: Dict[str, str] = {
    "youtube": "youtube: Data API viewCount (a view of a Short)",
    "tiktok": "tiktok: video/query view_count",
    "instagram": "instagram: Reels plays/views (a play)",
    "facebook": "facebook: video view (3-second threshold)",
}


def has_numbers(metrics: PlatformMetrics) -> bool:
    """True when the platform reported at least one value.

    All-``None`` is not a measurement of zero: it is a platform that said
    nothing about this post."""
    return any(
        v is not None
        for v in (
            metrics.views,
            metrics.likes,
            metrics.comments,
            metrics.shares,
            metrics.watch_time_seconds,
            metrics.avg_view_duration_seconds,
            metrics.impressions,
            metrics.click_through_rate,
            metrics.reel_initial_plays,
            metrics.reel_replays,
            metrics.reel_total_plays,
        )
    )


def _parse(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def _seconds_between(start: Optional[str], end: Optional[str]) -> Optional[float]:
    first, second = _parse(start), _parse(end)
    if first is None or second is None:
        return None
    return (second - first).total_seconds()


@dataclass
class MetricObservation:
    """One platform's answer (or refusal) about one public post, at one time."""

    platform: str
    status: str
    #: The real public post id. Required for a measurement: a number with no
    #: post bound to it cannot be checked, compared or superseded.
    post_id: Optional[str] = None
    #: Which account the observation came from (channel/page/IG user id).
    account_id: Optional[str] = None
    #: "short" / "long_form" / "reel" / "still" / "post" — formats are only
    #: comparable with themselves.
    post_format: str = ""
    #: Cumulative, lifetime-to-date counters. Never an interval; see
    #: :func:`interval_between`.
    cumulative: PlatformMetrics = field(default_factory=PlatformMetrics)
    #: When the post went up.
    event_at: Optional[str] = None
    #: When we asked the platform.
    fetched_at: Optional[str] = None
    #: The moment the platform's own numbers are complete through, when it
    #: says. ``None`` means the provider's lag is unknown — not that it is 0.
    data_through: Optional[str] = None
    #: Free text for the operator: the API error code, the missing scope.
    detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in ObservationStatus.ALL:
            raise ValueError(f"unknown observation status {self.status!r}")
        if self.status != ObservationStatus.MEASURED and self._has_numbers():
            raise ValueError(
                f"{self.platform} observation with status {self.status!r} carries "
                "numbers: only a measurement may report numbers, and presenting "
                "a refusal's numbers as measurements is the bug this prevents"
            )
        if self.status == ObservationStatus.MEASURED:
            if not self.post_id:
                raise ValueError(
                    f"{self.platform} measurement has no post_id: a metric not bound "
                    "to a real public post cannot be attributed to anything"
                )
            if str(self.post_id).startswith(_UPLOAD_HANDLE_PREFIX):
                raise ValueError(
                    f"{self.platform} measurement is bound to upload handle "
                    f"{self.post_id!r}, which is a file we uploaded, not a public post"
                )

    def _has_numbers(self) -> bool:
        return has_numbers(self.cumulative)

    @property
    def is_measurement(self) -> bool:
        return self.status == ObservationStatus.MEASURED

    @property
    def age_at_fetch_seconds(self) -> Optional[float]:
        """How old the post was when we asked. None if either time is unknown."""
        return _seconds_between(self.event_at, self.fetched_at)

    @property
    def reporting_delay_seconds(self) -> Optional[float]:
        """How stale the platform says its own answer is, ``fetched_at -
        data_through``. ``None`` means the platform did not say — which is
        not the same claim as "no delay"."""
        return _seconds_between(self.data_through, self.fetched_at)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "status": self.status,
            "post_id": self.post_id,
            "account_id": self.account_id,
            "post_format": self.post_format,
            "cumulative": self.cumulative.to_dict(),
            "event_at": self.event_at,
            "fetched_at": self.fetched_at,
            "data_through": self.data_through,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MetricObservation:
        known = {f for f in cls.__dataclass_fields__ if f != "cumulative"}
        fields = {k: v for k, v in data.items() if k in known}
        raw = data.get("cumulative")
        fields["cumulative"] = (
            PlatformMetrics.from_dict(raw) if isinstance(raw, dict) else PlatformMetrics()
        )
        return cls(**fields)


_DELTA_FIELDS = ("views", "likes", "comments", "shares")


@dataclass
class IntervalDelta:
    """The CHANGE in a cumulative counter between two observations.

    Distinct type on purpose: an interval and a lifetime total must not be
    able to flow into the same field. ``status`` says why an unknown delta is
    unknown — including ``COUNTER_RESET``, where the counter moved backwards
    (a deleted+reposted video, a platform restatement) and the true interval
    is unknowable rather than negative.
    """

    platform: str
    post_id: Optional[str]
    status: str
    start_at: Optional[str] = None
    end_at: Optional[str] = None
    views: Optional[int] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    shares: Optional[int] = None
    detail: str = ""


def interval_between(earlier: MetricObservation, later: MetricObservation) -> IntervalDelta:
    """Delta between two cumulative observations OF THE SAME POST.

    Unknown in, unknown out: if either end is not a measurement the delta
    carries that end's status, and a field either side never reported stays
    ``None`` rather than being read as a change of 0.
    """
    if earlier.platform != later.platform or earlier.post_id != later.post_id:
        raise ValueError(
            "an interval must describe the same post on the same platform, got "
            f"{earlier.platform}:{earlier.post_id} and {later.platform}:{later.post_id}"
        )
    base = {
        "platform": later.platform,
        "post_id": later.post_id,
        "start_at": earlier.fetched_at,
        "end_at": later.fetched_at,
    }
    for end in (later, earlier):
        if not end.is_measurement:
            return IntervalDelta(status=end.status, detail=end.detail, **base)
    deltas: Dict[str, Optional[int]] = {}
    for name in _DELTA_FIELDS:
        before = getattr(earlier.cumulative, name)
        after = getattr(later.cumulative, name)
        if before is None or after is None:
            deltas[name] = None
            continue
        if after < before:
            return IntervalDelta(
                status=ObservationStatus.COUNTER_RESET,
                detail=f"{name} fell from {before} to {after}; the interval is unknowable",
                **base,
            )
        deltas[name] = after - before
    return IntervalDelta(status=ObservationStatus.MEASURED, **base, **deltas)


@dataclass
class ViewComparison:
    """Per-platform view counts, and a blend only when one is defensible."""

    per_platform: Dict[str, List[int]]
    definitions: Dict[str, str]
    blended_views: Optional[float] = None
    #: Empty when ``blended_views`` is a real figure; otherwise the reason
    #: the blend was refused, in words an operator can act on.
    declined_reason: str = ""


def compare_views(
    observations: Iterable[MetricObservation], *, age_bucket_hours: float = 24.0
) -> ViewComparison:
    """Group measured view counts by platform; blend only like-with-like.

    A blended figure is produced ONLY when every measurement is the same
    platform, the same format, and the posts were the same age (within
    ``age_bucket_hours``) when fetched. Otherwise ``blended_views`` is None
    and ``declined_reason`` says which of those failed — averaging a
    YouTube view with an Instagram play, or a day-old post with a
    year-old one, answers a question nobody asked.
    """
    measured = [o for o in observations if o.is_measurement and o.cumulative.views is not None]
    per_platform: Dict[str, List[int]] = {}
    for obs in measured:
        per_platform.setdefault(obs.platform, []).append(int(obs.cumulative.views))
    definitions = {
        p: VIEW_DEFINITIONS.get(p, f"{p}: view definition undocumented") for p in per_platform
    }
    result = ViewComparison(per_platform=per_platform, definitions=definitions)
    if not measured:
        result.declined_reason = "no measured view counts to compare"
        return result
    if len(per_platform) > 1:
        result.declined_reason = (
            "view counts from "
            + ", ".join(sorted(per_platform))
            + " have different definitions and are not averaged: "
            + "; ".join(definitions[p] for p in sorted(definitions))
        )
        return result
    formats = {o.post_format for o in measured}
    if len(formats) > 1:
        result.declined_reason = (
            "mixed post format (" + ", ".join(sorted(f or "unknown" for f in formats)) + "): "
            "only like formats are comparable"
        )
        return result
    ages = [o.age_at_fetch_seconds for o in measured]
    if any(a is None for a in ages):
        result.declined_reason = (
            "post age at fetch is unknown for at least one observation "
            "(missing event_at or fetched_at): like ages cannot be shown to be alike"
        )
        return result
    if max(ages) - min(ages) > age_bucket_hours * 3600:
        result.declined_reason = (
            f"post ages at fetch span {(max(ages) - min(ages)) / 3600:.1f}h, wider than "
            f"the {age_bucket_hours:.0f}h bucket: unlike ages are not averaged"
        )
        return result
    views = [int(o.cumulative.views) for o in measured]
    result.blended_views = sum(views) / len(views)
    return result
