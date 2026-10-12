"""Dynamic scheduling -- ladder-driven weekly volume, region/recency
ordering, and FB/IG stagger. Deliberately pure -- no DB, no clock -- like
``response_policy``; callers own persistence and wiring the results into
the enqueue path (``enqueue_from_plan.enqueue_from_plan_sidecar``'s
``order=``/``stagger_hours=`` keywords).

Docs: ``docs/pipeline/149-dynamic-scheduling.md``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable, Mapping, Optional, Sequence

from .response_policy import Status, daily_cap_for

#: "Days since last posted" for a location this run has never seen at all.
#: Treated as maximally overdue (rather than 0) so a brand-new location is
#: never scored the same as -- or worse than -- one the ledger merely
#: posted to a long time ago.
_NEVER_POSTED_DAYS = 365

#: Only these two platforms are ever staggered off the YouTube slot --
#: matches the per-platform-map contract (``models.ScheduleWindow.
#: scheduled_at_utc_by_platform``): YouTube and TikTok publish at the
#: shared slot, Facebook and Instagram move ``stagger_hours`` later.
_STAGGERABLE_PLATFORMS = frozenset({"facebook", "instagram"})


def weekly_volume_from_ladder(policy, state: Status, healthy_days: int) -> int:
    """Weekly shorts volume implied by the daily ladder for ``state``.

    ``response_policy.daily_cap_for(state, policy.daily_ladder,
    healthy_days)`` gives the per-day cap; the weekly figure is that cap
    times 7, min'd with ``policy.weekly_ceiling`` when the brand sets one.
    ``policy`` may be ``None`` (no brand policy at all) -- treated the same
    as a policy with every field unset, matching ``brand_policy``'s own
    "missing = no override" contract.

    Wiring this into the actual enqueue ceiling is NOT this function's job
    (a separate step) -- it only computes the number.
    """
    ladder = getattr(policy, "daily_ladder", None)
    weekly = daily_cap_for(state, ladder, healthy_days) * 7
    ceiling = getattr(policy, "weekly_ceiling", None)
    if ceiling is not None:
        weekly = min(weekly, ceiling)
    return weekly


def order_shorts(
    records: Sequence[Mapping],
    *,
    region_weight: Optional[Mapping[str, float]],
    last_posted_by_location: Optional[Mapping[str, datetime]],
    now: datetime,
    score_of: Optional[Callable[[Mapping], Optional[float]]] = None,
) -> list:
    """Reorder ``records`` by score x region weight x recency, highest first.

    Each record is a mapping carrying (at least) ``region``, ``location``,
    ``scheduled_at`` and ``short_index``. Grounding for ``region``/
    ``location``: the publish-plan sidecar has no dedicated field for
    either (see ``plan_io.py`` / ``models.ShortPublishRecord``), so the
    caller is expected to derive ``region`` from a record's
    ``sampled_regions[0]`` (the same field ``youtube_adapter`` already
    reads as "region" in ``ShortPublishRecord`` and the plan builder) and
    ``location`` from its YouTube title -- documented in
    ``docs/pipeline/149-dynamic-scheduling.md``, not invented here.

    ``score_of`` absent, or returning ``None`` for a record, scores that
    record 1.0 -- the "no score yet" contract. ``region_weight``
    looks a record's ``region`` up directly; a region missing from the
    mapping (or ``region_weight`` itself being ``None``) scores 1.0, the
    neutral no-op. Recency multiplies by ``1 + days since that location
    last posted``; a location absent from ``last_posted_by_location``
    counts as ``_NEVER_POSTED_DAYS`` days overdue rather than 0, so a
    location this run has never touched is never scored behind one posted
    today. Ties break by ``(scheduled_at, short_index)`` ascending -- the
    planner's own rank order.
    """
    region_weight = region_weight or {}
    last_posted_by_location = last_posted_by_location or {}

    def priority(record: Mapping) -> float:
        score = 1.0
        if score_of is not None:
            raw = score_of(record)
            if raw is not None:
                score = raw
        weight = region_weight.get(record.get("region"), 1.0)
        last_posted = last_posted_by_location.get(record.get("location"))
        days_since = (now - last_posted).days if last_posted is not None else _NEVER_POSTED_DAYS
        return score * weight * (1 + days_since)

    def sort_key(record: Mapping):
        return (-priority(record), record.get("scheduled_at"), record.get("short_index"))

    return sorted(records, key=sort_key)


def staggered_by_platform(
    base_utc: datetime,
    *,
    stagger_hours: Optional[int],
    platforms: Sequence[str],
) -> dict:
    """Facebook/Instagram instants, staggered ``stagger_hours`` after
    ``base_utc`` (the YouTube slot). ``stagger_hours`` falsy (``None`` or
    0) -> no stagger, an empty dict. Only ``facebook`` and ``instagram`` in
    ``platforms`` are ever staggered -- ``youtube``/``tiktok`` (or any
    other name present) are ignored, matching the "a platform is
    listed only when its hour actually differs" contract
    (``models.ScheduleWindow.scheduled_at_utc_by_platform``, consumed by
    ``enqueue_from_plan`` / ``facebook_adapter``).

    Facebook's own lead cap (``facebook_adapter.MAX_SCHEDULE_LEAD``) clamps
    the result relative to ``base_utc``: a stagger long enough to push
    Facebook further than that past the base slot would make
    ``ScheduleWindowError`` permanent on that delivery, so it is clamped
    here rather than shipped to the adapter to fail on later. Instagram
    carries no such cap.
    """
    from .facebook_adapter import MAX_SCHEDULE_LEAD

    if not stagger_hours:
        return {}
    when = base_utc + timedelta(hours=stagger_hours)
    result: dict = {}
    for platform in platforms:
        if platform not in _STAGGERABLE_PLATFORMS:
            continue
        if platform == "facebook" and when - base_utc > MAX_SCHEDULE_LEAD:
            result[platform] = base_utc + MAX_SCHEDULE_LEAD
        else:
            result[platform] = when
    return result
