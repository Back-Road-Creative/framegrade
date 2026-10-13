### Audience-online table (`tracking/slot_performance.py`)

`audience_online(platform, weekday, hour_local, n_online, fetched_at)` stores raw
platform-reported presence — how many followers were online in a given local hour — and is
deliberately kept separate from `slot_performance`, which holds our own posts' measured
results. They answer different questions and must not be averaged together: one is the
audience's schedule, the other is ours. `upsert_audience_online` replaces on the same
`(platform, weekday, hour_local, fetched_at)` key rather than duplicating, so a re-fetch of
the same snapshot is idempotent.

Producers are the platform collectors (`InstagramCollector.fetch_audience_online`,
`FacebookCollector.fetch_audience_online`); no
collector imports this module, so the fetch and the persistence stay independently testable.
`metrics collect` is the call site: after the slot-performance fold, it walks
`MetricsCollector._collectors`, calls `fetch_audience_online()` on whichever ones have it
(YouTube has no hour dimension by design and TikTok exposes none, so both are skipped, not
treated as a fault), and upserts any non-empty histogram — same swallow-and-warn failure
contract as the fold, since the refresh runs after collection is already persisted. An empty
histogram (expected when the token lacks the insights scopes) writes zero rows and is
reported as such, not as an error. `metrics best-times` prints this curve's top hours beside the ranking; no GMS resolver
reads it to move a posting slot.

**The data path is retired.** `resolve_shorts_times_for_region` once moved `slot_performance`'s
top slot to the front of a region's shorts window where `audience_online` agreed, and phase D's
`_shorts_times_by_platform` carried a per-platform result into the publish plan. Both are gone with
`best_slot_from_data`: GMS reorders nothing from this table. `audience_online` is still collected;
`metrics best-times` prints its top hours beside the ranking.
