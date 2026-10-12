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
reported as such, not as an error. The consumer is the two-source best-time resolver, which
requires this curve and `slots_ranked` to agree before it will move a posting slot.

**`resolve_shorts_times_for_region(config, region, lf_day, *, conn=None, platform=None,
min_n=5)`** (`pipeline/schedule_constants.py`) gains an optional data path: passing a live
`conn` + `platform` moves `slot_performance`'s top-ranked slot for `(region, platform)` to the
front of the returned window when `n >= min_n` (default 5) **and** that slot's hour is in
`audience_online`'s top-3 hours for that platform where `audience_online` has any rows for it
(Meta only today — YouTube has no hour dimension by design, so B alone decides there and this
is never logged as a fault; TikTok is the same shape once a TikTok backfill exists). The candidate slot
must already be a member of `config.drive.shorts_slot_pool_by_region[region]` (falling back to
the configured window itself when the region has no pool) — the data only ever picks WHICH
configured slot to prefer, it never invents an hour. A caller that passes neither
keyword gets the previous behaviour unchanged. A missing/broken/absent database, disagreement between
the two sources, or `n < min_n` all keep the configured window and log an INFO line explaining
why; none of them raise. `metrics best-times` is a CLI table over this. The production
caller is phase D's `_shorts_times_by_platform` (`drive_phases/phase_d_scheduling.py`), which opens the spool
(`<seo.storage_path>/social_queue.sqlite3`, where `metrics ingest` writes `slot_performance`), asks
this once per first-party platform with the run's region and `--lf-day` (inverted from
`long_form_publish_days`), and hands only the platforms whose window differs from the configured one
to `build_publish_plan(shorts_times_by_platform=…)` — see `docs/social-publishing-queue.md`. It
skips itself (every platform keeps the slot, one INFO line) when the flat `shorts_publish_times` no
longer equals the region spec — an explicit `--shorts-times` — so data never reorders an operator's
window, the same precedence the drive handler enforces; a spool that cannot be opened is the same
no-op, never a failed drive.
