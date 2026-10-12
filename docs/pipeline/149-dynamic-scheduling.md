## Dynamic scheduling

`gomoveshift/publishing/dynamic_schedule.py` — pure, no DB, no clock, matching `response_policy.py`'s own shape.

- **`weekly_volume_from_ladder(policy, state, healthy_days) -> int`**: `response_policy.daily_cap_for(state, policy.daily_ladder, healthy_days)` times 7, min'd with `policy.weekly_ceiling` when the brand sets one (`brand_policy.BrandPolicy`). `policy=None` behaves like every field unset. Wired into the actual enqueue ceiling as the `ladder` rung — see "Enqueue ceiling rung `ladder`" below.
- **`order_shorts(records, *, region_weight, last_posted_by_location, now, score_of=None) -> list`**: sorts by `score x region_weight[region] x (1 + days since that location last posted)`, highest first, ties broken by `(scheduled_at, short_index)`. `score_of` absent or returning `None` for a record scores it 1.0. A region missing from `region_weight` (or `region_weight` itself `None`) scores 1.0. A location absent from `last_posted_by_location` counts as 365 days overdue, not 0 — a brand-new location must never rank behind one merely posted to a while ago.

  **Grounding note**: the publish-plan sidecar (`models.ShortPublishRecord` / `plan_io.py`) has no dedicated `region` or `location` field. `order_shorts` itself is field-agnostic — it reads whatever `record["region"]`/`record["location"]` the caller hands it. The documented convention for a sidecar-backed caller: `region` = `sampled_regions[0]` (the same field `youtube_adapter.py` already reads as a short's region), `location` = the short's YouTube title. Neither is a real new sidecar field; both are read off the raw JSON record already on disk.

- **`staggered_by_platform(base_utc, *, stagger_hours, platforms) -> dict[str, datetime]`**: Facebook and Instagram move `stagger_hours` after `base_utc` (the YouTube slot); every other platform name in `platforms` is ignored. Facebook's result is clamped to `facebook_adapter.MAX_SCHEDULE_LEAD` past `base_utc` — a stagger long enough to exceed it would make `ScheduleWindowError` permanent on that delivery. `stagger_hours` falsy (`None`/0) returns `{}`.

### Wiring into `enqueue_from_plan_sidecar`

`enqueue_from_plan.enqueue_from_plan_sidecar` gained two keyword-only parameters:

- **`order=`**: a callable taking the already-capped list of raw sidecar records (post `select_capped_shorts`) and returning them reordered. Changes insertion order only, never which shorts are selected. Typical caller: `functools.partial(dynamic_schedule.order_shorts, region_weight=..., last_posted_by_location=..., now=...)`.
- **`stagger_hours=`**: for each record, computes `staggered_by_platform` off `schedule.scheduled_at_utc` and fills `facebook`/`instagram` entries of `schedule.scheduled_at_utc_by_platform` **only when the sidecar didn't already carry one** for that platform (the best-times resolver's own explicit value always wins).

### Next-week digest

`gomoveshift/publishing/next_week.py`:

- **`next_week_report(conn, now) -> str`**: one line per `status='queued'` delivery scheduled in `[now, now + 7 days)`, grouped by day, each line naming the project/short/platform, an hourly slot label (`HH:MM–HH:MM`), and a reason. `'queued'` is `queue.enqueue`'s literal insert status; `TERMINAL_STATUSES = ("published", "failed_permanent", "cancelled")` (`queue.py`) never appear. Empty input returns `""` so a digest caller drops the section.
- **Reason field, grounded**: `queue.py`'s `deliveries` table stores no "why this slot" column, so the digest reports only what the stored rows prove: a Facebook/Instagram row whose `scheduled_at` differs from its short's own YouTube row is labelled `stagger`; everything else is `scheduled slot`. The region-weight/recency ordering `order_shorts` applies affects insertion order only, not the persisted `scheduled_at`, so it leaves no trace this reader can report on. A daily digest can call `next_week.main(config)` and emit the report as a "Next 7 days (queued deliveries — report only)" section; the empty-string contract above is what lets it drop the section when nothing is queued.
- **`main(config) -> str`**: resolves the DB the same way `cli.handlers.metrics` does (`open_queue_db(config.seo.storage_path)`) and returns the report.

## Enqueue ceiling rung `ladder`

`cli/handlers/enqueue_plan.py`'s rolling-week ceiling is set by three rungs, in order, each only ever trimming the ceiling the previous rung set — never raising it:

1. `config` — `config.seo.rolling_week_ceiling_per_platform` (or `_DEFAULT_ROLLING_WEEK_CEILING`).
2. `brand-policy-cap` — `load_brand_policy(current_brand()).weekly_ceiling`, when set and lower.
3. `ladder` — `weekly_volume_from_ladder`'s result for `youtube`, when lower still.

**youtube is the reference platform**: every delivery in the queue schedules off the youtube slot (`staggered_by_platform` moves only facebook/instagram off a youtube `base_utc`), so youtube's bounded-response state is the one used to gate a ceiling shared across platforms. `queue.get_platform_daily_cap(conn, "youtube")` reads the latest bounded-response decision (`platform`, `cap`, `state`, `mode`, `set_at`, `reason`); no row means the `ladder` rung is a no-op.

The rung only ACTS on a `mode="live"` row, matching `dispatch._select_due`'s own shadow-vs-live distinction — a `"shadow"` row never changes the ceiling, only logs what it would have set (`enqueue-plan: ladder rung would set ceiling ... mode=shadow`).

`healthy_days` (the `weekly_volume_from_ladder(policy, state, healthy_days)` argument) is derived from `queue.response_log`, not `response_policy.consecutive_healthy_days` directly (that function wants a full `{day: Status}` mapping the queue schema doesn't keep): `cli.handlers.enqueue_plan._healthy_days(conn, platform)` reads `response_log` rows for the platform ordered by `at DESC`, counts distinct calendar days (`at[:10]`) whose state is not `SUPPRESSED` (HEALTHY and RECOVERING both count as healthy, matching `daily_cap_for`'s own RECOVERING-ladder-step contract), and stops at the first `SUPPRESSED` row or the end of the log.

### Follow-ups (not in this change)

- **Report bullet**: shipped separately — see the `next_week_report` bullet above.
