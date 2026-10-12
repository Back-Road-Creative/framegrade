### Slot performance aggregate (`tracking/slot_performance.py`)

`(region, platform, weekday, hour_local) -> n, mean_views, mean_engagement` in a SQLite table
(`tracking.schema.ensure_slot_performance_schema`), recomputed on every fold from a per-short
observation table, `slot_observations` (`region, weekday, hour_local, views, engagement_rate`),
keyed `(project_name, short_index, platform)` — that key is this fold's slice of "content id +
platform" (source and metric set are fixed: one collector, two lifetime metrics), so recollecting
a short is an `INSERT OR REPLACE` of its one row rather than a new sample. `upsert_from_metrics`
derives each row's slot from the plan sidecar's `scheduled_at_utc`/`display_timezone` (local
weekday/hour) and `sampled_regions[0]` (region, matching `youtube_adapter._assign_playlist`'s
convention); an unresolvable sidecar/short is skipped, never guessed. `MetricRow.views`/
`.engagement_rate` are `Optional` because their `PlatformMetrics` sources are (a short with no
shares has no derivable engagement rate, which is routine on YouTube), and a row missing either is
skipped the same way: an unknown measurement is never folded in as a 0 — that would move a publish
slot on evidence nobody gathered — never overwrites an earlier observation, and never costs the
other rows their fold, since the whole fold runs in one `with conn:` transaction. Every
fold that records at least one observation rebuilds `slot_performance` in full — `n = COUNT(*)`,
`mean_views = AVG(views)`, `mean_engagement = AVG(engagement_rate)`, grouped by `(region, platform,
weekday, hour_local)` — so a replayed collection can never inflate `n`, and a re-fold with
corrected numbers replaces the sample rather than averaging it in on top of the stale one. This
retired the earlier incremental `n + 1` upsert; a `slot_performance` row with no backing
observation (only possible from that retired path) is dropped on the first fold after upgrade,
which is intended — it could only have come from the inflating path. `slots_ranked(region=None,
platform=None, min_n=3)` ranks by mean views then engagement.

**History (`slot_performance_history`, `tracking.schema.ensure_slot_performance_history_schema`).**
Same columns plus `folded_at TEXT NOT NULL` (ISO-UTC), PK `(folded_at, region, platform, weekday,
hour_local)`. Where `slot_performance` above is DELETEd and recomputed on every fold — the "latest"
view every ranking surface reads — this table is never deleted from: `_append_slot_performance_history`
runs immediately after `_rebuild_slot_performance` and appends a snapshot of the just-rebuilt
aggregate stamped with that fold's `folded_at` (`upsert_from_metrics`' own `folded_at` param, default
now — a test-only seam). `slot_performance_trend(conn, region, platform, weekday, hour_local, weeks=4)`
reads the last N snapshots for one slot, oldest first. `metrics best-times --trend` prints it per top
slot; see `docs/social-metrics-best-times.md`.

`metrics collect` folds every collected metric into it — inside `collect` itself, not as a
`metrics fold` subcommand, so there is no way to collect without folding. The handler projects the
saved tracking record into `MetricRow`s (one per short per platform; `engagement_rate` is the
record's derived property) and resolves each project's sidecar the way
`dispatch_queue.SidecarRecordLoader` does — newest `deliveries.sidecar_path` first, the flat
`<seo.storage_path>/<project>_publish_plan.json` convention as fallback — but returning `None`
rather than raising, so an unreachable sidecar costs the fold, never the collection. The aggregate
shares the publishing spool's database (`<seo.storage_path>/social_queue.sqlite3`), which is where
those sidecar paths already live. Collection is persisted before the fold runs, so a fold failure
is logged and swallowed; `collect` prints the folded count on its own line, leaving
`Collected metrics for N short(s) in project X` (parsed by the metrics cron) untouched.
`collect` then refreshes `audience_online` the same way and prints that count too.
