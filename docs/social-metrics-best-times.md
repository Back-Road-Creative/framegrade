# Best-time table — report-only

`gomoveshift metrics best-times` prints, per `(region, platform)`, the top-3 `slot_performance`
slots and which source agreed on the winner. It fetches nothing and writes nothing — no config,
no schedule. Empty store or no ranked rows prints one plain line saying so and exits 0.

`slot_performance` is keyed by `(region, platform, weekday, hour_local)`, aggregated from
collected metrics (`gomoveshift metrics collect`, see `docs/social-metrics-decline.md`'s sibling
doc for how a row gets there). `slots_ranked(conn, min_n=--min-n)` (default 5) ranks by mean
views then mean engagement, descending; a slot below `--min-n` samples never appears.

**Source agreement** reuses `schedule_constants.best_slot_from_data` — the same resolution
`resolve_shorts_times_for_region`'s data path already applies when it
moves a slot to the front of a region's shorts window:

- Our own ranking (Source B, `slot_performance`) has no `n >= --min-n` rows → "no ranked slot_performance rows".
- The top slot isn't in the region's configured `shorts_slot_pool_by_region` → "not in the region's configured ... pool" (the data only ever picks WHICH already-configured slot to prefer, never a new hour).
- The platform has no `audience_online` rows at all (YouTube by design; TikTok pending) → "resolved from slot_performance alone" — Source B alone decides, not a fault.
- `audience_online`'s (Source A) own top-3 hours for that platform don't include the Source B top slot → "sources disagree".
- Otherwise → "resolved: slot_performance and audience_online agree".

Where Source A exists for the platform, its own top-3 hours (summed `n_online` across every
recorded weekday) print alongside the table so a reader can see the agreement or disagreement
directly rather than take the verdict on faith.

```
ireland / instagram:
  #1 12:00 -- n=8 mean_views=500.0 mean_engagement=0.120
  #2 08:00 -- n=8 mean_views=100.0 mean_engagement=0.040
  agreement: resolved: slot_performance and audience_online agree
  audience_online top-3: 12:00, 09:00, 20:00
usa / youtube:
  #1 14:00 -- n=10 mean_views=1000.0 mean_engagement=0.080
  agreement: resolved from slot_performance alone (no audience_online for youtube)
```

**`--trend`.** Also prints, per group's rank-#1 slot, the last 4 `slot_performance_history`
snapshots for that exact `(region, platform, weekday, hour_local)`, oldest first — `n`/`mean_views`
as they stood at each fold, so a reader can see a slot's trajectory instead of only its latest,
overwritten aggregate (`slot_performance` is rebuilt from scratch every fold; the history table
never is — see `docs/pipeline/109-slot-performance-aggregate.md`). Omitted when the flag isn't
passed, and silently skipped for a slot with no history yet (a fresh install, or one fold old).

```
ireland / instagram:
  #1 12:00 -- n=8 mean_views=500.0 mean_engagement=0.120
  agreement: resolved: slot_performance and audience_online agree
  audience_online top-3: 12:00, 09:00, 20:00
  trend (12:00, last 3 fold(s)):
    2030-01-07T00:00:00+00:00: n=4 mean_views=200.0
    2030-01-14T00:00:00+00:00: n=6 mean_views=350.0
    2030-01-21T00:00:00+00:00: n=8 mean_views=500.0
```

Not wired into any nightly metrics run by default — the line
to add there is `gomoveshift metrics best-times`, same shape as the
`gomoveshift platform-decline` line.

## `metrics report` — the full `PerformanceReporter` markdown, on demand

`gomoveshift metrics report --project <name>` (or `--all` for the cross-project report) wires
`tracking.report.PerformanceReporter` — best/worst shorts, posting-time/weather/hook/duration
cohorts, platform breakdown, recommendations — into the CLI. Report-only, always exits 0; a
project with no tracking record prints one line and exits 0 rather than raising. `--project` and
`--all` are mutually exclusive and one is required. `--output <path>` writes the report to disk
(`PerformanceReporter.save_report` for a single project) instead of printing it to stdout.

```
gomoveshift metrics report --project example-project
gomoveshift metrics report --all --output /tmp/cross-project.md
```

The report's duration section (`⏱️ Duration vs Performance`) buckets shorts by
`ContentMeta.duration_sec` into `≤15s`, `16–30s`, `31–60s`, `>60s`, plus `unknown` for a short
whose duration was never measured — making a shorts-length experiment measurable the same
way the weather/hook sections already are.
