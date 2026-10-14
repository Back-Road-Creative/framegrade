# Best-time table — report-only

`gomoveshift metrics best-times` prints, per `(region, platform)`, the top-3 `slot_performance`
slots and the hours Waypost has decided for the region. It fetches nothing and writes nothing — no config,
no schedule. Empty store or no ranked rows prints one plain line saying so and exits 0.

`slot_performance` is keyed by `(region, platform, weekday, hour_local)`, aggregated from
collected metrics (`gomoveshift metrics collect`, see `docs/social-metrics-decline.md`'s sibling
doc for how a row gets there). `slots_ranked(conn, min_n=--min-n)` (default 5) ranks by mean
views then mean engagement, descending; a slot below `--min-n` samples never appears.

**Decided hours.** Waypost decides every region's shorts hours (`shorts_times` in its cadence: the
packaged `brand.yaml` seed, replaced by the latest `shorts_times` decision). The `decided hours:` line
prints that whole list for the group's region (`seo_slots.shorts_pool`); a region Waypost has no hours
for, or an unreadable decision store, prints `decided hours: unavailable (<reason>)` and the report
still exits 0. The ranking informs Waypost's decision; GMS moves no slot from it (the old
`best_slot_from_data` source-agreement check is gone).

Where the platform has `audience_online` rows, its own top-3 hours (summed `n_online` across every
recorded weekday) print on an `audience_online top-3:` line, so a reader can compare the platform's
curve with the ranking and the decided hours. A platform without rows (YouTube by design) prints no
such line.

```
ireland / instagram:
  #1 12:00 -- n=8 mean_views=500.0 mean_engagement=0.120
  #2 08:00 -- n=8 mean_views=100.0 mean_engagement=0.040
  decided hours: 08:00, 12:00
  audience_online top-3: 12:00, 09:00, 20:00
usa / youtube:
  #1 14:00 -- n=10 mean_views=1000.0 mean_engagement=0.080
  decided hours: 10:00, 14:00, 16:00, 18:00
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
  decided hours: 08:00, 12:00
  trend (12:00, last 3 fold(s)):
    2030-01-07T00:00:00+00:00: n=4 mean_views=200.0
    2030-01-14T00:00:00+00:00: n=6 mean_views=350.0
    2030-01-21T00:00:00+00:00: n=8 mean_views=500.0
  audience_online top-3: 12:00, 09:00, 20:00
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
