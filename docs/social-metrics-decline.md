# Platform decline detection

`gomoveshift platform-decline` reads the tracking store and reports platforms whose recent shorts are doing worse than the
ones before them. It fetches nothing, writes nothing, and always exits 0: with no metrics, fewer shorts than `2 x window`,
or no reportable drop it prints one plain line saying so.

**Day-7 views** = views over the first seven entries of `PlatformMetrics.daily`; a short with a shorter series is skipped, never
scaled up. The series comes from the tracking records, not `slot_performance` — that aggregate is keyed by `(region, platform,
weekday, hour_local)` and has no time axis and no day-7 column, so a rolling window over the last N shorts is not computable from
it. `velocity=` is the direction of `first_hours_velocity` across the two halves, and reads `unknown` unless BOTH halves
actually had a measured velocity — a half nobody measured is not a zero, and never reads as a fall.

**A finding needs a significant drop, not just a big one.** The median day-7 views of the newest `--window` shorts (default
14) must be at least `--min-drop-pct` (default 20%) below the `--window` before them, **and**
`PerformanceValidator.compare_strategies` — a Welch two-sample t-test over those groups' raw day-7 values — must return
`SignificanceLevel.MODERATE` or `STRONG` (p <= 0.05). Its real p-value is printed, and a dip that is only noise goes
unreported however large. The validator is built with `min_sample_size=window` because the window is the sample size; below
that it merely warns and still returns a result, so nothing relies on it — the `2 x window` gate refuses thinner data first.

```
PLATFORM_DECLINE platform=facebook window=14 metric=day7_views delta=-50% p=0.0000 significance=strong velocity=steady
```

A digest can grep the scheduler's log (last day) and mirror each finding verbatim; no finding, no section. To wire it, add one `gomoveshift platform-decline` line at the end of the nightly metrics run.
