# Post-level anomaly detection (shadow only)

`gomoveshift post-anomaly` reads the tracking store and reports the newest short on each platform
whose day-7 views are a statistically-backed dip or spike against its own recent references. It
fetches nothing, writes nothing, and always exits 0. **It is shadow only: nothing reads its lines
to act.** Every finding and refusal line carries `shadow=true`.

## The kernel does the qualification, this module supplies the policy

`seo_manager.analytics.anomaly_stats` is a pure statistics kernel with no thresholds of its own —
a robust deviation (median/MAD on a log1p scale), an empirical two-sided tail-rank p-value, and a
policy-feasibility check. It refuses by type rather than inventing a number: a reference with no
spread, too few references for the requested alpha, or a policy whose own cap forbids the evidence
it demands each come back as a typed `Refusal`, never a degraded statistic.

This module is the only caller. It carries a fixed shadow policy
as module constants — changing them is a new policy revision, never a runtime knob, so
`post-anomaly` takes no threshold flags:

- Total alpha 0.05 split over 2 registered tests (dip, spike) → 0.025 allocated per test, which
  needs 79 references (`min_references_for(0.025)`).
- Reference window: the platform's earlier shorts within a trailing 180-day lookback, newest 200
  at most.
- Candidate gate: `abs(robust_deviation) >= 3.5` **and** `abs(relative_effect) >= 0.25`, on top of
  the tail-rank p-value clearing the allocated alpha. A short that clears significance but not the
  practical-effect thresholds prints nothing — it is not a candidate, just noise the p-value alone
  would have overstated.

Feasibility is checked once per run, before any short is inspected. Direction comes from the sign
of the deviation: `dip` when negative, `spike` when positive — never both for the same short.

## Data

References reuse `tracking/decline.py`'s `platforms_seen`/`platform_series` — the same day-7
views series platform-decline already builds, oldest first. The target is the newest short in the
series; a platform with fewer than two entries has nothing to compare and is skipped, not refused.

## Output

```
POST_ANOMALY platform=facebook video=vid42 kind=dip metric=day7_views target=100 ref_median=1414 deviation=-14.92 relative=-93% p=0.0198 n=100 allocated_alpha=0.025 shadow=true
POST_ANOMALY_REFUSED platform=facebook reason=insufficient_resolution n=40 required=79 shadow=true
```

An empty store prints one plain "insufficient data" line; a policy-infeasible epoch (the alpha
split and reference cap can never together meet the required references) prints one
`POST_ANOMALY_INFEASIBLE` line and stops before any platform is inspected.

Cron/digest wiring is not included:
this module and its CLI are self-contained and produce no side effects on their own.

## Second surface: the console SEO tab

`console/insights.py::anomaly_view()` is a second, read-only reader of this same kernel — the CLI
above was the only call site until now, so a finding only ever reached a human who tailed a scheduler
log. The view opens `TrackingStore` itself, calls `check_feasibility()`/`detect_post_anomaly()`
unchanged, and translates each `AnomalyFinding`/`AnomalyRefusal` into a template row with the same
never-raise, never-invent-a-number discipline `performance_view()` already keeps for the tab: any
read failure degrades to the same "nothing to show" shape as a genuinely empty store, logged rather
than surfaced as an error.

One field the console adds that the dataclasses don't carry: `shadow` is stamped only into
`log_line()`'s string, not a dataclass field, so `anomaly_view()` adds `shadow: True` to every row
itself on the way into the template — the same `shadow=true` the CLI prints, just carried as a
value instead of parsed back out of a log line.

`seo.html`'s "Anomalies & suppression signals (hypotheses, not proof)" section is absent
entirely — not blank — when `anomaly_view()` found nothing at all to read (`has_data: False`);
present, hint text and all, the moment either an anomaly kernel platform or a suppression day
exists, even with zero current findings, so a quiet page can't be mistaken for a page that never
checked.

The view reaches the page through `app.py::_seo_context()`, which calls `context.update(anomalies=insights.anomaly_view())`; the template guards on `anomalies is defined`.
