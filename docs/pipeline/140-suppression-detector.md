### Per-platform suppression detector (`tracking/suppression.py`)

`tracking/channel_daily.py`'s Shorts-feed share says what happened to a day's channel
views; nothing classified whether that day's number is a healthy day or a platform-side
throttle. `tracking/suppression.py` fills that gap: `classify_youtube_shorts_feed(rows)` and
`classify_facebook_reel_reach(daily_reel_views)` each return a `{day: status}` map of
`"HEALTHY" | "SUPPRESSED" | "RECOVERING"` — **report-only, no callers act on the result** (by design, nothing acts on its own).

**YouTube.** `classify_youtube_shorts_feed` reuses `channel_daily.shorts_feed_share` for the
per-day share — it does not re-derive a share from raw rows — and separately sums each day's
total traffic-source views (`_daily_traffic_totals`) so it can see a day whose total collapses to
zero, which `shorts_feed_share` correctly omits rather than reporting as a fabricated `0.0`
share. A zero-total day is flagged `SUPPRESSED` immediately. Otherwise a day is flagged
`SUPPRESSED` when its share is below 50% OR below 0.5x its own trailing 28-day median share —
but only once 28 days of share history exist to compute that median against; `RECOVERING`
follows a `SUPPRESSED` day until the share holds at 70%+ for 7 consecutive days, at which point
the status returns to `HEALTHY`.

A strict "2 days in a row" rule would miss a single-day share collapse that recovers the very
next day, so this module flags on the first breaching day instead, gated on 28 days of share
history. The gate is required, not optional: without it, an early day (before any median can exist)
can breach the 50% threshold on its own and raise a false alarm inside an otherwise quiet window.

**Facebook.** `classify_facebook_reel_reach` takes a `{day: reel_views}` series and flags
`SUPPRESSED` when the trailing 7-day median falls below 0.3x the trailing 28-day median — both
windows computed over days strictly before the day being classified, never including it.

**Tests.** `tests/test_synthetic_suppression.py` replays constructed Shorts-share series (invented
daily view counts, not measured analytics) and asserts that a stable history raises no false alarm, that a low share inside the
28-day warm-up is never flagged, and that a single-day collapse after the warm-up is flagged on
that day.

**Not built here.** `objective_policy.py`'s `None` thresholds (`min_useful_improvement`, guardrail
degradation, `min_exposure`, `max_evidence_age_days`) are untouched — the "nothing acts on its own" invariant
treats that refusal as the intended safety behavior until a brand policy lands, and the brand policy's
keys (daily-cap ladder, per-location cap, weekly ceiling, stagger, region weights) don't
map to any of `objective_policy`'s fields regardless. This module's own thresholds are fixed constants, not a value read from a brand policy.

**Console surface.** `console/insights.py::anomaly_view()` reads `channel_daily.load_rows`
and calls `classify_youtube_shorts_feed` unchanged, then translates the returned `{day: status}`
map into the one thing worth showing on the SEO tab: the latest day's status and how many
consecutive days it has held. Nothing else is computed — this module produces no n, no p-value, no
deviation, so the console must not invent one for a status that carries none; that is the whole
difference from how the same view treats an `AnomalyFinding`. Facebook is not translated: `channel_daily`
now stores Page daily series under `facebook:<metric>` dimensions (`docs/facebook-metrics.md`), but
`classify_facebook_reel_reach` is not yet switched onto them. The row is still
report-only on the console exactly as it is from the CLI — the "nothing acts on its own" rule is
unchanged by having a second reader.
