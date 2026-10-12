### `PlatformMetrics` missing-vs-zero (`tracking/schema.py`)

Every measurement field (`views`, `likes`, `comments`, `shares`, `watch_time_seconds`,
`avg_view_duration_seconds`, `impressions`, `click_through_rate`) is `Optional[...] = None` —
`None` means "not reported by this platform"; a genuine reported `0` stays `0` and is never
confused with it. `to_dict()`/`from_dict()` round-trip `None` as JSON `null`. `engagement_rate`
returns `None` when views or any interaction count is unreported (never a fabricated `0.0`); it
still returns `0.0` for a genuinely measured 0 views, since that case is well-defined, not
missing. Cross-short **totals** (`ShortRecord.total_views`, `report._platform_breakdown`) treat a
missing per-platform value as contributing nothing to the aggregate — that's a property of
summing across platforms, not a claim that the platform's own value is zero. A legacy payload
persisted before this change, carrying plain zeros, still loads as literal zeros — there is no way
to retroactively distinguish a pre-fix measured zero from a pre-fix missing one, so no such
inference is attempted.

`ShortRecord.total_engagement` is a **rate**, not a total, and doesn't get the same fold-to-zero
treatment: it returns `None` ("unknown") when no platform ever reported a view count (every
`PlatformMetrics.views` is `None` — a credentials failure, scope denial, or a collector that never
ran). A total can validly absorb an unreported addend as "contributes nothing"; a rate's
denominator changes what its numerator means, so there is no zero-contribution reading of an
unknown rate that isn't a fabrication. If at least one platform reported (even a genuine 0 total
views), the result is unchanged: a well-defined `0.0`. `report.py`'s render/average call sites
treat `None` the same way the file already renders any other unknown value (the `'—'` dash) and
average only over shorts with a known rate.

The schema can only hold the distinction if the collector that populates it preserves it.
`tracking/collector.py`'s `YouTubeCollector` used `dict.get(key, 0)` when reading the Data/Analytics
API responses, which can't tell "key absent" from "key present with value 0" — so an unreported
metric silently became a measured zero on the way into `PlatformMetrics`. This matters because the
omissions are routine, not edge cases: the Data API drops `likeCount` entirely when a creator hides
likes, and drops `commentCount` when comments are disabled on that video. `_reported_metric(value)`
(`None` in, `None` out; otherwise `int(value)`) is used at every YouTube construction site instead,
and `fetch_video_metrics`/`collect_short_metrics` only overwrite a still-`None` field when the
source dict actually carries the key. The daily-series path (`DailyMetricPoint`) can't carry this
fix — its fields are plain `int = 0`, not `Optional`, so a missing per-day metric is structurally
indistinguishable from a measured zero there; widening that is a `schema.py` change, tracked
separately.

`InstagramCollector` had the same shape in two layers — `e["values"][0].get("value", 0)` building
the insights dict, then `int(values.get(name, 0) or 0)` constructing `PlatformMetrics` — and both
now yield `None` for a metric Graph did not report. The consequence worth spelling out is on the
failure path: `_get` returns `None` for a 403 scope denial, so `body` is `{}`, every name is absent,
and every metric is `None`. Before, that produced a full row of zeros. A missing-scope credential
is the platform declining to answer, and recording it as a measured zero made a permissions problem
indistinguishable from a Reel nobody watched — on `report.py`'s top/lowest-performer tables, and to
`decline.py`, which reads these as measurements and would see a cliff. A metric Graph genuinely
reports as `0` is still stored as `0`; only absence became unknown.

`test_scope_error_is_caught_logged_and_returns_empty` asserted `metrics.views == 0` from the
collector's first version. That assertion was the bug's fingerprint rather than a contract —
the test's own name calls the result "empty", and `PlatformMetrics` spells empty `None` — so it was
corrected rather than worked around, and now also pins `likes` and `impressions`.

`DailyMetricPoint` (`PlatformMetrics.daily`, one row per `dimensions=day` analytics query day)
carries the same missing-vs-zero distinction one level down: every measurement field (`views`,
`likes`, `comments`, `shares`, `watch_time_seconds`) is `Optional[...] = None`, `day` stays a
required `str` since it's the row's identity, not a measurement. `collector.py`'s
`collect_short_metrics` used to build these rows with `row.get("likes", 0)`-style defaults,
fabricating a zero for a column the API didn't return; it now passes the row value through
unchanged (`None` when absent). The two readers of `.daily` treat an unreported day as poisoning
the aggregate rather than as a zero, the same rule `engagement_rate` uses for `PlatformMetrics`
itself: `velocity.first_hours_velocity` returns `None` (not a number derived from a partial
window) when any day inside the early-hours window is unreported, or when the series lacks an
aggregate `views` total and any day anywhere in the series is unreported; `volume.short_day7_views`
skips a platform whose first 7 daily points include an unreported day, exactly as it already
skipped a platform with fewer than 7 points. Legacy on-disk rows loaded before this change carry
literal zeros and still load as zeros, per the same legacy rule as `PlatformMetrics` above.
