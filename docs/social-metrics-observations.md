# Observations: metrics that are comparable, or honestly labelled as not

`gomoveshift/tracking/schema.py` keeps "not reported" (`None`) apart from a
measured `0` field by field. `gomoveshift/tracking/observation.py` raises that
same distinction to the whole observation, so a caller can act on silence.

## A `MetricObservation` carries four things a bare metrics dict cannot

1. **Why there are no numbers** (`ObservationStatus`): `not_configured`,
   `unqueryable_id` (an upload handle, never a public post), `not_returned`
   (queried; deleted, private, or not on this account), `permission_denied`,
   `rate_limited`, `provider_error` — and `measured`. Only `measured` may
   carry numbers; constructing any other status with a value raises.
2. **Cumulative, not interval.** `cumulative` is a lifetime-to-date counter.
   A per-interval change comes from `interval_between(earlier, later)`, which
   returns a distinct `IntervalDelta` type — so a delta can never be stored
   in a lifetime field. A counter that moved backwards is `counter_reset`
   (unknowable), never a negative delta, and an unreported end is unknown,
   never a change of 0.
3. **Three timestamps, not one.** `event_at` (the post went up), `fetched_at`
   (we asked) and `data_through` (the platform's numbers are complete through)
   are separate; `reporting_delay_seconds` and `age_at_fetch_seconds` are
   derived from them, so they cannot disagree. An unknown provider lag is
   `None`, which is not a claim of "no lag" — a fresh zero and a stale zero
   are different observations.
4. **Binding.** Every observation names its `account_id` and the real public
   `post_id`. A measurement bound to an upload handle raises.

## Refusing to average incomparable things

`compare_views()` groups measured view counts per platform and publishes each
platform's view DEFINITION alongside them (a YouTube view, a TikTok
`view_count` and an Instagram play count different events). It returns a
blended figure only when every measurement is the same platform, the same
format, and the posts were the same age at fetch (default bucket 24h);
otherwise `blended_views` is `None` and `declined_reason` says which test
failed. Inventing a blended number no platform reported is the defect.

## Collectors

`TikTokCollector.collect_observations_batch()` returns one observation for
**every** requested id — the previous `collect_metrics_batch` answered `{}`
for "all upload handles", "no access token" and "queried, nothing came back"
alike. `YouTubeCollector.collect_observation()`, `InstagramCollector.
collect_observation()` and `FacebookCollector.collect_observation()` do the
same for one post, mapping a refusal (missing creds, upload handle, Graph
scope/throttle/404) to its own status. `collect_short_metrics` /
`collect_metrics_batch` survive as numbers-only projections for callers that
only want the bare values.

`MetricsCollector.collect_observations_for_short(short)` is the one place
production collection calls a platform's observation method — it batches for
TikTok, calls `collect_observation` for the rest, and never raises past a
single platform's failure. `collect_for_short` (bare `PlatformMetrics`, for
legacy callers) and `collect_all` (below) both project from it, so there is
exactly one call site per platform API.

## The series: `observation_log`, `store.append_observation`, `interval.py`

`ShortRecord.observation_log` (`schema.py`) is a per-platform, append-only
series of raw `MetricObservation.to_dict()` entries — unlike `ShortRecord.
metrics`, where a fresh collection replaces the last read and nothing
persists how a post moved between two fetches. `store.append_observation`
writes to it and is idempotent on `(post_id, fetched_at)`, so a re-run
collection can never double-count a reading it already has.

`MetricsCollector.collect_all` appends **every** observation it gets —
measurements and refusals alike — via `append_observation`, and updates
`ShortRecord.metrics[platform]` only from a `MEASURED` observation's
`cumulative` numbers. A refusal is recorded in the series but never
overwrites a prior measurement, because a refusal carries no numbers.

`gomoveshift.tracking.interval.latest_interval(short, platform)` is the
first production consumer of `interval_between`: it rebuilds the last two
entries of `short.observation_log[platform]` and returns the `IntervalDelta`
between them, or `None` when there are fewer than two entries — never a
zero-filled delta standing in for "no data yet". `gomoveshift metrics
collect` prints this trend (and every non-measured platform's own refusal
reason) below its scheduler-consumed summary line.
