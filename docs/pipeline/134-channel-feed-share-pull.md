### Channel-level daily feed-share pull (`tracking/channel_daily.py`)

Every existing collector in `tracking/collector.py` answers "how did THIS short do" -- nothing
pulled channel-wide views by how the viewer arrived, so there was no way to say what share of a
day's channel views the Shorts feed itself drove. `tracking/channel_daily.py` fills that gap:
a daily pull of two YouTube Analytics v2 dimensions -- `insightTrafficSourceType` (traffic
source, including the `SHORTS` feed value) and `creatorContentType` -- and a pure function,
`shorts_feed_share`, that derives the Shorts-feed share of a day's channel views from whatever is
already stored.

**Auth.** `YouTubeCollector.fetch_channel_daily_dimension(dimension, start_date, end_date)`
issues the `reports().query(ids="channel==MINE", dimensions=f"day,{dimension}",
metrics="views", ...)` call against the same `_analytics_client()` every other analytics fetch
on this collector already builds -- there is exactly one YouTube auth path in this service, not
a second one for channel-level pulls. It degrades the same way `fetch_analytics_metrics` does:
no token, no client, or an API exception all return `[]`, never raise.

**Storage.** One JSON file per brand, `<gms_output_dir()>/<brand>/tracking/channel_daily.json`,
holding `{"brand": ..., "rows": [...]}` where each row is `{"day", "dimension", "value",
"views"}`. This mirrors `tracking/store.py`'s own root and per-directory layout
(`<root>/<project>/tracking/<project>.json`) -- same root, same style, no new database engine --
but keys on brand instead of project, since this pull is channel-wide rather than scoped to one
drive. `pull_channel_daily(youtube_collector, brand, start_date, end_date, root=None)` fetches
both dimensions and upserts the result: a row sharing the same `(day, dimension, value)` key as
an existing one replaces it rather than duplicating, so replaying a day (a partial-day re-pull, a
backfill re-run) can never inflate the store.

**Shorts-feed share.** `shorts_feed_share(rows)` is a pure function over the stored row shape --
no I/O, no API access -- returning `{day: shorts_views / total_traffic_source_views}` for every
day that has at least one `insightTrafficSourceType` row with a nonzero total. A day with zero
total traffic-source views is omitted, never reported as a fabricated `0.0`. `creatorContentType`
rows are stored alongside but never folded into this total -- the two dimensions are counted
separately so one can't leak into the other's share.

**Wiring.** `cli/handlers/metrics.py`'s `_cmd_metrics_collect` (`gomoveshift metrics collect
<project>`, however it is scheduled) calls `_pull_channel_daily` after the slot-performance fold and the audience-online
refresh, for yesterday only -- the daily window. `brand` comes from
`console.settings.current_brand()` (`GMS_CONSOLE_BRAND`, default `"gomoveshift"`) -- the
console's own multi-brand resolution, never a literal in this module. Same swallow-and-warn
failure contract as the fold and the audience-online refresh: a channel-daily pull failure is
logged and reported as `0` rows, never fatal to a collection that is already saved. If a
schedule invokes `metrics collect` once per tracked project, this pull runs once per project
per night rather than once per night; the per-day upsert makes every invocation after the first
a no-op rather than a duplicate, so this is a redundant API call, not a correctness bug.

**Reading it: `metrics feed-share`.** `gomoveshift metrics feed-share [--days N] [--threshold F]`
(defaults `7` / `0.70`) prints the interim-cadence exit rule as a report line: the current brand's
`shorts_feed_share` value for each of the last `N` measured days, `OK`/`LOW` against `--threshold`,
and a final `verdict` line -- `MET`, `NOT MET`, or `INSUFFICIENT` when fewer than `N` measured days
exist yet. Report-only, always exits `0`; no `channel_daily` rows at all prints one line and stops.
A daily digest can shell to this exact command so the exit rule is a line the operator
reads, not a query they run.

**Not built here.** Meta (`post_views`, IG reach) is out of scope -- the Meta scopes this
service holds are not granted for it, and nothing here stubs it.

**Network.** `youtubeanalytics.googleapis.com` is the same host `fetch_analytics_metrics` and
`fetch_analytics_daily_metrics` already reach through the identical client; it is covered by the
existing `googleapis.com` allowlist entry (`docs/pipeline/84-network-allowlist-credential-stores.md`)
and adds no new host.

`gomoveshift metrics {collect,backfill-ids,best-times,report,feed-share,analytics-probe,enrol-channel}`: `enrol-channel` walks the whole uploads playlist (`list_uploads`, uncapped) and registers unknown uploads (shorts `duration <= SHORTS_MAX_DURATION`, private/unlisted included), printing `EVIDENCE_COVERAGE` or `ENROL_CHANNEL_FAILED reason=...` (exit 2); `analytics-probe` runs one 2-day `views` query through `YouTubeCollector.probe_analytics()` and prints the typed `AnalyticsStatus` line (`ANALYTICS_OK` exits 0, any reason line exits 2); a collector logs that reason line once per run instead of a warning per video.
