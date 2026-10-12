### Meta metrics collectors + audience-online table (`tracking/collector.py`)

`InstagramCollector` (`"instagram"`, replacing Metricool here) pulls per-Reel
`views`/`reach`/`likes`/`comments`/`shares` from `GET /{media_id}/insights` and best-time data
from `GET /{ig_user_id}/insights?metric=online_followers`. Credentials load lazily on first use
(never at construction — building `MetricsCollector` never touches SOPS); missing creds degrade
like `YouTubeCollector`. Metric name (`plays`/`views`, renamed at Graph v20.0) follows
`graph_version`. A 403/scope-mentioning Graph error is caught, logged, returns empty rather than
raised — live scope coverage is unverified. `saved` isn't fetched yet (no `PlatformMetrics` field).

`FacebookCollector` (`"facebook"`) is the same shape for the Page: per-video
`views`/`impressions`/watch-time (milliseconds, per Graph) from `GET /{video_id}/video_insights`
and the fan-online curve from `GET /{page_id}/insights/page_fans_online`, both followed through
`paging.next`. Metric NAMES are version-dependent (v21.0 deprecated the page-level
`total_video_*` family), so they live in one `_FB_VIDEO_METRICS` mapping keyed by the lowest Graph
major version using them — retargeting a version is one edit. A credential may report an older Graph version than the
`v21.0` default. Both edges need the `read_insights` scope; when the token does
**not** hold it, collection degrades by design, logging the missing scope NAME
once per collector — never a credential value — issuing no request and returning empty until the
token is re-consented.
