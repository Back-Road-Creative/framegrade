# Facebook metric collection

`gomoveshift metrics collect <project>` records a per-platform observation for
registered Facebook video IDs. Its aggregate short count means at least one
platform measured that short; it does not prove every platform is healthy.
The collector also logs `metrics health platform=... measured=... attempted=...
unanswered=...` for this run's registered platform IDs. Unanswered observations
produce a warning, remain in the observation log, and do not overwrite an
older measurement. A reported zero still counts as measured.

## Video insight contract

The collector requests `total_video_views`, `total_video_view_total_time`, and
`total_video_avg_time_watched` on `/{video-id}/video_insights`. These names are
Video insight metrics, not aliases selected by Graph version. Watch-time
values are converted from milliseconds to seconds.

Meta retired `total_video_impressions` for all API versions in June 2026.
Including it can reject the entire request. New observations therefore leave
`impressions` unknown (`null`); media views and unique viewers are different
quantities and must not be substituted into that field. Existing stored
measurements are not rewritten.

Meta's Video insight reference documents a three-second threshold for
`total_video_views`. Reel play metrics have different playback/replay
semantics. The collector does not substitute those into historical views or
invent video IDs from Page post IDs. A refused or unqueryable ID remains an
unanswered observation; confirm its object type and supported metrics before
repairing its registration.

The credential's Graph version is preserved. Meta retired v20 on September
24, 2026: review a credential pinned to that version separately, including its
publishing consumers, rather than silently changing it during collection.

## Monetization evidence

The current tracking schema does not store earnings or qualified views.
Meta documents `total_video_ad_break_earnings` (estimated US cents) and
`creator_monetization_qualified_views` for day/lifetime periods. Documentation
alone does not prove a particular Page, video, token, or monetization program
can read them. Verify authorized read access before adding explicit optional
fields, currency/period provenance, storage round-trip tests, and a Waypost
consumer. Missing access or data must remain unknown, never zero earnings.

References checked October 25, 2026:

- [Meta Video insights](https://developers.facebook.com/docs/graph-api/reference/video/video_insights/)
- [Meta v25 changes and deprecations](https://developers.facebook.com/blog/post/2026/02/18/introducing-graph-api-v25-and-marketing-api-v25/)
