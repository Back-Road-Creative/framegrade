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

A successful Video response with no recognized numeric metrics triggers one
Reel-metric query on the same public ID and endpoint, also for lifetime
values. A refused Video request does not trigger this fallback. Existing
pagination is bounded to ten pages per query. Numeric metric names in the
response determine `post_format` (`video` or `reel`); two empty responses keep
the format unknown and the observation unanswered. IDs are never guessed or
rewritten.

Reel observations keep their play counts in explicit optional fields:

| Field | Meta metric | Meaning |
| --- | --- | --- |
| `reel_initial_plays` | `blue_reels_play_count` | Initial plays, excluding replays |
| `reel_replays` | `fb_reels_replay_count` | Replays within the same session |
| `reel_total_plays` | `fb_reels_total_plays` | Initial plays plus replays |

These metrics count playback from one millisecond; they are not the
three-second `total_video_views` counter. Reel `views` and `impressions` stay
unknown, and existing view-based comparisons/rankings do not substitute play
counts. Reel watch-time metrics (`post_video_view_time` and
`post_video_avg_time_watched`) include replays and are converted from
milliseconds to seconds. A play-only zero is still a measured observation;
absent play fields remain `null`, including in old records loaded from disk.
The fields survive tracking-record and observation serialization. Collection
health reports measurement availability, not a claim that every metric or
monetization field is present.

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
