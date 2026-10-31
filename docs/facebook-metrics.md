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

## Reel reach

After a Reel's play counts are measured, the collector asks for its reach
(unique accounts) on the same endpoint, one lifetime metric per request, in
this order: `post_impressions_unique`, then `post_total_media_view_unique`.
The first numeric answer is stored as `reach`, and `reach_metric` records
which name answered. A name Graph refuses is skipped for the rest of the run,
and a refusal never makes the measured observation unanswered. No answer
leaves `reach` unknown (`null`), never zero. Video-format posts get no reach
request. Neither name has been confirmed against the live Page yet.

The follower vs non-follower split is not collected: its Graph shape for
Reels has not been verified, so no field is stored for it.

## Page daily series

Each nightly `metrics collect` also pulls Page-level daily insights into the
brand's `channel_daily` store (`tracking.channel_daily.pull_facebook_page_daily`):
one row per day under dimension `facebook:<metric>`, value `page`, the number
in `views`. Candidate metrics, each on its own request so a retired name
cannot reject the others: `page_media_view`, `page_total_media_view_unique`,
`page_video_views`, `page_impressions_unique`.

- **Day label.** Page insights close a day at midnight Pacific; each value is
  stored under that Pacific day (`end_time` minus one second), not UTC.
- **Backfill.** A metric with no stored rows backfills from 2026-06-01 in
  windows of at most 90 days; later runs re-pull from three days before its
  last stored day, because recent days settle. A re-pulled day replaces its
  row. Only one request is made per window; `paging.next` is never followed.
- **Refusals.** A refused metric stops after its first window, logs Graph's
  message and is reported `unavailable`; nothing is stored for it.
- **Cron line.** The run prints
  `FB_PAGE_DAILY page_media_view=<rows|unavailable> ...` for every candidate,
  which is how the answering names are confirmed.

**Detector source.** `classify_facebook_reel_reach` reads the first of
`page_total_media_view_unique`, `page_media_view`, `page_video_views`,
`page_impressions_unique` (`FACEBOOK_DETECTOR_PREFERENCE`) whose stored series
has more than 28 days, so no one picks the metric by hand. With none that long
it falls back to the interim reel-plays series
(`docs/pipeline/141-bounded-response.md`). The cron line
`SUPPRESSION facebook=<status> days=<n> source=<metric|reel_plays>` names which.

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
