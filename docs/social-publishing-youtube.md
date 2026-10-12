# YouTube Shorts publishing adapter

`gomoveshift/publishing/youtube_adapter.py::YouTubeShortsAdapter.publish()`
delivers one `ShortPublishRecord` through the durable queue
(`docs/social-publishing-queue.md`), orchestrating `YouTubeUploader` rather
than re-implementing its upload/retry/token/quota handling.

- **Guard:** shorts are canonical (`short_NN.mp4`), so
  `config.drive.youtube_upload_filename_pattern` widened to `^(?:lf|sl|short)_\d{2}\.mp4$`.
- **Ordering:** private upload with `status.publishAt`; `video_id` persists
  to the queue (`status=uploaded`) BEFORE the playlist call — a lost id there
  would re-issue `videos.insert` for content already live. A playlist failure after upload leaves `uploaded`
  with the id intact; resume retries only the playlist step, via
  `resolve_playlist` (`format: "shorts"` + the country derived from
  `sampled_regions` by `playlist_resolver.short_region`). `sampled_regions`
  hold PLACE labels ("West Cork", "County Kerry, Munster" — an Irish province
  rides as the "state" — or "Unicoi, Tennessee"), never a country, so
  the adapter passes the derived `ireland`/`usa` as `region` (place as
  `location`); `config.yaml` / the code defaults map it to "Ireland Shorts" /
  "USA Shorts". A short whose country cannot be derived (or whose config has
  no Shorts rule) is NEVER filed under `youtube_playlists.default`
  ("Unsorted (fallback)"): the playlist insert is skipped, a WARNING names the
  video id, and the delivery goes straight to `published` with no
  `platform_post_id`, so a resume never retries it forever. A scheduled
  `gomoveshift reconcile-playlists` run reports it. Long-form uploads keep the default.
- **Daily budget:** `videos.insert` has its own bucket
  (`config.drive.youtube_video_insert_daily_uploads`, default 100/day),
  separate from `youtube_quota_daily_budget` (unchanged, still governs
  `playlistItems.insert`). Once today's (UTC) uploads reach it, the
  adapter defers to next UTC midnight (`status=deferred`) rather than
  failing — `publishAt` tolerates the delay.
- **Reconciliation:** a timeout or lost response after `videos.insert` is an
  unknown outcome, never a failure. Before any new create call, `publish()`
  looks up the persisted `video_id` for the delivery (`find_delivery`) and,
  if one exists, confirms it with `videos.list(part="status,snippet",
  id=<video_id>)` before adopting it — a retry or duplicate dispatch never
  re-issues `videos.insert` for a delivery that already has a confirmed id.
  Only when reconciliation proves nothing was created (no persisted id, or
  `videos.list` returns no items) is a fresh insert permitted.
- **Create-idempotency gap, closed:** `videos.insert` has no create-idempotency
  key, so a lost response before anything is persisted (delivery stuck in
  `submitting`, no `video_id`) had nothing to reconcile against and could
  double-post. `_reconcile` now runs a channel-scan fallback in that one
  case — and ONLY that case, never on a first `queued` attempt: it walks
  `channels().list(mine=True)` → `relatedPlaylists.uploads` →
  `playlistItems().list` → `videos().list`, matching on exact title within a
  2-hour recency window (`youtube_adapter._ADOPT_RECENCY`). One match is
  adopted instead of inserting; two or more raise
  `AmbiguousUploadMatchError` rather than guess. `playlistItems.list` and
  `videos.list` cost units in the combined 10,000/day bucket, not the
  `videos.insert` daily-upload budget above, so the scan never eats into
  upload capacity.
- **Queue states:** `queued` → `submitting` (worker lease held, insert in
  flight) → `uploaded` (id persisted) → `published` (playlist assigned).
  `YouTubeQuotaError` lands in `retry_wait` (`error_class="quota"`), not a
  permanent failure — the daily quota window resets.
- **Worker lease:** a `submitting` delivery holds a lease
  (`queue.acquire_lease`, 15-minute ceiling); a live lease refuses a racing
  second insert, and an expired lease (killed worker) is reclaimable by the
  next attempt without a double `videos.insert`.
- The drive pipeline does not call `publish()` itself; the adapter is reached
  through the queue (`dispatch-queue`).
