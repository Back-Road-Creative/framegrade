# Facebook Reels publishing adapter

`gomoveshift/publishing/facebook_adapter.py`. Imports the shared credential
loader from `gomoveshift/publishing/meta_auth.py` (`page_id` /
`page_access_token`), see `docs/social-publishing-meta.md`.
Three-call resumable upload against `/<PAGE_ID>/video_reels`: **start**
(`upload_phase=start`) → `video_id` + `upload_url`; **upload** the local
bytes to that URL (exact HTTPS host allowlist, mirroring Instagram);
**finish** (`upload_phase=finish`) → `video_state` (`DRAFT` / `SCHEDULED`
/ `PUBLISHED`) and `description`. Graph version: the credential's
`graph_version`, falling back to `DEFAULT_GRAPH_VERSION`. `SCHEDULED`
requires `scheduled_publish_time`: more than 10 minutes and at most 29
days from now, else a permanent error names the bound, before any
network call — routing (in-window vs. stage-now/finish-later) is the
caller's call.
**Error classification → queue state:** `code=190`/auth subcodes →
`auth_blocked`; rate limit / 5xx → `retry_wait`; content/account
rejections → `failed_permanent`. Token never surfaces in an error or log.

## Permalink retrieval + durable queue wiring

`FacebookShortsAdapter` (bottom of `facebook_adapter.py`) durably queues a
publish through `queue.py`, mirroring `InstagramShortsAdapter` with
Facebook's three-phase upload (start/upload/finish) standing in for
Instagram's create/upload/publish. `finish` carries no permalink, and none
is fetched: Graph's own `permalink_url` for a reel is exactly
`/reel/<video_id>/`, so `reel_permalink()`
derives the public URL from the video id with no network call at all.

`platform_post_id` here stores the **bare Facebook video id**, never packed
JSON, and the derived permalink is written in the **same** `transition`
that marks the row `published`. That ordering is load-bearing:
`transition` refuses every write once a row is terminal (`queue.py`), so a
permalink obtained after the published transition can never be stored. Without this,
a delivery can publish to the Page and its row keep
`permalink=NULL`. A resumed already-confirmed
`publish()` returns `permalink=None` in memory; its row already carries one.

**`check_video_status` reads `status.publishing_phase.publish_status`, not
`status.video_status`.** A published reel reports
`video_status: "ready"` and never `"PUBLISHED"`; a check keyed
off `video_status` would answer "not published" for every real reel —
the one answer that lets reconciliation call `finish` a second time and
publish a duplicate. The fixtures carry that envelope shape.

Reconciliation before starting a second upload session
(`FacebookShortsAdapter._reconcile`, mirrors
`YouTubeShortsAdapter._reconcile`'s "a timeout is unknown, not failed"):
a delivery's persisted `platform_upload_id` is the video id, written
**before** upload/finish (the start call already put a live upload
session on the platform — Facebook's `video_reels` has no
create-idempotency key). On retry, before ever calling `start_upload`
again: if `platform_post_id` is already persisted, adopt it with no live
call; otherwise call `check_video_status` — `video_status=PUBLISHED`
means the previous `finish` succeeded server-side even though the client
never saw the response, so the adapter adopts that outcome instead of
starting a second upload session or re-finishing.
