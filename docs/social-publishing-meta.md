# Meta publishing (Instagram + Facebook)

`gomoveshift/publishing/meta_auth.py`: shared Meta credential loader (SOPS
`meta_publishing.json.enc` in `$SEO_INPUTS_DIR` (default `seo` under `GMS_INPUTS_DIR`; else beside the SEO credentials directory), same `sops -d`
pattern as `pipeline._yt_creds`; fields `app_id`, `app_secret`, `page_id`,
`ig_user_id`, `page_access_token`, `token_type`, `scopes`,
`user_token_expires_at`, `data_access_expires_at`, `issued_at`,
`graph_version`; never logged raw).
`instagram_adapter.py`: the Instagram Reels adapter. `facebook_adapter.py` is the Facebook
counterpart and imports `meta_auth` from here.

## Instagram Reels API contract

Create container (`POST /<IG_ID>/media`, `media_type=REELS`, `caption`
from `InstagramPayload.caption`, `upload_type=resumable` — local bytes
only, no public `video_url`) → upload to the returned
`rupload.facebook.com` URI (checked against an exact HTTPS host allowlist
first) → publish (`POST /<IG_ID>/media_publish`). Containers expire 24h
after creation; `ContainerHandle.is_expired()` refuses a stale publish
before the API call. Graph version: the credential's `graph_version`,
falling back to `DEFAULT_GRAPH_VERSION` — never unversioned.

**Publish only once the container is `FINISHED`.** Meta processes the
upload after `rupload` returns, and `media_publish` on a container still
processing fails (code 9007, subcode 2207027, "The media is not ready for
publishing"). So `InstagramShortsAdapter` polls `check_container_status`
(`CONTAINER_READY_POLL_S` = 10, bounded by `CONTAINER_READY_TIMEOUT_S` =
300, inside `LEASE_SEC`; `sleep`/`monotonic` are constructor kwargs) before
every `media_publish`, on the fresh and the resumed path (a resumed
container renews the 900 s adapter lease first, since drain_queue's own
lease is shorter than the wait). `FINISHED` →
publish. `PUBLISHED` → adopt the media id, never publish again. `ERROR` →
`InstagramPermanentError` naming the container's status text. `EXPIRED` on
a resumed container (read as `EXPIRED` up front, or turning `EXPIRED` while
we wait) → a fresh container (nothing was published from it); `EXPIRED`
right after a create → `InstagramPermanentError`. Still
processing at the bound → `InstagramRetryableError`; the container stays on
the row and the next pass resumes it. A container can turn `FINISHED` ~15 minutes
after a delivery was refused as permanent, so the bound must not be read as final.

**Every GET sends the token as an `Authorization: OAuth` header**, never
`access_token` in `params` (`requests` puts params into the URL, and a
connection error's message echoes the URL into logs and stored errors).

**Error classification → queue state:** `code=190`/auth subcodes →
`auth_blocked`. Rate limiting / 5xx / 9007+2207027 (media not ready) →
`retry_wait`. Everything else →
`failed_permanent`; code 100 in particular covers both content/account
rejections *and* plain invalid-field/programming errors, so the message
names only "rejected, not automatically retryable" rather than asserting a
cause it can't know. Token never surfaces in an error.

## Publishing-limit quota (live, per-publish)

Instagram's publishing limit is **per IG account, not per app** —
any other tool posting to the same account consumes the same rolling-24h pool,
so a cached or once-per-batch reading is wrong: unmetered API publishing can
exhaust the account's quota for the rest of the window.

`InstagramReelsAdapter.publish_container` therefore calls
`GET /<IG_ID>/content_publishing_limit?fields=quota_usage,config` **live,
immediately before every `media_publish`** — never cached between
publishes. `quota_total` is read from the response's `config.quota_total`
(never hardcoded 100 — the platform can change it).

- `quota_usage >= quota_total` → refuses with `InstagramPublishingLimitError`
  (subclasses `InstagramRetryableError`, `queue_state = "retry_wait"` — the
  rolling window self-heals, so this is retryable, never permanent). The
  message names the used/total count and the rolling window length; it
  never includes the token.
- **No headroom is reserved** — a reading of 99/100 is allowed to publish;
  only `>= quota_total` refuses. This matches the platform's own semantics
  (the 100th publish is still legal) and keeps the gate simple and
  predictable rather than guessing a safety margin.
- A malformed or empty `data` array, or a response missing
  `quota_usage`/`config.quota_total`, is a **refusal**, not "quota fine" —
  fail closed rather than assume.

## Meta app mode — reachability probe (read-only, no publish call)

The stored `meta_publishing.json.enc` credential never recorded which Meta app mode
(Development vs Live) the app is in, or whether App Review is required before this
pipeline's first real Reels publish. A read-only probe, run from the project
venv against the Graph API using the stored `page_access_token`, closes most of that gap.
No credential value — app secret, page access token, or any token string — is reproduced
here; the probe uses `page_id`, `ig_user_id` and `app_id` from the stored record. The probe is safe to re-run; repeat these three calls:

```
GET /{page_id}?fields=id,name
GET /{ig_user_id}?fields=id,username
GET /{ig_user_id}/content_publishing_limit?fields=quota_usage,config
```

All three return HTTP 200 for a working credential: the stored page token reaches both the Page
(`<page_id>`) and the linked IG Business account
(`<ig_user_id>`), and the Content Publishing API answers for that
IG account with a real rolling quota — a `quota_usage` below `quota_total` (the platform's
usual total is 100 over an `86400`s window).

**What a 200 confirms:** the credential is live and reaches the right assets, and the
`content_publishing_limit` gate this doc already documents above is reading a real,
current number for the correct account.

**What a 200 does NOT prove — do not overclaim:** the Graph API has no field that exposes
an app's Development/Live mode. A Development-mode app still returns 200 on assets whose
admin/developer/tester role the token holder occupies — which an account owner does for both the Page
and the IG account — so a clean 200 on every read-only GET is exactly what a
still-in-Development app would also produce. Reachability is confirmed; whether App
Review is required before first-party publishing can only be settled by the app dashboard
itself or by the first real `media_publish` attempt. This is now a **low-risk, non-blocking
unknown** with a specific, recognizable expected failure mode if wrong: an OAuth/permission
error surfacing on the first `media_publish` call (classified per the error-classification
rule above, `code=190`/auth subcodes → `auth_blocked`), not a mid-run corruption risk.

**Token scope and expiry:** `GET /debug_token` on the stored page token reports
`is_valid: true`, `type: PAGE`, `expires_at: 0` (non-expiring) for a long-lived Page token, and a live scope list that
includes both `instagram_content_publish` and `pages_manage_posts` — the two scopes Reels
publishing needs on the IG and Page surfaces respectively.

**The stored `scopes` field is a write-time snapshot, and `debug_token` is still the
authority.** No setup script ships in this tree; the credential record is written by the operator's own tooling. A record written by a tool that persisted a fixed `SCOPES` constant into `meta_publishing.json.enc` rather than what the token was granted
makes the field read back the same fixed strings forever — scopes such as `business_management`,
`instagram_manage_messages` and `public_profile` could be live on the token and absent from
the record, and a probe reading the field reports a constant back as evidence. The record should hold the sorted scope list `/debug_token` returned for that token, so the field is a
real write-time snapshot dated by the record's `issued_at`. It is still only a snapshot:
grants can be changed in the app dashboard afterwards without touching the file, so
`debug_token` remains the authority for what a token can do **now**. A record that carries a fixed constant should not be read as a scope list at all.

A token should be refused when it is missing any of the ten required scopes, which now include
`read_insights` and `instagram_manage_insights`: without those two the metrics collectors
in `gomoveshift/tracking/collector.py` log the missing scope and return empty forever, a
failure that looks exactly like "no engagement yet". They also include the three
first-comment scopes, `pages_manage_engagement`, `pages_read_user_content` and
`instagram_manage_comments`: without them `meta_comments.drain_meta_first_comments` is
refused on every Reel and a scheduled dispatch logs `first_comment_needs_reauth=meta`
(Graph error `#200` on Facebook and `#10` on Instagram). A permission
the app itself has not added cannot be ticked in the Explorer: add it to the app's use
case in the Meta App Dashboard first. Regenerate in the Graph API Explorer with all ten
ticked if the token is refused. The dispatch log keeps reporting this until a reconnect clears it.

**Reconnecting** (after the app gains scopes): obtain one token from the Graph API Explorer;
a short-lived one is fine, since it can be exchanged for a long-lived token. Keep the app ID and
secret from the existing record, keep the old record as a backup beside it, and swap the new one
in atomically. Refuse a Page token that expires within 30 days, since that means the user token
was not long-lived. Record `user_token_expires_at`, plus `data_access_expires_at` or `issued_at` as described below.

**Reconnect every 90 days, for Instagram.** A Page token from a long-lived user token
does not expire, but the app's access to the user's data lasts 90 days from their last
login, and Meta's list of permissions that "do not expire" names the Page ones and no
`instagram_*` one ([Meta: authentication vs data access](https://developers.facebook.com/docs/facebook-login/auth-vs-data)).
When it lapses, Instagram publishing, first comments and insights stop; Facebook keeps
working. The record should hold the date as `data_access_expires_at` (from `/debug_token`; nothing
when Meta reports none). A digest can run `python -m gomoveshift.publishing.token_expiry` for the day counts, falling back to `issued_at` + 90 days for older records; it warns 14 days ahead.

**Quota corroborates the shared-pool warning above:** a probe run before this pipeline has
published anything can still show non-zero `quota_usage`, consumed entirely outside it —
any other tool posting to the account shares the same rolling pool.

## Permalink retrieval + durable queue wiring (Instagram)

`InstagramShortsAdapter` (bottom of `instagram_adapter.py`) durably queues a
publish through `queue.py`, mirroring `youtube_adapter.YouTubeShortsAdapter`.

**A container object has no `permalink` field**, and neither does
`media_publish`'s own response (confirmed against the live API: `GET
{container_id}?fields=id,permalink` raises `(#100) Tried accessing
nonexisting field (permalink)`; only a published *media* object supports it
— e.g. `GET {ig_user_id}/media?fields=id,media_type,permalink`). Requesting
`permalink` on a container therefore makes the *whole* status/reconciliation
read fail with `InstagramPermanentError` — which previously turned every
post-timeout reconciliation into a false "failed_permanent", even when the
Reel had actually published (a human would then republish it, creating a
duplicate). `InstagramReelsAdapter._fetch_permalink` fetches the permalink
in a always-separate, best-effort GET on the published media id; any failure
there (including the field genuinely being absent yet) resolves to
`permalink=None`, never to a raised error — permalink is presentation-only
and must never fail an otherwise-successful publish or reconciliation. The
media id is stored bare in `platform_post_id` and the permalink in
`queue.py`'s dedicated `permalink` column (see
`docs/social-publishing-queue.md`'s column contract).

Reconciliation before re-creating (`InstagramShortsAdapter._reconcile`,
mirrors `YouTubeShortsAdapter._reconcile`'s "a timeout is unknown, not
failed" handling): a delivery's persisted `platform_upload_id` is the
container id, written **before** upload/publish (the create call already
put a live container on the platform; this write makes it recoverable). On
retry, before ever creating a second container: if `platform_post_id` is
already persisted, adopt it (and `permalink`) with no live call. Otherwise
call `InstagramReelsAdapter.check_container_status`, which requests only
`status_code,id,status` on the container — a `status_code` of `PUBLISHED`
means the previous `media_publish` actually succeeded server-side even
though the client never saw the response, so the adapter adopts that
outcome (fetching the permalink separately, as above) instead of creating a
second container or re-publishing. The `content_publishing_limit` quota
gate still runs live inside `publish_container` on every fresh publish
attempt — wiring never bypasses it.

**Open question, not settled here:** `check_container_status` GETs the
*container's own id* and returns that id back as `media_id`.
`media_publish` may mint a distinct media id for the published post rather
than reusing the container id — if so, the reconciliation path above hands
the caller the wrong id. This can only be settled by publishing a real Reel
end-to-end and diffing the container id returned by `create_container`
against the id `media_publish` returns for that same publish; not attempted
here. Until settled, treat `check_container_status`'s `media_id` as
provisional.

**Error classification note:** Graph's code 100 also fires on plain
invalid-field/programming errors (this defect: requesting the nonexistent
`permalink` field), not only content/account rejections — the raised
`InstagramPermanentError` message says only that the request was rejected
and is not automatically retryable; it does not assert a content-policy
cause it cannot know.

The same permalink retrieval + `queue.py` wiring for Facebook is described in
`docs/social-publishing-facebook.md`.
