# Social publishing dispatch (drain loop)

`gomoveshift/publishing/dispatch.py` drains `queue.py`'s delivery queue: each
pass selects due, non-terminal deliveries and, per row, acquires the worker
lease, calls a platform handler, and transitions the row on the result.

## Handlers are injected, never imported

`drain_queue(conn, handlers, ...)` takes `handlers: Mapping[str, Callable]`
keyed by platform. `dispatch.py` itself has no import edge onto any adapter.
`gomoveshift/publishing/handlers.py` is that wiring: `build_handlers(conn,
record_loader, *, youtube=..., instagram=..., facebook=..., tiktok=...)`
builds the mapping from whichever adapters/credentials the caller has
already constructed, keyed by tuples of `(adapter, media_loader)` (or, for
TikTok, `(TikTokCredentials, video_bytes_loader)`). Every adapter is
INJECTED — `build_handlers` never constructs one from config/credentials
itself, so a test never needs live credentials to exercise a handler.
Passing `None` (the default) for a platform leaves it unrouted; a platform
with no `handlers` entry is logged and skipped
(`DrainResult.skipped_no_handler`), left retryable. The CLI subcommand that
calls `drain_queue(conn, build_handlers(...))` is `gomoveshift dispatch-queue`
(see below).

`record_loader: Callable[[str, int], ShortPublishRecord]` resolves a
delivery row's `(project_name, short_index)` back to its immutable publish
record — the row itself carries no payload/media, only queue bookkeeping.
`youtube`'s media loader returns a `Path` (the adapter reads the file);
`instagram`/`facebook`/`tiktok`'s return `bytes` (their adapters take the
video in memory).

Facebook's CLI route currently supports video only. The concrete
`MetaPhotoGraphClient` targets the Instagram user and Instagram media endpoints,
so it must never be wired as a Facebook Page photo client. Facebook photo/carousel
records fail with `PermanentDeliveryError` before a media loader or adapter runs
when no Page-specific photo handler is supplied; they cannot fall back to the
Reels route. This is a visible unsupported-format failure, not a platform post.
Instagram's existing photo/carousel route and Facebook video delivery remain
available. A future Facebook Page photo implementation needs its own verified
client; setting an adapter's platform label is insufficient.

TikTok has no `*ShortsAdapter` class — `tiktok_adapter.py` is a
free-function toolkit (creator_info, chunk planning, Direct Post
init/upload/poll). `handlers.make_tiktok_handler` orchestrates those
functions against `queue.py` directly, mirroring the other three adapters'
"persist before the next irreversible call" pattern: `publish_id` is
written before the chunked upload starts, so a lost response is
recoverable, never duplicated.

## Retryable vs. permanent

- A handler raises `PermanentDeliveryError` to mark its delivery terminally
  failed on the FIRST raise. Use only for a response that will never
  succeed on retry.
- A handler raises `TransientMediaError` when it can't confirm its media
  still matches disk (missing file, moved run directory, unmounted drive —
  see `media_fingerprint.verify_media`). This retries exactly like the
  ordinary retryable path (same backoff, same lease release) but is
  **exempt from `max_attempts`**: an unmounted drive is operator-fixable
  and unrelated to the content or the platform, so the cap sized for API
  flakiness must never burn the delivery record for it. A row stuck on
  `TransientMediaError` retries indefinitely rather than requiring a
  manual re-enqueue once the drive is back.

  Both media loaders raise it: `_read_media` and `_media_path` run
  `verify_media` before handing anything to an adapter, so a missing,
  swapped or truncated short is caught immediately BEFORE an upload rather
  than discovered after one.

  The **sidecar** raises it too. Now that each row records its own
  `sidecar_path`, the sidecar sits on the run's drive beside the media and
  shares its offline exposure, so `SidecarRecordLoader` treats an absent
  sidecar as transient. This is a deliberate split, not a blanket change:
  a sidecar that EXISTS but is unparseable is a real defect and still
  raises `PublishPlanFormatError`, which is not retried forever.
- **Any other exception is retryable by default**: `attempt_count`
  increments and `next_attempt_at` is pushed out by an exponential backoff
  (`compute_next_attempt_at`, `drain_queue`'s `backoff_base_seconds` /
  `backoff_cap_seconds` params) before the row is left non-terminal for a
  later pass. Treating an unrecognised exception as permanent instead
  silently discards a publishable short — the failure mode this module
  exists to prevent.
- The retry loop is bounded: once `attempt_count` reaches `max_attempts`
  (a `drain_queue` param), the delivery transitions to `failed_permanent`
  with `error_class` prefixed `MaxAttemptsExceeded(...)` — visibly dead,
  never silently stuck retrying `queued` forever.
- Defaults: 60s base, doubling, capped at 3600s (1h), 8 max attempts — see
  the constants next to `compute_next_attempt_at` in `dispatch.py` for the
  quota math (Instagram's 100-posts/24h cap is shared with any other tool publishing to the same account; a stuck delivery must cost that quota almost nothing).
  All three are parameters with defaults, not hard-coded, so a canary run
  can tighten them without a code change.
- A per-platform adapter with its own exception hierarchy translates at the
  handler boundary in `handlers.py`: catch its genuinely terminal case(s)
  and re-raise `PermanentDeliveryError`; let everything else — including
  every adapter's own retryable subclasses and any exception the wiring
  doesn't recognise — propagate unchanged. Per platform:
  - **YouTube**: `AmbiguousUploadMatchError` (two-or-more channel uploads
    match a retrying short; only a human can resolve which is real) is
    terminal. `YouTubeQuotaError` and anything else propagate.
  - **Instagram** / **Facebook** (identical shape): `*PermanentError` (incl.
    `UnsafeUploadHostError`, and Facebook's `ScheduleWindowError`, both
    subclasses) is terminal. `*RetryableError` / `*PublishingLimitError` (a
    shared-quota refusal, which self-heals) propagate.
  - **A refused access token** (`*AuthBlockedError` on Instagram, Facebook
    and photos; `TikTokAuthBlockedError` for TikTok's `access_token_invalid`
    and `scope_not_authorized`) is never a row failure: it blocks every post
    to that platform, and only a person re-authorizing clears it.
    `handlers._pause_for_auth` pauses the platform (reason `auth:blocked`),
    puts the row back to `queued` when nothing of it reached the platform, and
    raises `PlatformAuthPausedError`, a `ThrottledDeliveryError` (never
    counted). The drain re-checks the pause before each row, so the platform's
    other rows stay queued and untouched in the same pass; `health check` rule
    16 raises one `platform_auth_blocked` incident naming the platform.
    Resuming the platform lets them go again on the next pass, the row that hit
    the refusal included: `queue.resume_platform` drops the 6 h throttle delay
    from rows whose `error_class` is `PlatformAuthPausedError`.
  - **TikTok**: no dedicated permanent exception subclass exists —
    `TikTokAdapterError` covers both transient and permanent failures with
    no way to tell them apart by type, so per the conservative default
    nothing else is translated from it (a refused token is above); only `confirm_publish`'s own terminal
    *return value* (`PublishOutcome.delivery_status ==
    "failed_permanent"`) becomes `PermanentDeliveryError`.
    `TikTokPollTimeoutError` is never translated — a poll timeout is an
    unknown outcome, never a failure, and the adapter's own docstring
    forbids resubmitting on it.

**Every Meta Graph error leads with its own classification fields.** The
Instagram, Facebook and photo mappers prefix each message with
`[graph http=<status> code=<code> subcode=<subcode> fbtrace=<fbtrace_id>]`
(`-` when Meta omits one; fbtrace is cut to `[A-Za-z0-9_-]`, 64 chars). A 4xx
with no error object still records its HTTP status. Because the tag leads,
the bounded `error_detail` always keeps it. Before this, five Instagram rows
were stored only as "Media ID is not available" or "unknown Graph API error",
and the cause could not be read back from the queue.

The Instagram mapper goes further. The tag also carries `op=<request>`
(`create_container`, `check_publishing_limit`, `publish_container`,
`check_container_status`, `fetch_permalink`, `list_recent_media`), so a failure
names which Graph call it was. When Graph answers with no `error` object (an
empty or non-JSON 4xx body), the message carries a 300-character,
whitespace-collapsed excerpt of the raw response body after
`(response body: ...)`, with the access token, any `access_token=` value and
any `Bearer`/`OAuth` string redacted. Classification (auth, retryable,
permanent) is unchanged. The Facebook and photo mappers keep the tag without
`op`.

`deliveries.error_detail` is bounded at 2000 characters for every dispatch
failure (it was 500, which cut a multi-reason `FinalTextReviewError` before its
last reason code). The console notify text keeps its own shorter clip.

## A failure is recorded from the row's live status

Adapters move their own row before they can fail (Instagram goes
`queued -> submitting -> uploaded`; the TikTok inbox handler goes
`submitting -> uploaded -> awaiting_manual_publish`). `queue.transition` is a
compare-and-set, so every `except` branch of `drain_queue` re-reads the row's
status and attempt count at failure time and transitions FROM that, never from
the status read when the row was selected. Without this, the summary can read `failed_permanent=1` while an Instagram delivery still read
`uploaded` with `error_class`/`error_detail` NULL -- the failure expected
`queued`, the row was `uploaded`, nothing was written, and the error text was
lost.

- A count (`failed_permanent`, `retried`, `throttled`) is incremented only
  when the transition actually wrote. The summary can no longer claim a
  failure the row does not carry.
- A row that is terminal by then (`published`, `failed_permanent`,
  `cancelled`), gone, or changed under the pass is never moved or
  overwritten. `result.errors` gets one line naming the delivery and its
  live status (`... but the delivery is now 'published', so the failure was
  not recorded: <message>`), and the lease this pass held is released. The
  one-line summary format is unchanged.

## A handler that returns has not always published

Two handlers return normally without publishing anything. The TikTok inbox
handler uploads a draft and parks the row at `awaiting_manual_publish`
(`next_attempt_at=9999-01-01`, reason `awaiting_manual`) for a human to post
from the app. The YouTube adapter, over its daily upload budget, moves the row
to `deferred` until the next UTC midnight (reason `platform_deferral`).

Without this, a TikTok delivery can read `published` without ever
being posted. The park kept the drain's lease; the lease expired; the
interrupted-attempt reconcile charged an attempt and set a retry time, which
un-parked the row; the next pass selected it, the handler returned early for
a parked row, and the drain recorded that return as a publish.

- After a handler returns, `drain_queue` re-reads the row. A row the handler
  left `awaiting_manual_publish` or `deferred` keeps its own wait: this pass's
  lease is released and nothing is counted as published.
- `_select_due` excludes `awaiting_manual_publish` by status, not only by its
  park time, so no write to `next_attempt_at` can put a parked draft back in
  front of the drain. `reconcile_awaiting_manual` is the only thing that
  closes one.
- An expired lease on a parked draft is released without charging an attempt.
  A `deferred` row's expired lease is still charged: it is selectable, so a
  worker dying on it every pass is a crash loop.

## Instagram "Media ID is not available" parks for an account scan

Graph sometimes answers a Reel publish with "Media ID is not available" when
the post did in fact go live, so neither a terminal fail nor a blind retry is
safe: the first loses a live post's record, the second can double-post. The
Instagram handler parks the row at `awaiting_ig_reconcile`
(`next_attempt_at=9999-01-01`, reason `awaiting_ig_reconcile`) instead.

- The park is the same as `awaiting_manual_publish`: `_select_due` excludes it
  by status, `drain_queue` counts nothing for it, an expired lease is released
  without charging an attempt, and `queue retry` refuses it.
- `dispatch-queue --apply` runs `reconcile_awaiting_ig` before the drain when
  Instagram is routed first_party. It scans the account's last 50 media with
  the same proof `queue recover-terminal` uses (`publishing/ig_media_scan.py`,
  `scan_for_post`): a caption-prefix or sha256 match closes the row
  `published` with that media id; proven absence re-queues it with one
  attempt charged, the old container cleared and the normal backoff (failing
  permanently at the attempt ceiling); a scan that cannot prove either (empty
  page, a full page short of the row's creation) leaves it parked.
- The pass prints `ig reconcile: considered=… published=… retried=… failed=…
  unprovable=…` when it had rows, and `publishing-status --validate` names a
  row parked longer than 24 h (`awaiting_ig_reconcile`).

## Other safety properties

- A platform with no handler registered is skipped WITHOUT acquiring the
  worker lease — the lease is only taken once a handler is confirmed, so a
  missing adapter never locks a row for the full lease duration for
  nothing.
- Lease contention skips, not retries, in the same pass.
- One handler's exception never aborts the pass (own `try`/`except` each).
- An empty (or fully-skipped) queue is success, not an error (a deliberate choice: an empty drain is not a failure).

## Acceptance lane (`tests/e2e/`)

A maintained journey lane that drives the real `drain_queue`, the reconcile
passes and a real temp sqlite queue, with recording adapters injected through
`build_handlers` (no network, no credentials, no real drive paths). Run it with
`pytest tests/e2e/`. Fixtures live in `tests/fixtures/acceptance/`;
`journeys.json` lists every journey and the test that covers it, and a test
fails if an entry points at a test that no longer exists.

Built: J07 (publish preparation: payload identity, caption full stop before
"Filmed", missing sidecar/asset refused, parked TikTok row never "published")
and J08 (delivery failure: transient retry, auth expiry, lost response after
remote acceptance, expired leases on `uploaded` / `awaiting_manual_publish`).
Not yet built: J01-J06 and J09-J12.

A handler that takes its own lease (TikTok) must be built with the same
`worker_id` that `drain_queue` gets as `owner`, as the CLI does; mismatched ids
refuse the row as "leased by another worker".

## Scheduling the drain

A scheduled drain runs `gomoveshift dispatch-queue --apply` from cron hourly
(for example at `:05`), from the checkout directory, under `flock -n` so an
overlapping run is a silent no-op; log to a file of your choice. Slots are
hour-aligned. `--apply` publishes for real to every platform routed
`first_party` — see the routing note below.

## The `dispatch-queue` CLI command

`gomoveshift dispatch-queue` (`cli/handlers/dispatch_queue.py`) is the entry
point a scheduled drain invokes. It builds the handler
registry and hands it to `drain_queue`.

- **Dry-run by default**, matching `publish-journal`. Without `--apply` it
  reports the non-terminal queue contents and which platforms are routed
  `first_party`, constructs no adapter, and calls no platform API.
- **Only routed platforms get an adapter.** A platform routed `off`
  is skipped before its credentials are decrypted through SOPS — so an
  absent or expired credential for an unrouted platform cannot fail the
  drain. With no platform routed `first_party` the command reports that and
  returns 0.
- **The class defaults are not a deployment's routing.** `config.py`'s
  `DriveConfig.publishing_route_*` default to `off` (nothing routed), and
  `Config.load` deep-merges the user override `config.yaml` in the config home,
  which is where a deployment routes a platform `first_party`. Read the merged config — `gomoveshift dispatch-queue`
  with no `--apply` prints the routed list — never the dataclass defaults.
- **Repair before the slot.** When YouTube is routed `first_party`, `dispatch-queue` first
  runs the Short final-text review (`short_review_refusal`) on every queued, non-shadow
  YouTube row (`publishing/queue_repair.py`). A refused row gets the two mechanical repairs, in
  order, on a scratch copy of its sidecar: the tag filter (as `queue sync-plan-tags`) and, when
  the refusal says the route record has no `bins`, the route re-measure (as `queue
  remeasure-short-route`, stages dir derived from the sidecar's output dir). The sidecar is
  replaced atomically only when the review then passes; otherwise it is untouched. Only tags and
  `route_record` change (never the stamp, schedule, recording date or experiment arms) and the
  stored `reviewed_plan_hash` is kept, never recomputed. One line per row:
  `QUEUE_REPAIR <delivery_id> <project> repaired|still_refused:<reason>|ok`; a repair is logged
  as a `queue_repair` `response_log` row. Without `--apply` the same lines print and nothing is
  written. A row still refused within 48 h of its slot raises the health rule
  `queued_short_refused` (`docs/pipeline/156-health-check.md`).
- **Slot hand-off and recovery.** Right after the repair (still before the loader caches a
  sidecar), `publishing/queue_slot_refill.py` acts on rows the review still refuses. A queued,
  non-shadow YouTube row still refused with its slot within `QUEUED_REFUSED_WITHIN_H` (48 h,
  the health rule 7 constant) never reaches publish: it swaps slots with the earliest LATER
  queued row that passes, so a run of refused rows drifts later while ready rows move up. With
  no passing row left (or a slot under 1 h away, too near for a new `publishAt`) it moves to the
  first free time of its region's daily Waypost shorts grid after the last queued slot (the
  `reslot-overdue-youtube` grid; the region comes from the sidecar, else from the grid that
  holds the row's own slot time). No hour that already holds a YouTube row is given out. Only
  `scheduled_at` (and the sidecar `publishAt`, when the sidecar is readable) changes, through
  `write_plan_slot` and `reschedule_delivery`; one `queue_slot` `response_log` row per move and
  one `QUEUE_SLOT <id> <old> -> <new> <reason>` line. The same step re-reviews `failed_permanent`
  rows the final-text review refused (no upload id, not already recovered): one that now passes
  is requeued into the next free slot through `recover_terminal_delivery`
  (`QUEUE_RECOVER <old id> -> <slot> queued_as:<new id>`); one that still cannot pass is
  `QUEUE_FAILED <id> still_refused:<reason>` and left. `JP_KEPT_FAILED_PERMANENT_IDS` (98, 102,
  106, 110, 114: JP keeps these failed) is never touched, and neither is a published row. Without
  `--apply` the same lines print and nothing is written. A new `failed_permanent` YouTube row
  raises the health rule `delivery_failed_permanent`.
- A queued row whose platform has no handler is left alone and counted as
  `skipped_no_handler`, never failed. Flipping a route back to `off`
  must not destroy deliveries that were already spooled.

### Where the `ShortPublishRecord` comes from

`build_handlers` needs a `record_loader: (project_name, short_index) ->
ShortPublishRecord`, but a queue row carries only `(project_name,
short_index, platform, plan_hash)`. `SidecarRecordLoader` reads the record
back from the Phase D sidecar that produced it
(`<project>_publish_plan.json`, rehydrated by
`publishing/plan_io.load_publish_plan`), caching per project because one
drain pass usually covers many shorts of the same project.

`--sidecar-dir` overrides where those sidecars are looked up; it defaults to
`seo.storage_path`. A queue row naming a `short_index` the sidecar does not
contain raises `LookupError` — the queue and the plan disagreeing is a real
inconsistency, not something to paper over.

Media needs no separate lookup: `record.media.path` is on the record
already (`MediaFingerprint.path`).

`gomoveshift enqueue-plan --apply` (`docs/social-publishing-queue.md`)
refuses to spool a delivery whose media fails
`media_fingerprint.verify_media` at enqueue time. The drain repeats the check
immediately before each upload: a handler that finds the media missing,
swapped or truncated raises `TransientMediaError`, so the delivery is retried
without counting against `max_attempts` (an unmounted drive is
operator-fixable).
