# Social publishing queue

`gomoveshift/publishing/queue.py` is a durable, per-delivery SQLite spool —
the storage half of publishing. The planning half (`models.py`/`plan.py`)
is deliberately **not** imported here; `plan_hash` is a plain `str`.

## Storage & schema

`<config.seo.storage_path>/social_queue.sqlite3`, WAL mode (`storage_path`
resolved by callers, same pattern as `content_ledger._default_ledger_path`).
The DB file and its `.schema.lock` are `chmod`'d group-writable (`0o664`)
by whichever process creates them, regardless of that process's umask —
the spool may be shared by more than one OS user, and SQLite/`os.open` both request
0o664 explicitly, so an unmasked request is the only way another user
can still open it read-write. One row per `(project_name, short_index, platform, plan_hash)` (unique). A
partial unique index over non-terminal statuses enforces **at most one
ACTIVE delivery per `(project_name, short_index, platform)`** in the DB —
`enqueue()` with a different `plan_hash` while one is in flight raises
`sqlite3.IntegrityError`. A second, independent partial unique index over
`status='published'` rows enforces **never republish a delivered short**:
once a `(project_name, short_index, platform)` reaches `published` (a
terminal status the active index deliberately excludes, so a cancelled row
stays re-enqueueable), `enqueue()` with a new `plan_hash` for that same
3-tuple returns the existing published row's id and inserts nothing — never
raises, same idempotent shape as an exact 4-tuple match.

**Column contract — `platform_post_id` vs `permalink`:** `platform_post_id`
is ALWAYS a bare platform id (a YouTube playlist id, an Instagram media
id, …) — never JSON, never any other encoding. Every adapter writes the
same shape, so a reader (reconciliation, a status export, the "prove a
post published" runbook) never guesses a per-platform encoding. `permalink`
is a dedicated column holding the platform's public URL for the delivery
when the platform returns one (Instagram does; YouTube's playlist id is
not a permalink and the column stays `NULL` there). Both are set together
via `transition(..., platform_post_id=..., permalink=...)`.

**Column contract — `shadow`:** `0` (the default) is a real delivery; `1`
marks a REHEARSAL that must never be published. `dispatch._select_due`
filters `shadow = 0`, so a shadow row is invisible to every drain path —
the CLI, a scheduled drain, and any future worker — rather than being skipped inside
the drain loop where a different caller could bypass the check. That is the
difference between "shadow mode is safe" and "shadow mode is safe as long
as nobody runs `dispatch-queue --apply`". Set it with
`enqueue-plan --shadow`; rows written before this column existed default to
`0`, because they are live ammunition.

**Column contract — `sidecar_path`:** the exact publish-plan sidecar file
(`<project>_publish_plan.json`) a delivery was enqueued from, recorded by
`enqueue()` at insert time. A real run's sidecar lives in the run's OWN
output dir, not one shared flat directory, and a queue can hold deliveries
from several runs (or drives) at once — see "Enqueue from the publish-plan
sidecar" and `cli/handlers/dispatch_queue.SidecarRecordLoader` below.
`NULL` for a row enqueued before this column existed or by a caller that
doesn't know the sidecar's path.

**Column contract — `published_at` / `published_at_source`:** `published_at`
is the moment THIS QUEUE recorded the publication, never the platform's own
publish time — a human confirming a TikTok inbox draft
(`console.pubqueue.mark_published`) can click hours or days after the post
actually went live, and TikTok's `status/fetch` payload carries no
publication timestamp of its own, so no true platform time is available and
none is invented. `published_at_source` names which of four provenances set it — `PUBLISHED_AT_OPERATOR_CONFIRMED` (human click, least trustworthy), `PUBLISHED_AT_PLATFORM_RECONCILED` (delayed poll, unknown lag), `PUBLISHED_AT_PLATFORM_CONFIRMED` (call returned success, seconds-accurate), or `PUBLISHED_AT_PLATFORM_SCHEDULED` (the platform accepted it and named a later time). `transition(..., new_status="published")` raises unless one is passed; `NULL` predates this column, never guessed.

**Column contract — `platform_published_at`:** when the platform itself goes live with the delivery, as the platform stated. Distinct from `published_at`, which is when THIS queue handed it over. The YouTube path uploads with `status.publishAt` + `privacyStatus=private`, so its insert returning success is an acceptance, not a publication — labelling that `platform_confirmed` ("within seconds of the real publish time") overstated the evidence by the whole length of the schedule, and the real instant, which YouTube had just been told, was thrown away. The adapter now records the same value it handed to `upload_video`, never a re-derived one, so the receipt and the API body cannot disagree.

**Facebook defers server-side as well.** `make_facebook_handler` never passed a
`scheduled_publish_time`, so `record.schedule` — the slot the scheduler picked for this short — was
ignored and every reel published the moment the queue drained it.
`facebook_adapter.due_schedule(record)` now resolves it: the per-platform slot
(`ScheduleWindow.scheduled_at_utc_by_platform["facebook"]`) wins over the shared one, because the plan builder
lists a platform there only when it actually resolved a different hour for it, so using the shared
slot would publish at a time the scheduler deliberately moved this platform off. A slot less than
`MIN_SCHEDULE_LEAD` (10 minutes) away returns `None` rather than raising: the queue drains *at* or
just after the slot, so "under ten minutes away" means the moment has arrived and publishing now IS
honouring the schedule — handing it to the API instead would raise `ScheduleWindowError`, which the
handler classifies as permanent, on every normal delivery. A slot past `MAX_SCHEDULE_LEAD` still
raises, because that is a genuine mistake and failing visibly beats publishing months early. The
`video_state`/`scheduled_publish_time` parameters stay an override for a caller that names both. A
reel the API merely *accepted* records `PUBLISHED_AT_PLATFORM_SCHEDULED` plus the same instant
handed to `finish_upload`; one that really published on the call keeps `platform_confirmed` and a
`NULL` `platform_published_at`.

`platform_published_at` is accepted by `transition()` only on the move to `"published"`, only alongside `PUBLISHED_AT_PLATFORM_SCHEDULED`, and that source in turn *requires* it: "scheduled" is a claim that a specific later time is known, and without one it says less than `platform_confirmed` while sounding like more. The other two platform sources already claim to be the publish moment, so carrying a second, different time beside one of them would be two claims about one event with nothing to say which a reader should believe. The value is parsed as ISO-8601 before it is stored — an unparseable string reads as a real platform receipt on every surface downstream. The console shows it: the delivery detail page labels the source "platform scheduled (publishes later)" and names both moments, and the queue list gives the `published` group a "Publishes at" column, because a scheduled row sits there for having finished with the QUEUE, not for being live. Until the console learned the source it fell through to "unknown (legacy)" — the most precise evidence this queue can hold, labelled as none at all. It ALTERs idempotently onto a live DB and stays `NULL` for legacy rows and for every path that publishes immediately: an honest unknown, never derived from `published_at`, which would be a fabricated platform receipt.

**Column contract — `destination_account`:** WHICH account the delivery landed on, as the adapter that published it held the identifier. `platform` says "facebook"; it does not say which Page, and until this column existed nothing did — `facebook_adapter` held `page_id` and `instagram_adapter` held `ig_user_id`, both taken from the brand's credentials at construction and both discarded once the post returned. So a published row proved a post existed and could not say whose feed it was in. That was tolerable while one operator ran one set of credentials; it is not, now that brands are a tenant boundary and credentials are rotated and mistyped. A delivery that went to the wrong Page was indistinguishable on every surface from one that went to the right one.

The value is read off the HTTP CLIENT (`FacebookReelsAdapter.account_id` / `InstagramReelsAdapter.account_id`), because the client is the object holding the credential that was actually used — the adapter above it never sees one. Both publish paths go through a local `_account_of`, which returns `None` for a client exposing no account identity (an injected test double, or a future client for a platform with no such notion); that tolerance is why a test pins `account_id` on each real client, since a rename would otherwise turn every receipt into `NULL` with nothing failing.

It is accepted by `transition()` only on the move to `"published"` — an account name on a queued or failed row claims something landed there when nothing did — and it is never re-derived later from whatever credentials are loaded now: those can have been rotated or corrected since, so a derived value would rewrite history to agree with the present configuration, which is exactly the question a reader is asking. A blank or whitespace-only value is REFUSED rather than stored: `""` is not an unknown, it is a stored answer that reads as one on every surface while being indistinguishable from a bug that dropped the real value. The once-only guarantee rides on the same CAS as `published_at` — the UPDATE is gated on a non-terminal status, so a reconcile running after a credential rotation cannot rewrite where the post actually went. It ALTERs idempotently onto a live DB and stays `NULL` for legacy rows.

**First comments — `first_comment_text` / `first_comment_posted_at` /
`first_comment_error` / `first_comment_attempts`:** every short published on YouTube, Instagram and
Facebook gets a first comment linking its full drive (the parent long-form
video): `🎬 Watch the full drive: https://youtu.be/<parent_lf_video_id>`,
built by the one function `first_comment.full_drive_comment_text`.
**TikTok gets none** — the Content Posting API has no comment endpoint, so
its rows never carry text.

- **Enqueue.** `enqueue_from_plan_sidecar` sets `first_comment_text` from
  the record's `parent_lf_video_id` for youtube/instagram/facebook rows and
  leaves it `NULL` for tiktok (a record without a parent id is warned about
  and left for backfill; it never blocks the enqueue).
- **Backfill.** `enqueue_from_plan.backfill_first_comment_text` fills rows
  whose text is `NULL`/empty from the row's `sidecar_path` record (matched on
  project + `short_index`). Idempotent; a posted row is never touched; an
  unreadable sidecar or missing parent id is skipped with a WARNING and never
  raises. `dispatch-queue` runs it once per run, before the drains, when any
  of the three platforms is routed `first_party`.
- **Selection.** `queue.select_due_first_comments` returns rows with a
  text, no `first_comment_posted_at`, a publish at least
  `FIRST_COMMENT_DELAY_S` (300s, 5 minutes) in the past, and a comment
  target. The publish age comes from a real labelled publish moment
  (`published_at` AND `published_at_source`). A `published` row with no
  source (published before `published_at_source` existed: early rows) is due once its `updated_at` and its
  `scheduled_at` (when set) are that old: the move to published stamps
  `updated_at` and later writes only move it forward, so it bounds the
  publish from above. The bare, unlabelled `published_at` is never
  trusted. At most `FIRST_COMMENT_BACKLOG_PER_PASS` (1) of these per
  platform per pass, untried rows first: all 20 at once would put the same
  link under every post within a minute. Each row carries
  `comment_target_id`: the YouTube **video** id (`platform_upload_id` —
  `platform_post_id` is the playlist there) or the IG media / FB reel id
  (`platform_post_id`). **A paused platform
  (`is_platform_paused`) is excluded**, so nothing comments while an
  operator hold is on.
- **Posting.** `youtube_comments.drain_first_comments` uses
  `commentThreads().list` then `commentThreads().insert` (needs
  `youtube.force-ssl`);
  `meta_comments.drain_meta_first_comments` calls
  `InstagramReelsAdapter.post_comment` / `FacebookReelsAdapter.post_comment`
  (`POST {graph}/{id}/comments`, needs `instagram_manage_comments` /
  `pages_manage_engagement`). Adapters are built lazily, only when a row is
  due, and only for platforms routed `first_party`.
- **Never a second copy.** Before each post the poster reads the comments
  already there: the Instagram/Facebook adapter `GET {graph}/{id}/comments`
  (at most `first_comment.COMMENT_CHECK_MAX_PAGES` pages of 100), YouTube
  `commentThreads().list` on the video (at most
  `youtube_comments.COMMENT_CHECK_MAX_PAGES` pages of 100, 1 quota unit a
  page). When one already carries exactly the text, it is recorded as the
  first comment and nothing is posted. So a comment the platform created
  whose answer was lost to a read timeout (Graph
  reads with a 30 s timeout, googleapiclient with 60 s) is found on the next
  pass, not posted again. A check that fails posts nothing and the row
  retries; past the page cap the post goes ahead. A 2xx answer that is not
  a JSON object with a `data` list counts as a failed check, never "no
  comments". The check sends the token as an `Authorization: OAuth` header,
  not in the query string, so a network error's URL cannot carry it into
  `first_comment_error`.
- **Errors.** A permission/scope error is recorded on `first_comment_error`,
  logged once per pass naming the permission, never raised, and the row stays
  retryable. Any other error is recorded and counted `failed` per row; one
  bad row never aborts the pass. Success sets `first_comment_posted_at`.
- **Giving up.** A refusal because of the post itself adds one to
  `first_comment_attempts`; after `FIRST_COMMENT_MAX_ATTEMPTS` (8) the row
  is never selected again. Its post is gone (a short taken down by hand)
  or closed to comments, and a scheduled pass would otherwise call the
  platform about it forever. YouTube: a 4xx such as 404 `videoNotFound` or
  403 `commentsDisabled`. Instagram/Facebook: the adapter's
  `*PermanentError` (a deleted reel is Graph code 100). Nothing else counts:
  a 5xx, a rate limit or spent quota, a lapsed token, a missing scope or a
  network failure says nothing about the post, so an overnight outage never
  uses up a live short's tries.
- **Output.** Each drain prints `first_comment: posted=N failed=N
  [needs_reauth=1]` above the summary. A scheduler may keep only a pass's
  last line, so the summary line adds the two drains up too:
  `first_comment_posted=N first_comment_failed=N
  first_comment_needs_reauth=youtube,meta`, each only when non-zero
  (`meta` is the one Meta app behind Instagram and Facebook).

`open_queue_db` upgrades an existing DB file created before the
`permalink`, `shadow`, `sidecar_path`, `published_at_source`, or
`first_comment_*` columns existed: schema creation stays
`CREATE TABLE IF NOT EXISTS` (which never adds a column to a pre-existing
table), so it is followed by a `PRAGMA table_info` check that runs
`ALTER TABLE deliveries ADD COLUMN permalink TEXT` (and the equivalent for
`shadow`, `sidecar_path`, `published_at_source`, and the four
`first_comment_*` columns) when the column is missing.
`first_comment_attempts` is `INTEGER NOT NULL DEFAULT 0`, so rows already
in the queue read 0: a `NULL` is never under the cap and would stop every
first comment. No data-migration
path is needed for the initial column adds.

States: `queued -> staging -> staged -> submitting -> processing ->
published`; failures `retry_wait`, `auth_blocked`, `failed_permanent`.
Terminal (`published`, `failed_permanent`) is enforced in the `UPDATE ...
WHERE` clause itself, not just via `expected_status`.

## API & network I/O

`open_queue_db`, `enqueue`, `transition` (compare-and-set), `acquire_lease`,
`find_delivery` — see docstrings in `queue.py`. No network I/O may occur
under a write transaction: every function is one short statement inside its
own `with conn:` block with no callback seam.

## Enqueue from the publish-plan sidecar

`gomoveshift/publishing/enqueue_from_plan.py` — `enqueue_from_plan_sidecar(conn,
sidecar_path)` is the reader that bridges Phase D's
`<project_name>_publish_plan.json` sidecar (writer:
`_emit_publish_plan_sidecar` / `_serialize_publish_plan` in
`gomoveshift/pipeline/drive_phases/phase_d.py`) into this queue. For every
short in the sidecar, it calls `enqueue()` once per platform in that
short's `routes.selected` (a platform absent from `selected` gets no row),
using the record's `reviewed_plan_hash` as the queue's `plan_hash` and the
sidecar's own resolved path as `sidecar_path` — so the drain reads back
exactly the file that produced the row, never a directory guess.

- **Idempotent by construction**: re-running over the same sidecar enqueues
  zero new rows — `enqueue()` is already idempotent on the exact
  `(project_name, short_index, platform, plan_hash)` 4-tuple. The reader
  reports which rows were new vs. already present via `EnqueueResult`
  (`.enqueued`, `.already_present`, and the matching `_count` properties).
- **Plan-changed conflict surfaces loudly**: when a DIFFERENT `plan_hash` is
  already ACTIVE for the same `(project, short_index, platform)` —
  `enqueue()`'s `sqlite3.IntegrityError` — the reader re-raises as
  `DeliveryPlanConflictError`, naming the project, short index, platform,
  and both the active and incoming `plan_hash`. Never swallowed.
- **Untrusted-input handling**: the sidecar is JSON on disk, so malformed
  JSON, a missing key, a wrong-typed field, or a platform outside
  `KNOWN_PLATFORMS` (`gomoveshift/publishing/models.py`) raises
  `PublishPlanSidecarError` naming the sidecar's file path — never a bare
  `KeyError`/`TypeError`.



## The `enqueue-plan` CLI command — and why the gate lives here

`gomoveshift enqueue-plan <pipeline_state.json>`
(`cli/handlers/enqueue_plan.py`) is what actually calls
`enqueue_from_plan_sidecar`. Before it existed the chain had a hole: Phase D
wrote the sidecar and `dispatch-queue` drained the queue, but nothing put
deliveries into it.

**Shorts publish automatically; no human review is required.** Every drive
runs `--no-review`, so no Gate C/D answer is recorded, and that absence does
not block the queue: shorts publish without a manual upload step.
A successful publish run (Phase D done, `youtube_upload` on, no skip flag)
calls this handler with `--apply` itself — `pipeline/auto_enqueue.py`, hooked
into `DriveOrchestrator.run` before the DoD validator. A refusal from it is a
run error, never a silent skip.

When a human *did* confirm Gate C/D, `review_gate` records
`gate_cd_confirmation_hash` over every `*_publish_plan.json` sidecar, and the
command still refuses when that confirmation carries no hash or the
recomputed hash no longer matches — the plan changed after it was reviewed,
so what was approved is not what would be published.

`--apply` also runs a series of whole-plan-refused gates, in order, before
any queue write — each reports every violation across every sidecar at
once (not just the first) and, on any hit, nothing is spooled: media
verification (`verify_media`), privacy status, the rolling 7-day
per-platform ceiling, the batch anti-spam gate, and
payload-field text limits (`publishing.models.validate_payloads`, limits
table in `pipeline.encoding_constants.PAYLOAD_LIMITS` next to the media
rendition limits): a short whose YouTube title/description/tags,
Instagram caption, Facebook description, or TikTok caption exceeds the
platform's documented length is refused here, before the platform API
ever sees it.

Dry-run by default, like `publish-journal` and `dispatch-queue`: without
`--apply` nothing is written. A run with no sidecar (every route `off`) is
success, not an error.

A `DeliveryPlanConflictError` stops the command rather than continuing to
the next sidecar: the plan moved under an in-flight delivery, and enqueuing
the remaining shorts would spool a partially-conflicting run.

### Shadow runs

`gomoveshift enqueue-plan <state> --apply --shadow` spools the whole plan
with `shadow = 1`. Every row is real — same plan hash, same platforms, same
idempotency — but no drain path can select it, so the run proves the chain
end to end and leaves nothing a scheduled drain can fire. `dispatch-queue` reports
the held-back count on its own line (`shadow (never published): N`) so the
rows do not read as stuck deliveries.

To promote a rehearsal to a real run, clear the shadow rows and re-enqueue
without `--shadow`; there is deliberately no in-place flip, because "make
this ammunition live" should be an explicit re-enqueue against a reviewed
plan, not a status update.

### End-to-end order

1. Phase D writes `<project>_publish_plan.json` (only when a route is `first_party`, and
   only on a publish pass: an audit pass never writes or rewrites it, so the plan this
   command reads is always one that the verified-place and hand-title gates enforced on).
2. Gate C/D — optional human review; when answered it records the confirmation hash.
3. `gomoveshift enqueue-plan <state> --apply` — run automatically at the end of a successful
   publish drive (`pipeline/auto_enqueue.py`); by hand only to re-spool. Spools deliveries (add `--shadow` to rehearse),
   recording each row's `sidecar_path`. `--max-shorts N` spools only the N lowest-ranked
   shorts (a scale-back). The cap is applied in memory over the approved plan: the sidecar is
   never rewritten, so a Gate C/D content hash keeps describing what a human reviewed, and
   enqueue-time media verification covers only the shorts that will actually be spooled.
   Editing a human-confirmed plan file to drop shorts instead flips that hash and the command
   refuses.
4. `gomoveshift dispatch-queue --apply` — drains them (also the scheduled path).
   `SidecarRecordLoader` resolves each delivery's record from its own
   `sidecar_path` when the column is set, falling back to
   `--sidecar-dir`/`<project>_publish_plan.json` only for rows that predate
   that column — so `--sidecar-dir` is no longer load-bearing for rows
   enqueued since the column was added, even when a run's output lives on a
   different drive than `seo.storage_path`.

## A delivery is not due before its planned publish time

`_select_due` gated only on `next_attempt_at`, which `enqueue` leaves NULL — so every queued row
was due the instant it was spooled. The plan's `schedule.scheduled_at_utc` lived only in the
sidecar and no drain path consulted it. **Only the YouTube adapter defers server-side**
(`publish_at`); Instagram and TikTok publish the moment they are called, and the Facebook handler
defaults to `video_state="PUBLISHED"`. (Facebook defers too now — see
`facebook_adapter.due_schedule` below.) One `dispatch-queue --apply` would therefore have dumped a
whole week of shorts at once.

- `enqueue` records `scheduled_at` on the row (nullable column, added by the same idempotent
  `ALTER TABLE` upgrade as `permalink`/`shadow`/`sidecar_path`).
- `_select_due` adds `AND (scheduled_at IS NULL OR scheduled_at <= now)`, so the gate lives in the
  SQL selection — no caller (CLI, a scheduled drain, a future worker) can publish early by taking a different
  path, exactly as with shadow rows.
- `enqueue_from_plan_sidecar` **refuses** a record with no `schedule.scheduled_at_utc` rather than
  spooling it as publish-immediately.
- `NULL` means a row enqueued before the column existed and stays due: stranding it silently is
  the same class of failure this guard prevents.
- `dispatch-queue` prints `not yet due (waiting for their slot): N, next <time>` so a correctly
  holding queue does not read as a stuck one.

The two times answer different questions: `next_attempt_at` is retry backoff, `scheduled_at` is
the plan's publish slot.

### Blackout dates

`_select_due` also excludes a row whose effective date (`scheduled_at`'s date, or `now`'s date for
a row with no recorded schedule) is one of the current brand's `policy.blackout` dates
(`publishing/brand_policy.py`, the brand-policy module). Same treatment as a paused
platform: no status change, no attempt bump, so the row is simply due again once the blackout date
passes. The console Publishing tab labels a held row `held: blackout`.

### Quiet after alert

`_select_due` also excludes, per platform, every non-terminal non-shadow row of a platform whose
most recent SUPPRESSED *alert* fired within `policy.quiet_after_alert_hours` hours of `now`
(`publishing/brand_policy.py`, the brand-policy module). The alert is the
`response_log` transition INTO `SUPPRESSED` (`dispatch.platform_quiet_alert_at`) — a scheduled
re-log while still SUPPRESSED does not restart the clock, and only `mode='live'` rows count. Same
treatment as blackout and a pause: no status change, no attempt bump. `quiet_after_alert_hours`
omitted/`None` costs zero queries. The console Publishing tab labels a held row `held: quiet`;
blackout wins when both apply.

### One slot, four platforms, four publish times

Instagram's best hour is not YouTube's — `resolve_shorts_times_for_region` resolves the
window per `(region, platform)`, and the plan carried ONE `scheduled_at_utc` every row copied.
`ScheduleWindow.scheduled_at_utc_by_platform` carries the difference: a platform appears **only
when its resolved hour differs** from the slot, so the map never restates the slot in every record.

`build_publish_plan(shorts_times_by_platform=…)` fills it, indexing each platform's window by the
short's ordinal **within its own publish day** (slots ranked by time) — the scheduler fills a day
from that window in order, so a platform's best hour moving to the front moves that day's first short.
A window shorter than the day's short count leaves the overflow shorts on the slot.

Phase D is the producer: `_emit_publish_plan_sidecar(region=…)` resolves the window once per
first-party platform against the spool's `slot_performance`/`audience_online` rows (the same
database `enqueue-plan` and `metrics ingest` use) and passes only the platforms the data moved. No
rows, no region, an unopenable spool, or an explicit `--shorts-times` all write the older single-slot shape —
every platform on the slot — and say so in the phase-D log; none of them fail the drive.

`enqueue_from_plan_sidecar` reads the map per row and falls back to `scheduled_at_utc` for any
platform it does not name — never to `NULL`, which `_select_due` reads as "due now". A malformed
entry is refused, not dropped. A sidecar written before per-platform times loads with an empty map. The slot never leaves the
region's configured grid: the best-time resolution picks *which* configured hour, this only carries it to the row.

## The queue must be the same database `enqueue-plan` wrote to

`seo.storage_path` defaults to the cwd-relative `./data/seo`, and the project `config.yaml` that
overrides it with the real absolute path is itself found as `Path("config.yaml")` (`Config.load`)
— also cwd-relative. `open_queue_db` creates what it does not find, so a drain started from the
wrong directory used to open an **empty** SQLite queue, print `non-terminal deliveries: none`
and exit 0: nothing published, forever, with a cheerful log line.

Two guards, both by construction:

- `dispatch-queue` refuses a **relative** `seo.storage_path` before opening any database.
- A scheduled drain should `cd` into the checkout directory before invoking the CLI (cron's cwd
  is the user's home), so the project `config.yaml` is the one found.

Run the drain from a released revision, not a development checkout that nothing keeps up to date.
Every tree that enqueues or drains must carry the same absolute queue and credentials paths in its
`config.yaml`. The CLI prints an `  error:` line above its summary for each failure, so a scheduler
that logs the output records a retry's reason.

## Where a delivery's media lives

Shorts **render** to `<stages_base>/day_<YYYY-MM-DD>/shorts/short_NN.mp4`, and `stages_base`
(`paths.gms_intermediate_dir`) is a staging directory on a fast local disk when the output sits on a
slow mount (by default under the cache home), so an unstable output mount cannot degrade a run.

That cache is **not** durable: a cache can be swept at any time, while a shorts ladder schedules out
to roughly **66 days**. A publish plan recording cache paths could therefore lose its media long
before its last delivery fires.

So Phase D copies each rendered short into the run's **output** dir
(`_persist_shorts_to_output` → `<output_base>/shorts/lf_NN/short_NN.mp4`) and the publish plan
records *that* path. Consequences worth knowing:

- The media sits beside the plan and its sidecar, on the drive you keep. Nothing the
  publish chain depends on lives in a cache any more.
- The copy is **unconditional**, not gated on a `first_party` route. A conditional artifact would
  make a run's output differ by config, and the Gate C/D hash covers that output.
- Only the finished shorts are copied (~37 MB each), never the multi-GB day masters — those stay in
  the cache and are still swept on the normal schedule.
- The copy is idempotent by size, so a resumed run does not rewrite gigabytes, and a truncated
  earlier copy is replaced rather than trusted.
- It never raises. A missing rendered short is logged and skipped: the CSV path does not need these
  copies, and a shorts shortfall already hard-fails at its own upstream gate.

## What a platform pause does and does not stop

`queue.pause_platform` stops **future pickup**: `dispatch._select_due` never selects a paused
platform's row, so the ordinary drain leaves it alone. `dispatch.reconcile_interrupted_attempts`
and `dispatch.reconcile_awaiting_manual` also check the pause, but each honors it differently,
because "stop touching this platform" means something different to each:

- `reconcile_interrupted_attempts` still **releases a stale lease** on a paused platform's row —
  that is bookkeeping about a worker that is provably dead, not a platform-facing action, and
  skipping it would leave the row wedged on a dead lease forever. What it will NOT do while
  paused is charge the attempt or drive the row toward `failed_permanent`: either would decide
  the delivery's outcome against a platform an operator deliberately told the system to leave
  alone, and a crash-looping worker must not be able to burn a paused platform's whole retry
  budget unattended.
- `reconcile_awaiting_manual` skips a paused platform's rows **entirely**, before it ever calls
  `poll`. Every action it takes — polling the platform's API, transitioning to `published` or
  `failed_permanent` — is itself a platform-facing decision, so unlike the lease case there is no
  bookkeeping-only half to salvage.

A row already leased or in flight when the pause takes effect is never touched by any of the
three paths: the platform call it may be mid-flight on cannot be un-sent.

## Standalone assets

A route-story film or other Waypost product with no drive run behind it has no
`pipeline_state.json`, so it skips `enqueue-plan` for its own door: `gomoveshift enqueue-asset
<video> --platform ... --at ... --title ... --confirm <hash>` (`cli/handlers/enqueue_asset.py`,
builder in `publishing/asset_plan.py`).

**The review hash is this door's Gate C/D equivalent.** `asset_plan.review_hash(title,
description, tags, sha256)` covers the four fields a human reviews. A dry run (no `--apply`)
prints that hash and writes nothing; `--apply --confirm <hash>` proceeds only when the confirm
value matches the hash recomputed from the CURRENT content, else refuses with "content changed
since review" (exit 2).

**Privacy is fail-closed, never self-attested** -- no `--attest-privacy-verified` flag exists.
`--receipt <film.receipt.json>` reads the film's own `privacy` block as-is; without it, every
required class (`dispatch.PRIVACY_REQUIRED_CLASSES`) is written `inconclusive`, holding the
delivery like any other unattested one.

**The bundle is a normal sidecar directory** -- `<project>_publish_plan.json` plus an attestation
file matching `dispatch.PRIVACY_ATTESTATION_GLOB`, written into `--bundle-dir` (default: next to
the video) so the existing drain serves it unchanged. There is no rollback
publisher (Metricool is retired); it is then spooled via
`enqueue_from_plan_sidecar(conn, sidecar_path, shadow=...)`.

The console's Publishing tab has a matching "Enqueue a standalone asset" form
(`POST /publishing/enqueue-asset`), containment-checked like `/publishing/enqueue`.
