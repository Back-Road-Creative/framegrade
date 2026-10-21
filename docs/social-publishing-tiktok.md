# TikTok Direct Post adapter

`gomoveshift/publishing/tiktok_adapter.py`. Design rules this adapter follows:

- **Timing:** TikTok has no scheduling field. The delivery queue submits at the
  due instant; the queue firing at `scheduled_at` is the scheduler.
- **Public routing:** a public post is attempted only when the audit gate
  (`publishing_tiktok_audit_approved`) is set and the requested privacy level
  appears in the account's `creator_info`; otherwise the adapter refuses
  rather than posting at a lower privacy level than asked.
- **Routing:** `DriveConfig.publishing_route_tiktok` is `first_party` or `off` and
  selects the adapter; the resolved mode (see "Pack mode" below) determines whether
  it calls a publishing API, so `first_party` alone does not establish automated
  TikTok publication.

## Scope of this module

`query_creator_info` returns the response's `data` object, NOT the
`{"data": ..., "error": ...}` envelope. It also raises when `error.code` is
anything but `"ok"` — TikTok answers auth and permission failures with HTTP
200 and an error code in the body, so `raise_for_status()` does not see them.
Returning the envelope would be a defect: every caller
indexes `creator_info["privacy_level_options"]`, which lives inside `data`, so
the offered list read as `None` and the routing gate rejected *every* privacy
level — `SELF_ONLY` included — as "not offered". A test must therefore feed the
gate the real call's return shape, not a hand-unwrapped dict, so the seam between
the call and its caller is exercised.

Implemented: `creator_info` query, the public-routing gate, chunk-plan
arithmetic, upload-host allowlisting, Direct Post init (`init_direct_post`),
the chunked FILE_UPLOAD transfer (`upload_video_chunks`), the
publish-status poll loop (`poll_publish_status`), and delivery confirmation
(`confirm_publish` — no privacy readback; see below).

**Shipped:** credential loading and rotated-refresh-token persistence in
`tiktok_auth.py` (`load_tiktok_credentials`, `get_valid_tiktok_credentials`);
the bounded publish-status poll loop (a timeout preserves `publish_id`) and
delivery confirmation in `tiktok_adapter.py` (`poll_publish_status`,
`confirm_publish`).

**Refresh persistence re-encrypt:** `get_valid_tiktok_credentials`
writes the rotated token to a random `mkstemp` name before handing it to
`sops -e -i`. The SOPS config's creation rule matches
filenames by regex (`...\.json$`), which a random `*.tmp` name never satisfies
— every refresh was `sops`-failing on "config file not found, or has no
creation rules" **after** TikTok had already issued (and possibly rotated) a
new token, so the encrypt failure discarded it: the credential could go
permanently unusable. Fixed by reading `.sops.yaml` fresh on every refresh and
passing the resolved `age` recipients explicitly. **Note:**
`--age` alone does not bypass creation-rule matching in SOPS 3.11. A refresh
still failed against a discovered config that rejected the temporary name.
With explicit recipients, encryption now uses `--config /dev/null --age
<recips> -e -i <tmp>`; the existing ciphertext's recipients take precedence
over config-derived recipients. A real SOPS/age regression test exercises
matching and unrelated creation rules, encrypted persistence and decryption
with throwaway keys, without platform requests. Recipients are re-read each call, never
hardcoded, so a rekey doesn't silently rot this path. `meta_auth.py` only
reads credentials (no refresh/persist path) so it isn't exposed to this
failure mode.

Queue reconciliation uses the persisted `publish_id` for API uploads;
pack rows have no upload ID and use the separate read-only video-list matcher
described below. Neither a failed local receipt nor an unmatched caption
authorizes a duplicate upload.

**Settled:** the status payload does NOT name a `privacy_level`
field, so delivery confirmation cannot read the audience back.

The audit gate ships with everything else in this file, never separately.

## Choosing the requested audience

`DriveConfig.publishing_tiktok_privacy_level` names the audience every short
in a plan asks for. It defaults to `SELF_ONLY`: the first run after the app
audit clears should be a trial, and a trial that publishes to everyone is not a
trial. Set it to `PUBLIC_TO_EVERYONE` only deliberately, once a private post
has been checked on the account. Values are validated against
`models.TIKTOK_PRIVACY_LEVELS` at config-construction time and again in
`build_publish_plan` — the plausible typo is `PUBLIC` for
`PUBLIC_TO_EVERYONE`. Because there is no readback (below), a wrong value is
not recoverable after the post lands.

## The public-routing gate

`assert_public_routing_allowed(...)` is the ENFORCEMENT POINT — a raising
guard, not a boolean a caller could forget to check; the deferred submit
path MUST call it before initializing a Direct Post. Every requested
`privacy_level` requires BOTH `audit_approved=True` (explicit
`DriveConfig.publishing_tiktok_audit_approved` flag, set only post-app-audit;
default `False` raises naming the flag and the live error code,
`unaudited_client_can_only_post_to_private_accounts`) AND the requested
`privacy_level` appearing in the live `creator_info` response's
`privacy_level_options` — neither alone is sufficient. An unaudited
client's Direct Post init is refused with HTTP 403 and
`unaudited_client_can_only_post_to_private_accounts` even for `SELF_ONLY` —
the constraint is on the ACCOUNT's public/private state, not the post's
audience, so the guard refuses every privacy_level while unaudited, not just
`PUBLIC_TO_EVERYONE`. Do not infer account state from `privacy_level_options`
— an unaudited sandbox client's `creator_info` may still offer
`PUBLIC_TO_EVERYONE`, so that list is a signal, never a
documented contract. There is no post-publish readback to fall back on (see
"Publish-status poll loop" below), so this pre-submit guard is the ONLY
audience control. `user/info/`'s null username means identity records use
`creator_info.creator_username`.

## TikTok uploads an H.264 rendition, never the HEVC master

`publishing/tiktok_rendition.py` transcodes the short to H.264 before the
upload. YouTube and Facebook keep the master, because they transcode
server-side; Instagram gets its own rendition only when the master is over
its cap (next section). The codec comes off the delivery's own `MediaFingerprint`, so
nothing probes the file, and an already-H.264 master is uploaded untouched.

**Why:** an HEVC short can upload cleanly (`status/fetch` → `SEND_TO_USER_INBOX`) while
its inbox notification will not open — tapping does nothing. The same clip as H.264 opens the composer
and publishes (`PUBLISH_COMPLETE`). Nothing rejects
the HEVC file: the Media Transfer Guide lists H.265 as supported and the limits
are 4 GB / 10 min / 4096 px, all far off. But H.264 is what that guide
*recommends*, and the app must pull the file down before the composer renders.

The rendition changes codec, pixel format, `+faststart` and size at
once, so the comparison does not isolate a cause; it establishes that the rendition
publishes where the master did not.

The failure shape is the point. The delivery looked perfect at every layer we
can see — accepted, delivered, parked awaiting a human — while being impossible
to publish. That is why this is fixed by construction rather than documented as
a caveat.

## Instagram gets a rendition when the master is over 25 Mbps

The Instagram reels reader (`_read_instagram_media`, `cli/handlers/dispatch_queue.py`)
verifies the master, then calls `ensure_instagram_rendition`. Meta caps Reels at
25 Mbps VBR. A master whose AVERAGE bitrate (fingerprint size x 8 / duration; ffprobe
only when that is unknown) is over 24 Mbps, or unknown, is uploaded as a sibling
`.instagram-h264.mp4`: 8-bit yuv420p H.264, `-b:v 16M -maxrate 20M -bufsize 40M`,
`+faststart`. At or under the cap the master goes up untouched.

**Why:** Instagram's publish latency tracks the master's
bitrate: a master over the cap can take hours to publish, or sit
`IN_PROGRESS` and then fail permanently ("Media ID is not available").
Facebook and YouTube take the same masters without delay. The cap is on the
average; masters are VBR with a 30 Mbps maxrate, so peaks can still exceed 25.

## Chunked upload contract

`plan_chunks(video_size, chunk_size=None)`: default chunk size is
`min(video_size, 64 MB)`; `total_chunk_count = floor(video_size /
chunk_size)`, a trailing remainder folds into the final chunk when it still
fits the 128 MB final-chunk ceiling, else becomes an extra chunk. A file
below 64 MB uploads as one chunk (no 5 MB floor — nothing to split). Hard
limits: 1–1000 chunks, video ≤ 4 GB; either violation raises
`TikTokAdapterError`.

**A file over 64 MB must upload in at least two chunks.** The Media Transfer
Guide says so, and `floor(video_size / 64 MiB)` is `1` for everything between
64 MiB and 128 MiB — so a naive default would propose a single chunk for
a file the API requires to be split. Every short in a typical project lands
in that window. Init answers:

    HTTP 400 invalid_params "The chunk size is invalid"

`plan_chunks` now halves such a file (both halves stay under the ceiling,
since the file is under 128 MiB), and refuses a caller-supplied `chunk_size`
that would yield one chunk — the invalid plan is unrepresentable rather than
rejected at init.

The sizes are **binary**, as single-variable
probes against the inbox init endpoint show (illustrative sizes):

| `video_size` | `chunk_size` | count | result |
|---|---|---|---|
| 97,000,000 | 67,108,864 (64 MiB) | 1 | 400 `invalid_params` |
| 97,000,000 | 64,000,000 | 1 | 400 `invalid_params` |
| 97,000,000 | 48,500,000 | 2 | 200, `publish_id` |
| 66,000,000 | 66,000,000 | 1 | 200 |
| 60,000,000 | 60,000,000 | 1 | 200 |

The 66,000,000-byte whole-file chunk being accepted is what settles it: that
is over decimal 64,000,000 and under 64 MiB, so the ceiling is `64 * 1024 *
1024`. Do not "fix" the constants to decimal.

`validate_upload_host(url)` refuses anything but `https://` to
a host in `ALLOWED_UPLOAD_HOSTS` — the host named
in an init response is untrusted input. No function here logs or raises
with token material.

## Error surfacing

Every HTTP call site (`query_creator_info`, `init_direct_post`,
`upload_video_chunks`, `poll_publish_status`) routes through the shared
`_tiktok_call` helper. A transport failure (timeout, DNS) or an undecodable
body raises with a fixed message naming only which call failed — there is no
response to quote. An **HTTP >= 400** raises with the status and TikTok's own
`error.code` / `error.message` / `log_id`: they are what tells a repeated init failure (for example a one-chunk plan for a file that must
be split) apart from a generic `Inbox upload init request failed`. A decoded body whose `error.code` isn't `"ok"`/absent (TikTok
answers many failures with HTTP 200) raises with TikTok's own `error.code`,
`error.message`, and `log_id` folded in — `log_id` is what TikTok support
asks for. The access token, if given to the helper, is scrubbed from that
echoed text before it's raised, so an API response that happened to echo it
back still never leaks it.

## Init and chunked transfer

`init_direct_post(access_token, *, title, privacy_level, audit_approved,
creator_info, video_size, chunk_size=None, ...)` calls
`assert_public_routing_allowed` FIRST, before any HTTP request, so an
unaudited PUBLIC request never reaches `POST .../video/init/`. Builds
`post_info`/`source_info` from `plan_chunks`, validates the returned
`upload_url` against the allowlist, and returns `{"publish_id",
"upload_url", "chunk_plan"}`.

`upload_video_chunks(upload_url, chunk_plan, chunk_reader)` PUTs each chunk
**sequentially, never in parallel**, with `Content-Range`/`Content-Length`
from `chunk_plan` (the final chunk may exceed `chunk_size`, up to the
128 MB ceiling). Re-validates the host itself before sending any bytes.
`chunk_reader(index)` supplies each chunk's bytes.

**Disclosure flags, all in `post_info`:** `is_aigc` defaults `False` — GMS
shorts are real captured driving footage, not AI-generated content; `True`
puts a "Creator labeled as AI-generated" badge on the post, so never flip
it without a production-process change. `brand_content_toggle` and
`brand_organic_toggle` both default `False`; whether GMS should set
`brand_organic_toggle` is a per-account decision — do not guess
it per post.

**`upload_url` is valid for one hour.** A transfer that outlives it fails
as an ordinary `TikTokAdapterError` — a distinct, retryable-by-re-init
condition: call `init_direct_post` again for a fresh `publish_id`/
`upload_url` rather than resubmitting chunks against the expired one.

## Publish-status poll loop and delivery confirmation

`poll_publish_status(access_token, publish_id, *, max_attempts=10, sleep=time.sleep)`
polls `POST .../status/fetch/` until `status` is terminal (`PUBLISH_COMPLETE`
or `FAILED`; anything else keeps polling), sleeping with capped exponential
backoff (2s doubling to a 30s cap) between attempts. A timeout is UNKNOWN,
never FAILED: exhausting `max_attempts` raises `TikTokPollTimeoutError`,
preserving `publish_id` for reconciliation. **Never resubmit a fresh Direct
Post on this error** — TikTok may finish publishing after the last poll, and
resubmitting risks a duplicate post.

`confirm_publish(access_token, publish_id, requested_privacy_level, ...)`
polls to terminal and returns a `PublishOutcome` with `delivery_status`
`"published"` or `"failed_permanent"`. Only a `FAILED` terminal status marks
`failed_permanent`; `PUBLISH_COMPLETE` is a successful delivery.

**There is no post-publish privacy readback, and there cannot be one.**
Publishing a `SELF_ONLY` video returns
`{"status": "PUBLISH_COMPLETE"}` — one key. TikTok does not report the
applied `privacy_level` on `status/fetch`, and the Display API endpoints
that might carry it (`video/list`, `video/query`) both answer HTTP 401
`scope_not_authorized` — an app holding only the `user.info.basic` and
`video.publish` scopes gets that answer. Adding `video.list` means an app-config change, re-auth and
another audited scope, and its documented field set does not include
`privacy_level` anyway.

An earlier implementation compared `privacy_level` regardless. Because the
key is never present, it marked EVERY successful publish `failed_permanent`
— the video live on TikTok, the queue claiming permanent failure, and a
human "fixing" it posting a duplicate.

A readback would be a **late gate** in any case: it runs after the post is
already public, so it can only report the harm, never prevent it. The
audience guarantee lives entirely upstream in `assert_public_routing_allowed`,
called before `init_direct_post` — the last point where a wrong audience can
still be refused. `PublishOutcome.requested_privacy_level` records what was
asked for, as an audit trail of the request, never as confirmation of the
result.

## Unaudited-path probe

A TikTok app that has not passed the audit (internal-use apps are not supported) can only
post privately. Two candidates for working without an audit: **inbox/draft upload**
(`INBOX_INIT_URL` — lands in the inbox for manual in-app publish, no
`privacy_level`, `init_direct_post(mode="inbox")` skips
`assert_public_routing_allowed`), and **Direct Post at `SELF_ONLY`** (worth
confirming — "private account" and "SELF_ONLY
visibility" may not be the same constraint).

`tools/tiktok_probe.py` makes ONE real call (and, unless refused, a real
upload) to find out which works — a manual runner, not a pipeline path, no
retries/queue/state, refuses without `--yes-live`. `--mode self-only` forces
`audit_approved=True` as a script-local diagnostic override so the request
reaches TikTok rather than being refused locally first; it never touches
`DriveConfig.publishing_tiktok_audit_approved`.

```
tools/tiktok_probe.py <video-file> --mode inbox --yes-live
tools/tiktok_probe.py <video-file> --mode self-only --yes-live
```

## Pack mode (default)

Until an app passes the TikTok audit, no API path can post publicly, so pack mode is the default. In `"pack"` mode
(`$GMS_TIKTOK_MODE`, the default) the pipeline writes each short into a folder
for manual scheduling in TikTok Studio. Nothing here calls TikTok.

`gomoveshift/publishing/tiktok_pack.py`:

- **Folder:** `$GMS_TIKTOK_PACK_DIR`, default `tiktok-pack` under the data home. If its parent
  is missing, or is a `/mnt/<drive>` directory with nothing mounted on it (a
  drive that is offline), or the drive answers with a dead-mount error (EIO, ENODEV, ENXIO,
  ESTALE, ENOTCONN, EACCES), the pack raises `TransientMediaError` and writes
  nothing. That error is exempt from `max_attempts`, so the row retries
  without ever failing for it, but the drain still increments `attempt_count`
  on each retry (`dispatch.py`, the `TransientMediaError` branch). A full disk
  (ENOSPC) is an ordinary error and counts toward `max_attempts`.
- **File name:** `<YYYY-MM-DD>-<HHMM>-<Place>-<delivery id>.mp4`. The date and
  time are the row's `scheduled_at` in the display timezone (falling back to
  `by_platform['tiktok']`, then `scheduled_at_utc`). `<Place>` is the leading
  phrase of the title (before ` — `, ` - `, `:` or `,`), CamelCased to ASCII;
  `Short<index>` when the title gives nothing.
- **Copy:** the H.264 rendition, written to `<name>.part.<pid>-<uuid>` (its own
  name per copy) then `os.replace`d; skipped when a file of the same size is
  already at that name.
- **Already packed (hand-copied files):** before copying, any `.mp4` in the
  folder with the same size AND SHA-256 as the source counts as packed, under
  whatever name it has. The short is not copied again, no caption block is
  added, and the row is parked `awaiting_manual_publish`. This is what keeps a
  file copied in by hand and already posted or scheduled from being
  posted twice under a new name. Only files of the source's size are read.
- **Re-enqueued short (plan change):** when the identical file has a packer
  name (`...-<delivery id>.mp4`) for another delivery, a plan change cancelled
  the old row and enqueued a new one. No second copy is made (it could get
  scheduled twice); instead `captions.txt` gets one block headed
  `<existing name> -- UPDATED for delivery <new id>` with the new slot and
  caption, written once.
- **Overdue pack row (`queue reslot-overdue-manual`):** a pack row still
  `awaiting_manual_publish` more than 24 h past its slot is moved by the
  operator command (dry run unless `--apply`; never run by the drain). It
  appends one block headed
  `<existing name> -- RESLOTTED for delivery <id> (<new slot YYYY-MM-DD HH:MM tz>)`
  whose Schedule line is the new slot, then moves the row's `scheduled_at` to the
  region's next Waypost shorts time (display timezone of the cadence, later than
  now + 20 minutes, not held by another live TikTok row; the region comes from the
  row's publish-plan record). The new slot is in the header so a second reslot on
  a later day is not skipped as an already-listed name. The `.mp4` keeps its
  original (now stale) name; the owner re-schedules by the block, and
  `mark-published` stays the only close. A row with no pack file, no resolvable
  region, a live lease, or an unmounted drive is refused and nothing is written.
- **`captions.txt`:** UTF-8 with CRLF line endings, one block per file (name,
  schedule line, caption), appended once; a name already listed is skipped.
  The schedule line is `Schedule: Post now (the slot was ...)` when the slot
  has passed or is under 20 minutes away at packing time. The 20 minutes is a
  margin so the operator is not racing TikTok Studio's 15-minute floor; it is not
  TikTok's rule.
- **Row state:** the row ends `awaiting_manual_publish` with the far-future
  `next_attempt_at`, the same park the inbox mode uses, so the console's
  mark-published button works unchanged. Packing never sets
  `platform_upload_id` or `platform_post_id` and never reports a publish;
  parking clears a `platform_upload_id` left by an earlier inbox upload, so
  `reconcile_awaiting_manual` (which polls only rows that carry one) never
  asks TikTok about a stale id. Only the console's mark-published closes a
  pack-parked row.
- **Rows parked before pack mode:** the sweep takes `queued` rows only, so a
  row parked in the app inbox by an earlier inbox-mode run stays parked, with its
  `platform_upload_id`, until it is published or dismissed in the TikTok app inbox and
  marked published in the console. The console's retry action does not requeue it
  (it only clears a backing-off row's `next_attempt_at`); re-queueing a row with a live
  draft would strand the draft.

`dispatch-queue` runs a sweep before the drain in pack mode: every queued
TikTok row scheduled within the next 9 days that the drain itself could pick
up is leased and packed, so the operator sees a week of files ahead rather than
one hour's. Selection is `dispatch._select_due` with the 9-day horizon as the
schedule ceiling, so the pause, shadow, retry-backoff, brand-blackout,
quiet-window and daily-cap rules all apply. The daily cap counts today's
published rows against rows up to 9 days out, which is conservative: it can
only pack fewer files. The sweep also applies the drain's privacy gate (the
same `PrivacyAttestationLoader` instance): a row whose required privacy
classes are not all verified, or whose attestation cannot be read, stays
queued and is reported, never copied to the upload machine. The sweep does nothing while
TikTok is paused, skips a row another worker holds, records a row failure
(releasing its own lease, whatever the row became) and carries on, and is
never fatal to the drain. It prints one `tiktok pack: packed=N
already_packed=N held=N leased_elsewhere=N errors=N dir=...` line, the hold
and error reasons under it, and puts its counts on the summary line a scheduler
keeps (`tail -1`): `tiktok_packed=`, `tiktok_pack_errors=`,
`tiktok_already_packed=` and `tiktok_pack_held=`, each only when non-zero.
The drain handler packs any row the sweep did not reach (no inbox cap, no API
call).

Pack mode needs no TikTok token: `dispatch-queue` builds the TikTok handler
without calling `get_valid_tiktok_credentials`, so an expired or unreadable
TikTok credential cannot stop the pass that drains the other platforms.

The console follows the pack: the phone push for a parked row reads "TikTok:
short ready to schedule" with the slot, the pack folder and the caption (no
inbox steps), and a parked TikTok row offers cancel and retry but not
reschedule. Its slot is fixed in the pack file's name and `captions.txt`, so
`pubqueue.reschedule` refuses it with a pointer to TikTok Studio.

**Reconcile (video.list).** Before the sweep, `dispatch-queue --apply` runs
`tiktok_pack.reconcile_packed`, which closes a parked row once TikTok's
read-only `video.list` shows the post. It considers `awaiting_manual_publish`
TikTok rows with no `platform_upload_id` (inbox rows stay with
`reconcile_awaiting_manual`). A listed video matches a row when its
`create_time` is within 12 hours of the row's slot AND its description,
normalised (NFKC, case-folded, whitespace collapsed), starts with the first
60 characters of the caption's first line (TikTok may trim what was pasted).
Exactly one match that matches no other row and whose id is on no other
delivery publishes the row with `platform_post_id` and the canonical
permalink, built before the transition. No match leaves it parked; two or
more candidates, or one video matching two rows, leave every row involved
parked and print `ambiguous` lines -- it never guesses. Paging stops at the
oldest slot minus the window. It needs the `video.list` scope and real
credentials, fetched only here (the drain keeps its empty token); any failure
prints `  error: tiktok reconcile: skipped (...)` and costs only the reconcile. The
console's mark-published remains the fallback for ambiguous or unmatched rows.

The cron-retained summary carries the resolved `tiktok_mode` and nonzero
`tiktok_reconcile_errors` / `tiktok_reconcile_ambiguous` counts. Pack mode
writes local files; inbox mode submits drafts; only direct mode requests an
API publication and retains the audit/audience gates. Mode resolution is an
explicit argument, then `GMS_TIKTOK_MODE`, then the existing pack default;
API errors never switch modes. `awaiting_manual_publish` is a local state,
not confirmation that a post remains unpublished remotely. Failed or
ambiguous reconciliation leaves it unchanged and visible while the other
platforms continue draining. Resolving expired API access does not establish
app-audit approval or permission to retry a post.

An unknown `$GMS_TIKTOK_MODE` takes only TikTok out of that pass: the other
platforms still drain, a `tiktok: skipped for this pass` line is printed, and
the summary line carries `tiktok_mode_invalid=1`.

## Routing the queue through inbox drafts

`handlers.make_tiktok_handler(..., mode=None)` resolves its mode from the
`mode` kwarg, else `$GMS_TIKTOK_MODE`, else `"pack"` (see "Pack mode" above;
`"inbox"` was the default before pack mode — no unaudited Direct Post path
exists). `"direct"`
stays available, unchanged, for a future audited app once
`publishing_tiktok_audit_approved` can be set.

In `"inbox"` mode the handler skips `query_creator_info` entirely (nothing
in `init_direct_post`'s inbox branch reads it) and calls
`init_direct_post(mode="inbox")`, which itself skips
`assert_public_routing_allowed` — there is no privacy_level to gate. Once
the chunked upload completes, the delivery transitions from `"uploaded"` to
`"awaiting_manual_publish"` instead of ever calling `confirm_publish`: an
inbox upload never auto-publishes, so there is no publish-status to poll.
The delivery also parks its own `next_attempt_at` far in the future so
`dispatch._select_due` never reselects it — nothing automated ever moves it
out of `"awaiting_manual_publish"`; only a human, after publishing the draft
from the TikTok app, can transition it onward via `queue.transition`.

**`"uploaded"` means the bytes are on TikTok, and a resume
can tell that from "init returned an id".** The publish id is still persisted
before the chunked upload -- that is what makes an interrupted upload
recoverable instead of a duplicate post -- but it is now written with the row
left at `"submitting"`, and `"uploaded"` is set only once
`upload_video_chunks` returns. Previously both were one write, so a worker
killed mid-upload left a row indistinguishable from a finished one, and the
resume branch (keyed on the publish id alone) skipped straight past the
upload: in inbox mode it parked the delivery as a draft *awaiting manual
publish* that was not in anyone's app inbox -- nothing reprocesses that status
automatically, a human looking for the draft finds nothing, and it holds one
of the `_MAX_PENDING_INBOX_DRAFTS` slots for good; in direct mode it polled
`confirm_publish` for a publish that could never complete until the attempt
budget ran out. A resume that finds `"submitting"` with a publish id now
raises `PermanentDeliveryError` naming the interrupted upload: re-running
`init_direct_post` would create a SECOND post, so the safe move is to fail
where a person can see it and re-enqueue. A resume that finds `"uploaded"`
behaves exactly as before -- no re-init, straight to the park or the confirm.

The console's Publishing tab renders the
`awaiting_manual_publish` group with a "publish from the TikTok app inbox"
hint and a mark-published button per row. `POST
/publishing/mark-published/{delivery_id}` calls
`console/pubqueue.py::mark_published`, which transitions the row to
`"published"` via `queue.transition` (never raw SQL) -- refusing (409) any
delivery not currently `awaiting_manual_publish`, and 404 for an unknown
id. Once `"published"`, the row is one of `dispatch._select_due`'s own two
terminal statuses, so it is doubly unreachable by drain.

**A parked delivery announces itself.** The console tab only
speaks to whoever opens it, and a draft can sit in the inbox for
hours while its owner searches the TikTok app's *Drafts* folder — where an API
upload can never appear, because drafts are on-device and the upload arrives as
an **Inbox notification**. Two surfaces now carry it without anyone going
looking:

- `dispatch-queue` puts `awaiting_manual_publish=<n>` ON its summary line.
  It has to be that line: a scheduler that logs only the last line of output
  (`tail -1`) would drop a separate line after the summary, and one
  before it would never be logged.
- A daily digest section, "Waiting on a manual publish", can list each parked
  delivery with its slot, saying Inbox rather than Drafts, and stays silent
  when nothing waits.

`notify.check_queue_regressions()` watches `awaiting_manual_publish`, and
`dispatch-queue --apply` runs it after every drain, so a draft parked in a pass is pushed in that
pass. Nothing sets `GMS_CONSOLE_NOTIFY_URL` (a scheduler wrapper is not the place): run
`python3 tools/phone_alerts_setup.py` once, which writes `notify_url` under the config home (read
when the env var is unset), subscribe the phone, then `--test`. The push gives the slot in Eastern
time, the Inbox steps and the caption to paste. The first scan only records what is already
parked.

**`status/fetch` DOES close this loop** — the status is not terminal at
`SEND_TO_USER_INBOX`. Per the Get Post Status reference it advances to
`PUBLISH_COMPLETE` ("the user has clicked on the inbox notification and has
successfully posted the media using TikTok editing flow") and the response then
carries `publicaly_available_post_id` (TikTok's spelling) — an exact id, so
reconciliation needs no matching heuristic. `FAILED` carries `fail_reason`.

Do not conclude otherwise from a poll that keeps answering
`SEND_TO_USER_INBOX`: that means the draft has NOT been published yet, which is
the state of a draft still sitting in the app, not yet posted. The status does advance once the human publishes, so that poll alone is not evidence that it never will.

Matching on the caption is impossible in any case, and that IS structural:
`init_direct_post(mode="inbox")` sends only `source_info` — no `post_info`, no
title — because TikTok takes the caption from the human in the editing flow.
Whatever the published video says is what they typed.

When the `video.list` scope is granted it is still worth having:
`/v2/video/query/` turns the id from `publicaly_available_post_id` into a
`share_url` for the delivery's `permalink`. Note the `share_url` comes back
tagged with `utm_campaign=tt4d_open_api&utm_source=<client key>`; store the
canonical `https://www.tiktok.com/@<username>/video/<id>` instead.

## Reconciling a draft the human published

The parked-delivery notification above relies on the fact that `status/fetch` advances past
`SEND_TO_USER_INBOX`. This is the code that acts on it. A typical answer:

    {'status': 'PUBLISH_COMPLETE',
     'publicaly_available_post_id': [7000000000000000001]}

(`publicaly_` is TikTok's spelling.) `PUBLISH_COMPLETE` lands on the tap; the
post id follows a minute or two later, after moderation.

`dispatch.reconcile_awaiting_manual(conn, poll=..., permalink_for=...)` runs
at the top of every `dispatch-queue --apply` pass, **before** the drain. For
each `awaiting_manual_publish` TikTok row carrying a `platform_upload_id`:

| `status/fetch` | result |
|---|---|
| `PUBLISH_COMPLETE` with a post id | `published`, with `platform_post_id` and `permalink` |
| `PUBLISH_COMPLETE`, no post id yet | left parked — moderation is still running |
| `FAILED` | `failed_permanent`, `error_detail` = `fail_reason` |
| anything else, or a poll that raises | left parked |

Four things are deliberate:

- **It cannot live inside `drain_queue`.** `_select_due` never returns an
  `awaiting_manual_publish` row: it excludes the status itself, not only the
  `next_attempt_at=9999-01-01` park time, since any write that moved that
  time un-parked the draft. That guard is what
  stops the drain re-uploading a draft or recording it as published;
  reconcile reads the rows directly instead of weakening it.
- **The permalink is built before the transition and passed into it.**
  `queue.transition` refuses every write once a row is terminal, so a
  permalink fetched after the publish transition is unstorable. That is
  exactly the Facebook defect where a cosmetic post-publish
  call classified a successful publish as `failed_permanent` and only the
  terminal-status guard saved the row.
- **The poll runs with `max_attempts=1`.** `poll_publish_status`'s default is
  10 attempts with backoff — about three minutes of sleeping — and it
  *raises* when nothing terminal arrives, which is the normal answer for a
  draft still in the inbox. One request; a timeout means "still parked".
- **The permalink is the canonical URL**, built from the creator username, not
  the `share_url` that `video/list` returns: that one carries
  `utm_campaign=tt4d_open_api&utm_source=<client key>`, and the client key
  does not belong in a user-visible column.

Best effort throughout: a row that fails to reconcile is recorded and the
pass continues, and credentials that will not decrypt cost the parked rows
only — never the shorts that are due today. The console's mark-published
button stays, for the case where TikTok has not caught up yet.

## The pending-draft throttle

An inbox draft is a *reservation*, not a free artefact. TikTok caps how many
uploaded-but-unpublished drafts an account may hold and refuses further inits
with HTTP 400 `spam_risk_too_many_pending_share` once past it. Once enough drafts are pending, every subsequent upload on
the account is blocked. The ceiling is undocumented.

Two guards, deliberately in that order:

- **Upstream (`handlers.make_tiktok_handler`, inbox mode only).** Counts
  `awaiting_manual_publish` TikTok rows and raises `ThrottledDeliveryError`
  *before* `init_direct_post` once ≥ `_MAX_PENDING_INBOX_DRAFTS` (3). Validating
  before the irreversible act, not after — an upload that must not happen is
  cheaper never made than diagnosed from a 400. Direct Post is exempt: it
  creates no draft, so pending drafts cannot throttle it.
- **Backstop (`tiktok_adapter`).** `THROTTLE_ERROR_CODES` maps TikTok's own
  refusal to `TikTokThrottledError` on both failure paths (HTTP ≥ 400 and the
  HTTP 200 + `error.code` shape), and `handlers` translates it to the same
  `ThrottledDeliveryError`. This catches the case where our count is wrong —
  a draft dismissed in the app leaves the row parked but the reservation gone.

`drain_queue` treats `ThrottledDeliveryError` like `TransientMediaError`:
retried, lease released, **exempt from `max_attempts`**, but on a fixed
`_THROTTLE_BACKOFF_SECONDS` (6h) instead of the backoff curve. The curve caps
at an hour, which is right for API flakiness and far too eager for a state
only a person can change; and the 8-attempt bound would declare
`failed_permanent` a short whose media and credentials are both fine.

`dispatch-queue` appends `throttled=<n>` to its summary line when non-zero
(a scheduler logging `tail -1` records it), and `DrainResult.throttled` counts it as a subset
of `retried`.

**Clearing it is manual, by construction.** Publish or dismiss the parked
drafts in the TikTok app inbox. Do not re-queue an `awaiting_manual_publish` row
while its draft is live for the same reason: it strands the draft (nulling
`platform_upload_id` leaves the reconciler nothing to poll) and sets up a duplicate
post when someone later taps the orphan.

## Scheduling and captions — what the API cannot do

Per the Content Posting API reference:

- **There is no scheduling.** No `scheduled_publish_time`, `publish_at` or
  equivalent on any endpoint. TikTok reserves scheduling for its own tools.
  The delivery queue firing at `scheduled_at` IS the scheduler, and that is
  the whole of the design — nothing is missing here.
- **The inbox endpoint takes no caption.** Its parameter table is
  `source_info` only. The caption is whatever the human types in the editing
  flow, so a post can go live with
  an empty one. The parameter table is the basis for this.
- **Direct Post takes a full caption** in `post_info.title` (2200 UTF-16
  runes, hashtags included) plus `disable_comment`/`disable_duet`/
  `disable_stitch`, `video_cover_timestamp_ms` and the brand toggles — and is
  gated behind an app audit (internal-use apps are not
  supported; see "Unaudited-path probe").

**Fallback when a specific slot has to land while the inbox is throttled:**
TikTok Studio on desktop web schedules 15 minutes to 10 days ahead with
caption, hashtags and cover image, on a Creator or Business account, with no
audit and no API. Fully manual, so not a pipeline path — but it is the answer
for one post that cannot wait, and is written down here so nobody re-derives
it under pressure.

## Reconnecting

The credential record is `tiktok_publishing.json.enc`, beside the SEO credentials
directory (the parent of `seo.credentials_dir`; a brand with its own
`brands/<brand>/creds` directory keeps it at `brands/<brand>/`). It is SOPS-encrypted. The loader reads these fields: `client_key`, `client_secret`,
`access_token`, `refresh_token`, `access_expires_at` and `refresh_expires_at`
(ISO-8601 timestamps). The access token is refreshed automatically and a rotated
refresh token is written back; once the refresh token itself has expired the record
must be re-created from a fresh authorization. No setup helper ships in this tree,
so build the record by hand with those fields and encrypt it in place.
