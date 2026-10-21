# Social publishing status — ground truth

This is the authoritative "what actually works today" table for the
first-party `gomoveshift/publishing/` stack.
Cells below cite the source file they rest on, and cross-document references
name a section rather than a line. Its companion command is
`gomoveshift publishing-status` (`gomoveshift/cli/handlers/pub_status.py`),
which prints a live, no-decrypt snapshot of credential-file presence and delivery
queue row counts.

**Routing: `first_party` or `off`; the built-in default is `off`.** Metricool is
no longer a publishing route. `DriveConfig.publishing_route_{youtube,instagram,facebook,tiktok}`
all default `"off"` (`gomoveshift/config.py`), so a deployment's config opts each
platform in. A route is either
`"first_party"` (the adapters below receive traffic,
`gomoveshift/cli/handlers/dispatch_queue.py`, `_routed`) or `"off"` (ships
nothing). A `"metricool"` value now refuses at config construction. A route
flips two ways: a `config.yaml` value (wins outright), or the per-platform
`GMS_PUBLISHING_ROUTE_{YOUTUBE,INSTAGRAM,FACEBOOK,TIKTOK}` env var, which only
supplies the *default* — set from the console Settings tab so a brand can
flip a platform from a phone without editing config.yaml.

## Publication refuses media whose privacy classes were not verified

All four privacy classes — **people, faces, plates, vehicle occupants** — are
required for a publishable export. The day-master encoder writes one
attestation per day into the run's output dir
(`day_<date>_privacy_attestation.json`, `pipeline/day_master_encoder.py`,
schema `gms.privacy_attestation/v1`), recording per class one of three
states with the reason:

| state | meaning |
| --- | --- |
| `verified` | the detector model resolved and ran for this class |
| `failed` | a detector ran and the class is not covered |
| `inconclusive` | **nothing ran** — the model would not resolve, or the class is switched off by config |

`inconclusive` is deliberately not `failed`: "no detector ran" and "a detector
ran and found nothing" are different claims, and neither is a pass. A class
turned off by config (`blur_person_enabled=false`, `blur_vehicle_cabin_frac=0`)
is `inconclusive`, never absent — a config flag cannot talk a required class
away.

The gate is at **publication, not the encode**: a render costs GPU hours, a
missing model file is fixed in seconds, so the encode proceeds and the master
is retained. `dispatch.drain_queue` consults the attestation immediately after
the handler check and **before** `acquire_lease` — the same "check before
locking" rung `skipped_no_handler` sits on — and holds any delivery whose
required classes are not all `verified` (`publishing/dispatch.py`,
`privacy_hold_reason`). A hold is **not** a failure: no status transition, no
attempt bump, no lease, so restoring the model and re-running the day
publishes the very same delivery row.

**Linkage** — `deliveries.sidecar_path` (the row's own publish-plan sidecar)
locates the run's output dir, which is where Phase D writes publish plans and
where the encoder writes attestations. `PrivacyAttestationLoader`
(`cli/handlers/dispatch_queue.py`) merges a run's days worst-state-wins: a
delivery names a project, not a day, so one unverified day holds the run.
Fail-closed at every edge — an absent, unreadable or malformed attestation is
a hold, not a pass.

The dispatch summary carries `held_privacy_unverified=<n> Held: privacy
classes unverified` **on the existing summary line** (only when non-zero), so
a scheduler that logs the last output line (`tail -1`) keeps it. Per-delivery `error:` lines print
ABOVE the summary, so the last line is always the summary.

## The plan's data model carries a format, not an assumed short

`gomoveshift/publishing/models.py`'s `PublishRecord` (alias `ShortPublishRecord`,
kept for `dispatch.py`/`handlers.py`/every adapter) now carries an explicit
`format` field (`KNOWN_FORMATS`: `short`/`photo`/`carousel`/`long_form`) and
an `additional_media` tuple for a carousel's extra items — `media` stays the
one item every other format takes. `format=None` means "not stated" and is
never treated as `short` by anything that reads it; `plan_builder.py` (the
only producer today) sets `format="short"` explicitly on every record it
emits. `plan_io.load_publish_plan` applies one deliberate exception for
sidecar files: a file whose `format` key is missing ENTIRELY predates this
field and this package only ever produced shorts before it existed, so that
absence loads as `short`; a file whose `format` key is present but `null`
loads as `None`, same as a fresh in-memory record.

`gomoveshift/publishing/photo_adapter.py` (`PhotoCarouselAdapter`) handles
photo and carousel records: a `format=photo`/`format=carousel` record now has a real
publish path — create container(s), publish, record the platform's own
post id as the receipt, or fail loudly on no id. A carousel publish is
atomic from the operator's point of view: any item-container failure
raises `PartialCarouselError` naming the failed index and how many items
already succeeded, and that failure is **always** permanent, never
auto-retried (Instagram's item containers have no idempotency key, so a
retry would recreate every item and risk a duplicate post). `handlers.py`
wires it in via `build_handlers(..., instagram_photo=…, facebook_photo=…)`,
routing the existing `"instagram"`/`"facebook"` queue platform keys on the
loaded record's `format` — the video handlers are unchanged when no photo
tuple is supplied. `PhotoGraphClient` is a narrow, self-defined interface
(create-container/publish shape modeled on Meta's documented Content
Publishing API, matching how `instagram_adapter.py`/`facebook_adapter.py`
already handle a single Reel's container) — **not exercised against a
live account** in this repo.

`gomoveshift/publishing/photo_graph_client.py` (`MetaPhotoGraphClient`)
is a concrete implementation of that interface, and
`cli/handlers/dispatch_queue.py::_build_routed_handlers` wires it for
any `instagram`/`facebook` platform already routed `first_party` — a
`format=photo`/`format=carousel` delivery on such a platform reaches this
client instead of `skipped_no_handler`. Only two of its calls reuse a shape
`instagram_adapter.py` already uses:
`publish_container` (`media_publish` with a `creation_id`) and the
publishing-limit read (`content_publishing_limit`, live-read every
publish, never cached, shared with any other tool posting to the account — same
discipline as `InstagramReelsAdapter._check_publishing_limit`).
`create_carousel_parent_container` (`media`, `media_type=CAROUSEL`,
`children=<item ids>`) mirrors Meta's *documented* shape but is itself
unexercised. `create_single_container`/`create_carousel_item_container`
(`photo_image_url.py`) take an `image_url` instead of raw
bytes and POST `media` with `image_url=`/`is_carousel_item=true` — documentation
inference, unexercised against a live account. The URL is
derived from what `pipeline/photos/selector.py` already knows (the Hugo
site's own `baseURL`, read from its `hugo.toml`, + journey slug +
filename — the photos pipeline reads photos FROM the site, so once it's
deployed this is exactly where a photo is served), with a Drive
`uc?export=download&id=` fallback (shared with
`csv_generator._drive_link_for_metricool`, not duplicated) when no site
baseURL is available. Both container calls verify the URL is actually
reachable with a real GET — distinguishing not-yet-deployed (`404`,
naming the site deploy step) from a wrong content-type from an
outright-unreachable host — **before** creating any Graph container,
refusing via `PhotoImageUrlUnverifiedError` rather than guessing at a
broken link.

`photo_adapter.PhotoGraphClient`'s contract now matches this `image_url:
str` shape end to end: `photo_adapter.PhotoCarouselAdapter
.publish` calls it through a `media_image_url_loader`, and
`cli/handlers/dispatch_queue.py::_build_routed_handlers` wires one —
`photo_image_url.image_url_for_media_path` — bound to the routed brand's
own `config.photos.resolve_site_root()`, never
`photos.selector.default_site_root()`'s import-time fallback. A record's
`MediaFingerprint.path` that isn't under the site's served
`static/images/journeys/<slug>/<filename>` layout (no builder in this
codebase constructs a `format=photo`/`format=carousel` `PublishRecord`
yet, so what a real one carries is unproven; a resized social copy under
`photos.resolve_resized_base()` is NOT under this layout and the site
does not serve it) is refused as a named, permanent failure rather than
guessed at. Wiring `pipeline/photos/orchestrator.py`'s CSV export path
onto this adapter, and a plan builder that produces
`format=photo`/`format=carousel` records, are not implemented.

## Photos are exported for scheduling, never published here

`photos publish` / `photos batch` end at a Metricool import CSV.
`pipeline/photos/orchestrator.py:_publish_batch` runs resolve -> schedule ->
resize -> Drive upload -> CSV -> ledger booking and calls **no platform API**;
nothing in that path returns a post ID or any other receipt. Metricool posts
these on its own schedule, so between the CSV and a live post there is a step
this codebase cannot observe.

Both CLI routes therefore render `describe_scheduler_export()`
(`pipeline/photos/orchestrator.py`), which states the export and the absent
receipt in one place.

The persisted `batch.published` field is kept for on-disk compatibility and
means "this batch is spent" — holds consumed, slots booked. It is not a claim
that anything went live. Do not read it as the `result.published` counter in
`publishing/dispatch.py`, which only increments after a registered
first-party handler really posts.

| Platform | Adapter | Credential source | Live-exercised? | Scopes / app-review | Quota notes | Gap to go-live |
|---|---|---|---|---|---|---|
| YouTube (long-form) | `gomoveshift/pipeline/drive_phases/youtube_uploader.py` `YouTubeUploader`, called directly from `external_systems_coordinator._upload_lfs_to_youtube` | SOPS `youtube_upload_token.json.enc` at `<seo.credentials_dir>/../` (`gomoveshift/pipeline/_yt_creds.load_youtube_credentials`), falls back to Fernet `BrandCredentials` (same function) | **Yes** — the long-form uploader (`--publish DAY` implies `--youtube-upload`) | `YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"` (`youtube_uploader.py`), overridable by the credential's own `scopes` field | `YouTubeQuotaError` retried by the day-budget gate `_defer_if_over_budget` (`gomoveshift/publishing/youtube_adapter.py`, config field `youtube_video_insert_daily_uploads`) | None for long-form |
| YouTube (shorts, via the queue adapter) | `gomoveshift/publishing/youtube_adapter.py` `YouTubeShortsAdapter` (wraps the same `YouTubeUploader`) | same as above | **No** — only reachable when `publishing_route_youtube="first_party"` (the route defaults to `off`, `config.py`); exercised only by `tests/test_publishing_youtube_adapter.py` against a fake uploader | same as above | same day-budget gate, plus the channel-scan fallback in `_find_existing_upload` costs `playlistItems.list`/`videos.list` units, never `videos.insert` quota (`youtube_adapter._find_existing_upload`) | Not exercised against the real API end-to-end through the queue/dispatch path |
| Instagram Reels | `gomoveshift/publishing/instagram_adapter.py` `InstagramReelsAdapter` / `InstagramShortsAdapter` | SOPS `meta_publishing.json.enc` at `<seo.credentials_dir>/../` (`gomoveshift/publishing/meta_auth.load_meta_credentials`) | **Partial** — a read-only reachability probe against the real Graph API is documented (`docs/social-publishing-meta.md`, "Meta app mode"): `GET /{page_id}`, `GET /{ig_user_id}`, `GET /{ig_user_id}/content_publishing_limit` return 200 with a quota reading. The probe makes no `media`/`media_publish` call (an actual Reel post); all publish-path tests are fixture-only (`tests/test_publishing_instagram_adapter.py`) | The stored `scopes` field is a write-time snapshot; `debug_token` is the live source and shows `instagram_content_publish` + `pages_manage_posts` (`docs/social-publishing-meta.md`, "Meta app mode"). Whether the app is in Development or Live mode — and therefore whether App Review is required before the first real publish — **cannot be read from Graph**; a clean 200 on every read-only GET is what a still-Development app also produces as long as the token holder is an admin/developer/tester | 100 posts/rolling-24h **shared with any other tool publishing to the same account** — read fresh via `content_publishing_limit` immediately before every `media_publish`, never cached (`docs/social-publishing-meta.md`, "Publishing-limit quota"; `instagram_adapter.py`) | App Review requirement depends on the app's mode (see above); open question on whether `check_container_status`'s `media_id` equals the id `media_publish` actually mints (`docs/social-publishing-meta.md`, "Permalink retrieval") |
| Facebook Reels | `gomoveshift/publishing/facebook_adapter.py` `FacebookReelsAdapter` / `FacebookShortsAdapter` | same `meta_publishing.json.enc` as Instagram (shared loader, `meta_auth.py`) | **Yes** — a reel published through this adapter was read back; the readback confirms `publishing_phase.publish_status=published`. That readback is also what proved the fixtures wrong (`video_status` is `ready`, `permalink_url` is relative) — see `docs/social-publishing-facebook.md` | Same Meta app/token as Instagram; Page-surface scope is `pages_manage_posts` per the same `debug_token` read (`docs/social-publishing-meta.md`) | No Facebook-specific quota documented in-repo; not the same rolling pool as Instagram's `content_publishing_limit` (that endpoint is IG-account-scoped) | No reachability probe; the App Review requirement is the same as Instagram's |
| TikTok Direct Post | `gomoveshift/publishing/tiktok_adapter.py` (free-function toolkit) + `tiktok_auth.py` | SOPS `tiktok_publishing.json.enc` at `<seo.credentials_dir>/../` (`gomoveshift/publishing/tiktok_auth.load_tiktok_credentials`) | **Partial** — several individual calls were confirmed against TikTok's live sandbox: `creator_info` envelope shape (`docs/social-publishing-tiktok.md`, "Scope of this module"), the `unaudited_client_can_only_post_to_private_accounts` 403 on Direct Post init for every privacy level (`docs/social-publishing-tiktok.md`, "The public-routing gate"), and the publish-status payload carrying no `privacy_level` field (`docs/social-publishing-tiktok.md`, "Publish-status poll loop"). No successful end-to-end Direct Post (init → chunked upload → publish) is documented; all of `tests/test_publishing_tiktok_adapter.py` / `test_publishing_tiktok_auth.py` are fixture-only | `unaudited_client_can_only_post_to_private_accounts` means an app that has **not cleared TikTok's content-posting API audit** is refused on every privacy level, not just public ones (`docs/social-publishing-tiktok.md`, "The public-routing gate"); the display scopes needed for any post-publish readback (`video.list`/`video.query`) are separate from `user.info.basic` and `video.publish` | No numeric daily quota documented in-repo; `upload_url` expires after 1 hour (`docs/social-publishing-tiktok.md`, "Chunked upload contract") | **App audit for Direct Post** — an unaudited app cannot post publicly; `DriveConfig.publishing_tiktok_audit_approved` must stay `False` until the app clears the audit (`gomoveshift/config.py`). Without the audit, `handlers.make_tiktok_handler` runs in `pack` mode by default (`inbox` mode is the alternative; `docs/social-publishing-tiktok.md`, "Pack mode" and "Routing the queue through inbox drafts"): inbox mode lands each delivery `awaiting_manual_publish` after the chunked upload and requires a human to publish the draft in the TikTok app, and the console's mark-published action records that |

## Gaps before go-live

1. **TikTok's content-posting audit is a prerequisite for public posting.**
   An unaudited client is refused with a 403
   (`unaudited_client_can_only_post_to_private_accounts`) on every requested
   privacy level, not just `PUBLIC_TO_EVERYONE`
   (`docs/social-publishing-tiktok.md`, "The public-routing gate").
   `publishing_tiktok_audit_approved` must stay `False`, and an unaudited
   app can only post privately, until the audit clears. `tools/tiktok_probe.py`
   is a one-shot, `--yes-live`-gated runner for probing the two unaudited-client
   candidates (inbox/draft upload, Direct Post at `SELF_ONLY`) with one real
   video (`docs/social-publishing-tiktok.md`, "Unaudited-path probe"). The
   queue handler (`handlers.make_tiktok_handler`) also has an inbox/draft mode
   (`$GMS_TIKTOK_MODE=inbox`), so a delivery lands `awaiting_manual_publish`
   after the chunked upload rather than being claimed published; a human
   publishes each draft from the TikTok app. The default is `"pack"` (copy the
   file into the TikTok Studio pack folder for manual scheduling; no API call;
   `docs/social-publishing-tiktok.md`, "Pack mode").
2. **Meta app mode (Development vs. Live) is not readable from Graph**, so
   whether App Review is required before the first real `media_publish`/Reels
   post depends on the app and has to be checked in the Meta developer
   dashboard. A clean 200 on read-only probes is exactly what a
   still-Development app also returns for an admin/developer/tester token
   holder (`docs/social-publishing-meta.md`, "Meta app mode"). Facebook
   additionally has no reachability probe.
3. **Only the Facebook publish path has a documented real-API readback**
   (`publishing_phase.publish_status=published`; the Facebook Reels row above
   and `docs/social-publishing-facebook.md`). **Every other platform's publish
   path (container/media create, chunked upload, `media_publish`/Direct Post
   `init`) is exercised in this repo only by tests against mocked HTTP
   responses** (`tests/test_publishing_{instagram,tiktok,youtube}_adapter.py`);
   Facebook's fixtures are the captured real envelope rather than a guess,
   which is how the `video_status`/`publish_status` bug was found. The
   queue/dispatch/plan_builder chain runs only when a
   `DriveConfig.publishing_route_*` is `"first_party"`: Phase D then emits the
   publish-plan sidecar (`gomoveshift/pipeline/drive_phases/phase_d.py`) and
   `pipeline/auto_enqueue.py` enqueues it. With every route at its default
   of `off`, nothing is produced.

## Restore validation: `publishing-status --validate`

Counting rows proves a backup restored; it does not prove the restored queue
can still do anything. The queue DB, the publish-plan sidecars and the
credential files are three separate things on disk, and a restore can easily
bring back one without the others — the rows look healthy right up until the
first drain pass fails on them.

`gomoveshift publishing-status --validate`
(`gomoveshift/cli/handlers/pub_status.py`, `validate_restored_state`) is the
check to run on a restored instance **before** letting a dispatcher near it.
It reads; it never writes, never opens an adapter, and imports nothing from
`gomoveshift/publishing/dispatch.py`, so running it cannot itself publish
anything. It walks every **non-terminal** delivery — a `published`,
`failed_permanent` or `cancelled` row owes no further work, so a sidecar it
no longer needs is not a restore defect — and names one of:

| kind | what it means | what to do |
|---|---|---|
| `sidecar_path_unset` | the row records no publish-plan sidecar at all | re-enqueue the short from its drive run; the delivery cannot be read back |
| `sidecar_unreadable` | the recorded sidecar is missing or is not a valid publish plan (`plan_io.load_publish_plan` wraps a missing file as `PublishPlanFormatError`) | restore the drive run's output dir alongside the queue, or re-enqueue |
| `short_not_in_sidecar` | the sidecar loads but carries no short matching this row's `(project_name, short_index)` — a *different* run's plan file | point the row at the right sidecar, or re-enqueue |
| `credential_absent` | no credential file on disk for that platform (existence only — nothing is decrypted or read) | restore the SOPS `.enc` file before the first drain |
| `stale_lease` | `lease_owner` set with `lease_expires_at` still in the future: a worker that died with the old instance | wait out the lease or clear it; until then `queue._has_live_foreign_lease` refuses the operator's cancel and reschedule and the row reads as busy |
| `missing_remote` | a YouTube row marked terminal because the channel returns no video for its upload id (post-upload verify, or `queue mark-missing-remote`) | find out why and tell the operator; re-queue only on the operator's per-row decision: `queue recover-terminal --id N --authorized-by ...` dry-runs every gate, and `--apply` writes the recovery row, which resolves the problem. See `docs/social-publishing-queue.md` § `missing_remote` |
| `awaiting_ig_reconcile` | an Instagram row parked on Graph's "Media ID is not available" for more than 24 h: every hourly account scan since has been unable to prove the post is live or absent | check the account by hand; `queue recover-terminal` only applies once the row is terminal |
| `manual_publish_overdue` | a TikTok inbox draft (`awaiting_manual_publish`) whose `scheduled_at` slot passed more than 24 h ago (`MANUAL_PUBLISH_GRACE_HOURS`) and still has not been posted from the app | open the TikTok app's inbox notification and tap Post; the hourly reconcile closes the row. To give a missed slot a new time instead, `queue reslot-overdue-manual` (dry run; `--apply` moves it and, for a pack row, adds a `RESLOTTED` caption block) -- the row then drops off the overdue list because its slot is no longer past; it never closes the row. The operator digest's "Waiting on a manual publish" section prints the overdue count every morning (zero included) and marks each listed row; the row has no expiry and stays named until the reconcile closes it |

Exit status is `1` when any problem is found and `0` when the restore is
coherent, so a restore script can gate on it. Without the flag the command is
the unchanged presence-and-counts report and always exits `0` — a broken
restore must not start failing a status check that never claimed to validate.

The same problems also **gate publishing** (every kind except `stale_lease`,
which on a running box is just work in flight, and `manual_publish_overdue`, a
reminder for the human who taps Post rather than a pipeline defect): `enqueue-plan --apply` and
`dispatch-queue --apply` refuse the affected platform/project pairs. See
`docs/social-publishing-queue.md` § "Validate problems gate enqueue and dispatch".

### `--live-catalog`: published YouTube deliveries vs the live channel

`gomoveshift publishing-status --validate --live-catalog <live_catalog.json>`
(`validate_against_live_catalog`) adds the remote check the restore walk
skips: every **`published`** YouTube delivery is looked up in a catalog
written by `gomoveshift catalog-fetch --out <live_catalog.json>` (read-only
`videos.list`). The status command makes no network call itself; the catalog
file is the evidence. To make absence conclusive rather than an enumeration
miss, fetch with `--known-ids-from-queue` (or `--known-ids` listing the queue's published
ids); `queue mark-missing-remote` then turns each proven-missing row terminal.

| kind | what it means | what to do |
|---|---|---|
| `published_video_missing` | the row's `platform_upload_id` is not in the live catalog — the queue says published, YouTube returns no such video | find out why (deleted, removed by YouTube, re-uploaded under another id) and tell the operator; **never** re-upload or flip the row to re-queue |
| `published_title_drift` | the live title differs from the publish plan's YouTube title | report it; a human decides which title is accurate. Never auto-retitle |
| `plan_stale_retitled` | the live title differs from the plan, but it is the newest `APPLIED` title in a `tools/retitle_apply.py` audit log passed as `--retitle-audit` — our own retitle, the plan lagging behind it | report-only (exit stays 0): run the `gomoveshift queue sync-plan-titles --audit <log> --apply` the line names. Without `--retitle-audit` it reads as drift |

`retitle_apply` now writes each applied title into the plan sidecar in
the same step, so a fresh `plan_stale_retitled` means a retitle made before
that, or a sync that failed (it warns, naming the command).

The same two kinds also gate `enqueue-plan` / `dispatch-queue` when a
`catalog-fetch --known-ids-from-queue` snapshot no older than 24 h sits at
`<storage_path>/live_catalog.latest.json`; older or missing is a named warning
(`docs/social-publishing-queue.md` § "Validate problems gate enqueue and dispatch").

A row whose sidecar is gone has no local title to compare, so it is unknown,
not drift. Detection only: nothing is re-queued, re-uploaded, deleted or
retitled. Long-form title drift is the `uploaded_title_matches_live` column of
`gomoveshift catalog-inventory`.

## Parity: where the Metricool CSV's rows come from vs. `plan_builder`

The Metricool shorts CSV (`gomoveshift/pipeline/csv_generator.py`,
`build_consolidated_shorts_csv`) and the first-party publish plan
(`gomoveshift/publishing/plan_builder.py`, `build_publish_plan`) are
built from the **same upstream Phase D data** — `shorts_by_lf` (per-LF social
posts), `slots_by_short_index` (schedule), `parent_lf_video_id`/
`parent_recording_date`/`parent_publish_at` (per short) — but never both for
the same short: a platform's `publishing_route_<platform>` flag routes it to
exactly one path. When a route is `"first_party"`,
`csv_generator.platforms_for_publishing_routes` forces that platform's CSV
toggle `"FALSE"` so Metricool never re-posts it
(`gomoveshift/pipeline/drive_phases/phase_d.py`), and
`_emit_publish_plan_sidecar` builds a `PublishPlan` for it instead
(`phase_d.py`). For CSV → first-party field parity, the
fields to reconcile are: CSV `Text`/caption vs. `InstagramPayload.caption` /
`FacebookPayload.description` (both derive from the same post caption plus
the "Filmed <date>." suffix — `plan_builder._with_filmed` mirrors
`csv_generator._format_recording_date_pretty`); CSV `First Comment Text`
(parent-LF link) vs. `plan_builder`'s `parent_link` embedded directly in the
YouTube description (`plan_builder.build_publish_plan`); CSV's per-platform boolean
toggles vs. `RouteSelection.selected`; and the CSV's TikTok-only
columns vs. `TikTokPayload.privacy_level` plus the audit-gate fields
(`disclose_branded_content`, `disclose_your_brand`) that the CSV path has no
equivalent for today.
