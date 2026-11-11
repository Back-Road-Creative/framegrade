# Durable single-photo delivery

`FacebookPhotoDelivery` adds an explicit, unwired queue consumer around the
[Page photo client](facebook-page-photo-client.md). It supports one approved JPEG
with immediate public intent per delivery. Production CLI and adapter factories
do not construct it; the media bridge's signed approval consumer is not connected.
This change does not enable posting or make a live Facebook request.

## Authority and immutable content

Construction accepts a trusted `authorize(row)` dependency, defaulting to refusal.
Every new write calls it; passing an `AuthorizedPhoto` value directly is not a
publishing API. The authority must verify the exact current signed decision and
plan, consume/recheck approval, validate fresh privacy, delivery and account
evidence, and provide the approved immutable bytes. A returned dataclass or a
matching hash alone does not prove that these checks happened. No default
implementation of this authority ships here.

The consumer verifies binding to the delivery and queue plan, Page, signed-plan
and receipt identities, item index, JPEG hash, exact caption, publication-master
purpose, immediate timing and public intent. Those five intent fields have no
defaults: the authority must supply each explicitly. The authority supplies an aware
`valid_until` timestamp no more than 15 minutes ahead. Freshness is checked after
the authority returns and immediately before dispatch. Expiry during a database
wait leaves the recorded attempt held without sending a request. The callback
must do the actual signature/owner/evidence checks; these structural checks are
not a substitute.

The caption goes unchanged to the transport. The approval producer must render
title and description into that exact per-item caption before review; Framegrade
does not recompose them or silently drop a title. Reconciliation never reloads
the current draft or local JPEG and never requires fresh approval.

## Atomic write boundary

The queue adds nullable `deliveries.upload_kind`; old rows remain unknown. New
photo writes use `facebook_page_photo` and the existing write-once `upload_account`.
The `facebook_photo_attempts` table retains the immutable request snapshot,
signed-plan/item identity, write boundary timestamp, returned creation receipt
and any verified story ID. A unique signed-plan/item pair prevents replay through
another delivery row. Queue plan hash and signed publication-plan hash remain
separate fields.

With a current lease, `begin_facebook_photo_attempt` atomically checks the queued
row, shadow/pause/timing constraints and identity, inserts the attempt and moves
the delivery to `awaiting_facebook_photo_id`. It commits **before POST**. An insert
conflict rolls back the delivery change. The timestamp means dispatch may have
started, not proof of network transmission.

If creation returns a valid receipt, a separate atomic operation persists its
Photo ID in `platform_upload_id` and the complete receipt, then moves to
`awaiting_facebook_photo_reconcile`. A crash or unknown response without a usable
receipt leaves the no-ID state parked. This includes a crash before the POST:
the conservative hold avoids guessing whether a write happened.

Known IDs permit only the client's exact Photo-to-story/Page readback. The story
ID is retained even when a later read fails. Only literal `is_published=True`,
matching Photo/story IDs and Page ownership can close the row. The queue records
the story ID in `platform_post_id`, Page in `destination_account`, and source
`platform_reconciled`; it does not fabricate a platform publication timestamp.
False, unreadable, conflicting or malformed evidence remains unknown.

## Retry and mutation protection

No-ID attempts are nonterminal and excluded from normal selection. Expired
leases are released without consuming their hold. Retained-ID attempts continue
read-only reconciliation with existing backoff beyond the usual retry budget.
Neither kind is reset into another upload. Ordinary transition, cancellation,
rescheduling and retry helpers cannot discard the durable attempt or rewrite
its identity. Terminal recovery cannot create a replacement for one either.
There is no automatic no-ID recovery, caption/time search or object adoption.

`drain_queue(..., facebook_photo_reconciler=consumer.reconcile)` is a separate,
explicit read-only port. For persisted photo kinds it bypasses current-draft
validation and privacy loaders, which might be unavailable after a write, while
queue pause/shadow/backoff selection still applies. New writes retain their
normal publication checks. Without this injected reconciler, the row remains
held; it cannot fall through to an existing Reels or Instagram handler. The
current-record format router independently refuses durable Page-photo attempts.
`make_facebook_page_photo_handler(consumer)` is an opt-in construction seam, not
a production factory change.

Confirmed Page photos are excluded from the existing first-comment selector.
Text inherited from enqueue or later mutable-sidecar backfill cannot cause an
additional comment POST: the approved photo contract contains no comment intent.
Legacy Facebook/Instagram/YouTube comment behavior remains unchanged.

The queue database remains trusted application state. These operations protect
supported code paths, not arbitrary SQL edits or a malicious injected callback.
Python traceback locals and snapshots may contain private content; diagnostics
must not serialize credentials, JPEG bytes or arbitrary exception locals.

## Verification and remaining work

Offline tests cover atomic checkpoints visible from another connection, competing
approved-item claims, rollback, crash windows, no-ID holds, retry exhaustion,
account/kind binding, exact captions and hashes, approval expiry, and recovery
without draft/media files. The client qualification limits still apply: the live
diagnostic covered an unpublished synthetic photo; public creation remains
contract-based and offline-tested.

Activation requires the real signed-approval consumer, verified publication
masters and current eligibility evidence, deliberate route wiring, and approval
for the exact first public item. Scheduled publishing, carousels and resolving
unknown no-ID attempts are outside this unit. There is no exactly-once guarantee
for the remote service.
