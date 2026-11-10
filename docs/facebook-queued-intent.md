# Facebook queued publication intent

The queue adapter supports immediate publication and scheduled publication.
It refuses `DRAFT` before taking a lease or starting an upload: the current
queue has no draft receipt and later-publish lifecycle, so recording a draft as
published would misstate the result. Low-level draft request construction remains
available for a future caller that supplies that lifecycle.

An explicit timestamp requires `video_state="SCHEDULED"`; a scheduled state
requires a timezone-aware timestamp inside the existing adapter bounds. These
checks run before any new-upload queue mutation or API call, and again before the finish
request. An invalid request cannot start an upload that would later fail merely
because its state/time combination was inconsistent.

Without an explicit override, the Facebook-specific record slot takes precedence
over the shared slot. At or after that instant, the existing driving queue may
publish immediately. A future slot within the adapter's minimum scheduling lead
raises a retryable not-due error before taking a lease or making a network call.
It must not be converted into an immediate post. Normal queue selection already
filters by scheduled time; this guard also protects direct/early callers. The
caller remains responsible for honoring retry timing and any stale-approval rules.

## Uncertain finish and recovery

The adapter checkpoints the credential client's Page ID in nullable
`upload_account` before starting an upload. This write-once provenance field is
separate from `destination_account`, which remains a published-receipt field.
Reopening an existing queue adds the column without guessing accounts for old
rows. The adapter persists `platform_upload_id` before uploading bytes and commits
`awaiting_facebook_reconcile` before sending finish. Only a literal boolean
`success: true` permits a confirmed finish receipt. A lost response, exception,
missing success, false value, or malformed body raises `FacebookFinishUnknownError`
and retains the ID without a published receipt. The checkpoint also survives a
worker exit between sending finish and recording the answer.

Every later attempt with a retained ID only reads that video's status. It never
creates another upload, sends bytes again, repeats finish, or resets the ID. A
negative, pending, unreadable, or malformed status remains unknown; only confirmed
publication permits a reconciled receipt. The retained `upload_account` must
match the current client's Page ID before any status read, and the receipt uses
that original account. Missing provenance or changed credentials holds the row
for account reconciliation without contacting Facebook or relabeling the upload.
This also applies to legacy uploaded rows whose old code did not record whether
finish started or which Page was used. Those rows require operator investigation;
the worker never fills old provenance from current credentials. Missing retained
IDs cannot be reconciled by guessing or creating a replacement.

The dispatcher preserves this state beyond normal retry limits, including expired
worker leases, and keeps the existing retry backoff and lease release. A retained
ID protects the unknown outcome even if the handler fails before reaching the
adapter. Ordinary failures without an ID still exhaust their retry budget. Paused
platforms, privacy checks, validation holds, and existing queue timing remain in
force. An unknown row is eligible when its retry time is due; it is not a published
receipt when a handler returns without changing state.

A successful scheduled finish retains the existing `platform_scheduled` receipt
and exact requested timestamp; it is evidence of scheduled acceptance, not a live
post. An uncertain scheduled finish is polled until actual publication is reported.
The current parser does not separately confirm scheduled acceptance during
reconciliation, so polling can continue through the scheduled time. Reconciliation
does not change the request's timing or issue a new publication request.

## Remaining delivery boundary

The current Facebook retained-upload-ID convention belongs to Reels; the photo
adapter does not write `platform_upload_id`. Any future photo implementation that
uses that field must qualify its own reconciliation behavior before reuse.

The existing handler still loads the record and local media before reaching the
adapter; missing files or a validation hold can delay status reads. Unknown rows
stay held rather than becoming permanent failures. Losing the start-upload
response before an ID is persisted is outside this retained-ID guarantee. This
change does not promise exactly-once remote delivery.

The media bridge still requires exact-plan approval and verified platform contract
wiring before automated delivery is enabled. Tests use mocked existing request
and status shapes; current Meta documentation could not be audited because the
documentation endpoint returned HTTP 429. No request fields were changed based on
assumed API behavior, and no live Facebook calls were used for validation.
