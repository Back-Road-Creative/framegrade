# Facebook queued publication intent

The queue adapter supports immediate publication and scheduled publication.
It refuses `DRAFT` before taking a lease or starting an upload: the current
queue has no draft receipt and later-publish lifecycle, so recording a draft as
published would misstate the result. Low-level draft request construction remains
available for a future caller that supplies that lifecycle.

An explicit timestamp requires `video_state="SCHEDULED"`; a scheduled state
requires a timezone-aware timestamp inside the existing adapter bounds. These
checks run before any queue mutation or API call, and again before the finish
request. An invalid request cannot start an upload that would later fail merely
because its state/time combination was inconsistent.

Without an explicit override, the Facebook-specific record slot takes precedence
over the shared slot. At or after that instant, the existing driving queue may
publish immediately. A future slot within the adapter's minimum scheduling lead
raises a retryable not-due error before taking a lease or making a network call.
It must not be converted into an immediate post. Normal queue selection already
filters by scheduled time; this guard also protects direct/early callers. The
caller remains responsible for honoring retry timing and any stale-approval rules.

## Remaining delivery boundary

This change does not qualify the complete Facebook publishing path for the media
bridge. The existing queue adapter still needs a separate confirmed-finish and
unknown-outcome lifecycle: it currently ignores the finish response's `success`
field, and a non-published status read does not prove an earlier finish failed.
That repair must preserve the upload ID, distinguish pending/unknown outcomes,
and prevent ordinary retry exhaustion from labeling a possibly published post
as a permanent failure. The new media bridge must remain disconnected until those
checks and exact-plan approval are in place. No live Facebook calls were used to
validate this change.
