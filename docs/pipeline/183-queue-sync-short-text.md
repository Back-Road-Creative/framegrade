## Queued Shorts follow the current generator: `sync_short_text`

A queued Short keeps the text the generator made when it was queued, so a generator fix never
reached it. `sync_short_text(conn, apply=...)` in `gomoveshift/publishing/queue_short_text.py`
rewrites those rows. Dry run unless `apply`, and a dry run writes no file. The
`gomoveshift queue sync-short-text` command arrives with the next change; nothing calls this
function yet.

It touches every `queued`, unleased YouTube row whose slot is more than 24 hours away
(`SYNC_LEAD_TIME` in `gomoveshift/publishing/queue_short_text.py`), regenerating title,
description and tags through the per-LF path Phase D uses (`plan_bracket_sync._generate_lf`,
now also usable without a bracket record). The title stamp, recording date, schedule, media and
`experiment_arms` stay as stored: the sync never picks, swaps or excludes an arm.

The whole Short is refused (nothing written) when any delivery row of it, on any platform, is
neither `queued` nor terminal, holds a live lease, has no `scheduled_at`, or has a slot inside
the lead time, or when a recovery row exists.

The no-repeated-place chain is seeded from the stored title and description of the Short before
each rewritten one, not from regenerated text, and the generator's date is the LF's
`parent_lf_publish_at`. Each rewritten Short
must pass `short_review_refusal`; a refused one is left unchanged and reported.

A changed Short gets `reviewed_plan_hash` recomputed (`asset_plan.review_hash`). The sidecar is
written to a temp file and replaced inside the queue transaction, which moves the plan hash of
the Short's non-terminal, unleased rows (`updated_at` is not touched) and writes the audit row
(`response_log`, `action` = sync_short_text, old hash kept).

Tests: `tests/test_queue_sync_short_text.py`.
