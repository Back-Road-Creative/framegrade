## Queued Shorts follow the current generator: `queue sync-short-text`

A queued Short keeps the text the generator made when it was queued, so a generator fix never
reached it. `gomoveshift queue sync-short-text [--apply]` rewrites those rows. Dry run unless
`--apply`, and a dry run writes no file. Exits 1 when any Short is refused or any sidecar is
unusable.

It touches every `queued`, unleased YouTube row whose slot is more than 24 hours away
(`SYNC_LEAD_TIME` in `gomoveshift/publishing/queue_short_text.py`), regenerating title,
description and tags through the per-LF path Phase D uses (`plan_bracket_sync._generate_lf`,
now also usable without a bracket record). The title stamp, recording date, schedule, media and
`experiment_arms` stay as stored: the sync never picks, swaps or excludes an arm.

The whole Short is refused (nothing written) when any delivery row of it, on any platform, is
neither `queued` nor terminal, holds a live lease, has no `scheduled_at`, or has a slot inside
the lead time, or when a recovery row exists. A Short with a hand title in
`<plan dir>/social/titles.yaml` (the file Phase D reads) is `unchanged` with reason `hand_title`.

The no-repeated-place chain is seeded from the stored title and description of the Short before
each rewritten one, not from regenerated text, and the generator's date is the LF's
`parent_lf_publish_at`. Each rewritten Short
must pass `short_review_refusal`, `models.validate_payloads` and `anti_spam_gate.check_batch`
(with the queue's recent text and the brand's `per_location_cap`), the gates enqueue runs; a
refused one is left unchanged and reported.

A changed Short gets `reviewed_plan_hash` recomputed (`asset_plan.review_hash`). The audit row
(`response_log`, `action` = sync_short_text, old hash kept) and the queue update (non-terminal,
unleased rows; `updated_at` is not touched) commit first, then the sidecar is replaced through a
temp file, after a sha256 check that it did not change meanwhile. A crash between the two leaves
the sidecar old and the queue new; the next run writes the same text and converges. Each run also
moves a queued row still on a stale hash to the sidecar's when the sidecar already holds the
synced text (`reconciled`, with an audit row); a row on an unknown hash refuses the Short.

Tests: `tests/test_queue_sync_short_text.py`.
