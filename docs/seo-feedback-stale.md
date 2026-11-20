# `seo_feedback_stale`: is the Shorts feedback loop still turning?

`gomoveshift health check` rule 12 (`gomoveshift/health/seo_feedback.py`, listed in
`docs/pipeline/156-health-check.md`). It only files incidents: it edits no plan, queue row,
tracking record or decisions file. Each condition has its own stable key, is pushed once as
`health_seo_feedback_stale`, and resolves on a later check once the condition clears.

## The decisions file

`gomoveshift metrics collect` prints the per-arm `ARM_DECISION` lines and, once **every**
factor (`experiments.SHORTS_FACTORS`) has been decided, writes `<seo storage>/shorts_arm_decisions.json`
(`experiments.ARM_DECISIONS_FILENAME`) atomically:
`{"schema": 1, "decided_at": <UTC ISO>, "decisions": [{factor, arm, decision, n, score,
reference, gap, reason}]}`. A run that fails part-way writes nothing, so `decided_at` is not
refreshed; that missing refresh is what condition (a) catches.

## Conditions

| Key | Fires when | Constant |
|---|---|---|
| `seo_feedback_stale:decisions` | the file is missing, unreadable, or `decided_at` is older than the limit | `seo_feedback.DECISIONS_STALE_AFTER_H` = 36 |
| `seo_feedback_stale:metrics:<project>:<index>` | a YouTube Short whose day-7 window is settled and that was posted within the lookback has a null `views` or `avg_view_percentage` | settled = `tracking.day7.window_settled` (posted + 7 d <= today - 2 d); `seo_feedback.CORE_METRICS`; `seo_feedback.METRICS_LOOKBACK_DAYS` = 28 |
| `seo_feedback_stale:queued:<project>:<index>` | a queued, unleased, non-shadow YouTube row whose slot is at least the lead time away (so `sync-short-text` would rewrite it) still repeats the previous Short's place text although the clip measured another place, or carries an arm the decisions file marks `demote` or `pause` | `queue_short_text.SYNC_LEAD_TIME` = 24 h; repeat judged by `social_generator.choose_place_body` (reason `varied`) on `queue_short_text._stored_places`; `seo_feedback.DEMOTED` |

A repeat with no other measured place is not flagged: `sync-short-text` could not vary it
either. A row inside the lead time is left alone, as the sync leaves it.

## Unreadable inputs

An input that cannot be read is a note, and the open incidents it feeds stay open instead of
closing: a tracking record that fails to load holds every `metrics` row; a missing queue
database or an unreadable sidecar holds the `queued` rows (one row for a bad sidecar); an
unreadable decisions file also holds the `queued` rows, because demoted arms cannot be judged.
A **missing or unreadable decisions file is itself condition (a)'s finding**, not a held input.
A relative `seo.storage_path` skips the whole rule with a note.
