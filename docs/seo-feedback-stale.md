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
| `seo_feedback_stale:metrics` | one grouped row: at least one YouTube Short whose day-7 window is settled and that was posted within the lookback has a null `views` or `avg_view_percentage`. The detail gives the count, up to `LISTED` = 10 `project#index (missing fields)` entries, then "and N more"; it resolves at a count of 0 | settled = `tracking.day7.window_settled` (posted + 7 d <= today - 2 d); `seo_feedback.CORE_METRICS`; `seo_feedback.METRICS_LOOKBACK_DAYS` = 28 |
| `seo_feedback_stale:queued` | one grouped row: at least one queued Short has the verdict `changed` from `queue_short_text.sync_short_text(conn, apply=False, now=now)` on a read-only connection, i.e. text the sync would rewrite but the nightly `health sync-short-text` did not apply | the sync's own verdicts (it owns the lead time, lease and review checks) |

A Short that ages out of the metrics lookback is simply no longer counted; it is not reported as
a fix. The rule asks the sync itself instead of re-implementing it, so it can never flag a row
the sync refuses.

**Refused rows are not this rule's job.** A `refused` verdict (for example a TikTok row of the
same Short awaiting manual publish) is filed by the nightly `health sync-short-text` as its own
`short_text_sync_refused:*` incident.

Arm assignment does not yet exclude demoted arms; that is a planned unit (C4c), not this rule.

## Unreadable inputs

An input that cannot be read is a note, and the open incident it feeds stays open instead of
closing: a tracking record that fails to load holds the `metrics` row; a missing queue database
or a sync/database error holds the `queued` row. A **missing or unreadable decisions file is
itself condition (a)'s finding only**: it holds nothing else. A relative `seo.storage_path`
skips the whole rule with a note.
