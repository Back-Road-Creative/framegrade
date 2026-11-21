# `seo_feedback_stale`: is the Shorts feedback loop still turning?

`gomoveshift health check` rule 13 (`gomoveshift/health/seo_feedback.py`, listed in
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

## The sync run record

`gomoveshift health sync-short-text` (the nightly queued-Short text sync) writes
`<incidents dir>.short_text_sync_run.json` atomically
(`short_text_sync.run_record_path`): `{"schema": 1, "ran_at": <UTC ISO>, "changed": n,
"refused": n, "crashed": bool}`. It is written on every run, even when the sync crashed or the
incident work is skipped because the store lock is held. A crash is covered by
`short_text_sync_crash` alone, so condition (c) means only that the job has not run (cron line
missing, cron dead). A failure to write the record is logged and never changes the exit code.

## Conditions

| Key | Fires when | Constant |
|---|---|---|
| `seo_feedback_stale:decisions` | the file is missing, unreadable, or `decided_at` is older than the limit | `seo_feedback.DECISIONS_STALE_AFTER_H` = 36 |
| `seo_feedback_stale:metrics` | one grouped row: at least one YouTube Short whose day-7 window is settled and that was posted within the lookback has a null `views` or `avg_view_percentage`. The detail gives the count, up to `LISTED` = 10 `project#index (missing fields)` entries, then "and N more"; it resolves at a count of 0 | settled = `tracking.day7.window_settled` (posted + 7 d <= today - 2 d); `seo_feedback.CORE_METRICS`; `seo_feedback.METRICS_LOOKBACK_DAYS` = 28 |
| `seo_feedback_stale:sync` | the run record is missing, unreadable, or `ran_at` is older than the limit or more than `SYNC_SKEW_H` = 1 h in the future (unreadable); a naive `ran_at` is read as UTC; the detail says which and gives the age. Reads only that file; resolves on a fresh stamp | `seo_feedback.SYNC_STALE_AFTER_H` = 36 |

A Short that ages out of the metrics lookback is simply no longer counted. Condition (c) says only that the nightly sync is not running; it does not compare queued Shorts with the generator.

**Open rows follow the finding.** A row already open has its `detail`, `count` and listing
refreshed on each check; `pushed` stays true and nothing is pushed again. When the `metrics`
row drops a Short it listed because that Short aged out of the lookback still without metrics
(not because metrics arrived), the check notes it once, on the tick it happens, whether the row
stays open or resolves; Shorts the row never listed are not counted.

Arm assignment does not yet exclude demoted arms; that is a planned unit (C4c), not this rule.

## Unreadable inputs

An input that cannot be read is a note, and the open incident it feeds stays open instead of
closing: a tracking record that fails to load holds the `metrics` row. A **missing or unreadable decisions file is
itself condition (a)'s finding only**: it holds nothing else. A relative `seo.storage_path`
skips the whole rule, condition (c) included, with a note (the run record path comes from the
incidents dir, so it does not depend on it; the skip is kept so the rule stays one unit).
