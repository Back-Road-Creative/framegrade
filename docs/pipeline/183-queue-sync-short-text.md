## Queued Shorts follow the current generator: `queue sync-short-text`

A Short queued days ago keeps the title, description and tags the generator made then, so a
generator fix (the no-repeated-place title rule) or an experiment-arm decision never reached it.
`gomoveshift queue sync-short-text [--apply]` rewrites those rows. Dry run unless `--apply`.

### What it touches

Every `queued` YouTube row whose slot is more than 24 hours away (`SYNC_LEAD_TIME` in
`gomoveshift/publishing/queue_short_text.py`). It regenerates the Short's text through the same
per-LF path Phase D uses (`plan_bracket_sync._generate_lf`, now also usable without a bracket
record) and writes title, description and tags to the sidecar.

Kept as they were: the title stamp (`<date>, Nth stretch`), recording date, schedule, media and
the arm tags.

### Arm decisions

The nightly `metrics` run prints `ARM_DECISION` lines (`decide_shorts_arms`). The sync reads the
same decisions. An arm whose latest decision is `demote` or `pause` is never used for the new
text: the Short takes the first other arm of that factor that is not barred (its tag moves with
it), and with none left the Short is left unchanged. No decision for an arm means the Short keeps
it. The generator takes the effective arms through the new `pinned_arms` constructor argument.

### Review and plan hash

Each rewritten Short must pass `short_review_refusal`. A refused Short is left unchanged and
reported with the reason. Its Facebook and Instagram rows share the one record, so a refusal holds
all of them.

A changed Short gets `reviewed_plan_hash` recomputed (`asset_plan.review_hash`). Its non-terminal
queue rows move to the new hash in one transaction with the sidecar replace, and `response_log`
(`action` = sync_short_text) keeps the old hash. The sidecar is written through a temp file and
`os.replace`. A second run with an unchanged generator changes nothing.

### Nightly path

`metrics` runs the sync with `--apply` semantics right after the `ARM_DECISION` lines and prints
`SYNC_SHORT_TEXT changed=N refused=N rows=N`. A failure there is logged and never stops the run.

Tests: `tests/test_queue_sync_short_text.py`.
