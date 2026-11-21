## Queued Shorts follow the current generator: `queue sync-short-text`

A queued Short keeps the text the generator made when it was queued, so a generator fix never
reached it. `gomoveshift queue sync-short-text [--apply]` rewrites those rows. Dry run unless
`--apply`, and a dry run writes no file. Exits 1 when any Short is refused or any sidecar is
unusable.

It touches every `queued`, unleased YouTube row whose slot is more than 24 hours away
(`SYNC_LEAD_TIME` in `gomoveshift/publishing/queue_short_text.py`), regenerating title,
description and tags through the per-LF path Phase D uses (`plan_bracket_sync._generate_lf`,
now also usable without a bracket record). The title stamp, recording date, schedule, media and
`experiment_arms` stay as stored: the sync never picks, swaps or excludes an arm. The
generator renders each Short's stored Shorts arms (`yt_title_body`, `yt_hashtag_set`) verbatim,
and a Short whose known stored arm the new text no longer carries is refused whole, so the
experiment is never mislabeled; an arm id outside `SHORTS_ARMS` is never rendered.

The whole Short is refused (nothing written) when any delivery row of it, on any platform, is
neither `queued` nor terminal, holds a live lease, has no `scheduled_at`, or has a slot inside
the lead time, or when a recovery row exists. A Short with a hand title in
`<plan dir>/social/titles.yaml` (the file Phase D reads) is `unchanged` with reason `hand_title`.

The no-repeated-place chain is seeded from the stored title and description of the Short before
each rewritten one, not from regenerated text, and the generator's date is the LF's
`parent_lf_publish_at`. Each rewritten Short
must pass `short_review_refusal`, `models.validate_payloads` and `anti_spam_gate.check_batch`
(with the queue's recent text: the queue's published and queued text, read once per run, minus
the Shorts this run rewrites and a sidecar's own projects; earlier sidecars' final text is added),
the gates enqueue runs, plus
`plan_gates.assert_plan_text_grammar` and `assert_shorts_metadata_in_region` on the new post;
a refused one is left unchanged and reported. The brand's per-location cap is
loaded by the CLI and the health entry point and counted per sidecar: a separate `check_batch`
over that ONE sidecar's whole, untrimmed `plan.shorts` with the cap. This is an approximation of
enqueue, not a mirror: enqueue runs `check_batch` over every `*_publish_plan.json` in the
directory, with `select_capped_shorts` minus the refused keys (`cli/handlers/enqueue_plan.py`),
so the sync can refuse a Short enqueue would have accepted and the reverse. The cross-sidecar
passes (earlier sidecars' records, later sidecars' records) run without the cap.

### Order, recent text and idempotence

Sidecars are planned in order of their earliest queued slot, ties broken by sidecar path.
Sidecar i's recent text is the pre-run snapshot of every project OUTSIDE the sync set (one
`recent_platform_texts` read per run, before anything is applied, so dry run and apply give the
same verdicts) plus the final in-memory text of sidecars 0..i-1, and never a later one's. Its
`check_batch` runs over the earlier sidecars' final records followed by its own trial plan, so
the cross-sidecar rules (shared title prefix, repeated text) bind the later sidecar and only its
own violations refuse it. The cross check sits in the same refusal loop as the neighbour-place
check (`_successor_clashes`), so after any refusal the generator re-runs and both checks look
at the sidecar's FINAL text. The published-titles ledger and the nightly title snapshot
(`title_evidence.json`, written by `metrics collect`) are read once per run, not per title.
Because each sidecar sees the same inputs on every run, a rerun is a no-op: two queued projects
that share a place no longer flip description leads on every apply.

### Nightly: `gomoveshift health sync-short-text`

The nightly entry point (`gomoveshift/health/short_text_sync.py`) runs the sync with `--apply`
and files incidents through the health incident store (`health-incident/1`, see
`156-health-check.md`): `short_text_sync_refused:<project>` (one row per project, listing each
refused Short and why) and `short_text_sync_crash`. Every refusal files a row and counts EXCEPT
the few that clear on their own: each carries the structured flag
`ShortTextChange.transient`, set where the reason is created and never inferred from its words
(a reason can quote a Short's text). An unclassified or new reason is unflagged, so it is
actionable and the check fails closed. Classified by reading `queue_short_text.py`:

- Transient (flag set; no row, no exit 1, until the same Short has been refused on
  `TRANSIENT_ESCALATE_RUNS` = 3 consecutive runs, an actionable night in between counting too, and
  is transient on this one: it is then filed like an actionable one, the
  reason naming the run count; the count is one row per Short in `<incidents dir>.streaks`, reset
  by the first run that no longer refuses it. It counts runs, not nights: a crash run and a run that finds the incident
  store lock held neither count nor reset it): a sibling row's slot still ahead
  inside the lead time (it passes or is moved), a sibling slot already past on a still-queued row whose `next_attempt_at` is still ahead (the dispatcher's retry backoff; a row stuck there escalates), a sibling row holding a live lease (the dispatcher releases it), "retried next
  run" (the cascade after another sidecar's failed write, itself actionable), and a write that
  lost a race (`SidecarChangedError`, `LeaseRaceError`: rolled back whole) or met a busy
  database (`sqlite3` BUSY, e.g. "database is locked"; LOCKED is a conflict inside one connection, so it repeats and stays actionable). A failed write whose reason
  carries the "clashes with this sidecar's stored text" note is never transient.
- Actionable, needs a person: "generator: ...", an LF sidecar unreadable or without a location,
  a stored arm the generator cannot render, "no such Short in the plan", "already holds the new
  plan hash", a recovery row, a sibling slot already past on a row still queued and past its retry backoff ("is past but still queued"),
  a row in another state ("is <status>"), "has no usable
  scheduled_at", "no clip place to compose a title from", an old title with no verifiable
  "Nth stretch" stamp (nothing edits it for the sync, so it is permanent), "its ... place
  repeats ...", a review, payload, plan or anti-spam gate refusal, "sidecar unusable: ...",
  an unknown queue hash, a failed write (unless it lost a race or met a busy database with no clash).
- Not refusals, by design: "already holds the synced plan hash" is reported `unchanged` (the
  text is already the generator's and the queue's UNIQUE key says another row carries the
  target hash: nothing for the sync to write), and a place the generator could not vary
  (`place_variety` reason `no_alternative`) is a sanctioned repeat, not a neighbour clash.
Each row is pushed once (`health_short_text_sync_refused`, `health_short_text_sync_crash`; a push
that fails is retried the next night) and a refusal row resolves on the first run that no longer refuses that project (the crash row, on the first run that completes). A crash leaves
the refusal rows open (it proves nothing about them). `health check` lists these rows but never
writes, resolves or pushes them: this command is their only writer, and it takes an exclusive
non-blocking file lock (`pipeline/_file_lock.py`, portable to Windows) on `<incidents dir>.lock` around the store write (held: log a line, skip
the incident work, exit 0). An open sync-owned row keeps `health check` at exit 1 like any open
incident. Exit 0 clean, 1 when a Short was refused for an actionable reason, 2 on a crash; the
entry point never raises (a failure in the store or the push is a crash). It takes no `--jobs-dir`.
`queue sync-short-text` itself still exits 1 on any refusal. Every run, a crash and a held lock
included, stamps `<incidents dir>.short_text_sync_run.json` atomically (`schema`, `ran_at`,
`changed`, `refused`, `crashed`); a write failure is logged and never changes the exit code, and
`health check` rule 13 (`seo_feedback_stale:sync`) reads it to notice a job that stopped running.

Failure handling: a new text is also checked against the final text of every LATER sidecar
(including a Short it leaves as stored); a clash refuses the earlier sidecar's Short and the
run is re-planned. After a failed write the sidecars still to be written are refused ("an earlier
sidecar's write failed") and converge the next night. Any exception in a write (not only I/O
errors) rolls the queue back and refuses the rest; the failed sidecar's Shorts get a refused verdict (actionable unless the write lost a race or met a busy database with no clash; exit 1 when actionable), not a crash verdict; the text the
earlier sidecars already wrote is re-checked against the failed sidecar's STORED text and a
clash is named in its refusal. A sidecar skipped by the cascade stays as stored too, so the text
written earlier is checked against ITS stored text as well: a clash is named in its reason and
makes it actionable (otherwise it is transient).

A sidecar that cannot be planned (a bad `titles.yaml`, a trial plan the gates reject, a missing
`lf_index`) is refused but stays queued as stored, and its STORED text reaches the gates of the
sidecars before and after it. Caveat: when a later sidecar's stored text was itself changed
by an earlier run, settling may take two runs.

Schedule: run `gomoveshift health sync-short-text --data-root <workspaces>` from the service
directory, from the staging mirror's venv, AFTER the nightly `metrics collect` (which writes
the title snapshot the sync reads). The scheduler is the workspaces nightly metrics cron
(`bin/gms-metrics-cron.sh`, 03:30, installed from `_ops/cron/`; workspaces #5919): it runs the
sync once after the collect loop, logs exit 1 (refusals, already filed) as a line, and names
any other non-zero exit `SHORT_TEXT_SYNC_FAILED`. A night it does not run is alerted by health
rule 13 (`seo_feedback_stale:sync`).
A Short is also refused when its final place equals its neighbour's final place in the LF
(new text if rewritten, stored otherwise; the later of two rewritten ones), when the generator
refuses it alone (the LF is retried Short by Short; Shorts not being rewritten ride along as
discarded stamped stand-ins, so their own composition cannot fail the others), or when a row on
the same platform already holds the new plan hash (the queue's UNIQUE key; also checked before a
reconcile).

A changed Short gets `reviewed_plan_hash` recomputed (`asset_plan.review_hash`). The audit row
(`response_log`, `action` = sync_short_text, old hash kept) and the queue update (non-terminal,
unleased rows; `updated_at` is not touched) commit first, then the sidecar is replaced through a
temp file, after a sha256 check that it did not change meanwhile. The queue update must move
every row it planned to; if a row was leased meanwhile it moves fewer, the whole sidecar's
transaction rolls back (no audit row, no replace) and the sidecar is refused ("a row was leased
during the write"). A crash between the two leaves the sidecar old and the queue new; the next
run converges even if the generator has moved on (a row on a hash this sync's audit row moved
from the sidecar's own is accepted). Each run also
moves a queued row still on a stale hash to the sidecar's when the sidecar already holds the
synced text and this sync's own audit row names that old and new hash (`reconciled`, with an
audit row); a row on any other hash refuses the Short ("plan hash ... is unknown"). A residual
race remains between the sha256 check and the replace; a concurrent writer in that window is
not detected.

Tests: `tests/test_queue_sync_short_text.py`, `tests/test_queue_sync_short_text_nightly.py`.
