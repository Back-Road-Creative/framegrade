## `health publish`: one command to upload an audited drive

`gomoveshift health publish <project> --region R --stages-root S [--jobs-dir J] [--code-root C]
[--lead-hours N] [--expect-sha SHA] [--dry-run] [--json]` is the single approval for
uploading a drive `health backlog` audited; `health backlog` runs the same call each tick for its
region's drive whose audit passed. It goes through Waypost's claim, so the schedule store
never goes stale (a hand-launched `drive ... --youtube-upload --phase D` bypasses it). Exit 2 on
`refused`, else 0.

1. **Checks, no writes.** The drive must be `auditing` in `schedule.sqlite3`, or `failed` (a publish
   job that ended without uploading; it is reopened to `auditing` at the first write, step
   `reopen:`); any other status is `refused` with the status named. No upload receipt under its output dir;
   the long-form count is `regular_lfs` in the `drive_plan.json` beside its `resume.json` (missing,
   unreadable or empty is `refused` before any write). `health backlog` registers every drive with 1, so a
   count that differs is a `count:` step and is written back with the reopen: the claim books one weekly
   slot per long-form, the same series Phase D dates them by;
   no live drive on its own stages or project dir (`heal.drive_conflict`; a render of any other
   drive never holds a publish back, note 191); its `resume.json` under `<stages root>/<project>` must rebind to the code
   root's venv, and its `pipeline_state.json` must hold no completed phase whose stamped code
   version the current code re-runs (`heal.stale_phases`; Phase D would refuse the missing
   prerequisite only after the slot is booked), else `refused` naming the phases: re-audit it;
   and no older `ready` drive may sit ahead of it, because `claim_or_reason` claims
   the oldest ready drive of the region, never a named one. An older `ready` drive that already
   holds an upload receipt with `publish_at` (uploaded outside a claim) does not block: it is
   listed as a `retire:` step, and before the approve Waypost's `retire_published` books its
   receipt (adopting a booking of the same video under its run dir's name) and marks it
   `complete`. A retirement Waypost refuses is `refused`.
2. **Approve.** `set_status(project, "ready")`.
3. **Claim.** `claim_or_reason(...)`. A miss (`outside_lead_window`, a taken slot, a stale sha) puts
   the drive back to `auditing`. If the claim names another drive (it became ready first), that
   claim is failed (bookings released), the other drive is made `ready` again and this one
   `auditing`.
4. **Launch.** A detached `gms-publish-<project>` job (job runner under `--code-root`, record in
   `--jobs-dir`) running the recipe's own drive args with `--skip-upload`, `--no-youtube-upload`
   and the audit's `--publish`/`--lf-day` replaced by: `--publish D --lf-day D --lf-publish-date
   <claimed slot's local date> --youtube-upload --force --no-tmux --phase D`, with the recipe's
   `GMS_INTERMEDIATE_DIR` and `GMS_LAUNCHED_BY=health-publish`. A failed launch fails the claim and
   returns the drive to `auditing`.
5. **Close.** `health check` (the hourly tick) calls `publish.reconcile_publishes`: for each
   `claimed` drive whose `gms-publish-<project>` job (started after the claim) has ended, it calls
   `finish_claim` with `ok` = the output dir holds an upload receipt (`youtube_uploads.json` with a
   `video_id`; a receipt wins over the exit code, so a live video keeps its slot). Otherwise the
   drive is `failed` and its unconfirmed booking is released. A running job, or a claim with no
   publish job, is left alone; a repeat pass is a no-op. It then retires every `ready` or
   `auditing` drive that already holds an upload receipt (`publish.retire_uploaded`, the same
   `retire_published` call; an `auditing` drive that was uploaded is published, never left in the
   claim line, and a drive with no receipt is never retired). The closed claims and retirements are `publish_reconciled` in `run_check`'s result.

`--dry-run` runs every check, then reports the four steps and Waypost's own dry-run answer
(`next_claim`); it writes nothing and launches nothing.

**A refused publish retry does not end the backlog tick.** `health backlog` retries the publish
of a drive whose publish job failed (`156-health-check.md`, `publish_only`). A retry refused
for anything but `no claim:` is listed under `publish_refusals` (printed `publish retry refused
<project>: <reason>`) and the tick goes on to the `auditing` drives, so one drive's standing
refusal never starves the region's next passed audit; the tick still exits 2 while the refusal
stands. Before, Kerry-to-Cork's blur-mask frame-gate refusal returned first every hour and Old
Head to Courtmacsherry, audited 2026-11-24, never published (2026-11-25). Tests:
`tests/test_health_backlog_launch.py`, `tests/test_health_backlog_cli.py`.

**A passed audit's refused publish does not block an operator's supersede of another row.**
Without `--supersede`, an `auditing` drive whose audit passed and whose publish is refused is
still the tick's refusal and holds the region, as before. With `--supersede <other project>`,
that refusal is listed under `publish_refusals` the same way and the supersede goes on; the
tick still exits 2 while the refusal stands. Before, the supersede re-rendering Kerry-to-Cork
on the fixed blur mask was refused with Old Head to Courtmacsherry's own mask-gate refusal
(2026-11-26). A failed audit still stops an explicit supersede of another row.

**A passed audit behind a refused one still gets its publish.** Without `--supersede`, the first
passed audit whose publish is refused no longer ends the auditing loop: every later passed audit
is offered for publish. The first one that publishes is the tick's result, with the refusals ahead of
it under `publish_refusals` (exit 2). If none publishes, the first refusal is still the tick's
`refused` result and the region is still held (no new row launches), with the later refusals
listed. Before, Old Head to Courtmacsherry's mask-gate refusal held Galley Head's passed audit
(2026-11-27).

**A supersede re-renders a drive whose own passed audit cannot publish.** `--supersede <project>`
naming a `failed` store row whose publish retry is refused sets that run aside and relaunches the
row (the retry is still offered first; a retry that publishes wins). Naming an `auditing` drive
whose audit passed asks `health publish` as a dry run: if it is refused, the run is set aside and
the row relaunches; if it could publish, the supersede is refused with `its passed audit can
publish`, and nothing is moved. Either relaunch re-registers the same store row `auditing` and
reports the refusal under `relaunch`. Before, the mask gate asked for re-renders of Kerry-to-Cork
(`failed`) and Old Head to Courtmacsherry (`auditing`) that `--supersede` refused as "in the
schedule store (failed)" and "audit in progress" (2026-11-27).

**The relaunched row's own unpublished ledger entries do not refuse it.** With the relaunch
allowed, the supersede was still refused `in the content ledger`: Kerry-to-Cork's failed publish
had left 7 Shorts entries naming the project, none holding a video id. An operator's
`--supersede` relaunch now skips that check for the row's own entries when none of them holds a
video id (`launch.load_inputs` returns the ledger projects that do, `ledger_live_projects`); an
entry with a video id is live and still refuses. Every other row in the ledger is still skipped,
and the automatic failed-row relaunch (`190-health-backlog.md`) still never takes a ledger row
(2026-11-27). Tests: `tests/test_health_backlog_launch.py`.
