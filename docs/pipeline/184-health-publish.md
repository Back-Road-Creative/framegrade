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
   no drive job running; its `resume.json` under `<stages root>/<project>` must rebind to the code
   root's venv, and its `pipeline_state.json` must hold no completed phase whose stamped code
   version the current code re-runs (`heal.stale_phases`; Phase D would refuse the missing
   prerequisite only after the slot is booked), else `refused` naming the phases: re-audit it;
   no live-catalog video may be an unbooked upload of this drive (`longform_metadata.
   orphan_upload_refusal`): a publish that died after `videos().insert` leaves a scheduled video with no
   receipt and no ledger row, and the only key tying it to the drive is its title. Each LF expects the
   exact title Phase D wrote to `<output>/<project>_lfNN_youtube.json`; with none, the derived forms
   (each `per_lf_ost_text` label and each `per_lf_pre_render_title_decisions[...]["input_label"]`,
   assembled, bare and dated ` (Mon D, YYYY)`, with a ` — Part N of M` / `(Part k [of N])` suffix
   stripped from the catalog title), minus any form another drive derives too. Never a known id: any
   drive's receipt whatever its `publish_at` (a replace's new upload is saved with none), any drive's
   `replace_plan.json` id (`replace`/`kept` `video_id`, `progress` `old_id`/`new_id`) and any
   superseded run's receipt. Only a private/unlisted (or privacy-unknown) video, or a public one that
   went live after this drive's first `gms-publish-<project>` job, counts: a public historic video
   with the title does not block. A match is `refused` naming its id, title, privacy and
   `publish_at` (verify which drive uploaded it, then delete it in YouTube Studio or book it into the
   ledger) and opens one `orphan_upload` incident per LF (`replace_scheduled_partial` kind, never
   duplicated by a rerun and never closed by the replace flow's LF-done resolve; closed once a snapshot
   that lists other videos no longer holds it, or when the checkpoint drops the LF; a dry run records
   none). After a failed publish (a dead `gms-publish-<project>` job: failed, cancelled, stalled or killed,
   the one rule `launch.dead_publish_result` that the backlog's retry uses too, or a `failed` drive) the
   snapshot must have been fetched after the newest dead job ENDED (`result.ended_at`; `now` when the
   record has none; a failed drive with no job record also uses `now`), since the upload happens inside
   the run: `backlog_publisher` and `health publish` run `catalog-fetch --known-ids-from-queue` first
   (`catalog_refresher`, stdout kept clean for `--json`; the tick builds one refresher, so a failing fetch
   is tried once per tick), and a snapshot still older, or a failed fetch, refuses with a reason saying
   the next tick retries. A clean first publish needs no fresh snapshot. A video matching only a title
   another drive derives too (every form of an LF, or just its bare form) is refused naming both drives,
   unless a receipt names it or it went up before this drive's first attempt. A missing ledger refuses
   ("missing or unreadable") and opens the incident too.
   The refusal writes nothing to the store, counts no attempt and touches no other drive; nothing on
   YouTube is deleted or unlisted. An unreadable or missing ledger refuses too; no checkpoint or no
   snapshot has nothing to match;
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
