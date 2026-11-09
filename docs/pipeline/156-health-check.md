## 156 - Health check and heal

`gomoveshift health check` detects failed or stalled pipeline states, writes one incident row per
incident and pushes each NEW one once (`console/notify.py`, ntfy). `check` never launches, resumes or
repairs; `health heal` (below) does. Exit 0 = no open incident, 1 = at least one.
An unreadable job record or incident file is counted and reported (`N unreadable`), never raised.

Paths: data root `--data-root` / `$GMS_DATA_ROOT` (default: the checkout); incidents dir
`--incidents-dir` / `$GMS_HEALTH_INCIDENTS_DIR` (default `gms/incidents` under the data root);
job-runner records `logs/jobs/<run_id>/job.json` under the data root (schema `job/1`, parsed as JSON).
The job runner is the process supervisor named by `$GMS_JOB_RUNNER` (default `bin/job-runner.py`).

## Rules

1. **Drive failed.** The latest `job/1` record of a drive job has `result.state == "failed"`, or a
   nonzero `exit_code` and an `expect` log containing `Drive pipeline failed`. A drive job is one
   whose argv or name contains `drive`, or whose log carries the drive orchestrator's logger (real
   drives run through launcher scripts). Row file: `<drive>.json`. A later `succeeded` run of the
   same job name resolves it, and so does a later succeeded drive job under ANY name that shares
   its `expect` log path or its stages dir (audit failed, publish succeeded).
   A job that ended exit 5 (`EXIT_WAITING_AT_GATE`) with `DRIVE WAITING:` as its log's last
   closing line stopped at a human gate (the OST audit) and is not a failure: no row is filed, an
   open one is resolved, so `health heal` never re-resumes it onto the same gate. `health
   backlog` resumes it once the gate is confirmed (below).
2. **LF slot inside 48 h with nothing booked while a drive is ready.** A free slot is a
   `next_available` row of `build_schedule_report` (the scheduler already excludes booked slots);
   "ready" is `ScheduleStore.ready_depth(region) > 0` over `<seo.storage_path>/schedule.sqlite3`.
   No store file means no readiness signal: the rule is reported as not evaluated, not guessed.
3. Not in this change (route derivation belongs to `health heal`).
4. **Dispatch stalled.** The queue (`social_queue.sqlite3`, opened read-only) has DUE work
   (non-shadow, non-terminal, `scheduled_at` null or past, platform not paused) and no delivery
   row has `updated_at` within 24 h. Every dispatch attempt rewrites `updated_at`, so a quiet queue
   with due work means the drain is not running. Row file: `queue_stalled.json`.
5. **Unverifiable placeless label.** A `drive_backlog.yaml` row carries a `placeless_label` the
   launcher refuses (`launch.placeless_label_skip_reason`): it claims scenery, weather, time of
   day or road kind, which nothing can measure on a GPS-less drive, so it cannot be said. The row
   never launches (the rows behind it still do) and the incident stays open while the yaml carries
   the label; editing the row to a claim-free label (e.g. `Somewhere On The Road`) closes it. An
   absent or unreadable backlog closes nothing. Row file: `placeless_label_<project>.json`.

A relative `seo.storage_path` skips rules 2 and 4 with a note (it would resolve against the cwd).

## Push and idempotency

Each row carries a stable `key` and `pushed`. A push that returns `False` (attempted, failed) leaves
the row unrecorded and the next pass retries it. A condition that clears sets `resolved_at`.
An open row found again keeps its `pushed` state and `resume_attempts`; only a `resume_recipe` or
`log_path` it was filed without is filled from the current detection (a row an older reader filed).

## Incident row, schema `health-incident/1`

Common: `schema`, `kind` (`drive_failed` | `lf_slot_unbooked` | `queue_stalled` | `placeless_label_unverifiable`), `key`, `file`,
`detected_at`, `pushed` (bool), `pushed_at`, `resolved_at` (null while open).

`drive_failed`: `drive`, `job_name`, `phase` (or null), `gate` (or null), `error`, `log_path`,
`job_run_id`, `failed_at`, `staging_sha`, `resume_recipe` (path of the drive's `resume.json`, or null), `resume_attempts`
(`[]`, owned by `health heal`).
`lf_slot_unbooked`: `region`, `slot`, `ready_drives`. `queue_stalled`: `due_deliveries`,
`last_updated_at`, `quiet_hours`. `placeless_label_unverifiable`: `project`, `label`, `claims` (the words
outside `PLACELESS_NO_CLAIM_WORDS`), `reason`.

## Resume recipe and `health heal`

**Recipe.** At start (`cmd_drive`, before the orchestrator runs) a drive writes
`<stages dir>/resume.json` (schema `resume-recipe/1`; under the cache `_stages` dir, never
tracked): `argv` (`python -m gomoveshift` + the drive's own args, `--fresh` removed), `cwd`, `sha`
(the product's own commit: HEAD of a source checkout, else the commit `pip` recorded for a git-ref install, else null; never an enclosing repo's HEAD), `started_at`, `phase_reached` (updated by the
orchestrator's `_save_state` as phases complete). `env` records `GMS_INTERMEDIATE_DIR` when it was set: the staging
override decides where the saved state lives, so `health heal` replays it as an `env K=V` prefix on
the jobrun argv (shown in the dry run). No other variable is recorded or replayed. It also logs one stable line,
`resume recipe: <path>`; `health check` parses that line from the drive log (a job's `expect` logs, and the job's own `output.log` beside its `job.json`, which is the only place a backlog drive job prints it) into the incident's
`resume_recipe`, and uses its directory as the stages dir that links sibling runs. A
single-`--phase` run logs the line but does not rewrite the file (so a heal run can't overwrite
the full recipe it came from). A discovery-only run (`--dry-run`, `--list-lfs`,
`--label-dry-run`) renders nothing, so it neither writes the recipe nor logs the line, and its
saves never move `phase_reached`: an existing recipe stays byte-identical. Writing never fails a
drive (warning only).

**Heal.** `gomoveshift health heal [--dry-run] [--json]` takes open `drive_failed` rows and
launches at most one resume per recipe per mirror SHA: the recipe argv + `--skip-upload --no-youtube-upload`
(each added only when the recipe lacks it), via
the job runner (`--name gms-heal-<drive>-<sha7> --heavy --detach --stall 86400`). A failed heal job becomes an
incident row of its own, but it shares the drive's recipe, so it waits for new code like the
first row. It is an
audit-only resume; publish is never launched here. It never passes `--phase`, the drive CLI's
single-phase mode: `--phase A` would pass phase A and then fail the DoD gate (`lf_count 0 < 1`)
because B-D never ran. The plain resume runs every phase the checkpoint does not hold, so a
failure in any phase heals, and a failed phase that cannot be mapped no longer refuses. When the
checkpoint beside the recipe (`pipeline_state.json`) holds a phase whose stamped code version is
no longer current (see `PHASE_CODE_VERSIONS`), that plain resume re-runs the stale phases and the
phases built from them, reusing renders by fingerprint. That attempt records `stale_phases`. No
readable recipe is `refused` (argv is never guessed). Each attempt is appended to `resume_attempts` as
`{at, mirror_sha, slot, action: launched|skipped|refused, reason, job_name}` (+ `outcome`,
`outcome_at` once the heal job ends); an unchanged repeat skip is not re-recorded.

**Operator hold.** A file named `no-auto-resume` beside the recipe (`<stages dir>/no-auto-resume`,
contents free-form notes) makes heal refuse that drive on every mirror SHA, dry run included, with
reason `operator hold: no-auto-resume beside the recipe`. Use it for a run authorized for one
attempt only: guard (a) alone would relaunch a failed drive after each new staging landing.
Delete the file to hand the drive back to heal.

**Code-root binding.** A resume runs the code it records as `mirror_sha`, never the recipe
writer's: the recipe's interpreter is replaced by `<code root>/.venv/bin/python`
(recorded as the attempt's `interpreter`) and the cwd is the code root, not the recipe's `cwd`.
Host-only overrides (`--allow-cross-fs-source`) are stripped; a resume re-runs the source checks.
Stripping it cannot refuse a resume: source footage on another filesystem than staging is never
refused (it logs a `PERFORMANCE:` WARNING instead of INFO); only dangling or unreadable source
files are, with or without the flag.
A recipe whose argv is not `<python> -m gomoveshift …`, or a code root with no service
interpreter, is `refused` with the reason.

**One drive.** `health heal --recipe <stages dir>/resume.json` heals that drive only: it registers
the recipe as a `drive_failed` row keyed `drive_failed:<drive>:recipe` (reusing an existing row, so
attempt history is kept; a resolved row is reopened, stamped `reopened_at`, so an operator's
`--recipe` is never a silent no-op), then runs the pass above restricted to that key. For a drive no
`health check` saw fail, e.g. an evidence run. An unreadable recipe exits 2 and writes nothing;
with `--dry-run` the row is registered in a scratch copy and the incidents dir is untouched.

**By hand, from the recipe.** `gomoveshift drive --from-recipe <path-or-project>` is the manual
relaunch that never retypes the environment. The argument is the recipe file, a `_stages` or
project dir holding one, or a bare project name resolved to the one live recipe under the stages
root (`GMS_INTERMEDIATE_DIR`, else the cache home; `<root>/<project>/_stages` and
`<root>/<run>/<project>/_stages`, never `_superseded`; none or several refuse, listing them).
Before argparse runs, the recipe's argv becomes the command line (`--fresh` was already stripped
when it was written) and its recorded `GMS_INTERMEDIATE_DIR` is set in the process, so
`_stages_base` lands on the dir the run used — the same env, not a second lookup. Only `--phase`,
`--publish`, `--lf-day`, `--lf-publish-date`, `--skip-upload`, `--no-youtube-upload`, `--force` and
`--no-tmux` may be added; a directory, `--fresh` or any other flag refuses naming it, as does an
environment that already sets a different `GMS_INTERMEDIATE_DIR` ("unset it or drop
--from-recipe"), a recipe that is not a `<python> -m gomoveshift drive …` run, and an unreadable
one. Refusals exit 2 before anything is touched; a relaunch rewrites `resume.json` through the
normal path. 2026-11-08: a hand relaunch of nctopa-2026-03-05-rec0063 with a retyped env chose a
fresh stage dir and orphaned the earlier 139 GB one (`cleanup <project> --abandoned` reclaims it).

A `--dry-run` pass launches nothing and writes no row, but each would-launch attempt in
`attempts` carries the exact `argv` (the job-runner wrapper and the rebound drive command), `cwd`,
`interpreter`, `job_name` and `mirror_sha`, so the resume can be graded before it runs.

Guards (each a recorded skip): (a) this SHA already has a launched attempt; (b) a drive job is
running (job/1 record with no result, or `pgrep`); (c) the publish slot has passed or is unknown
(the recipe's `--lf-publish-date`, else the soonest free slot of `gomoveshift schedule`, recorded
as `slot`); (d) a metadata-gate failure (gate matches `/metadata/i`) when
`git diff --name-only <staging_sha> <mirror sha> -- GENERATOR_PATHS` is empty
(`health/heal.py`: the title/longform-metadata/content-validator modules); (e) one push at launch (`health_heal_launched`) and one at
outcome (`health_heal_outcome`), the latter observed on the next heal pass.

## `health backlog`: launch the next backlog day

`gomoveshift health backlog --region R --stages-root S --output-root O [--sources-root P]
[--dry-run] [--json]` launches at most one audit-only drive for the region's next uncovered LF
slot. It is run by an operator or an agent session; no cron calls it. Exit 2 on `refused`, else 0.

**Selection.** The region's free LF slots in the next 7 days (from `build_schedule_report`) are
sorted; every `ready` or `auditing` drive in `schedule.sqlite3` covers one, and the first slot
left over is the target. An `auditing` drive for the region, or a drive job already running
(job/1 record with no result, or `pgrep`), is a `noop`: a repeat call never starts a second run.
An `auditing` drive whose newest `gms-backlog-<project>` job record ended in anything but
`succeeded` (for example `failed` on an ffmpeg error) is `refused`, naming the run id and exit
code: the failed audit still holds its slot, so the operator relaunches it (the drive resumes
from its stages) or supersedes it (below); a relaunched job that is running is a `noop` again.
An audit whose newest job stopped at the OST audit gate (exit 5, `DRIVE WAITING:`; every
placeless drive does, since a no-GPS LF gets no system review) is not failed: it is a `noop`
naming each `ost_audit_<date>.json` sidecar under its stages and the `.confirmed` marker to
create. Once every sidecar has its marker and one marker is newer than the job's end, the next
run resumes the drive from its `resume.json` (rebound to the code root's venv, audit only, under
the same `gms-backlog-<project>` job name; nothing is registered, the store row is already
`auditing`). Markers all older than the stop did not pass the gate (the audit changed), so the
drive stays a `noop` asking for a re-review instead of re-resuming every tick. A `no-auto-resume`
hold keeps a confirmed drive where it is; `--supersede` sets a waiting audit aside like a failed
one, and a waiting audit with no backlog row is never retired.
A failed audit whose project no longer has a `drive_backlog.yaml` row for the region can be
neither, so it is retired instead: its store row is set `failed`, the result lists it under
`retired` with the failed job, and selection goes on to the next row (`--dry-run` lists it and
writes nothing). A retirement also writes a `no-auto-resume` hold beside each of the drive's
`resume.json` recipes under the stages root, so `health heal` refuses it on every later mirror
SHA instead of re-resuming a drive that can never pass. A drive whose store row is already
`failed` and has no row (retired before holds existed, or set `failed` by hand) gets the same
hold on the next run; the result lists every stage dir held under `held`, and an existing hold
is left as it is. An audit with a row, one still running,
or one with no job record is never retired. Retiring and holding launch nothing, so they run
before the one-heavy-drive check and happen even while a drive runs. On 11-08, heal's own resume
of the dead 03-04 drive started two seconds before each hourly backlog, so the hold was never
written. The text output, which is what the cron logs, prints `retired <project>: <reason>` and
`held <stages dir> (no-auto-resume)`, or `would retire` and `would hold` under `--dry-run`.
The row is the first `drive_backlog.yaml` row for the region that is not skipped. Each skip is
reported with its reason: already in the store (with its status), already in the content ledger,
a missing source file, a missing `day_gpx` track, a `placeless_label` that claims something, typed
labels (`per_lf_locations` / `location`) the drive's own `--per-lf-location` rules would refuse
(`location labels: <the parser's reason>`), an operator hold (a `no-auto-resume` file under `<stages root>/<project>`,
or a held earlier stage whose sources overlap the row's), or every source already staged by an
earlier run's `resume.json`.

**Storage.** Need is summed per physical device (`st_dev`), so two roots on one drive share one
free-space figure. Staging needs `max(30 GB, sources × 1.5)` and output 50 GB; free equal to need
passes. The ×1.5 is the existing preflight contract (`estimate_peak_cache_gb`), not a bound
validated independently of it, and retained artifacts (earlier stages, outputs, the day masters
they keep) count only through the measured free space. A launch still runs `run_preflight` with
the row's own sources before the job starts, under the code root's drive interpreter (a child
`python -c`, cwd the service dir) so tool checks such as gopro-overlay test the venv that will
render; its blockers are `refused`, and a crashed child is an error, never a pass.

**Render gate.** Before anything is moved, launched or registered, and in a dry run too, the row
is checked against the drive's own pre-ingest refusal (`day_under_part_min_message`) under the
row's `ingest_mode` and `lf_mode`: a `full_drive` day whose raw footage (with its neighbouring
calendar days) is under `full_drive_part_min_sec` is skipped with `render gate: <message>`, and the
next eligible row is tried, so one short day never stalls the region. When no row behind it is
eligible, or the refused row is the `--supersede` target, the result is `refused` with that reason.
The scan reads a throwaway dir of links to the row's sources until its subset dir exists.

**Launch.** Under the `gms-backlog-launch` lease (the script named by `$GMS_LEASE_SCRIPT`, default `bin/lease.sh`; held → `noop`, released in a
`finally`), it writes the row's chapters as a symlink dir `<sources root>/<project>` (reused when
identical, `refused` on any mismatch, never rewritten; the day-GPX slicer's own `<stem>.gpx`
beside a chapter, which a `--day-gpx` drive writes there before ingest, is not a mismatch), then runs
the job runner (`--name gms-backlog-<project> --heavy --detach --stall 86400 --expect
'O/<project>/**/shorts/**/*.mp4' -- <code root venv
python> -m gomoveshift drive <subset> --region R --publish D --lf-day D --ingest-mode --lf-mode
… --output-dir O/<project> --skip-upload --no-youtube-upload --ost-system-review --no-review
--force --no-tmux`) with `GMS_INTERMEDIATE_DIR=S/<project>`, and registers the drive `auditing`
in the store. A row with a `day_gpx` (footage with no GPS of its own) adds `--day-gpx <track>
--camera-utc-offset-hours <N>` from the row; Waypost refuses a row that sets one without the
other. A row with `per_lf_locations` (footage whose place the camera never recorded, with the
operator's label typed per LF into the row) adds `--per-lf-location <i>=<label>` per entry and
`--location <label>` when the row sets one, so nothing is typed at launch time; the labels are
checked against the drive's own rules before any render (`location_labels_skip_reason`) and the
drive still waits for a person at the OST audit gate, since no GPS gives it a system review. The `--expect` makes an exit 0 that wrote no fresh shorts `failed-no-output`. Uploads stay off; publishing still needs the normal approval and re-run.
`--dry-run` takes no lease, writes nothing and prints the exact argv, env and storage table.

**Supersede.** `--supersede <project> --reason <text>` replaces that row's own earlier run, for
example a render made on an older release: without it the row is skipped as already staged, and a
relaunch would resume the stale stages instead of rendering fresh. The project must be the
region's next eligible row once its own stage root is set aside; a row in the store or the content
ledger, or an output whose `youtube_uploads.json` holds
records (or is unreadable) is `refused`, and so is an empty reason. `<stages root>/<project>` and
`<output root>/<project>` are renamed (one device, no copy) to `<root>/_superseded/<project>-<UTC
stamp>/`, each with a `superseded.json` (`backlog-supersede/1`: project, reason, time, moves, the
old recipe's `sha`/`started_at`/`phase_reached`/`argv`, and whether a `no-auto-resume` hold sat
under it). A hold under the row's own root does not block the supersede; it moves with the run.
A failed audit (the row's store entry is `auditing` and its newest job did not succeed) can be
superseded too: its run is set aside the same way and the row relaunches as the manifest now reads
it (for example a changed `lf_mode`), re-registering the same store row; the result names the
failed job under `relaunch`. An audit that is still running, a `ready` row, or a failed audit when
`--supersede` names another row is never replaced.
A dead sibling (an earlier run of the same footage under another project name, found by
overlapping recipe sources) is set aside the same way, under its own name, and listed in the same
`moves`; a held sibling, one in the store or ledger, or one whose output holds upload records is
`refused` naming it.
Nothing moves until every check up to the release-pin check has passed, so a refusal leaves the
old run in place; `--dry-run` prints the planned moves. `_superseded/` is never read as staging.

## `health publish`: one command to upload an audited drive

`gomoveshift health publish <project> --region R --stages-root S [--jobs-dir J] [--code-root C]
[--lead-hours N] [--expect-sha SHA] [--dry-run] [--json]` is the single operator approval for
uploading a drive `health backlog` audited. It goes through Waypost's claim, so the schedule store
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
   publish job, is left alone; a repeat pass is a no-op. It then retires every `ready` drive
   that already holds an upload receipt (`publish.retire_uploaded`, the same `retire_published`
   call). The closed claims and retirements are `publish_reconciled` in `run_check`'s result.

`--dry-run` runs every check, then reports the four steps and Waypost's own dry-run answer
(`next_claim`); it writes nothing and launches nothing.
