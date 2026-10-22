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
2. **LF slot inside 48 h with nothing booked while a drive is ready.** A free slot is a
   `next_available` row of `build_schedule_report` (the scheduler already excludes booked slots);
   "ready" is `ScheduleStore.ready_depth(region) > 0` over `<seo.storage_path>/schedule.sqlite3`.
   No store file means no readiness signal: the rule is reported as not evaluated, not guessed.
3. Not in this change (route derivation belongs to `health heal`).
4. **Dispatch stalled.** The queue (`social_queue.sqlite3`, opened read-only) has DUE work
   (non-shadow, non-terminal, `scheduled_at` null or past, platform not paused) and no delivery
   row has `updated_at` within 24 h. Every dispatch attempt rewrites `updated_at`, so a quiet queue
   with due work means the drain is not running. Row file: `queue_stalled.json`.

A relative `seo.storage_path` skips rules 2 and 4 with a note (it would resolve against the cwd).

## Push and idempotency

Each row carries a stable `key` and `pushed`. A push that returns `False` (attempted, failed) leaves
the row unrecorded and the next pass retries it. A condition that clears sets `resolved_at`.

## Incident row, schema `health-incident/1`

Common: `schema`, `kind` (`drive_failed` | `lf_slot_unbooked` | `queue_stalled`), `key`, `file`,
`detected_at`, `pushed` (bool), `pushed_at`, `resolved_at` (null while open).

`drive_failed`: `drive`, `job_name`, `phase` (or null), `gate` (or null), `error`, `log_path`,
`job_run_id`, `failed_at`, `staging_sha`, `resume_recipe` (path of the drive's `resume.json`, or null), `resume_attempts`
(`[]`, owned by `health heal`).
`lf_slot_unbooked`: `region`, `slot`, `ready_drives`. `queue_stalled`: `due_deliveries`,
`last_updated_at`, `quiet_hours`.

## Resume recipe and `health heal`

**Recipe.** At start (`cmd_drive`, before the orchestrator runs) a drive writes
`<stages dir>/resume.json` (schema `resume-recipe/1`; under the cache `_stages` dir, never
tracked): `argv` (`python -m gomoveshift` + the drive's own args, `--fresh` removed), `cwd`, `sha`
(git HEAD of the code root, null if unknown), `started_at`, `phase_reached` (updated by the
orchestrator's `_save_state` as phases complete). `env` records `GMS_INTERMEDIATE_DIR` when it was set: the staging
override decides where the saved state lives, so `health heal` replays it as an `env K=V` prefix on
the jobrun argv (shown in the dry run). No other variable is recorded or replayed. It also logs one stable line,
`resume recipe: <path>`; `health check` parses that line from the drive log into the incident's
`resume_recipe`, and uses its directory as the stages dir that links sibling runs. A
single-`--phase` run logs the line but does not rewrite the file (so a heal run can't overwrite
the full recipe it came from). A discovery-only run (`--dry-run`, `--list-lfs`,
`--label-dry-run`) renders nothing, so it neither writes the recipe nor logs the line, and its
saves never move `phase_reached`: an existing recipe stays byte-identical. Writing never fails a
drive (warning only).

**Heal.** `gomoveshift health heal [--dry-run] [--json]` takes open `drive_failed` rows and
launches at most one resume per incident per mirror SHA: the recipe argv + `--phase <A|BC|D>`
(from the failed phase) + `--skip-upload --no-youtube-upload`, via
the job runner (`--name gms-heal-<drive>-<sha7> --heavy --detach --stall 86400`). It is an
audit-only resume; publish is never launched here. No readable recipe, or an unmappable phase,
is `refused` (argv is never guessed). Each attempt is appended to `resume_attempts` as
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
attempt history is kept), then runs the pass above restricted to that key. For a drive no
`health check` saw fail, e.g. an evidence run. An unreadable recipe exits 2 and writes nothing;
with `--dry-run` the row is registered in a scratch copy and the incidents dir is untouched.

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
from its stages) or clears the row; a relaunched job that is running is a `noop` again.
The row is the first `drive_backlog.yaml` row for the region that is not skipped. Each skip is
reported with its reason: already in the store (with its status), already in the content ledger,
a missing source file, an operator hold (a `no-auto-resume` file under `<stages root>/<project>`,
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

**Launch.** Under the `gms-backlog-launch` lease (the script named by `$GMS_LEASE_SCRIPT`, default `bin/lease.sh`; held → `noop`, released in a
`finally`), it writes the row's chapters as a symlink dir `<sources root>/<project>` (reused when
identical, `refused` on any mismatch, never rewritten), then runs
the job runner (`--name gms-backlog-<project> --heavy --detach --stall 86400 --expect
'O/<project>/**/shorts/**/*.mp4' -- <code root venv
python> -m gomoveshift drive <subset> --region R --publish D --lf-day D --ingest-mode --lf-mode
… --output-dir O/<project> --skip-upload --no-youtube-upload --ost-system-review --no-review
--force --no-tmux`) with `GMS_INTERMEDIATE_DIR=S/<project>`, and registers the drive `auditing`
in the store. The `--expect` makes an exit 0 that wrote no fresh shorts `failed-no-output`. Uploads stay off; publishing still needs the normal approval and re-run.
`--dry-run` takes no lease, writes nothing and prints the exact argv, env and storage table.
