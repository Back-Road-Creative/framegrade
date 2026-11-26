## 190 - Health backlog: launch the next backlog day

Split out of [156-health-check.md](156-health-check.md) (the health check, its rules and `health heal`).

`gomoveshift health backlog --region R --stages-root S --output-root O [--sources-root P]
[--dry-run] [--json]` launches at most one audit-only drive for the region's next uncovered LF
slot. The hourly `bin/gms-health-cron.sh` tick (workspaces) calls it per region; an operator may too. Exit 2 on `refused`, else 0.

**Failed row, no resume (relaunch from source).** `health heal` resumes only a drive with a readable
`resume.json`, so a `failed` store row (or an `auditing` one whose newest audit failed or has no job
record) whose stage dir holds none (rec0060-0061: `calibration.json` only) was reached by nothing.
`launch.FailedRelaunch` asks `retry_policy.refusal` (the same class, fingerprint and budget rules as heal) whether a
relaunch is due: the failure's fingerprint (`failure_fingerprint` on its `drive_failed` incident row) against the one now,
a legacy row (no stamp, or no incident row) getting one launch. Due: the stale stage and output dirs move to
`_superseded/` (the `--supersede` helper), the row is re-registered `auditing` and it launches from source on that tick,
no operator command. Not due: nothing moves, and one `refused` attempt (deduped per fingerprint and reason) lands in
`resume_attempts` of the project's `backlog_relaunch` incident row, written already resolved, as is each launch. Never a
published row (upload receipt), a booked or `ready` row, or one in the content ledger; a drive running still blocks it
(one heavy drive at a time), and a running audit is never replaced. `--dry-run` plans the moves and writes nothing.

**Selection.** The region's free LF slots inside the render horizon (from `build_schedule_report`) are
sorted; every `ready` or `auditing` drive in `schedule.sqlite3` covers one, and the first slot
left over is the target. An `auditing` drive for the region, or a drive job already running
(job/1 record with no result, or `pgrep`), is a `noop`: a repeat call never starts a second run.
The render horizon is `launch.render_horizon_h`: the wider of 7 days and the cadence's
`lead_hours` (the packaged `brand.yaml` sets 1680 h, 70 days). A finished drive can claim a
slot only inside `lead_hours`, so the horizon matches it: each drive renders, publishes into the
next free slot and frees its scratch, and the next tick launches the next row, instead of one
drive a week. The `covered` noop names the horizon in hours.
An `auditing` drive whose newest audit job record (`gms-backlog-<project>`, or a `health heal`
resume of it, `gms-heal-gms-backlog-<project>-<sha7>`, nested once per resume, or a `health heal
--recipe` resume named by the stages dir, `gms-heal-<project>-<sha7>`) ended in anything but
`succeeded` (for example `failed` on an ffmpeg error) never holds its region: the tick walks the
region's `auditing` rows in order, lists each failed one under `failed_audits` (project, run id,
exit code; the text output prints `failed audit <project>: <reason>`), and goes on. The row stays
`auditing`, `health heal` keeps resuming it from its stages, and once its newest audit
`succeeded` the publish path below takes it. Behind it, an audit in progress, one at the OST gate
or a passed one is handled as below, and when every `auditing` row is failed the next eligible
backlog row launches (exit 0). Only `--supersede` of another row still stops at a failed audit
(`refused`); superseding the failed row itself is unchanged (below).
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
a missing source file, a missing `day_gpx` track, any `placeless_label` (a last resort: give `bracket_gpx`), typed
labels (`per_lf_locations` / `location`) the drive's own `--per-lf-location` rules would refuse
(`location labels: <the parser's reason>`), an operator hold (a `no-auto-resume` file under `<stages root>/<project>`,
or a held earlier stage whose sources overlap the row's), or every source already staged by an
earlier run's `resume.json`.

**Publish.** An `auditing` drive whose newest audit job `succeeded` is handed to `health publish`
(`184-health-publish.md`) by the same call, with the call's own `--stages-root`, `--jobs-dir` and `--code-root` and
Waypost's default lead window: the hourly tick publishes each audited drive once its slot is inside
that window. `launched` is `published: <project> for <slot>`. A claim miss (`no claim: ...`, for
example `outside_lead_window`) is a `noop`, `audit passed: <project> waits for its Waypost slot:
...`, and the drive stays `auditing` until a later tick. Any other publish refusal (a stale
checkpoint, no long-form plan, an older `ready` drive) is `refused`, `publish refused: ...`. A
running audit, one with no job record, a failed one and one at the OST gate are never published
(a failed one is skipped, not a stop);
`--supersede` of the drive itself is unchanged. `--dry-run` asks `health publish --dry-run`.

**Failed publish.** A drive still in the backlog whose store row is `failed` because its newest
`gms-publish-<project>` job ended in anything but success (`failed`, or `cancelled`, `stalled`,
`killed` with no exit code; its audit succeeded) is handed to `health publish` again, before
the `auditing` drives; `health publish` reopens it (`failed -> auditing -> ready`) and claims a slot.
A `failed` row with no failed publish job (set by hand, or a publish still running) is left alone.
Before, the tick skipped it `in the schedule store (failed)` and the audited drive never aired
(Kerry-to-Cork, 2026-11-19). The retry first keyed on `failed` or a non-zero exit code, so a publish
job the runner ended with no exit code was never retried (unproven as Kerry's exact end state). It
retries the PUBLISH only (`publish_only` class): never a render, and never a drive that already
holds an upload receipt. Tests: `tests/test_health_backlog_launch.py`.

**Storage.** Need is summed per physical device (`st_dev`), so two roots on one drive share one
free-space figure. Staging needs `max(30 GB, sources × 3.0)` and output 50 GB; free equal to need
passes. The ×3.0 is preflight's `PEAK_CACHE_FACTOR` (`estimate_peak_cache_gb`): the episode
concats plus the day's raw concat and master that the per-day encode gate asks for, so a drive
the encode cannot fit is refused here instead of failing hours in. Retained artifacts (earlier
stages, outputs, the day masters they keep) count only through the measured free space. A launch still runs `run_preflight` with
the row's own sources before the job starts, under the code root's drive interpreter (a child
`python -c`, cwd the service dir) so tool checks such as gopro-overlay test the venv that will
render; its blockers are `refused`, and a crashed child is an error, never a pass.

**Render gate.** Before anything is moved, launched or registered, and in a dry run too, the row
is checked against the drive's own pre-ingest refusal (`day_under_part_min_message`) under the
row's `ingest_mode` and `lf_mode`: a `full_drive` day whose raw footage (with its neighbouring
calendar days) is under the 480 s minimum long-form (`min_lf_duration`; a day under
`full_drive_part_min_sec` but over it is planned as `best_moments`, not refused) is skipped with `render gate: <message>`, and the
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
drive still waits for a person at the OST audit gate, since no GPS gives it a system review. The `--expect` makes an exit 0 that wrote no fresh shorts `failed-no-output`. Uploads stay off; the publish is a separate Phase D job, launched by a later tick once the audit has passed (**Publish.** above).
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

**Replace scheduled uploads.** `--replace-scheduled` re-renders a drive whose uploads are still scheduled and swaps the new render in: see [185-replace-scheduled-uploads.md](185-replace-scheduled-uploads.md).
A dead sibling (an earlier run of the same footage under another project name, found by
overlapping recipe sources) is set aside the same way, under its own name, and listed in the same
`moves`; a held sibling, one in the store or ledger, or one whose output holds upload records is
`refused` naming it.
Nothing moves until every check up to the release-pin check has passed, so a refusal leaves the
old run in place; `--dry-run` prints the planned moves. `_superseded/` is never read as staging.
