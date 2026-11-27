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
   its `expect` log path or its stages dir (audit failed, publish succeeded). A later succeeded
   heal resume OF that job (`gms-heal-<job name>-<sha7>`, nested once per resume) resolves it by
   name alone: a resume can run under a moved scratch root, sharing no log or stages dir.
   A job that ended exit 5 (`EXIT_WAITING_AT_GATE`) with `DRIVE WAITING:` as its log's last
   closing line stopped at a human gate (the OST audit) and is not a failure: no row is filed, an
   open one is resolved, so `health heal` never re-resumes it onto the same gate. `health
   backlog` resumes it once the gate is confirmed ([190-health-backlog.md](190-health-backlog.md)). The `DRIVE WAITING:` line ends with `[gomoveshift <version>]`, the code that stopped. When the
   drive's recipe carries `--ost-system-review` and that version differs from the installed one (or the line
   records none, a legacy stop), `health backlog` resumes it once per installed version so the gate re-checks;
   it never writes or bypasses a `.confirmed` marker, and the gate inside the drive decides.
2. **LF slot inside 48 h with nothing booked while a drive is ready.** A free slot is a
   `next_available` row of `build_schedule_report` (the scheduler already excludes booked slots);
   "ready" is `ScheduleStore.ready_depth(region) > 0` over `<seo.storage_path>/schedule.sqlite3`.
   No store file means no readiness signal: the rule is reported as not evaluated, not guessed.
3. Not in this change (route derivation belongs to `health heal`).
4. **Dispatch stalled.** The queue (`social_queue.sqlite3`, opened read-only) has DUE work
   (non-shadow, non-terminal, `scheduled_at` null or past, platform not paused) and no delivery
   row has `updated_at` within 24 h. Every dispatch attempt rewrites `updated_at`, so a quiet queue
   with due work means the drain is not running. Row file: `queue_stalled.json`.
5. **Placeless label in the backlog.** A `drive_backlog.yaml` row carries a `placeless_label`.
   The launcher refuses every such row (`launch.placeless_label_skip_reason`): a placeless label
   is a last resort, and the better answer is `bracket_gpx` (the fixes files of the drives before
   and after). The row never launches (the rows behind it still do) and the incident stays open
   while the yaml carries the label; replacing it with `bracket_gpx` closes it. `claims` lists any
   scenery, weather, time-of-day or road-kind words in the label. An
   absent or unreadable backlog closes nothing. Row file: `placeless_label_<project>.json`.

   A `drive_failed` row that `health heal` closed with a `resolution_reason` (a recipe-less
   publish or heal job, never resumable) stays closed: the check does not re-file it from the same
   failed job. A row resolved without a reason is re-filed while its job is still failed.

6. **Failed drive left unattended.** `health heal` retries every open `drive_failed` row on a
   backoff capped at 24 h; this rule is the independent alarm should anything still let one sit. An
   open `drive_failed` row is unattended when its newest `launched` attempt (the newest across every
   open row sharing its `resume_recipe`) is older than `UNATTENDED_AFTER_H` = 26 h (the cap plus two
   cron ticks), or it has no attempt and `failed_at` is that old, or its newest attempt is `refused`
   (heal can never relaunch it). It is also unattended, however recent the last launch, when the
   drive's last `UNATTENDED_AFTER_FAILURES` = 3 launches in a row failed (a `succeeded` launch resets
   the count; a refusal, a running launch and a pre-flight failure neither add nor reset), or when
   the retry policy (`retry_policy.refusal`) says the budget of the failure class is spent under the
   fingerprint the row was filed with and the newest launch failed (publish-only failures and rows
   filed before fingerprints are exempt). The incident text names which condition fired and the
   count. One `failure_unattended` incident per drive: its heal resumes
   (`gms-heal-<drive>-<sha7>`, nested) and the rows sharing its `resume_recipe` are that drive, so
   they raise one alarm named for the original drive (key `failure_unattended:<drive>`, row file
   `failure_unattended_<drive>.json`). It names the drive, the failed phase/gate, the last attempt time and why. It prints as `[failure_unattended] ...`, is
   pushed once (`health_failure_unattended`) and resolves on a later check once a newer launch lands
   or the drive resolves. An operator hold (`no-auto-resume` beside the recipe) is deliberate: it is
   reported as a `held` note, never an incident. A job the drive's tracked-launcher gate refused
   (its log carries `run_origin.LAUNCHER_GATE_REFUSAL`) never started its drive, so it files no
   `drive_failed` row and raises no alarm: the drive is still the backlog's to launch.
7. **Queued Short the review refuses.** One `queued_short_refused` incident per queued,
   non-shadow YouTube delivery whose slot is within `QUEUED_REFUSED_WITHIN_H` = 48 h of now (or
   past) and whose final-text review (`short_review_refusal`, run on the row's sidecar) still
   refuses it after `dispatch-queue`'s repair: it would fail at its slot and the slot is never
   refilled. Key `queued_short_refused:<delivery id>`; pushed once (`health_queued_short_refused`);
   resolves once the review agrees or the row leaves `queued`. Read-only (queue opened `mode=ro`).
   A sidecar that cannot be read (drive offline) is a note, never an incident.

8. **YouTube delivery failed permanently.** One `delivery_failed_permanent` incident per
   non-shadow YouTube delivery that ended `failed_permanent` within
   `FAILED_PERMANENT_NEW_WITHIN_H` = 72 h (`updated_at`) and has no recovering row. JP's kept rows
   (`JP_KEPT_FAILED_PERMANENT_IDS`: 98, 102, 106, 110, 114) never raise. Key
   `delivery_failed_permanent:<delivery id>`; pushed once (`health_delivery_failed_permanent`);
   resolves when the row is recovered (`dispatch-queue` requeues a review refusal that now
   passes) or ages out. Read-only (queue opened `mode=ro`).

9. **Fast guard for a bad Shorts run** (`gomoveshift/health/bad_run.py`). The newest
   `GUARD_N` = 3 settled YouTube Shorts (settled as in `tracking/day7.py`: posted + 7 d <= today - 2 d,
   ordered by `posted_at` across every tracked project) each have day-7 views under `LOW_FRACTION`
   = 0.25 of the trailing median of the `TRAILING_N` = 10 settled Shorts before them (at least
   `MIN_BASELINE` = 5 measured, else not evaluated). The measured base rate, 4 of the last 10
   under 5 views, trips it against any median above 20; one good Short in the window breaks the
   run. A Short with no settled series counts as low (the API omits zero-view days). One
   `short_bad_run` incident (key `short_bad_run`, row file `short_bad_run.json`) lists each Short
   with `causes`, each `{code, detail, c5_fixable}`, from recorded data only: `delivery_status`
   (newest queue delivery not `published`), `duplicate_place_text` (`c5_fixable: true`: unit C2's
   composer fixes it for new plans, and the later queued-row re-write unit acts on that flag),
   `late_publish` (`published_at` over 3 h after `scheduled_at`), `bunched_publish` (another Short
   within 1 h), `place_county_mismatch` (place text names a county the sidecar's
   `sampled_regions` did not measure; skipped without a sidecar county) and
   `missing_core_metrics`. YouTube's own processing/privacy state is not recorded, so it is not
   swept. Detection only: nothing is edited. It resolves on a later check once the run is broken.
   Pushed once as `health_short_bad_run`. `run_check(tracking_root=...)` overrides the tracking
   store root (default `gms_output_dir()`).

10. **Store project in no backlog row** (`backlog_store_orphan`,
   `gomoveshift/health/accounting.py`). A `ready_drives` project in `schedule.sqlite3` (read
   through `ScheduleStore.all_drives`) that is in no `drive_backlog.yaml` row and in no
   `drive_backlog_excluded.yaml` entry's `project`. Row file `backlog_store_orphan_<project>.json`
   (`project`); pushed once as `health_backlog_store_orphan`. A missing or unreadable store, or an
   unreadable manifest, is a note, never a finding.
11. **Camera footage in no backlog row** (`backlog_unlisted_footage`). For every folder holding at
   least one source of a manifest row or an exclusion entry, the camera files (case-insensitive
   `.mp4`, `.insv`, `.mov`; files only, not recursive) that no row's `sources` and no exclusion's
   `sources` names. One incident per file (key `backlog_unlisted_footage:<absolute path>`, fields
   `directory`, `name`, `path`; row file `backlog_unlisted_footage_<safe name>_<10 hex of
   sha256(key)>.json`, the hash so `/a_b/x.mp4` and `/a/b_x.mp4` never share a file), so each
   opens and resolves on its own. New ones push as ONE `health_backlog_unlisted_footage` per tick
   when there are several (count plus up to `check.FOOTAGE_PUSH_NAMES` = 10 names, then "and N
   more"; a lone file keeps its own push); every row still names its own file. Directory listing only: a media file is never opened,
   probed or decoded. Paths compare after `os.path.normpath`, and case-folded under
   `/mnt/<single letter>/` (a Windows drive). A folder that cannot be listed (D: offline: a
   listing error, a `/mnt/<letter>` with nothing mounted, or no answer within
   `accounting.LIST_DEADLINE_S` = 20 s for the whole tick, since the cron tick holds a lock with
   no timeout) is a note naming it, never a finding and never "empty": its open incidents stay
   open, so a drive that goes offline and returns does not close and re-notify them.

**`drive_backlog_excluded.yaml`** sits beside `drive_backlog.yaml`
(`paths.drive_backlog_excluded_path()`). It is a separate file on purpose: the backlog loader and
`seo-manager schedule backlog-next` never read it, so an exclusion can never launch. A top-level
list (empty is valid; a missing file is the same); each entry is a mapping with `sources` (non-empty
list of absolute path strings), `reason` (non-empty string: the measured reason and where it was
measured) and optionally `project` (a store project deliberately not a manifest row); no other
keys. A source is an absolute POSIX path (a Windows `D:\...` path is refused), appears once across
the file and in no manifest row (compared as in rule 11). An entry that breaks this opens a
`backlog_exclusion_invalid` incident (key `backlog_exclusion_invalid:<the entry's first source
path>`, or the entry number when it has no usable source; fields `entry`, `detail`) and excludes
nothing, so its footage still shows under rule 11 (the folders of an invalid entry's absolute
sources are still listed, but account for nothing, so open incidents there stay open). An unreadable file or a non-list top level opens
`backlog_exclusion_invalid:file` and rules 10 and 11 are not evaluated that tick (a note; their open
incidents stay open, as do open per-entry `backlog_exclusion_invalid` incidents: only the `:file` finding is live)
instead of flooding every excluded file in as a finding.

**Backlog launch refused for space.** Not a check rule: `health backlog` writes it. A launch
refused with `storage:` (a device's free space is under the staging plus output need,
`launch.storage_plan`) opens one `backlog_storage_refused` row per region (key
`backlog_storage_refused:<region>`, row file `backlog_storage_refused_<region>.json`, fields
`region`, `project`, `reason`), pushed once as `health_backlog_storage_refused`. A later
`health backlog` tick for the region that launches, or that measures every device with room,
sets `resolved_at`; a refusal for another reason, or a tick that never reached the space check
(lease held, running drive), leaves it open. `health check` never resolves it. A dry run writes
nothing.

A relative `seo.storage_path` skips rules 2, 4, 7, 8, 9, 10, 13 and 15 (it would resolve against the cwd). The cadence rules' note names rules 2 and 4; rule 10 adds its own note. Rules 10 and 11 run independently: one raising is noted by name and does not skip the other.

12. **Nightly Short text sync** (`gomoveshift health sync-short-text`,
   `gomoveshift/health/short_text_sync.py`; not part of `health check`, which never runs it).
   `short_text_sync_refused` (key `short_text_sync_refused:<project>`, row file
   `short_text_sync_refused_<project>.json`, `shorts` = each refused Short's `delivery_id`,
   `short_index`, `reason`) and `short_text_sync_crash` (key `short_text_sync_crash`, `detail` =
   the exception). Pushed once (`health_short_text_sync_refused`, `health_short_text_sync_crash`);
   a refusal row resolves on the first later run that no longer refuses THAT project (other
   projects' rows stay open), the crash row on the first run that completes; a crash leaves
   refusal rows open. Every refusal files a row except the named transient ones (a sibling slot
   still ahead inside the lead time or past on a row under the dispatcher's retry backoff, a live lease, a write race or busy database, or the cascade
   without a clash); a sibling slot already past but still queued (past its retry backoff), and an old title with no stamp,
   are permanent, so they file. A Short refused on 3 consecutive runs (an actionable one counts, the latest must be transient)
   (`TRANSIENT_ESCALATE_RUNS`) files too, its reason naming the run count; the count is one row
   per Short in `<incidents dir>.streaks` (not an incident: never listed or pushed), reset by the
   first run that no longer refuses it (a crash, or a run that finds the store lock held, neither counts nor resets: runs, not nights).
   `health check` lists these rows (an open one keeps it at exit 1) but never writes or pushes
   them; the sync takes a non-blocking file lock on `<incidents dir>.lock` around its store write.
   Every run (a crash and a held store lock included) also writes the run record
   `<incidents dir>.short_text_sync_run.json` atomically (`{"schema": 1, "ran_at", "changed",
   "refused", "crashed"}`; a write failure is logged, never changes the exit code); rule 13 reads it. Details: `183-queue-sync-short-text.md`.

13. **SEO feedback gone stale** (`seo_feedback_stale`, `gomoveshift/health/seo_feedback.py`).
   Three conditions, each one stable key and its own resolution. `seo_feedback_stale:decisions`:
   `shorts_arm_decisions.json` missing, unreadable or older than `DECISIONS_STALE_AFTER_H` = 36
   h. `seo_feedback_stale:metrics`: ONE grouped row counting the settled YouTube Shorts, posted
   within `METRICS_LOOKBACK_DAYS` = 28, with null `views` or `avg_view_percentage` (the detail
   gives the count, then up to 10 `project#index (missing fields)` entries, then "and N more";
   it resolves when the count is 0). An open row's detail, count and listing are refreshed
   without a re-push; a listed Short that ages out of the lookback while still empty drops out
   of the listing and the check says so once in a note ("not a fix"). `seo_feedback_stale:sync`: the item-12 run record is missing, unreadable (a `ran_at` over 1 h in the future
   counts) or its `ran_at` is over `SYNC_STALE_AFTER_H` = 36 h old (the detail says which, and the
   age); it means only "the job has not run" (a crashed run is `short_text_sync_crash`'s alone),
   reads only that file and resolves on a fresh stamp. Pushed once as `health_seo_feedback_stale`; detection only. An
   unreadable tracking record is a note and keeps the open metrics row live; a missing
   decisions file is condition (a)'s finding only and holds nothing else. Detail:
   `docs/seo-feedback-stale.md`.
14. **Pinned version not installed** (`install_deferred`). The code root's `pyproject.toml` pins one
   `gomoveshift-video` version and its service venv has held another for over
   `INSTALL_DEFERRED_AFTER_H` = 6 h. Every launch path refuses meanwhile (`heal.install_pending`,
   below), so past that something still holds the staging sync's deferred install back. The
   clock is one file beside the incidents dir (`<incidents dir>.install_pending.json`, `since`),
   started the first tick the versions differ and removed when they match. One row (key and file
   `install_deferred`; `installed_version`, `pinned_version`, `pending_since`), pushed once as
   `health_install_deferred`, resolved once the versions match. Either version unreadable is a
   note: an open row stays open.
15. **Social queue runs dry** (`social_queue_gap`, `cadence.queue_gaps`). Rules 2 and 2b see 48 h
   and a week ahead, too late to render and enqueue Shorts for a hole. For each posting platform
   (one with a non-shadow delivery slotted no earlier than `QUEUE_ACTIVE_LOOKBACK_DAYS` = 28 days
   ago) that is not paused, the first day from tomorrow through `QUEUE_RUNWAY_DAYS` = 42 days
   ahead with no non-shadow delivery slotted on it. A day is the UTC date of `scheduled_at` (as
   dispatch's blackout reads it); any status but `cancelled`, `failed_permanent` and
   `missing_remote` fills it (a YouTube upload is `published` ahead of its slot), and the
   brand's `policy.blackout` dates are never a gap. Key `social_queue_gap:<platform>:<date>`
   (`platform`, `date`), pushed once as `health_social_queue_gap`; it resolves when that day is
   filled or arrives, and the next empty day is a new incident. Read-only (queue opened `mode=ro`).

16. **Platform paused: access token refused** (`platform_auth_blocked`). One incident per
   platform in `platform_pauses` whose `reason` is `auth:blocked` (the dispatch handlers pause a
   platform when Facebook, Instagram or TikTok refuses its access token, and leave its queued posts
   queued). Key `platform_auth_blocked:<platform>`, fields `platform`, `paused_at`, `reason` (the
   refusal); pushed once (`health_platform_auth_blocked`); resolves when the pause is lifted.
   Read-only (queue opened `mode=ro`).

## Push and idempotency

Each row carries a stable `key` and `pushed`. A push that returns `False` (attempted, failed) leaves
the row unrecorded and the next pass retries it. A condition that clears sets `resolved_at`.
An open row found again keeps its `pushed` state and `resume_attempts`; only a `resume_recipe` or
`log_path` it was filed without is filled from the current detection (a row an older reader filed).

## Incident row, schema `health-incident/1`

Common: `schema`, `kind` (`drive_failed` | `lf_slot_unbooked` | `queue_stalled` | `placeless_label_unverifiable` | `failure_unattended` | `queued_short_refused` | `delivery_failed_permanent` | `short_bad_run` | `backlog_storage_refused` | `backlog_store_orphan` | `backlog_unlisted_footage` | `backlog_exclusion_invalid` | `seo_feedback_stale` | `install_deferred` | `social_queue_gap` | `platform_auth_blocked`), `key`, `file`,
`detected_at`, `pushed` (bool), `pushed_at`, `resolved_at` (null while open).

`drive_failed`: `drive`, `job_name`, `phase` (or null), `gate` (or null), `error`, `log_path`,
`job_run_id`, `failed_at`, `staging_sha`, `resume_recipe` (path of the drive's `resume.json`, or null), `resume_attempts`
(`[]`, owned by `health heal`).
`lf_slot_unbooked`: `region`, `slot`, `ready_drives`. `queue_stalled`: `due_deliveries`,
`last_updated_at`, `quiet_hours`. `placeless_label_unverifiable`: `project`, `label`, `claims` (the words
outside `PLACELESS_NO_CLAIM_WORDS`), `reason`. `failure_unattended`: `drive`, `last_attempt_at`, `detail`. `queued_short_refused`: `delivery_id`, `project`, `short_index`, `scheduled_at`, `reason`. `delivery_failed_permanent`: `delivery_id`, `project`, `short_index`, `scheduled_at`, `error_class`. `short_bad_run`: `n`, `median_views`, `threshold_views`, `shorts` (`project`, `index`, `youtube_id`, `posted_at`, `views`, `causes`). `seo_feedback_stale`: `detail`, plus `decided_at` (decisions), or `count` and `shorts` (metrics: `project`, `short_index`, `missing`), or `ran_at` (sync; null when the record is missing or cannot be parsed). `install_deferred`: `installed_version`, `pinned_version`, `pending_since`. `social_queue_gap`: `platform`, `date`.

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
launches resumes by failure class and fingerprint, never by mirror SHA (`health/retry_policy.py`; guard (a)). An unrelated
commit to the mirror no longer relaunches a drive whose failure it cannot change. The **fingerprint** is the installed
versions of `gomoveshift-video`, `seo_manager` and `redactcam`, the sha256 of the drive's `drive_backlog.yaml` row, the sha256
of the resolved drive config and the failed phase; each attempt records `fingerprint`, `fp_hash` and `failure_class`, and the
row's `failure_fingerprint` is stamped by `health check` when it FILES the row, so it names the code the drive failed under, never
the code heal runs later. A row with no `failure_fingerprint` is legacy (filed before fingerprints): heal never stamps it, and it gets
exactly one launch under the current fingerprint, which the attempt then records and the launched-count rule bounds (a legacy
deterministic row is thus not refused forever by the very release meant to recover it). A version that cannot be read counts as
`null`, which is itself a distinct value). The **class** comes from a first-match regex table over the
failure text (`CLASS_TABLE`; unmatched = `unknown`): `deterministic` (blur gate, final-text review refusal, duration floors,
batch duration integrity, GPS fix density), `transient` (disk full, ENOMEM/killed, drive not mounted/offline, I/O error, place
lookup unavailable, network timeouts), `publish_only` (YouTube quota, Drive upload). Budget per recipe and fingerprint:
`deterministic` 0 until the fingerprint changes, then one; `transient` 3, spaced by the backoff; `unknown` 1; `publish_only`
never (heal has no publish path: it records `refused`, naming class, a 12-char fingerprint hash and `use health publish`).
A refusal is one `refused` attempt (deduped per fingerprint and reason) and one log line. While the checkout's `pyproject.toml`
pins a version the venv lacks, heal skips with `install pending: checkout pins <v>, venv has <v'>`
(the venv is the published one, `retry_policy.service_python`: the staging sync moves
`.venvs/current` and leaves `.venv` alone while a process runs through it)
(`heal.install_pending`; `health backlog`'s new drive and confirmed resume and `health publish` refuse
with the same reason, so every render boundary is a gap the deferred install lands in; unknown either way launches).
Otherwise heal launches
the recipe argv + `--skip-upload --no-youtube-upload`
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
readable recipe is `refused` (argv is never guessed), except for an incident whose job is a publish
(`gms-publish-<project>`) or heal (`gms-heal-...`) job: those never have a recipe, so heal closes
the row (`resolved_at`, `resolution_reason: no resume recipe: ...`, one `closed` attempt) instead
of refusing it on every tick. A heal of a heal reuses the base drive name
(`gms-heal-<drive>-<sha7>`, never `gms-heal-gms-heal-...`). Each attempt is appended to `resume_attempts` as
`{at, mirror_sha, installed_version, fingerprint, fp_hash, failure_class, slot, action: launched|skipped|refused, reason, job_name}` (+ `outcome`,
`outcome_at` once the heal job ends); an unchanged repeat skip is not re-recorded.

**Operator hold.** A file named `no-auto-resume` beside the recipe (`<stages dir>/no-auto-resume`,
contents free-form notes) makes heal refuse that drive on every mirror SHA, dry run included, with
reason `operator hold: no-auto-resume beside the recipe`. Use it for a run authorized for one
attempt only: guard (a) alone would relaunch a failed drive after each new staging landing.
Delete the file to hand the drive back to heal.

**Code-root binding.** A resume runs the code it records as `mirror_sha`, never the recipe
writer's: the recipe's interpreter is replaced by the code root's published venv
(`.venvs/current/bin/python`, else `.venv/bin/python`, resolved to its physical `.venvs/<id>`;
recorded as the attempt's `interpreter`) and the cwd is the code root, not the recipe's `cwd`.
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

Guards (each a recorded skip that ends by itself; an open `drive_failed` incident always gets
another resume, and only an operator hold or an unreadable recipe stops one): (a) retry backoff:
with n launched attempts of the recipe (counted across every row sharing it) on the current mirror
SHA and installed version, the next waits `min(3600 * 2**(n-1), 86400)` s (`HEAL_RETRY_BASE_S`,
`HEAL_RETRY_MAX_S`) after the newest, skip reason `retry backoff: next attempt at <iso>`; a new SHA
or version retries at once, and pre-flight failures do not count (after `HEAL_PREFLIGHT_RETRIES`
of them on one code the retry waits the 24 h max from the newest). A run killed by a host
restart (job record left at `running`, nanny and worker gone per `_proc_alive`) gets attempt outcome
`interrupted` and one `health_heal_outcome` push; it uses neither budget nor backoff for the first
`HEAL_INTERRUPTED_RETRIES` (3) on a fingerprint, then counts like a failure; (b) a live drive on the recipe's own stages or project dir
(`heal.drive_conflict`: the run lock of either dir held by a live pid, or a live drive job with the
same `--output-dir`); a drive of any other project is never a reason (note 191); (c) none: the publish slot (the recipe's
`--lf-publish-date`, else the soonest free slot of `gomoveshift schedule`) is recorded as `slot`
but never gates, since an audit-only drive keeps only the weekday check and `health backlog` /
`health publish` claim a fresh slot after a passed audit; (d) a metadata-gate failure (gate
matches `/metadata/i`) when `git diff --name-only <staging_sha> <mirror sha> -- GENERATOR_PATHS` is
empty (`health/heal.py`: the title/longform-metadata/content-validator modules) retries once per
24 h; (e) one push at launch (`health_heal_launched`) and one at outcome (`health_heal_outcome`),
the latter observed on the next heal pass.

## `health backlog`: launch the next backlog day

Moved to [190-health-backlog.md](190-health-backlog.md).

## Blur-mask frame-count gate (`mask_index_mismatch`)

A drive rendered before the redactcam frame-count fix carries a blur mask built on more frames than
its source container holds (Kerry-to-Cork: 306,385 vs 306,377), so the blur lags the picture.
`gomoveshift/publishing/mask_frame_gate.py` is the one check every upload entry point calls:
`health publish` (and the backlog republish, which runs `run_publish`) refuses before any store
write, and Phase D raises `MaskFrameCountError` at the top of an upload run (an audit pass is not
gated). Per `day_<date>` stage dir it compares `source_frame_count` in
`day_<date>_blur_regions.json` with the frames of the container the detect pass read: ffprobe
`nb_frames` of `day_<date>_raw.mp4`. The encode deletes the raw and cleanup's master-drop deletes the
master once the finals exist, so a finished day is counted from the raw's inputs: the
`_stages/episode_*_<date>.mp4` files, summed, but only when their size:mtime fingerprints are the
`source_fingerprints` that `day_<date>_color_cache.json` recorded at encode (the concat is a stream
copy). `day_<date>_master.mp4` counts last, and only when its frame rate is within 1% of the
sidecar's `source_fps`: a master encoded at another rate (Old Head: 184,421 frames at 29.97 fps
against a 368,837-frame 59.94 fps mask) is in a different frame space. When nothing else can
count such a day, its `mask_frame_count_unknown` reason names both rates (row key `rate_note`;
`tests/test_mask_gate_master_rate.py`), so a master set aside for its rate is not read as one
that could not be probed. Unequal counts refuse, naming the day, both counts and the difference. A
missing sidecar, a missing count, episodes that are not the recorded ones, or no probeable
container refuses too, as
`mask_frame_count_unknown` (fail closed). Each refused day gets one `mask_index_mismatch` incident
(`mask_index_mismatch_<drive>_<date>.json`, key `mask_index_mismatch:<drive>:<date>`), never one
per tick; it is resolved once the counts agree. There is no override flag or variable: re-render
the day on the fixed renderer, which rewrites the sidecar. A `--dry-run` refuses but records no
incident. Queue-dispatched Shorts are not re-checked; they are enqueued by Phase D, behind it.
