### Jobs (`console/jobs.py`, `GET/POST /jobs`)

`JobStore` (SQLite row per launch: id/name/argv/status/rc/created_at/finished_at/log_path/brand)
under `<GMS_CONSOLE_DATA_DIR>/jobs.sqlite3`. `brand` records the brand `launch()`
actually launched as -- `_ensure_brand_column` ALTERs it into a pre-existing table on open
(`CREATE TABLE IF NOT EXISTS` is a no-op against an existing table, and existing installs already have the table), so a legacy row keeps `NULL` (rendered "unknown", never guessed). `launch()` only Popens
`sys.executable -m gomoveshift <subcommand>` for a `subcommand` in `ALLOWED_SUBCOMMANDS` (argv list
only, never a shell string; `_validate_argv` raises `ValueError` on anything else), logging to
`<GMS_CONSOLE_DATA_DIR>/job_logs/<id>.log` and keeping the live `Popen` in an in-module registry
keyed by the store's database file and the job id. `launch(..., brand=None)` defaults to
`settings.current_brand()`, matching its `env=None` default of the current environment. No
background thread: `get_job`/`list_jobs` reap status/rc/finished_at on read. Every form control on the page carries a real `<label for>`: both `subcommand` selects and the schedule's name/interval inputs. A `placeholder` is a hint, not an accessible name -- it disappears on the first keystroke and is not reliably announced -- so the schedule inputs dropped theirs for labels. The two selects share the field name `subcommand` but have distinct ids, so they are distinguishable to a screen reader. `GET /jobs` lists jobs + a launch form; `GET /jobs/{id}` tails the
last 100 log lines; `POST /jobs/{id}/rerun` relaunches the same argv as a new row, replaying the
recorded launch brand (`brand_launch_env(row["brand"])`) rather than the console instance's
current brand -- a legacy `NULL` row falls back to `brand_launch_env()`. It refuses a row that is
still `running` (`rerun_blocked_reason`, 409 naming the job and why): the argv is replayed
verbatim, so re-running a live `dispatch-queue --apply` starts a second drain over the same queue
and the second worker picks up rows the first is mid-publish on -- the double-post the delivery
lease exists to stop. The route reaps first (`get_job`), so a job whose child has since exited is
never refused merely because nothing had looked at it yet, and the detail page asks the same
function, hiding the button and printing the reason instead of offering a control that will
refuse. Every other status -- `done`, `failed`, `interrupted` -- is exactly what the button is
for and is unaffected. A rerun is a hybrid, and the page now says so instead of leaving a
reader to infer it from the word: the `argv` is the original job's, written before any setting
saved since, while the environment is resolved from this console's settings as they stand at the
moment of the click. Neither "retry that job" nor "start again with current settings" describes
it alone. The new row records `rerun_of` (the job it came from; `NULL` for an ordinary launch
and for every row predating the column, which arrives by the same idempotent `ALTER TABLE` as
`brand` and the liveness columns -- `_ensure_rerun_column`), and `job_detail.html` prints both
halves with a link back to the original. The generic launch/schedule form only
ever offers `GENERIC_SUBCOMMANDS` (a strict subset of `ALLOWED_SUBCOMMANDS` that genuinely runs
with zero extra args -- `dispatch-queue`, `schedule`, `publishing-status`, `reconcile-playlists`)
-- every other allowlisted subcommand needs a real argument the form never collects (`metrics`
requires `collect <project>` and is deliberately excluded until a parameterized argv form
exists). `sweep_orphaned_jobs()` runs once at startup (before the ticker starts) to reconcile any
DB row still `running` from a prior process, since `_PROCESSES` (keyed by store file + job id: every
store numbers its jobs from 1, so another store's job 1 is never this one's) is in-memory only and
doesn't survive a restart -- but an empty `_PROCESSES` is not evidence the child died. `launch()`
records the child's pid plus a fingerprint (`/proc/<pid>/stat` field 22, `starttime`, so a *reused*
pid isn't mistaken for the original child) on the same `store.update` that sets `status="running"`
(`pid`/`started_fingerprint` columns, ALTERed onto a pre-existing table the same way as `brand`).
The sweep has exactly three outcomes for a `running` row it doesn't hold a `Popen` for: pid alive
with a matching fingerprint -> genuinely still running, left `running`, no notification; pid gone,
or alive under a mismatched fingerprint (pid reuse) -> `interrupted`, notified; no pid recorded (a
legacy row predating this column) -> `interrupted`, notified, unchanged from before -- there is no
evidence either way, so the old behaviour stands. `rc` is never guessed for a row reconciled this
way; an out-of-process child's exit code is unknowable and stays `NULL`, same as `interrupted`
already means elsewhere. A row left `running` has no `Popen` to reap, so `reap_finished` (called by
`get_job`/`list_jobs` and the `Ticker`) re-checks every untracked `running` row on each pass, so a
child that exits later is still eventually marked `interrupted` instead of stranded `running`
forever.

`jobs.html` no longer carries a `<meta http-equiv="refresh">` that reloaded the whole page every 5
seconds while a job was running. The same page has required text/number inputs (schedule name,
interval minutes) that a full reload silently discarded mid-entry, and an unpausable, unadjustable
auto-reload also fails WCAG 2.2.1 (Timing Adjustable). The page now offers a plain
`<a href="/jobs">Refresh</a>` instead: an explicit, JS-free navigation the operator triggers, never
one that fires underneath typing they haven't submitted yet.

**`interrupted` records what was SEEN, not a verdict** (`outcome_evidence`, same
idempotent-`ALTER TABLE` contract as `pid`). The console launches jobs out of process, so for a
row it did not fork it can never obtain an exit code -- `wait()` belongs to the parent -- yet it
wrote `interrupted`, which reads as "this job was cut short", collapsing three different
observations into one word. In two of them the job may have finished all its work. The status
stays `interrupted` (the terminal bucket the list page and the rerun guard already understand)
and the column carries which observation was made: `gone_at_startup` (absent, or alive under a
mismatched fingerprint -- the same knowledge either way), `no_pid_recorded` (a legacy row; no
observation was ever possible), `exited_while_watched` (the sweep saw it alive and a later pass
found it gone -- the strongest evidence here, and still not an exit code).
`interrupted_evidence_sentence` turns that into the sentence the detail page prints; a row from
before the column reads "marked interrupted before this console recorded why". Every sentence
says the outcome is unknown and none claims the job failed. A row with a real `rc` gets no
sentence at all -- there is nothing to hedge about.

**A `running` row also carries a `liveness` label -- evidence, never a verdict** (`liveness_label`,
`STALL_THRESHOLD_SEC = 600`). It compares the current time against the mtime of the job's own log
file, the one per-job signal every subcommand's log already provides: a log written within the
last `STALL_THRESHOLD_SEC` seconds reads `working (log updated <age> ago)`; one older reads
`quiet <age> -- may be stalled`. `600` is modeled on `ffmpeg_runner.py`'s per-stage `stall_timeout`
default -- a console job launches a whole `gomoveshift` subcommand, at least as coarse-grained as a
single ffmpeg stage, so the same window isn't too eager to flag a merely-quiet-but-healthy job
(`uploader.py`'s tighter `_DRIVE_STALL_TIMEOUT` of 120s watches one upload's byte progress, a
finer-grained signal than a job log, so it isn't the precedent used here). The label never claims
the process is dead -- that verdict stays `sweep_orphaned_jobs`'s alone -- and a missing or
unreadable log yields no label at all, never a "stalled" claim it can't back up. `get_job` and
`list_jobs` add the `liveness` key only to a row whose `status == "running"`; every other status
renders exactly as before, with no new key. `jobs.html` and `job_detail.html` print it beside the
status when present.
