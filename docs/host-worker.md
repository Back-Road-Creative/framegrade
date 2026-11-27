# Host GPU worker

`python -m gomoveshift.hostworker [--spool DIR]` runs on the host and executes
requests the console container spools, so the console needs no GPU. Spool dir
defaults to `$GMS_HOST_SPOOL_DIR`; the shared GPU lock is `$GMS_GPU_LOCK_PATH`.

Spool layout:

| Path | Writer | Meaning |
|---|---|---|
| `pending/<job_id>.json` | console | request: `job_id, brand, argv, dedupe_key, env, submitted_at` |
| `running/<job_id>.json` | worker | claimed by atomic rename, oldest first, by `GMS_HOST_WORKER_LANES` (default 2) claim loops |
| `done/<job_id>.json` | worker | finished, any outcome |
| `cancel/<job_id>` | console | cancel marker |
| `logs/<job_id>.log` | launcher | child stdout+stderr (a `log_path` in the request is ignored) |
| `jobs/<job_id>.launch.json` | worker | argv, env, log, whether to take the GPU lock (mode 0600) |
| `jobs/<job_id>.proc.json` | launcher | `pid`/`start` of the launcher and `child_pid`/`child_start` (start ticks from `/proc/<pid>/stat`), `waiting_gpu` |
| `jobs/<job_id>.exit.json` | launcher | `exit_code`, `error`, `waiting` (the lock was held; never started); its last write |
| `capabilities.json` | worker | `{adopts_running_jobs: true, pid}`; host-reconcile 5u restarts the worker at once only when `pid` is the unit's MainPID |
| `<job_id>.status.json` | worker | `state` (queued/running/succeeded/failed/cancelled), `pid`, `phase`, `progress`, `exit_code`, `error`, and `waiting` while it waits for the GPU lock |
| `heartbeat.json` | worker | `{ts, pid}` refreshed every ~3 s while alive |

Behaviour:

- Refuses to start when `nvidia-smi` fails, when no spool dir is given, or when no GPU lock path is configured.
- Re-validates argv: `[python, -m, gomoveshift, <ALLOWED_SUBCOMMANDS member>, ...]`. Anything else is marked `failed` and never spawned. `argv[0]` only has to look like a Python (the console writes `python3` there, never its container interpreter path, which does not exist on the host); the child always runs under the worker's own `sys.executable`. A request with an invalid `job_id` or unparseable JSON is moved to `done/` with no status file.
- A `drive` request requires an absolute source directory that this host worker can list. It opens each top-level supported video file before spawning, including symlink targets. Missing, inaccessible, or broken sources fail with a named status error before taking GPU time. This is a launch check; the drive pipeline still inventories the footage and checks its own media contracts.
- Jobs outlive the worker. Each job runs `python -m gomoveshift.hostworker.launch <spool> <job_id>` in its own transient unit (`systemd-run --unit=gms-job-<id> --collect`, as the worker's own uid/gid and cwd). Where systemd refuses a transient unit (an unprivileged worker, no polkit rule), the worker logs it and starts the launcher in its own session instead; the unit then needs `KillMode=process` so a stop leaves it alone. On SIGTERM/SIGINT the worker stops claiming, stops watching and exits; it never signals a job, and the job's status stays `running`.
- On startup the worker re-adopts every `running/` entry and only watches it (identity by pid plus start ticks, so a reused pid is not the job). Nothing is relaunched. A launcher that died without an exit record is `failed`; an entry claimed by an older worker that ran jobs as its own children (no `launch.json`) is `failed` as interrupted, since that worker killed its child when it stopped.
- Applies the request's `env` (the brand's setting overrides) on top of its own environment: registry names with string values only; `GMS_HOST_SPOOL_DIR` and `GMS_GPU_LOCK_PATH` are never overridable; a malformed `env` is ignored.
- For `drive`, passes the configured shared GPU lock path to the child; the drive orchestrator takes the whole-run lock after its run locks, queueing behind another render (note 191). For other commands, the launcher takes the lock before starting the child; while it stays busy past `GMS_GPU_LOCK_WAIT_S` (a render holds it) the request goes back to `pending` at the end of the line with status `queued` and a `waiting` note, and is tried again: it is never failed because a render is running. A cancel while it waits is honoured before the next try. Only an `OSError` on the lock fails it (`GPU lock unavailable`). There is no CPU fallback.
- Lanes: `GMS_HOST_WORKER_LANES` (default 2) claim loops. With two or more, the first claims only `drive` requests and the others everything else, so a render never holds up another job's claim: renders run one after another in the drive lane while other requests keep moving. With one, a single loop claims everything in order.
- Progress: a child log line `GMS_PROGRESS <0-1> <phase>` updates `progress`/`phase` in the status file.
- Cancel: SIGTERM to the job's child process group (or to its launcher while it still waits for the GPU lock), a grace period, then SIGKILL, each only while the recorded pid still has its recorded start ticks. Only that job is signalled. On Windows the child starts in a new process group and is stopped with CTRL_BREAK, then `taskkill /T /F` (`gomoveshift/_process_group.py`).
- The worker's source check prevents unusable host paths from entering the pipeline; media and publication gates stay in the code the child runs.
