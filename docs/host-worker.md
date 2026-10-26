# Host GPU worker

`python -m gomoveshift.hostworker [--spool DIR]` runs on the host and executes
requests the console container spools, so the console needs no GPU. Spool dir
defaults to `$GMS_HOST_SPOOL_DIR`; the shared GPU lock is `$GMS_GPU_LOCK_PATH`.

Spool layout:

| Path | Writer | Meaning |
|---|---|---|
| `pending/<job_id>.json` | console | request: `job_id, brand, argv, dedupe_key, env, submitted_at` |
| `running/<job_id>.json` | worker | claimed by atomic rename, one at a time, oldest first |
| `done/<job_id>.json` | worker | finished, any outcome |
| `cancel/<job_id>` | console | cancel marker |
| `logs/<job_id>.log` | worker | child stdout+stderr (a `log_path` in the request is ignored) |
| `<job_id>.status.json` | worker | `state` (running/succeeded/failed/cancelled), `pid`, `phase`, `progress`, `exit_code`, `error` |
| `heartbeat.json` | worker | `{ts, pid}` refreshed every ~3 s while alive |

Behaviour:

- Refuses to start when `nvidia-smi` fails, when no spool dir is given, or when no GPU lock path is configured.
- Re-validates argv: `[python, -m, gomoveshift, <ALLOWED_SUBCOMMANDS member>, ...]`. Anything else is marked `failed` and never spawned. `argv[0]` only has to look like a Python (the console writes `python3` there, never its container interpreter path, which does not exist on the host); the child always runs under the worker's own `sys.executable`. A request with an invalid `job_id` or unparseable JSON is moved to `done/` with no status file.
- A `drive` request requires an absolute source directory that this host worker can list. It opens each top-level supported video file before spawning, including symlink targets. Missing, inaccessible, or broken sources fail with a named status error before taking GPU time. This is a launch check; the drive pipeline still inventories the footage and checks its own media contracts.
- On SIGTERM/SIGINT mid-job the worker stops the child's process group (TERM, grace, KILL), writes `failed` with a "worker stopped" error (the console reports it and notifies), releases the lock, then exits. Stopping the unit never leaves a child rendering without the GPU lock.
- Applies the request's `env` (the brand's setting overrides) on top of its own environment: registry names with string values only; `GMS_HOST_SPOOL_DIR` and `GMS_GPU_LOCK_PATH` are never overridable; a malformed `env` is ignored.
- For `drive`, passes the configured shared GPU lock path to the child; the drive orchestrator takes the whole-run lock after its run locks. For other commands, the worker takes the lock before spawning and fails with `GPU lock unavailable` if it stays busy past `GMS_GPU_LOCK_WAIT_S`. There is no CPU fallback.
- Progress: a child log line `GMS_PROGRESS <0-1> <phase>` updates `progress`/`phase` in the status file.
- Cancel: SIGTERM to the child's own process group, a grace period, then SIGKILL. Only the worker's own child is signalled. On Windows the child starts in a new process group and is stopped with CTRL_BREAK, then `taskkill /T /F` (`gomoveshift/_process_group.py`).
- The worker's source check prevents unusable host paths from entering the pipeline; media and publication gates stay in the code the child runs.
