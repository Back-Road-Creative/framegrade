## 179 - Status snapshot

`gomoveshift status [--json]` prints one read-only snapshot of everything live. It is the single
reader (`gomoveshift/status/snapshot.py`, `build_snapshot`) that the console's status page is built
on. It never launches, repairs or writes: it does not run `health check`, it opens the publishing
queue read-only, and it never creates a missing file. Exit code is always 0.

Every section carries `as_of` (UTC) and `source` (the path or command it read). A store that cannot
be read carries `unavailable: <why>` and none of its data keys, so it never reads as empty or healthy.
One unreadable store does not blank the others.

| Section | What it reports | Read from |
|---|---|---|
| `jobs` | `running` (no result recorded yet) and the 10 most recent ended job-runner records, each with drive, phase and error parsed from its `output.log`; `counts` by state; `unreadable` | `$GMS_JOBS_DIR`, default `logs/jobs` under the data root (`$GMS_DATA_ROOT`) |
| `incidents` | `open` incident rows as written by `health check` ([156](156-health-check.md)), `resolved` count, `unreadable` | `$GMS_HEALTH_INCIDENTS_DIR`, default `gms/incidents` under the data root |
| `queue` | `counts` by platform and status, `overdue_manual` (drafts past their slot plus the 24 h grace of `publishing-status --validate`), the `next` 5 scheduled deliveries | `social_queue.sqlite3` under `seo.storage_path` |
| `runway` | the `gomoveshift schedule --json` report (what is booked, when the queue ends, next free slots) | the content ledger |
| `backlog` | `rows`, `by_region`, the first 20 `projects` | `drive_backlog.yaml` under the inputs dir (`$GMS_INPUTS_DIR`) |
| `machine` | `volumes` (free and total bytes per labelled path), `gpu` (who holds the shared GPU lock), `cron` (per `gms-*-cron.log`: last timestamped line, and the last line containing `FAILED`) | `$GMS_STATUS_VOLUMES`, `$GMS_STATUS_GPU_LOCK_PATH`, `$GMS_CRON_LOGS_DIR` |

Environment:

- `GMS_JOBS_DIR`: job-runner records directory. Unset: `logs/jobs` under the data root.
- `GMS_CRON_LOGS_DIR`: directory holding `gms-*-cron.log`. Unset: `logs` under the data root.
- `GMS_STATUS_VOLUMES`: `label=path,label=path`. Unset: one volume, the data root.
- `GMS_STATUS_GPU_LOCK_PATH`: where to read the GPU lock holder from. Unset: the lock path itself
  (`GMS_GPU_LOCK_PATH`). It is separate so a container can be shown the host's lock file read-only
  without changing where its own processes take the lock.

The cron section reports what the log says, not an exit code: the cron logs do not record one in a
uniform way, so only the last line and the last `FAILED` line are shown.

On a read-only mount the queue is read with SQLite `immutable=1`, which reads the database file as it
stands and can lag writes still in the write-ahead log.
