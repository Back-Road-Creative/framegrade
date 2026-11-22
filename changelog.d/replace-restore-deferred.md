### Fixed

- A restore that is put off (a `gms-backlog-<project>` job since the attempt is running and alive, or a drive is running) returns `deferred: <reason>`, never the `""` that means restored: a permanent refusal or an unclaimed give-up that meets it keeps the marker and bumps it, with no `restore_failed` incident, and the old run is restored on the first tick the block lifts. A marker put off on 8 consecutive ticks opens a `restore_deferred` incident naming the project and the job (it never clears the marker).
- A job record left `running` by a killed nanny (a WSL shutdown, an out-of-memory kill) no longer puts a restore off: the job must have a live worker or nanny (`heal._proc_alive`).
- `stages_root` is a required keyword of `run_tick`, `_run_tick`, `give_up_unclaimed_replace_retries` and `restore_set_aside`, so a caller can no longer bury a live stage dir by leaving it out.
- `default_launcher` puts the run_id from the launcher's stdout first in its detail, so a uuid in stderr text is never the one pinned into the marker.
