### Fixed

- `health check` rule 6 (`failure_unattended`) now alarms on failures, not only time: a drive whose last 3 launches in a row failed, or whose retry budget (the heal policy's own count) is spent, stays alarmed however recent its last launch. Before, every relaunch reset the 26 h clock, so a drive that kept failing never stayed alarmed.
- `health check` no longer re-files a `drive_failed` row that `health heal` closed with a `resolution_reason`.
