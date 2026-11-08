### Fixed

- `health check` and `health backlog` no longer treat a drive stopped at the OST audit gate (exit 5, `DRIVE WAITING:`) as failed. No `drive_failed` row is filed, so `health heal` stops re-resuming it onto the same gate on every mirror SHA. The backlog names the sidecar to review and the `.confirmed` marker to create, then resumes the drive from its stages on the next run once the marker exists.
