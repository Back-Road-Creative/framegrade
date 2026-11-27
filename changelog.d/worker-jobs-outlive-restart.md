### Fixed

- A host worker restart, deploy or unit change no longer kills a running render. Each job runs in its own transient unit (`gms-job-<id>`), or in its own session where systemd refuses one, under `gomoveshift.hostworker.launch`. A stopping worker signals no job, and the next worker re-adopts each running job and only watches it, never relaunching it. The worker writes `capabilities.json` so host-reconcile knows it may restart it at once.
- The worker runs two claim loops (`GMS_HOST_WORKER_LANES`, default 2): drive requests in their own lane and everything else in the other, so a short job no longer waits behind a render. The launcher now takes the GPU lock for non-drive jobs, and a busy lock still sends the request back to pending as `queued` (note 191).
