### Fixed

- A host worker restart, deploy or unit change no longer kills a running render. Each job runs in its own transient unit (`gms-job-<id>`), or in its own session where systemd refuses one, under `gomoveshift.hostworker.launch`. A stopping worker signals no job, and the next worker re-adopts each running job and only watches it, never relaunching it. The worker writes `capabilities.json` so host-reconcile knows it may restart it at once.
- A route-story-film (or any job) that finds the GPU lock busy now waits for it as `queued` instead of failing after 120 s with "GPU lock unavailable".
- The worker runs two claim loops (`GMS_HOST_WORKER_LANES`, default 2), so a short job no longer waits behind a long one in its queue.
