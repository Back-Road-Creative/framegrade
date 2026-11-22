### Fixed

- `--replace-scheduled` fixup: a rebooked slot is reserved in the ledger and Waypost before any YouTube
  call and re-verified on re-entry, so two LFs never share a slot (a refused final booking is an
  incident, never done); the rebook search keeps the 6 h margin and never picks today. The real
  reopen now runs before the launch: a refusal or failed launch restores the old run, leaves a retry
  marker the next `health backlog` tick re-runs with the heal backoff, and opens incidents. An LF
  inside the 6 h margin is kept (incident) instead of refusing the whole plan. The publish backoff counts
  only this plan's jobs, a failed replace whose manifest row was removed is republished rather than
  held, and the live-catalog path moved to a leaf module so phase D tests no longer import the CLI.
