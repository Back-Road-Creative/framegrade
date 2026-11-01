### Fixed

- `drive` now refuses a real render unless `health backlog` started it (`GMS_LAUNCHED_BY=health-backlog`); `--dry-run`, `--list-lfs`, `--label-dry-run` and `--preview` are exempt, and `GMS_ALLOW_MANUAL_DRIVE=1` is the logged operator override. `health backlog` also refuses to launch when the render interpreter's framegrade is older than the newest `vX.Y.Z` release tag (`GMS_RELEASE_REMOTE` overrides the remote), or when either version cannot be read.
