### Fixed

- `drive` now refuses a real render unless `health backlog` started it (`GMS_LAUNCHED_BY=health-backlog`); `--dry-run`, `--list-lfs`, `--label-dry-run` and `--preview` are exempt, and `GMS_ALLOW_MANUAL_DRIVE=1` is the logged operator override. `health backlog` also refuses to launch when the render interpreter's framegrade is older than the newest `vX.Y.Z` release tag (`GMS_RELEASE_REMOTE` overrides the remote), or when either version cannot be read.
- The same gate accepts every tracked launcher, not only the backlog: `GMS_LAUNCHED_BY` may be `health-backlog`, `health-heal` (a heal resume), `console` (the console job store) or `hostworker` (set by the worker itself on spooled drives, never taken from the request). An unknown value is refused and named.
