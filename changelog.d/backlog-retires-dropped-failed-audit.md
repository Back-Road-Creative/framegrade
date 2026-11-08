### Fixed

- `health backlog` no longer stalls a region on a failed audit whose row was dropped from `drive_backlog.yaml`. That drive could not be relaunched or superseded: `--supersede` refused because the row was not in the backlog. So every call refused until someone ran `set-status failed` by hand. Now the store row is set `failed`, the result lists it under `retired`, and the next row launches. A dry run only lists the retirement.
