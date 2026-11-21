### Fixed

- `gomoveshift health check` rule 11 keeps listing the folders named by an invalid exclusion entry (the entry still excludes nothing), so its open `backlog_unlisted_footage` incidents no longer close and re-notify when the entry is fixed; an unreadable `drive_backlog_excluded.yaml` keeps open per-entry `backlog_exclusion_invalid` incidents instead of closing them.
- Several new unlisted camera files in one tick send one grouped push (count plus up to 10 names) instead of one per file; each file keeps its own incident row.
- `backlog_unlisted_footage` incident files carry a short hash of the key, so two paths that flatten to the same safe name (`/a_b/x.mp4`, `/a/b_x.mp4`) no longer share one file.
