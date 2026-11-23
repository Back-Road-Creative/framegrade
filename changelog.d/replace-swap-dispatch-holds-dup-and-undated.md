### Fixed

- `load_publish_plan` refuses a plan that lists a `short_index` more than once
  (`DuplicateShortIndexError`) for every reader. Dispatch holds a Short whose plan record has no
  `recording_date`, and every row of a project with a duplicate index, as `RecordHeldError` (a hold: no attempt counted, no status change; one `delivery_record_held` incident per hold), so a refused
  Short's "rows held" incident is true and one Short's video is never uploaded under another's row.
  The replace rebuild refuses every position of a duplicated index with one incident.
- The replace gate's swap-invariant Gate C/D hash projects only the replace run's own plan (recorded
  per plan name); a sibling plan's repointed media or changed consent is refused.
  A legacy single-string swap hash is refused with a re-confirm Gate C/D reason.
- A `delivery_record_held` incident resolves itself when the hold clears, so a later hold files a
  fresh, unpushed incident instead of being swallowed by a stale open one. One Short on several
  platforms files one incident (keyed `project:short_index`), and the cron summary line carries
  `held_record=N` when non-zero.
