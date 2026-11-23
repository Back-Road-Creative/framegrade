### Fixed

- `--replace-scheduled` retry marker: it records its region (a marker of another region no longer
  hijacks a tick) and is written before the renames, so a crash mid-replace is still retried. A marker
  tick that does not launch now clears the marker on a permanent refusal (nothing to replace, margin-kept
  records, row gone) and bumps it on a transient one, then falls through to the plain backlog in the same
  tick, so a stuck marker no longer stalls the region. A locked store on the dry-run reopen is retried
  instead of crashing the tick, a failed restore keeps the marker, audits older than the replace plan are
  ignored, and the margin-only refusal says so.
