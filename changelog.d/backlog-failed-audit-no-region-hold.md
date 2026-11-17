### Changed

- `health backlog` no longer stops a whole region at one failed audit. The tick walks the
  region's `auditing` rows, lists each failed one under `failed_audits` (printed as `failed audit
  <project>: ...`) and launches the next eligible row, exiting 0. The failed row stays `auditing`
  for `health heal` to resume, and is published once its audit succeeds. An in-progress, gated or
  passed audit behind it is handled as before.
