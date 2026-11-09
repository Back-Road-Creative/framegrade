### Added

- `gomoveshift cleanup <project> --abandoned` reclaims a killed or superseded run's
  whole `_stages/` dir, which the plain `cleanup` could only skip as "Pipeline not
  complete". The dir qualifies only when every rule holds — not a symlink, no
  `no-auto-resume` hold, no run lock held by a live pid, no `gomoveshift drive`
  running anywhere, old enough for `--min-age` — and the first failing rule is the
  `SKIP` reason. Dry run by default; `--confirm` re-checks and removes that one dir.
  A project name is required. Without the flag the incomplete-project `SKIP` line
  now names `--abandoned`.
