### Fixed

- `cleanup <project> --abandoned` now reclaims a run that `health backlog` superseded even when it is younger than `--min-age`. The supersede record (`superseded.json`) already says that run is dead, so the 7-day gate only held its space while the live run waited. The lock, hold and running-drive checks still apply.
