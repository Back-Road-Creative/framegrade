### Fixed

- `health backlog` ignored a `health heal --recipe` resume when finding a drive's newest audit job: a recipe-registered row names the drive by its stages dir, so the job is `gms-heal-<project>-<sha7>`, not `gms-heal-gms-backlog-<project>-<sha7>`. A recipe resume that succeeded still left every tick refusing `audit failed` on the older failed job. It now counts as the audit; a bare `<project>` name, never wrapped by a heal, still does not.
