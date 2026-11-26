### Fixed

- Every replay of a drive's saved `resume.json` (`health heal`, a confirmed `health backlog` resume, `health publish`, `drive --from-recipe`) now re-resolves its input files first: a gone `--day-gpx` / `--bracket-gpx` takes the `drive_backlog.yaml` row's file and is written back into the recipe, and a gone input with no replacement (GPX, source directory, `--map-inset-bundle`, `--trail-profile`) is refused with the file named instead of launching a drive that dies in pre-flight. Before, `health publish` and `drive --from-recipe` replayed a renamed GPX as written.
