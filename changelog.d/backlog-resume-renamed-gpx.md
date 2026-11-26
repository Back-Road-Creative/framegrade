### Fixed

- `health backlog`: resuming a drive whose OST audit was confirmed now takes the backlog row's
  `--day-gpx` / `--bracket-gpx` file when the recipe's no longer exists, as `health heal` already
  does. A GPX rename left three Ireland drives' recipes naming gone files.
