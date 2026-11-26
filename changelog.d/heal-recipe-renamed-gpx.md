### Fixed

- `health heal`: a resume whose recipe names a `--day-gpx` or `--bracket-gpx` file that no longer
  exists takes the drive's backlog row's file instead, when that file exists. A GPX rename left
  Galley Head's resume pointing at a gone file, and the drive refused it with exit 2.
