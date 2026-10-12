### Shorts (`console/shorts.py`, `GET/POST /shorts`, `GET /shorts/csv`, `GET /shorts/csv/download`)

Shorts have no standalone CLI subcommand -- only `drive` (and the legacy single-video `pipeline`)
produce them -- so `shorts.build_shorts_argv()` forwards to `drives.build_drive_argv()`, adding
only the two shorts-specific knobs `drive` exposes (`shorts_target`/`shorts_gap`, blank-omitted).
`POST /shorts/run` runs the real `drive` argv but records it as a `"shorts"` job (distinct from
the Drives tab's `"drive"` row name), no new `ALLOWED_SUBCOMMANDS` entry needed. Same brand
selector/validation as the Drives tab (unknown slug is a 400).

`GET /shorts/csv?output_dir=` is read-only: `shorts.find_shorts_csv()` globs the run's
`*_shorts_metricool.csv` (written by `drive_phases/phase_d.py`) and `shorts.read_shorts_csv_rows()`
parses it for an HTML table preview -- the file is only ever read, never regenerated. 404s when
`output_dir` doesn't exist or the CSV hasn't been written yet (same two-step gate as
`/drives/preview`). `GET /shorts/csv/download?output_dir=` serves that same file's raw bytes
(`FileResponse`, `text/csv`) so the download is byte-identical to what the CLI wrote.
