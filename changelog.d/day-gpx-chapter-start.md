### Fixed

- `--day-gpx` slices each later chapter of a split Insta360 recording at its own start: the filename time plus the earlier chapters' durations. Before, every chapter was sliced at the recording start, so the second 30-minute chapter carried the first chapter's GPS (Phase A's 60 min start tolerance let it through). A chapter whose predecessor is missing from the source dir is refused. See `docs/pipeline/71-generic-source-ingest-full-drive-lf-mode.md`.
