### Added

- `health backlog --supersede <project> --reason <text>`: a row whose own earlier run is stale
  (for example rendered on an older release) moves that run's stage root and output dir to
  `_superseded/` with a `superseded.json` record and launches fresh, instead of being skipped as
  already staged. Refused for a published or booked project, an output with upload records, or
  any row but the region's next.
