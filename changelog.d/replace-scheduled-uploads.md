### Added

- `health backlog --supersede <project> --reason <text> --replace-scheduled` re-renders a drive
  whose long-form uploads are still scheduled: every upload record must be public (kept) or
  scheduled at least 6 h ahead and live-confirmed private. Phase D then uploads each replacement
  with the same publish time and title and deletes the old video only after the new one is up.
  `superseded.json` is now `backlog-supersede/2` (`/1` still reads).
