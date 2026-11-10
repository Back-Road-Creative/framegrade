### Fixed

- The upfront staging estimate (preflight and the hourly backlog) sized a drive at 1.5x its
  source while the day-master encode asks for the raw concat plus the master on top of the
  episode concats it keeps, 3x in all. A drive could launch, spend hours on scoring, then stop
  at the encode for lack of space. Both now read one `PEAK_CACHE_FACTOR` (3.0), so a drive that
  cannot fit is refused before it starts.
