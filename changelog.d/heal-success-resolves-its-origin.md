### Fixed

- A failed drive stayed an open incident after a heal resume of it succeeded, when the resume ran
  under a moved scratch root and shared no log or stages dir with it. Every new mirror SHA then
  launched another multi-hour audit-only resume, which held the region's backlog ("one heavy drive
  at a time"). `health check` now resolves a failed job once a later heal resume named after it
  (`gms-heal-<job name>-<sha7>`, at any nesting) succeeds.
