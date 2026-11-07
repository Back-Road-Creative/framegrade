### Fixed

- `health heal` resumes a drive at most once per recipe per mirror SHA. Before, `health check` filed each failed heal job as a new incident row and heal resumed that row again on the same code, so one deterministic failure re-ran every hour and its live job kept deferring the staging sync that carried the fix. A heal of a heal no longer repeats `--skip-upload --no-youtube-upload`.
