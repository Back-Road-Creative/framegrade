### Fixed

- `health heal --recipe` on a drive whose recipe row `health check` had already resolved did
  nothing and said nothing. Registering the recipe now reopens that row (stamped `reopened_at`,
  attempt history kept), so the heal runs.
- An open `drive_failed` row filed before `health check` read a job's own `output.log` kept a null
  `resume_recipe` forever, so `health heal` refused it ("no readable resume.json") on every pass.
  `health check` now fills a missing `resume_recipe` or `log_path` from the current detection.
