### Fixed

- `health heal --recipe` on a drive whose recipe row `health check` had already resolved did
  nothing and said nothing. Registering the recipe now reopens that row (stamped `reopened_at`,
  attempt history kept), so the heal runs.
