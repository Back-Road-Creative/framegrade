### Fixed

- A failed destination place lookup (every Overpass mirror down) no longer drops the LF title to the bare county, which duplicated a live video. The render now raises `PlaceLookupUnavailableError`, a transient failure `health heal` retries; a lookup that answered with no place keeps the fallback, and `--label-dry-run` reports the LF as `place lookup unavailable (transient)`.
