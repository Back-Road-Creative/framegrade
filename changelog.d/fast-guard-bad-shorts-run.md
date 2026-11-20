### Added

- `gomoveshift health check` rule 9, a fast guard for a bad Shorts run. When the newest 3 settled YouTube Shorts each fall under a quarter of the channel's trailing median day-7 views, it raises one `short_bad_run` incident listing each Short with the causes recorded data shows: delivery status, duplicate place text (marked `c5_fixable`), late or bunched publish, place text against the sidecar's measured county, and missing metrics. It changes nothing itself.
