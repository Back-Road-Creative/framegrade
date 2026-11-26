### Fixed

- Tracking: a day reel that was queued before day reels got a tracking record is no longer
  dropped from metrics. Its publish, or `metrics backfill-ids`, now writes the missing record
  instead of skipping the platform id, so it is measured and counted by the growth review.
