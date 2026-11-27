### Fixed

- `metrics growth-review` matches `creatorContentType` values whatever their case or underscores (`VIDEO_ON_DEMAND`, `videoOnDemand`), so long-form watch hours and Shorts views are counted. An Analytics answer with rows but no known content type now logs `PARTNER_PROGRESS_UNMAPPED` and reports the counts as unmeasured instead of 0.
