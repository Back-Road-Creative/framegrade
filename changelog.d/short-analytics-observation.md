### Added

- Short observations now carry per-video Analytics `engagedViews`, watch time, average view duration and `averageViewPercentage`, an `age_bucket` (24h/72h/7d) and, when Analytics fails, an `analytics_unavailable` reason. A later pass never nulls stored values.
