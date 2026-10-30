### Added

- `metrics collect` pulls Facebook Page daily insights (`page_media_view`, `page_total_media_view_unique`, `page_video_views`, `page_impressions_unique`) into `channel_daily` under `facebook:<metric>`, labelled by the Pacific day, backfilling from 2026-06-01 on the first run. A metric Graph refuses is reported unavailable, never stored as zero; the run prints `FB_PAGE_DAILY <metric>=<rows|unavailable> ...`.
- Facebook Reel observations record `reach` and the metric that answered (`reach_metric`); reach stays unknown when no candidate answers.
