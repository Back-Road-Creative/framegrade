### Changed

- The Facebook suppression detector reads the Page daily series from `channel_daily` once one has more than 28 days, trying `page_total_media_view_unique`, `page_media_view`, `page_video_views`, `page_impressions_unique` in that order; otherwise it keeps the reel-plays series. The cron line gains a trailing `source=<metric name|reel_plays>`.
