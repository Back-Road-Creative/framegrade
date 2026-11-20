### Fixed

- `queue sync-short-text` refuses a queued Short whose known stored Shorts arm (`yt_title_body`, `yt_hashtag_set`) the regenerated text no longer carries, and leaves it unchanged, instead of writing text that mislabels the experiment. An arm id outside `SHORTS_ARMS` is never rendered and never causes a refusal.
