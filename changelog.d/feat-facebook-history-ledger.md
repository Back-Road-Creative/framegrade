### Added

- `metrics collect` builds a one-time ledger of Page Reels posted before first-party publishing (2026-09-10) at `<brand>/tracking/facebook_history.json`, tagged `source=metricool` with lifetime metrics, and prints `FB_HISTORY listed=<n|unavailable> stored=<n>`. Reels whose captions were hand-edited on 2026-10-02 carry `caption_edited_on`, and `in_matched_cohort` excludes them.
