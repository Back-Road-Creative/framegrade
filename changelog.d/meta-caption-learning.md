### Added

- Facebook and Instagram caption openers now learn from Meta performance. The nightly
  `metrics collect` writes each opener's day-7 Facebook initial plays and Instagram reach to
  `caption_evidence.json`, and the caption generator tries openers that clearly beat the pool
  first and those that clearly trail it last, through Waypost's `rank_captions`. Nothing changes
  until two openers each have 10 settled posts, and the anti-spam rule against repeating a recent
  opener still wins. Each post records the ranking it was built from (`caption_rankings`).
