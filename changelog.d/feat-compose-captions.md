### Changed

- Facebook, Instagram, TikTok and YouTube captions are composed from independently rotated parts (an opener, a data-only detail or an engagement prompt for interactive shorts) instead of four fixed templates per platform, so a 12-short batch never repeats an opener.
- Generic hashtags rotate per short from a pool of six instead of every post carrying `#POVDriving #RoadTrip`; the place and region tags stay mandatory.
- `enqueue-plan` refuses a batch whose caption or YouTube title opens with the same six words as text already spooled in the last 60 days (`cross-batch-repeat`).
