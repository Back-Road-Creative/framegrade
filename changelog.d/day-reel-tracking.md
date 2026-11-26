### Fixed

- A day reel now gets a tracking record when it is queued, so its platform ids are stored
  when it publishes and `metrics collect` can measure it. Before, the route enqueued
  straight from its sidecar and the publish step's id registration failed for every reel.

### Added

- The weekly growth review scores `source:day-reel` against `source:drive`, so day reels
  and Shorts cut from recorded drives can be compared on the same day-7 views.
