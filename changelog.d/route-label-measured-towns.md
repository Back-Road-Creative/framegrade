### Fixed

- A road-only LF label ("L6099, County Cork") now broadens to the measured towns with the most retained time ("Courtmacsherry & Timoleague, County Cork", level `places`) before falling back to the county, so Cork drives stop colliding on the generic county title. A cross-drive exact title clash gets the drive date (" (Sep 19, 2025)") instead of a permanent refusal; a same-project or undated clash still refuses.
