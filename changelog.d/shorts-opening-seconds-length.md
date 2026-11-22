### Changed

- Shorts are ranked by their opening seconds: a window needs its first 4 seconds and its whole length above the day's bar, and each clip's end is trimmed back to the last second above the bar (never under the minimum). The per-Short window mean, opening mean and bar are recorded in the shorts audit sidecar, outside its hash.
