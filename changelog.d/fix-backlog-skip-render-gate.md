### Fixed

- `health backlog` skips a row the render gate refuses (a `full_drive` day too short to fill a
  part) and launches the next eligible row, instead of refusing the region on every tick. The
  skip is listed with its `render gate:` reason; with no eligible row behind it the region is
  still `refused`.
