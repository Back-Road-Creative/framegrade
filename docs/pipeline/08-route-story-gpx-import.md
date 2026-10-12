## Route Story GPX import (`gomoveshift/pipeline/route_story/gpx.py`)

`load_gpx_segments` uses `defusedxml`, retains GPX `trkseg` order, and expresses
timestamps relative to the first recorded point. It rejects missing, timezone-ambiguous, or invalid timestamps,
coordinates, and elevations rather than sorting or synthesizing an unsupported route clock.

Each point also carries the optional Garmin `TrackPointExtension` fields `hr`, `cad`, and `atemp`
(heart rate, cadence, ambient temperature) onto `RoutePoint` when present; a track with no extension
data leaves them `None` rather than inventing zeros.

`select_gpx_day` is the project-facing date-selection seam. It selects only complete segments whose
explicit source-timezone calendar date matches the requested date, then removes byte-for-byte-equivalent
point sequences once. It returns source, other-date, and duplicate exclusion counts for the project
receipt. It does not alter `load_gpx_segments`, sort a source, join segments, treat overlapping
geometry as a duplicate, or bridge a timestamp gap.

The day selection retains the selected source start as an aware instant. `captured_at_route_seconds`
in `route_story.anchors` translates only two explicitly zoned instants onto that source clock.
It refuses naive camera times and filesystem timestamps by construction. Location interpolation
remains governed independently by `anchor_media`'s observed-edge gap policy.
