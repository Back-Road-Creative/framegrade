## Route Story media anchoring (`gomoveshift/pipeline/route_story/anchors.py`)

`anchor_media_to_segments` evaluates every selected segment, so media after a device outage is not
accidentally searched only against the first segment. Embedded coordinates remain the display
location while an explicitly zoned capture clock may place media on the timeline. Timestamp-only
media can be interpolated inside one observed track edge when EITHER the edge duration is under the
configured time gap limit OR the edge's ground distance is under the configured `_m` gap limit — a
recorder that only samples every 30-60 s still produces a wide-in-time edge at a rest stop, and the
ground distance there is the honest test. An edge wide in both time and ground remains unanchored.
An edge accepted only through the ground-distance rule sets `interpolation_method` to
`"stationary_edge"` for audit. If embedded coordinates and the clock-derived position differ by more
than the configured safety radius, the speed needed to reach the embedded fix from the nearest track
fix on each side of the clock time decides the outcome: beyond `implausible_speed_factor` times the
track's own 95th-percentile edge speed (floored at 3 m/s so a near-stationary track can't make every
disagreement look impossible), the embedded fix is rejected as physically impossible — the result is
`timestamp_interpolated` at the clock-derived position, with the rejected fix and required speed
recorded (`rejected_latitude`/`rejected_longitude`/`required_speed_mps`). A merely-far but reachable
disagreement keeps `coordinate_time_conflict`, embedded coordinates on screen, and the film labels it
for review rather than placing a false arrival. Private surfaces may retain the anchor decision and
timeline position while omitting latitude and longitude (including any rejected fix) entirely.

`capture_start_from_filename` reads a phone clip's recording START time out of a recognised
filename pattern (Pixel `PXL_YYYYMMDD_HHMMSSmmm`, UTC), returning `None` for any non-matching name
rather than guessing. A container's `creation_time` is file-finalisation time, not recording start,
so callers deriving a video's capture time should prefer the filename start when the name matches.

`place_media` produces a `MediaPlacement` that decides WHEN and WHERE independently instead of
collapsing both into one status. `timeline_s` is set whenever the (optionally calibrated) capture
clock is known and falls inside the recording window, regardless of whether a position could be
derived — a positional uncertainty never discards a certain time. `position_source` records how the
marker location was derived: `embedded_gps`, `clock_interpolated` (a short observed edge),
`nearest_observation` (any raw observation, including an isolated fix a continuous-edge search would
drop, within a caller-set time radius of the clock; the time distance is recorded as `radius_s`), or
`none`. Embedded GPS that disagrees with a clock-corroborated position by more than the configured
distance is a conflict; the same track-derived implausible-speed test used by
`anchor_media_to_segments` decides whether WHERE defers to the clock (`conflict_rejected=True`, the
rejected fix kept as `conflict_latitude`/`conflict_longitude`, the required speed as
`required_speed_mps`) or keeps the embedded fix on screen, still flagged `conflict` for review.
`calibrate_clock_offset_s` derives one camera body's clock offset from a single still
whose embedded GPS actually corroborates against the track; the resulting offset is passed to
`place_media` as `clock_offset_s` (with `clock_source` naming the calibrating still) to correct that
body's other originals.
