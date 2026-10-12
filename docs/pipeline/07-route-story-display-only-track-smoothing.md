## Route Story display-only track smoothing (`gomoveshift/pipeline/route_story/display_track.py`)

Consumer GPS jitters side to side at walking pace, so the raw drawn line and moving marker
saw-tooth across a path the hiker actually walked smoothly. `smooth_display_segments` produces a
smoothed copy of the display segments for the drawn line only. It changes nothing about what is
*measured*: distance, pace, elevation gain, grade and every other figure in `RouteSummary` and the
Route Story receipt are always computed from the original recorded points via `metrics.py`, never
from a smoothed point. `gpx.py`'s "never manufacture route evidence" rule is untouched by this
module for the same reason — a smoothed point is display pixels, never written back onto the
recording and never reported as a measurement.

The smoother is a distance-windowed (not point-count-windowed, since sampling rate varies)
weighted average of lat/lon in a local metre-scale plane. Elevation, `time_s`, and every other
`RoutePoint` field pass through untouched, and every output point keeps its original `time_s` — the
moving marker is looked up by time against the same segments used to draw the line, so a changed
clock would reopen the "dot lags the line" bug. It never smooths across a segment boundary, and the
first and last point of every segment is pinned exactly (the start/finish pins and the video
arrival point must not move). `max_shift_m` bounds how far any point may move from its recorded
position; a correction that would exceed it is clamped onto that radius. This is also what keeps a
genuine switchback readable: the window necessarily reaches into the far leg near a sharp turn, but
the clamp bounds the correction there too, so the apex only moves a bounded amount instead of being
rounded off.

The module is a pure function — no I/O, no clock, no randomness. `film.py`'s `drawn_segments`
helper calls it on `clean_display_segments`'s output (default on, `--no-display-smoothing` to draw
the raw cleaned line instead) and is the single place both `render_route_story_film` and `main`'s
preflight sampling get their drawn line from, so a preflight frame can never sample a different
line than the film actually draws. `render_route_story_film` draws the smoothed line for every map
leg, the video-arrival hold and the recap map, but every measurement — `summarize_track`, leg
bounds, pacing and the dashboard rail — stays keyed to the un-smoothed cleaned segments. The map
renderer's own dashboard rail is normally computed from the very segments it draws the line from, so
both renderers carry `measured_segments` and a `measuring(segments)` method: the film hands them the
recorded points to measure and the smoothed ones to draw. It is done inside the renderer on purpose
— the rail fill is translucent, so re-stamping the right numbers over a rail already drawn leaves
the wrong digits ghosting through. A renderer with no `measuring()` is refused while smoothing is
on, never silently measured off the smoothed line. The receipt's
`display_smoothing` block (present in both the full and `--preflight-only` receipts) carries
`enabled`, `recipe` (`DISPLAY_SMOOTHING_RECIPE_VERSION`), `window_m`, `max_shift_m`,
`max_shift_m_observed` and `mean_shift_m`/`points_clamped` from the returned `SmoothedTrack`.
