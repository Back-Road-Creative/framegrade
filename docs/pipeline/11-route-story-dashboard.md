## Route Story dashboard (`gomoveshift/pipeline/route_story/dashboard.py`)

`dashboard_state_at` combines a frame's source-clock position with cumulative route facts for both
the preview and final-render HUD. Total distance remains the accepted original-route total; traveled
distance, elapsed/moving time, ascent, descent, and moving average stop at the frame's recording
time. A route gap yields unavailable current telemetry and no fabricated connecting edge. Display
code owns formatting, units, precision, layout, and text; this layer owns only SI facts.

Displayed grade is windowed over trailing travel rather than the last observed edge, so one short
noisy edge cannot swing the shown number; with less history than the window it reports the grade
over whatever travel it has instead of going unknown. The two windows are exposed publicly as
`GRADE_WINDOW_M` (75.0 m) and `PACE_WINDOW_S` (20.0 s) so the export receipt's dashboard block
(`hud.dashboard_receipt`) can quote them as an auditable smoothing-lag figure. Both the grade and
pace/speed windows are bounded to the current segment only — a window that would otherwise reach back past the segment's
own first point truncates there rather than pulling in an earlier segment's points across a
recording gap. Inside a gap, `current_heart_rate_bpm`, `current_cadence_rpm`,
`current_temperature_c`, `current_speed_mps`, and `current_grade_percent` all go `None`; cumulative
facts (traveled/elapsed distance, ascent/descent) keep their last observed value. `cadence_frames`
(default 1, no-op) snaps the
frame down to the start of its bucket before resolving telemetry, so every frame in a bucket reports
identical values — a readable HUD update rate instead of every-frame digit flicker — deterministically
by frame time. `RoutePoint`'s optional heart rate, cadence, and ambient temperature (parsed from a
GPX's Garmin `TrackPointExtension`, see below) are interpolated onto the frame the same way as
elevation and stay `None` when the source GPX never carried them.

<a id="route-story-dashboard"></a>
