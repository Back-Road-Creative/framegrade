## Route Story dashboard rail (`gomoveshift/pipeline/route_story/hud.py`)

The HUD rail (`_draw_dashboard`/`_dashboard_cells`) shows one MOTION cell, ELEVATION, GRADE,
DISTANCE, optional HR/CAD/TEMP, and an optional TIME cell — no WARP badge (removed; a viewer had
no way to know what the ×N figure meant). The honest speed-up factor it showed stays in the export
receipt's `time_compression` key, which was always separate from the badge.

**Column set** — `dashboard_fields_for(segments)` decides the whole film's dashboard columns
once, before any frame renders: MOTION/ELEVATION/GRADE/DISTANCE always appear; HR/CAD/TEMP each
appear only when *some* point in *some* segment of the film carries that telemetry (a mid-film
sensor dropout does not remove the column); TIME appears only when a clock anchor is active. Once
frozen, a column's position never moves, and a frame whose telemetry is temporarily `None` renders
`—` in that column rather than shifting the columns after it. The value font is fit once per film
against each column's widest plausible content (e.g. DISTANCE against the film's total, not the
current frame's traveled distance) instead of being refit per cell per frame, so digit width never
changes column layout mid-film. That fit starts from the rail's base value font (`max(16, width // 48)`)
and only shrinks, so freezing never enlarges the text past the panel.

**Motion cell** — exactly one of PACE (min/mi) or SPEED, decided once for the whole film by
`motion_cell_mode(moving_average_speed_mps)`: PACE when the *trip's* overall moving-average speed
is ≤ `_MOTION_PACE_THRESHOLD_MPS` (4.5 m/s, a brisk-run ceiling), else SPEED; unknown reads SPEED.
This is a whole-render decision (module state set once, like the badge it replaces), not a
per-frame one, so the label never flips mid-film as the cumulative average crosses the threshold.
On screen the cell reads "RECENT PACE"/"RECENT SPEED" (`hud._dashboard_cell_label`), disclosing
that the value is eased over the trailing window (`dashboard.PACE_WINDOW_S`), not the instantaneous
edge — the underlying "PACE"/"SPEED" field-key strings used for column freezing and
`_FIELD_VALUES` lookups are unchanged.

**TIME cell** — local time of day at the frame's route-clock moment, e.g. `2:41 PM` (no seconds, no
leading zero, no timezone text), or omitted entirely rather than guessed. Resolution order for the
UTC offset: (1) every zoned photo's EXIF offset, when they all agree — a disagreement is logged and
falls through; (2) `--utc-offset` (`+05:30`/`-04:00`, strict `argparse` validation); (3) neither →
omitted. The offset itself is recovered per photo from `extract_exif_capture_timestamp` (camera-local,
naive) minus `extract_exif_capture_metadata(...).capture_utc` (UTC), since the latter's `tzinfo` is
always UTC and does not retain the original `OffsetTimeOriginal`. `render_route_story_film` anchors
the clock once, at `recording_start_utc` converted to the resolved offset — route-second 0 is
exactly `source_started_at` (`gpx.py`'s `_to_route_points` epoch) — and each frame adds
`DashboardState.recording_time_s`, the source recording clock `dashboard_state_at` already reads.
It must never add `elapsed_seconds`: that sums observed segments only, so every recorder pause
would leave the readout that much behind the real time of day for the rest of the film.

**Panel placement** — the panel is painted from `layout.dashboard_panel_top` for
`layout.dashboard_panel_height` rows. On portrait that puts its bottom edge at
`height - layout.short_bottom_safe_px`, clear of the platform-reserved bottom safe zone, which
lifts its top rows into the map field; on landscape, where that safe zone is zero, the panel
still runs to the frame's full height. Preflight frame QA masks the composed frame from
`preflight.map_field_rows` — the earlier of the map field's end and `dashboard_panel_top` — so
the rail's white value text is never counted as page-white (it alone read as 0.60% on a clean social frame and would fail every portrait preflight). The fractions' denominator stays the
map field's full area (`check_frame(field_pixels=…)`): masking the rail can only lower a fraction,
never tip a frame that was under a cap over it (the rail's 72 px floor reaches into the map field
on small landscape exports such as the 640x360 test renders).

The export receipt's `"dashboard"` block (`hud.dashboard_receipt`) records what was actually drawn:
`{"motion_cell": "pace"|"speed", "time_cell": "exif_offset"|"cli_offset"|"omitted", "utc_offset":
"-04:00"|null, "pace_window_s": 20.0, "grade_window_m": 75.0}` — the last two quantify how much
smoothing lag the motion and grade cells carry (`dashboard.PACE_WINDOW_S`/`GRADE_WINDOW_M`).

**Profile** (`profile_geometry.py`, `hud.draw_profile`) — a synchronized trip telemetry strip
drawn directly above the dashboard panel, x-axis DISTANCE travelled, never time: a long stop or a
slow crawl never stretches the chart. `build_profile(segments)` walks every source segment once and
turns it into one or more `ProfileSegment`s — a gap between two source segments (recorder pause,
ferry, device swap) is a visible break, never a joined synthetic climb, and adds no distance of its
own; a point with unknown elevation ends the current segment the same way, without discarding its
own distance contribution. A run of fewer than two drawable points is dropped from the chart but
still advances the running distance for whatever comes after it. The vertical scale never shrinks
below `MIN_PROFILE_SPAN_M` (30 m), so a dead-flat route still reads as a profile instead of a wall
of GPS noise filling the whole chart height.

The cursor (`ProfileGeometry.cursor_distance_m(at_time_s)`) reads the exact same recording clock as
the map marker, so the two never drift apart through a stop, a hold, or a seek: inside a segment it
interpolates distance linearly between neighbouring points; inside a gap between segments it HOLDS
at the previous segment's end distance rather than drifting through the gap or jumping ahead to the
next one — this is deliberately "synchronized trip telemetry," never a live navigation instrument.
`SceneMapRenderer.measuring(segments)` builds the geometry once per film, from the same display
segments the dashboard measures; it is not rebuilt per frame.

Deliberately not drawn yet (first release only): climb bands, a grade trace, and a
dedicated climb summary card.

**Caps and units** — `hud.set_active_display_caps`/`set_active_display_units` scope the HUD's
speed/grade caps and speed/pace/elevation units to the whole render, the same module-state shape
as the dashboard clock and motion-cell mode above. `render_route_story_film`'s CLI (`film.main`)
sets both right after resolving the film's `ActivityProfile` (`route_story/activity.py`) and its
units: the caps come from `hud_speed_cap_mps`/`grade_cap_percent` on that profile (hiking 4.0 m/s /
35%, cycling 25 / 20, driving 60 / 12), and the units come from the resolved `--units`/region value
(`"imperial"` → mph/`ft`, `"metric"` → km/h, min/km pace, meters). Both reset to `None` once the
render finishes, restoring the legacy 8.0 m/s / 30.0% caps and imperial units — what a lone
`render()`/single-frame call outside this CLI path always draws with. The export receipt's
`display_filter.hud` block records the resolved caps: `{"speed_cap_mps": <float>,
"grade_cap_percent": <float>}`.
