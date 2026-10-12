# Route Story shot planner

`gomoveshift.pipeline.route_story.planner` is the renderer-independent camera math for the
Route Story drone move: pure Python producing one `CameraPose` (from
`gomoveshift.pipeline.route_story.opening`) per frame. It does not import MapLibre or any
renderer; a preview and a final export consume the same frame-addressable plan.

## Shapes

- `Shot` — one named move (`descent`, `reveal`, `arc`, `arrival`, `return`, `pullback`, or the
  not-yet-implemented `follow`) over `[start_frame, end_frame)`. `pose_at(frame)` resolves a pose.
- `ShotPlan` — an ordered tuple of shots covering the whole film. `frame_count` is the last
  shot's `end_frame`. `pose_at(frame)` looks up the owning shot. `validate()` raises on a gap,
  an overlap, or a pose discontinuity at a join (compared within `1e-9`).

## Builders

- `descent(start, target, fps, seconds=16.0, *, final_zoom=16.0)` — 14-18s zoom-in, target-locked
  by interpolating the camera centre in normalized Web Mercator space (not raw lat/lon), pitch
  held at 0.
- `reveal(at, fps, seconds=7.0, *, final_pitch=45.0)` — 6-8s pitch tilt from `at.pitch` to
  `final_pitch` (35-50°), position held fixed.
- `arc(center, base, fps, seconds=7.5, *, degrees=30.0)` — 6-9s orbit of `base` around `center`
  by 20-40°, radius held constant (haversine/great-circle math), always looking inward.
- `arrival(base, target, fps, seconds=2.0)` / `return_to(base_after, current, fps, seconds=0.7)`
  — short settle/push-in transitions. `return_to`'s pose function returns `base_after` verbatim
  for `frame >= end_frame`, so a join back onto the pose it left is exact, not just within
  tolerance.
- `pullback(end, route_bbox, fps, seconds=6.0, *, width_px, height_px, padding_px)` — rises from
  `end` to a zoom computed by `fit_zoom` so the whole route bbox fits the given output aspect.

All builders raise `ValueError` outside their documented duration/angle ranges, and every
builder's easing endpoints have zero velocity and acceleration (`opening._smootherstep`, reused
rather than duplicated).

## Geometry primitives

- `fit_zoom(bbox, width_px, height_px, padding_px)` — MapLibre-convention zoom (512px world
  tiles at zoom 0) that fits a lon/lat bbox inside a viewport, clamped to `[0, 22]`.
- Web Mercator projection helpers (`_mercator_xy` / `_mercator_lonlat`) and great-circle helpers
  (`_haversine_m`, `_initial_bearing_deg`, `_destination_point`) used by the builders above.

## Motion bounds

- `motion_report(plan, fps)` returns per-frame bearing speed/acceleration/jerk, pitch speed, and
  zoom velocity (bearing differences taken on the shortest arc, never a 359→0 spike).
- `assert_smooth(plan, fps, limits=None)` raises `ValueError` naming the worst frame against
  `DEFAULT_MOTION_LIMITS` (bearing ≤25°/s, bearing accel ≤40°/s², pitch ≤15°/s, zoom ≤1.2/s).

## Not implemented

- `follow()` — chase-drone shot with look-ahead heading smoothing over an arbitrary track.

Terrain-aware clearance (`camera_altitude_m`, `clear_terrain`) and the stable-key-order JSON
serialization of a `ShotPlan` (`plan_to_json`, `write_plan_json`) live in
`gomoveshift.pipeline.route_story.terrain`.
