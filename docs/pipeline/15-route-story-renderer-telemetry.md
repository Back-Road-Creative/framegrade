## Route Story renderer telemetry (`gomoveshift/pipeline/route_story/frame_state.py`)

`frame_state_at` is the renderer-facing adapter from a storyboard frame to a source-clock GPS
observation. It interpolates position, elevation, speed, grade, and cumulative distance only within
an accepted observed segment. A timestamp between segments has no location or measurements, so a gap
transition cannot quietly draw a fictional road or display an invented speed. Map preview and final
rendering must use this payload for their marker and HUD rather than calculate presentation-time
metrics.

`FrameState.distance_m` is the cumulative route distance at that frame: 0 before the first observed
time, the route total after the last, and held at the previous segment's end distance through a gap
-- the same gap-holding rule `ProfileGeometry.cursor_distance_m` (`profile_geometry.py`) applies to
the HUD's profile cursor, so the two can never disagree.

The Pillow default renderer's `_render_map` (`layers.py`) resolves its drawn marker from the same
`frame_state_at` call it already builds for its dashboard at that frame, instead of a second,
independently-interpolated position -- marker and HUD are provably the same resolution.

**Not yet done:** `scene_renderer.py`'s own `_point_at_time`/`_marker_at` usage (its
alternate map renderer path) still resolves its marker independently of `frame_state_at` and has no
duplicate-timestamp guard. `layers._point_at_time` is kept only because `scene_renderer.py` imports
it; retiring it there is not yet done.
