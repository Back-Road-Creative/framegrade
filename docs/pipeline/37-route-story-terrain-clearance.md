## Route Story terrain clearance (`gomoveshift/pipeline/route_story/terrain.py`)

`TerrainSampler` reads a bundle's Terrain-RGB tiles back to metre heights (bilinear, pixel-centre
aligned, one decoded tile cached per `(x, y)`), at the manifest's `terrain.maxzoom` and scaled by
the manifest's own `terrain.exaggeration` — clearance must measure the heights the scene actually
renders, not the raw DEM. A lon/lat with no tile **raises** rather than answering 0 m: a guessed
sea-level answer is what flies a camera into a real mountain.

Camera geometry mirrors MapLibre's projection for 256px tiles and its default 36.87° vertical
field of view: `distance = (viewport_height/2) * metres_per_pixel / tan(fov/2)`, altitude is that
distance times `cos(pitch)`, and a pitched camera's own ground point sits `distance * sin(pitch)`
**behind** the centre along the reverse bearing — which is the point clearance samples, because
that is where the mesh can swallow the lens.

`clear_terrain` only ever pulls zoom out; longitude, latitude, pitch and bearing are the story and
dodging a hill by moving them would rewrite the shot. Per frame it computes the zoom that would put
the camera at `ground + margin_m` (default 150 m), then takes a **Lipschitz lower envelope** of
those caps at the plan's own `zoom_velocity_per_s` limit (forward then backward pass), so the
correction inherits `assert_smooth`'s bound instead of stepping frame to frame, and the pose stays
continuous at every shot join. `min_zoom` floors the result. Frames whose ground height is unknown
keep their planned pose — a missing tile is the fetch unit's failure to report, not flat ground.

`plan_to_json` / `write_plan_json` (schema `route-story-shot-plan/1`) serialize one pose per frame
in MapLibre units plus the shot boundaries; this is the handoff the headless scene capture consumes,
and it is written temp-then-rename so a killed render cannot leave a half-written plan.

**The enlargement cap** (`planner.py`): no shot may exceed the imagery `maxzoom` by more than
1.25x — `max_shot_zoom(imagery_maxzoom, enlargement=ENLARGEMENT_MAX)` returns
`imagery_maxzoom + log2(enlargement)`, and `ShotPlan.assert_within_zoom(max_zoom)` samples each
shot's boundary frames plus evenly spaced interior frames and raises naming the shot, frame and
zoom of the worst offender (the proof once arrived at z15.6 on z15 imagery, a 1.52x enlargement).
`render_proof_clip` (`proof.py`) reads the manifest's `imagery.maxzoom` and runs this check before any
render starts.
