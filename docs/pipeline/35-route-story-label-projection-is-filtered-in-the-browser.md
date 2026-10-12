## Route Story label projection is filtered in the browser (`gomoveshift/pipeline/route_story/scene/scene.js`)

`projectLabels` used to `map.project` every feature in `labels.geojson` and return all of them.
For the example-route bundle that file is 18,261 features / 3.92 MB, so a 4,312-frame film paid roughly
79 million projections and marshalled ~6.3 GB of JSON across the browser boundary — to draw about
nine labels per frame.

- **The filter is behaviour-preserving because every consumer already required `visible`.**
  `proof.place_labels` skips a label without it, and both marker lookups (`frame_qa
  .marker_xy_from_labels`, `proof`'s own) test `entry.get("visible")`. Dropping the off-canvas
  entries in the browser therefore cannot change a pill, a marker position or a QA verdict.
  `tests/test_route_story_label_filter.py` pins that contract from the Python side, so a future
  consumer that starts reading invisible entries fails instead of silently changing the film.
- **The marker is pushed AFTER the filter.** It is not a label — it carries `kind: "marker"` and no
  name — and must ride back whatever its own `visible` flag says.
- **Only labels that can reach the canvas are projected at all.** Returning fewer entries did not
  make a frame cheaper: `map.project` is terrain-aware, and a CPU profile of the page (example-route bundle, consumer GPU) put **2,269 ms of a 2,801 ms pose — 80% of every frame of the film — inside
  that one `labels.map` callback**; MapLibre's own render was ~60 ms. `cullLabelsToRadius` now drops
  every label farther from the view centre than `labelCullRadiusM`, a bound taken from the camera's
  own geometry (zoom, pitch, field of view, centre elevation): the distance at which the frame's
  top-corner ray reaches ground `LABEL_CULL_FLOOR_M` below sea level, plus the camera's offset from
  the centre, plus 10%. The same poses settle in **120 ms**. Measured equivalence: 48 of 48 real poses
  (zoom 9–16.5, four bearings, pitch 0 and 60) return identical visible labels at identical pixels.
  - **Never cull on `map.getBounds()`.** With terrain on, the bounds come from where the corner rays
    hit the mesh, so a near ridge shrinks them; that version dropped far-field labels the frame
    plainly shows (5 of 48 poses, a named gap on the far ridge among them).
  - The cull stands down (`Infinity`) when the horizon is in frame or the radius passes 300 km, where
    one latitude's metres-per-pixel no longer describes the view.
- **This edit invalidates the frame cache by design.** `_scene_code_identity` hashes `scene.js`, so
  every previously captured frame misses. Land it BETWEEN renders, never mid-render.
