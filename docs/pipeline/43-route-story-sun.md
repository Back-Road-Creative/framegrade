## Route Story sun (`gomoveshift/pipeline/route_story/sun.py`)

`light_at` computes the real solar position (elevation from `sun_track`, azimuth from the NOAA
formula implemented in-module) instead of the fixed `315°`/0.6 constants; `hillshade_exaggeration`
zoom-interpolates relief so low zoom keeps help reading terrain and high zoom doesn't double-light
NAIP's own shadows; `sun_sweep` and `light_for_pose` serve the time-lapse pullback and per-frame
chase light respectively. `SceneLight.to_scene(zoom)` serializes a light into the dict the scene
reads, folding in `hillshade_exaggeration` for that pose's zoom so callers never compute it
separately.

`SceneCapture.render(pose, light=None)` (`scene/capture.py`) takes that `SceneLight` directly: when
given, it replaces `pose["light"]` with `light.to_scene(pose["zoom"])` (raising if the pose has no
`zoom`) without mutating the caller's `pose` dict; with no `light` argument, any `light` already on
`pose` passes through untouched — `proof.py:render_frames` always supplies one via its per-frame
`light_fn` (see the Route Story terrain-proof render section below). `scene.js`'s
`setLight` mirrors that split: it always sets `hillshade-illumination-direction` from
`light.azimuth`, and only when `elevation`/`intensity` are both present does it additionally call
MapLibre's `map.setLight` (`position` in `[radial, azimuthal deg, polar deg from zenith]`, anchor
`"map"` so the azimuth reads as a compass bearing) and set `hillshade-exaggeration` from
`light.hillshadeExaggeration` — a night light's already-low intensity flows straight through, since
nothing on the scene side may invent moonlight.
