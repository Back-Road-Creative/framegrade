### Travelled-line minimum width

`travelled_line_min_width` measures the actual on-screen width, in px, of the
bright travelled-line colour within a radius of the marker (default 200px) —
not the layer's `line-width` paint value. MapLibre drapes `travelled-line`
onto 3D terrain by extruding its width in the map's own 2D plane rather than
in screen space, so at a grazing camera angle (a steep pitch looking nearly
along the road — common on steep, switchback terrain) the extruded offset
itself foreshortens on screen; a real capture at pitch 65 measured a hairline
there even with the paint `line-width` raised from 3 to 8. `check_frame`
takes an optional `marker_xy` and, when given, also runs this check against
`FrameQaThresholds.min_travelled_line_width_px` (default 2.0); with no
`marker_xy` the check is skipped, never guessed, and
`FrameQaResult.travelled_line_width_px` stays `None`. `summarize` rolls the
thinnest measured width (and its frame) into the receipt as
`travelled_line_width`.

**The distance transform runs on the whole colour mask; only the RIDGE is sampled inside the ring.**
Cropping the mask to the ring first makes the ring's own cut read as background, so a stroke crossing
that boundary measures about a pixel across at the cut while measuring its true width everywhere
else. A render was refused 1,841 frames in for exactly that: 975 line px in range, a 4.6 px median
ridge, reported width 1.0 px. Measured on the whole mask the same frame reads 3.39 px, and 60 cached
frames from the same film re-check clean.

**A width is only measured when there is a length to measure it along.** A ridge shorter than
`_MIN_MEASURABLE_RIDGE_PX` (8) returns `None`, the same answer as a fully hidden line. An earlier
render was refused for a frame that was correct: the walker was still ~50 m from the start, so the
stroke sat almost wholly under the marker disc and the start pin, leaving a six-pixel island that
read as a "1.0 px hairline". A genuine 1 px hairline measures a ridge as long as it is -- 13 px
across a small proof frame, 30 px in the unit fixture -- so the floor keeps the detector's teeth.

Method: a colour mask (tolerance around `#ffcf3f`) is distance-transformed;
its local ridge (ties for the local max in a small neighbourhood)
approximates the line's medial axis, and twice a ridge pixel's distance
approximates the line's true width there (corrected by 1px for the
pixel-centre-to-pixel-centre convention `cv2.distanceTransform` uses). The
reported width is the 5th percentile over the ridge, not the strict minimum:
a real switchback puts several sharp bends within the sampling radius, and a
bend's own join pixel reads a couple of px narrower than the stroke either
side of it from corner geometry alone, not a rendering defect.

The shipped fix is `scene.js`'s `drawRouteOverlay`: the WHOLE on-screen route
(dim, `pose.route`) and travelled line (bright, `pose.travelled`) are redrawn
on a plain 2D `<canvas id="overlay">` (`scene/index.html`), positioned
absolutely on top of MapLibre's own canvas and sized once in `init()`. A 2D
canvas stroke has a constant screen-pixel width by construction, immune to
terrain draping. An earlier, radius-limited version (redrawing only within
`OVERLAY_RADIUS_PX` of the marker) left two visible defects: a segment whose
far vertex fell outside the radius was dropped whole (a real hairline could
still reach the marker), and a hard seam where the redraw stopped and the
draped hairline resumed. Drawing the entire visible route/travelled line in
one pass removes both by construction; the draped `route-casing`/
`route-line`/`travelled-line` layers are set `visibility: "none"` (never
removed, so their paint stays documented) and no longer draw anything.

`drawRouteOverlay` draws, in order: the not-yet-travelled route as a light
line over a darker halo (one clean stroke replacing the old
dark-casing-under-dim-line double-track look); the travelled line on top; then
the marker's
own disc is knocked back out of whatever line pixels landed under it with
`globalCompositeOperation = "destination-out"`, at a radius read from the
marker layer's own `circle-radius`/`circle-stroke-width` paint (never
duplicated as a guessed constant) plus 1px, so the marker always stays
visually on top without a z-order trick. It is called from `renderPose`'s
`onIdle` success path only — the sole path that ever resolves with a frame;
a failed settle rejects instead, and `capture.py` re-renders the whole pose
rather than screenshot a half-settled one, so there is no second path that
could leave a stale overlay on a fresh frame.

#### Render speed: stop retiling a hidden line

`renderPose` used to also call `setRoute(pose.route)`/`setTravelled(pose.travelled)` every frame,
feeding the (now-hidden, see above) `route`/`travelled` MapLibre GeoJSON sources -- a leftover from
before `drawRouteOverlay` took over painting the visible line. `setRoute` deduped its `setData` when
the route was byte-identical (the whole-trip geometry, normally unchanged pose to pose), but
`setTravelled` had no such guard and called `setData` on every frame with the already-travelled
portion of the route -- a coordinate list that only ever grows over a leg (and over the whole film),
so MapLibre retiled a strictly bigger `LineString` every frame for a layer nothing ever painted.

A render measured the effect directly: per-frame map-leg time climbed
from ~0.7 s early in the film to ~11.0 s by frame ~1450, tracking the growing travelled line rather
than any per-leg property (`gomoveshift.pipeline.route_story.assembly`'s "map leg N/M frames, X
s/frame" progress line, `_append_map_leg`). Removing the two calls (`renderPose` no longer calls
`setRoute`/`setTravelled` at all; the functions stay defined, just unused, so the hidden layers' paint
properties stay documented) cannot change a single output pixel: `drawRouteOverlay` never reads the
MapLibre sources, only `pose.route`/`pose.travelled` directly, and those are untouched.
`tests/test_route_story_scene.py::TestSceneJsTravelledLine::test_render_pose_no_longer_feeds_the_hidden_maplibre_sources`
pins the removal.

#### A single hidden vertex is an artifact, not a ridge

`strokeRouteParts` hides the part of a line terrain stands in front of, one
vertex at a time: `isVertexOccluded` projects the vertex to its own screen
point and asks MapLibre to unproject that pixel back, calling the vertex hidden
when the raycast lands somewhere else on the ground (tolerance
`OCCLUSION_TOLERANCE_PX`, 4px of ground resolution at the vertex's own
latitude and zoom).

That round trip is not perfectly stable. Near a terrain tile edge — or wherever
the DEM sample the ray happens to hit belongs to the next post over — it can
come back a few metres off for ONE vertex while both neighbours recover to
float precision. The drawing pass used to stream `moveTo`/`lineTo` per vertex,
so that single false verdict ended the subpath and the next visible vertex
started a fresh one. When the artifact landed two vertices short of the marker
the remaining stub was a fraction of a pixel long and drew nothing, leaving the
travelled line short of a marker that is always drawn — `frame_qa`'s
`max_line_end_to_marker_px` defect, reported by an operator as the red dot
lagging behind the bright line.

So every vertex of a part is now sampled (projected, viewport-tested and
occlusion-tested) **before** anything is drawn, `repairIsolatedOcclusions`
clears any hidden vertex whose immediate neighbours are both visible, and only
then are the visible runs stroked. Off-canvas and occluded are tracked
separately because only the occlusion verdict is repairable: an off-canvas
vertex has no pixels to draw whatever its neighbours say.

Measured on the pose that refused a render (zoom 15.32, pitch 60): of the travelled line's last eight vertices, seven recovered
0.0m and one recovered 13.8m against a 12.5m tolerance — 1.3m over, with
visible vertices either side and the marker itself visible. The drawn line
ended 7.2px from the marker against the 4.0px ceiling; with the repair the same
pose measures 0.01px.

Repairing a run of ONE cannot draw a line through a mountain: a real ridge
hides a contiguous run (the same capture recovered 37–990m on genuinely hidden
vertices, and every span is subdivided to `OCCLUSION_SUBDIVIDE_PX`, so a real
crossing always produces several in a row). Any run of two or more is left
hidden, and a hidden vertex at either end of a part has no pair of neighbours
to vouch for it. The repair walks in place, which never lets a real occlusion
through: a just-repaired vertex can only vouch for the next one if that one's
other neighbour is visible too.

#### A run of false occlusions right before a visible marker

`repairIsolatedOcclusions` only clears a LONE hidden vertex; a false-verdict
run of two or more, sitting right next to a visible marker, still stopped the
line short. A render refused on exactly that: "travelled line ends 4.01px
from the marker" at frame 10 (bearing 200.99, pitch 60, zoom 15.32) — two
consecutive false verdicts adjacent to a marker that was itself visible.

The travelled line's own last vertex is always the marker's own coordinate
(`_travelled_coordinates` in `scene_renderer.py`), so a visible marker means
the ground right under it is, by construction, not occluded.
`forceTrailingSpanToMarker` runs after `repairIsolatedOcclusions` and before
the part is stroked: when the marker is visible, it walks back from the
vertex just before the marker, clearing the occlusion verdict of every
vertex in that trailing run until it hits one already visible. A hidden run
further back along the same part — not adjacent to the marker — is left
alone, since that is a real ridge; an invisible marker forces nothing, so a
hidden marker is never given a fabricated line. It runs only on the
travelled line's own last part (the one that actually ends at the marker),
never on the dim/bright whole-route strokes, which are not tied to the
current marker.

#### A shallow false-occlusion run between visible samples

A render refused on "travelled line ends 7.01px from the marker" (frame 55,
zoom 15.3): the marker recovered 0.0 m, but two samples 8-13 px before it
recovered 17.0 m and 21.3 m against a 12.5 m tolerance, with one visible
sample between them and the marker. That run was too long for
`repairIsolatedOcclusions` and not adjacent to the marker, so
`forceTrailingSpanToMarker` never reached it.

Every sample now carries its raycast miss (`missM`), and
`repairShallowOcclusionRuns` (after the lone-vertex repair, before the
trailing force) clears any hidden run bounded by visible samples on both
sides whose every miss is under `SHALLOW_OCCLUSION_MAX_M` (30 m). Measured
false verdicts reach 21.3 m; the shallowest genuinely hidden vertex recovered
37 m, so a run with even one deep sample stays hidden, as does a run touching
either end of the part.

#### Stroke widths scale with the frame, in both renderers

All three route strokes are a fraction of the frame's **width**:
`layers.route_halo_width_px` (÷220, floor 5), `layers.route_line_width_px`
(÷500, floor 2) and `layers.travelled_line_width_px` (÷280, floor 4) — 8 / 3 /
6 at 1920, unchanged from what 1080p already shipped, and 11 / 5 / 9 at 2560.
`scene.js` computes the same three from the same divisors
(`routeHaloWidthPx` / `routeLineWidthPx` / `travelledLineWidthPx`).

They used to be flat constants in `scene.js` while the pins beside them
already scaled with frame height, so a 1440p master grew its pins and kept its
1080p line: the travelled line stayed 5px against everything else a third
larger. `tests/test_route_story_line_width_parity.py` reads the divisors out
of Python and asserts the exact JavaScript expressions, so a width fixed in
one renderer and not the other is red rather than a film that changes
thickness when the renderer changes. The travelled line's floor is the highest
of the three because frame QA measures that stroke
(`FrameQaThresholds.min_travelled_line_width_px`) — scaling it below the gate
would make a small proxy fail on its own thinness.

Each source-to-source span is walked through `strokeRouteParts`/
`plotVertex`: a long span is resampled every `OCCLUSION_SUBDIVIDE_PX` (12)
screen px (capped at `MAX_SUBDIVIDE_STEPS`, 256 — a route point that projects
to a wild or non-finite screen coordinate, e.g. one behind the camera, would
otherwise make the per-span subdivision loop run effectively unbounded on the
page's main thread and hang `renderPose`), and each resampled vertex is both
viewport-clipped (`withinViewport`, `VIEWPORT_MARGIN_PX` 60) and
occlusion-tested (`isVertexOccluded`) before being stroked, so a span behind
a ridge for only part of its length is cut exactly there rather than kept or
dropped whole.

`isVertexOccluded` uses only MapLibre's public API: for a vertex `p`,
`s = map.project(p)` then `q = map.unproject(s)`; with terrain enabled,
`unproject` raycasts the actual visible terrain surface under that screen
pixel, so a visible point's project→unproject round-trips back to itself
(confirmed empirically on MapLibre 6.10.0: near-zero float error, ~1e-8°),
while an occluded point's round trip recovers the occluding ridge's own
surface point instead — a real, internally self-consistent different lng/lat
(measured 37m–990m away on real example-route terrain at pitch 65-70, zoom 14-15.5),
not a raycast failure. `OCCLUSION_TOLERANCE_PX` (4px, scaled to metres via
`groundResolutionMetersPerPixel`'s standard Web Mercator formula at the
vertex's own latitude/zoom) is well under the smallest measured occluded
drift and comfortably above float-precision noise on a visible point.

<a id="route-story-viewer-checks"></a>
