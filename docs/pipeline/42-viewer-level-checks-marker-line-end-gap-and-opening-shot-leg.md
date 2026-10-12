### Viewer-level checks: marker/line-end gap and opening-shot legibility

Two checks close a gap the pixel-fraction and travelled-line-width detectors
above leave open: both are things an operator's eye catches instantly on a
rendered frame that every existing detector reads as clean, because both are
about what a *viewer* sees rather than a count of matching pixels.

`line_end_to_marker_px` measures the gap, in px, between the edge of the
marker's exclusion disc and the nearest travelled-line-coloured pixel outside
it, sharing `_travelled_line_colour_mask` (the colour-tolerance mask plus
per-pixel marker distance) with `travelled_line_min_width` rather than
duplicating that mask logic. 0.0 means the line reaches right up to the disc
edge, as intended; a growing gap is the "the red marker dot lags behind the
end of the bright travelled line" defect an operator rejected a film for — a
stale/late line-progress update draws the marker correctly but stops the line
short of it. Returns `None` when no travelled-line pixel is visible within
the search radius at all (e.g. the very first frame), never a guessed 0, the
same convention as `travelled_line_min_width`. `check_frame`, when given
`marker_xy`, also runs this check against
`FrameQaThresholds.max_line_end_to_marker_px` (default 4.0, PROVISIONAL) and
records the result on `FrameQaResult.line_end_to_marker_px` (appended last so
existing positional/keyword construction is unaffected).

**Cost.** A render profile put `check_frame` at 453 ms of a ~1 s
film frame: the line mask and a full-frame distance grid were built twice per
frame over all 1920×923 pixels, and the three pixel detectors each made an int16
copy of the frame. Now `check_frame` builds one `_LineWindow` — the mask and
marker distances over the window within `_LINE_RADIUS_PX` +
`_LINE_WINDOW_MARGIN_PX` (64) of the marker — and both line checks read it; the
detectors use `cv2.inRange` on the uint8 frame. 12 ms per frame, same verdicts:
cv2's distance transform treats beyond-the-edge as line, so the crop can only
change a distance already larger than the margin (a solid line-coloured blob
~120 px across, not a stroke). `tests/test_route_story_frame_qa_speed.py` pins
equality with the frozen full-frame implementation on 50 film-sized frames.

`check_frame` also grew a `marker_geometry_only` flag: when true, it skips
the void/unloaded/flat-black pixel-fraction detectors (reporting their
fractions as 0.0, no problems from them) and runs only the two marker checks
above. `run_preflight` uses it for a `checks_raw_frames` renderer (the scene
renderer): that renderer already QA's void/unloaded/flat-black on its own
*raw* capture before composing the dashboard/label pills on top, so
re-running those detectors on the composed frame would spuriously trip on
the dashboard rail — but the marker-geometry checks look only near the
marker, not at the whole frame, so they are safe to run there and, before
this change, were the one check that never ran at all for that renderer
(the branch skipped every check).

`check_opening_frame(rgb, thresholds)` covers the second defect: an opening
shot that renders (no void/unloaded/black pixels, a normal-width line) but
reads to a viewer as dark, flat and blurred. Three independent floors, each
producing its own plain-English problem string when tripped: mean luma (0-255
grayscale mean; a dark opener), colourfulness (the Hasler-Süsstrand metric on
RG/YB opponent colour channels; a flat/desaturated opener), and sharpness
(variance of the Laplacian of luma via `cv2.Laplacian`; a blurred/
out-of-focus opener). `min_opening_mean_luma` (45.0) and `min_opening_colourfulness` (12.5) are
retuned against the actual rejected film's own bundle
(from the route-story cache), rendered at fractions 0/0.25/0.5/0.75/1 for
the real trailhead target: the OLD trailhead-locked, ungraded camera measured mean luma 36.6-89.3
(worst 36.6), colourfulness 12.0-44.3 (worst 12.0); the graded travel camera measured mean luma
60.6-118.2 (worst 60.6), colourfulness 13.8-55.0 (worst 13.8). Both floors sit strictly between the
two worst-cases, so the old opening fails (on luma, at fraction 0) and the graded one clears every
sampled fraction. The graded camera's own sharpness over that bundle measured 5.4-11258.3 (worst 5.4), and that
worst case is exactly why it does NOT set the floor: those samples come from rasters at ratio
0.17-0.36, below `check_opening_sharpness`'s own 0.5 floor, which the fixed camera never selects.
`min_opening_sharpness` (50.0) is calibrated differently — see the "sharp handoff"
paragraph above: the same bundle's `naip` band still carries the old, too-narrow bbox
`bands._sharp_handoff_focal_margin_m` now widens, so it still exhibits this defect and
cannot stand in for the fixed opening. `check_opening_sharpness` (proven by geometry, not this
bundle) is the actual gate against raster-resolution blur; this floor is a late, pixel-content
backstop, held at its original 50.0 rather than left at the 4.0 the still-broken camera
needed to pass. The synthetic dark/bright fixtures in `tests/test_route_story_viewer_checks.py`
still exercise all three floors' extremes with wide margin either side.
`run_preflight` takes an optional `opening_frames` sequence and, after the
contact sheet is written (so a failure still leaves the sheet for a human or
an agent to look at), raises `RoutePreflightError` naming every floor
tripped and its measured value.
