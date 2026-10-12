### Route Story opening title and high-point callout (`hud.py`, `assembly.py`)

Two captions, both of which may only name things the run can evidence.

**The opening card** (`hud.resolve_opening_title`, drawn by `hud.draw_opening_title`) is two
lines burned over the opening camera. `--title` is the operator's own name for the trip and is
used verbatim -- never edited, appended to, or reordered. Absent one, the title is derived from
the labels file: the closest label to the track's first point, an arrow, and the closest to its
last, collapsing to one name when both resolve to the same place. A label further than
`hud.OPENING_TITLE_MAX_LABEL_KM` from the point is not eligible, and when nothing is inside that
ceiling the title line is empty rather than wrong -- the labels file carries every name in the
regional bbox, so an unbounded "nearest label" would caption a mountain ride with a town two
counties away. The subtitle is the recording day, taken straight from the required
`--recorded-on` argument rather than re-derived from the source video's UTC instant plus an
offset; an instant with no offset names a local day only by luck near midnight.

Both lines are measured and shrunk to `hud._TITLE_SAFE_WIDTH_FRACTION` of the frame width before
being drawn, through the same `hud._fitted_value_font` the dashboard uses. The card is centred, so
text wider than the frame is clipped at *both* edges and reads as a mangled title rather than a
long one: a render opened on "Example Ridge Trail -- Alpha Gap to Beta Summit" and burned "ge Trail -- Alpha Gap to Beta Su" for hours of render. The size had been
derived from the frame width alone, with no measurement of the string. The subtitle is fitted
against its uppercased form, because that is what gets drawn. The leftover margin is not slack:
the glyph stroke is drawn *outside* the measured advance, and a title-safe TV crop eats the rest.
The frame QA detectors look for void/unloaded/black pixels and cannot see text running off an
edge, so the fit is the only thing standing between a long trip name and an unusable opening.

The card is drawn opaque onto its own transparent RGBA layer and that whole layer is then faded
and composited. Passing an alpha inside `fill` does not work: on an RGB image
`ImageDraw.text` ignores the fill's alpha channel and draws at full strength, so the fade
silently becomes a hard cut. Filled rectangles -- including the opening's own dimming wash --
do honour it, which is why the trap only appears once a text overlay is the thing being faded.

The card fades in over the first tenth of the opening, holds, and is fully clear of the frame
before the descent ends (`OPENING_TITLE_FADE_OUT_END`). This is not cosmetic: the first map leg
starts the instant the camera lands, and a title still burning in at that hand-off covers the
route the leg exists to show. `film.main` resolves the same card for the preflight review pack's
opening frames, so what the operator approves is what the long render burns.

**The high-point callout** (`hud.high_point_callout`) is the track's highest recorded fix,
reported in feet at the badge style `hud.draw_high_point` draws -- a different accent colour and
a lower position than a media-arrival badge, so the two never read as the same event. A track
whose points carry no elevation gets no callout at all rather than a zero-foot one. Ties go to
the earliest fix. `assembly._append_map_leg` takes the callout as `(time_s, text)` and draws it
only on frames within `hud.HIGH_POINT_CALLOUT_SECONDS` of that route moment: a badge on every
frame of the leg would stop being a landmark and become permanent chrome, and a leg that never
passes the summit shows nothing. Both map-leg call sites in `film.py` pass the same callout, so
a summit that falls inside a leg split by a placed video is still captioned exactly once.
