## Route Story review pack (`gomoveshift/pipeline/route_story/review_pack.py`)

The preflight contact sheet above catches a map/tile defect; it says nothing about the things an
operator actually rejects a *finished* multi-hour film for — an unrecognisable opening shot, a
dull photo, or a photo/video landing in the wrong route position. `--preflight-only` now writes a
small review pack alongside the contact sheet so those are visible in seconds, before the long
render: `<output stem>-review-pack/opening-sheet.jpg` (one tile per opening-camera fraction, `0.0`
through `1.0`, labelled), `stills-sheet.jpg` (one row per kept photo — the raw original beside the
photo composited through the exact same `_load_still` → `prepare_still` → `still_card_frame` pipeline
`assembly._append_photo` uses, called at `progress=0.0` so the pack shows the film's own first
presentation frame of that photo and a colour-grade or framing change to that shared pipeline shows
up here for free), `placement-table.png` (file,
kind, status, route clock, reason — red for not placed, amber for before/after the trail), and
`review-pack.json` (the sheet file names, the `photo_selection_block()` receipt block verbatim, and
kept/dropped/unplaced counts derived from it).

Every builder is pure and independently testable (`opening_sheet`, `stills_sheet`,
`placement_table`, `write_review_pack`) — no CLI parsing or GPX/EXIF loading inside. A row's
placement status comes from `classify_placement`, which takes `film.classify_bookend`'s own verdict
for the photo and never re-derives before/after from the clock: only a photo the film actually
plays as a bookend reads "before the trail"/"after the trail", a photo outside the track that is
not a bookend (too far outside, or `--no-bookends`) reads "not placed" with that reason, and a
bookend best-pick dropped is left out like any dropped route photo. The stills sheet lists kept
bookends in the film's play order — before, the route's photos, after. Assembling the sheets' inputs
(rendering opening frames, pairing kept photos, building placement rows) is `gather_review_inputs`
— it lives here rather than inline in `film.main` because it needs more objects in scope than that
function's `--preflight-only` branch has editing room for; `film.main` passes through the
`PhotoMoment`s, `RoutePoint`s and rasters it already built for the preflight sample above.

The pack's still recipe is a CALL into the film's own, never a second composite: `render_still_as_shown`
is two lines because a pack frame that differs from the render is worse than no pack at all — the
operator approves a frame the render never produces. It drifted exactly that way once, still
compositing a 0.91 letterboxed inset after the film had moved to a full-bleed push-in, and the test
that pins it asserts pixel equality against `still_card_frame(prepare_still(...), w, h, 0.0)` rather
than re-describing the recipe.
