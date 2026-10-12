## Route Story bookend photos (`gomoveshift/pipeline/route_story/film.py`)<a id="route-story-bookend-photos"></a>

A same-day photo shot just outside the GPX recording — the camera came out
before the GPX logger started, or stayed out after it stopped — used to be
ordinary unplaced media: refused before any render unless `--allow-unplaced`,
and even then never drawn. `classify_bookend(photo, track_start_s,
track_end_s)` is the one pure rule every site shares: a photo whose anchor
status is `unanchored_outside_track` (never any other unplaced reason — a gap
inside the recording or a coordinate conflict stays ordinary unplaced media),
with a known EXIF capture time, within `BOOKEND_MAX_OUTSIDE_S` (6 hours)
before the track's first point or after its last, is a "bookend" —
`"before_recording"` or `"after_recording"`. Further out than that is not the
same outing and is refused/dropped exactly as before.

A bookend never claims a route position: it plays with the same unplaced-style
still card shape as any other unmapped photo (`_append_photo(...,
mapped=False, ...)`, no pin, no map position) but its own caption, never the
generic "not placed on the route map" one — `UNPLACED_PHOTO_CAPTION` on a
deliberate bookend reads as an error to a viewer. `_append_photo` takes an
optional `caption: str | None` (default `None` keeps today's
`UNPLACED_PHOTO_CAPTION` for an ordinary unplaced photo); `film.py` threads
`BOOKEND_BEFORE_CAPTION` ("Before the trail") and `BOOKEND_AFTER_CAPTION`
("After the trail") through its two bookend render calls. `before_recording`
bookends, in capture order, play right after the opening and before the first
map leg; `after_recording` bookends, in capture order, play right after the
last map leg and before the recap. A bookend never triggers the
unplaced-media refusal and is never left out of `unplaced_media`/
`photo_selection.not_considered` — those two now only name photos that are
neither mapped nor a bookend.

Bookends are best-of-look-alike-group selection candidates too — `main()`
runs `select_kept_photos` over mapped photos **and** bookends together
(`select.py` already treats a candidate with `placement=None` correctly, so
two look-alike bookends collapse to one winner exactly like two look-alike
mapped photos, and `--keep NAME` rescues a bookend the same way). Kept
bookends count toward `resolve_photo_seconds`'s `kept_count`, so they are
covered by the 35% stills-runtime cap same as any other still.

The receipt's `photos` entries for a kept bookend carry `"placement":
"before_recording"`/`"after_recording"` (instead of `"unlocated_memory"`) plus
`"seconds_outside_track"`; a top-level `"bookend_photos"` list names every
kept bookend in play order (present, possibly empty, on both the full and the
`--preflight-only` partial receipt). `--no-bookends` restores today's
behaviour — every non-mapped photo goes back through ordinary unplaced
handling, bookend or not.

**A `before_recording` bookend is excluded by default** (a product decision:
a driving film shouldn't open with pre-trip photos). Leaving it out is a
deliberate exclusion, not unplaced media: the default run renders without it
and without `--allow-unplaced`, it never appears under `"unplaced_media"`, and
its name is recorded on the top-level `"excluded_pretrip_photos"` list
(present, possibly empty, on both receipts). `--no-bookends` still sends every
bookend, pre-trip or not, back through ordinary unplaced handling. `--pre-trip-photos`
opts back into the old default: a `before_recording` bookend plays exactly as
described above. `after_recording` bookends are unaffected either way.
