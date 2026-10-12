### Route Story map-leg pacing and the honest speed badge (`assembly.plan_map_legs`)

Marker motion across a map leg is now pace-true under one global route-clock
time-compression factor, not a per-leg log-scale reading-time clock: `assembly.plan_map_legs`
takes every map leg's `(start, end)` route-clock bounds up front and returns one factor `N` —
total route seconds across every leg divided by their total on-screen seconds — plus each leg's
frame count (`route_seconds / N * fps`, always at least 1, never dividing by zero for a
zero-length leg). Every leg then gets exactly `route_seconds / N` film seconds, so a stop stays
a stop and a fast descent moves fast, everywhere in the film; `map_leg_seconds` (the old
minimum/log1p/maximum curve) no longer decides an individual leg's duration — it only supplies
`plan_map_legs`'s *default* total on-screen seconds (the sum it would have produced over the
same legs), so a film's length does not change unless `render_route_story_film`'s `map_seconds`
keyword (CLI: `--map-seconds`) overrides that total explicitly. `film._map_leg_bounds` computes
the ordered leg list once (mapped-photo legs, an optional video-arrival leg, the final leg to
`end_time`) so `render_route_story_film` and `film_receipt` agree on the same bounds and the
same factor; the receipt records both under `time_compression` and `map_seconds` so the on-screen
badge is auditable against the export. `--target-seconds` (e.g. 90) solves that total instead
(`still_selection.solve_map_seconds`, closed form): intro, recap, each kept still and each mapped
beat's settle hold keep their seconds, the pacing cap still picks the per-still seconds, and the
map legs share the rest. It is solved once from the final kept photos and written into
`args.map_seconds`, so preflight, the film and the receipt read one number. A placed source
video counts as fixed runtime too (`extra_seconds`): its ffprobe duration plus
`VIDEO_ARRIVAL_HOLD_SECONDS`, and an unreadable duration is refused rather than guessed. A target
the fixed scenes already fill is refused. Mutually exclusive with `--map-seconds`.

The badge itself (`×N`, integer-rounded, floored to `×1` below 1.5) is read from
`hud._active_time_compression`, module state `hud.set_active_time_compression` sets immediately
before and clears immediately after each `_append_map_leg` call (`film._append_map_leg_with_badge`)
— not a `DashboardState` field — because the dashboard for every `_render_map` call (opening,
photo-arrival overlay, recap) is built inside `layers.py`'s renderer call chain, which this scoping
does not need to touch: `None` (cleared) draws exactly as before, so only the frames actually
produced by a leg's `render_leg` call pick up the badge.
