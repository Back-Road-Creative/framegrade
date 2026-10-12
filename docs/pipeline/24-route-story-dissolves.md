### Route Story dissolves (`assembly._dissolved`, `film._last_saved_frame`)

Map → still and still → map cuts are `DISSOLVE_SECONDS` (0.5 s) cross-dissolves. They sit inside
frames the scenes already own, so no scene gets longer and `--target-seconds` stays exact: a
still's first frames blend from the settled map frame (`_append_photo(dissolve_from=)`), and the
next map leg's first frames blend from the still the viewer last saw, read back from disk by
`film._last_saved_frame` (`_append_map_leg(dissolve_from=, fps=)`). Only the saved copy is
blended; the leg's own `last_image`, which the settle hold replays, stays the pure map frame.
A stand-in scene that writes no pixels yields no previous frame and therefore no dissolve.
