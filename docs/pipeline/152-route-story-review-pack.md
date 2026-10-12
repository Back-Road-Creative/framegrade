## Route Story review pack (`gomoveshift/pipeline/route_story/review_pack.py`)

`gather_review_inputs`/`write_review_pack` build the operator's fast, pre-render look at a
Route Story film — an opening sheet, a stills sheet, a placement table, and one JSON tying
them together — so a `--preflight-only` run can be approved or rejected in seconds, before
the multi-hour long render starts. Every builder here is pure: it takes already-rendered/loaded
inputs and writes files; `film.py` computes those inputs and calls in. `--review-pack` opts a
full (non-`--preflight-only`) render into writing the same pack before the long render, at the
same point in `main()` -- it requires `--preflight-frames > 0` or `--preflight-only` (argument
validation fails fast otherwise, there being nothing rendered yet to build a pack from).

`write_review_pack`'s `motion_clips` field carries continuous short clips of the delivered
film's most hectic camera sections, cut post-encode by `write_motion_clips` (a lossless
`ffmpeg -c copy` per window) once `film.py`'s `_review_motion_clips` sees the scene renderer
exposed at least two `rendered_poses`. `planner.hectic_windows` picks the highest-magnitude
`bearing_speed_deg_s` peaks, spread at least `min_gap_s` apart so one violent turn can't claim
every slot, and expands each to a `window_s`-wide window clipped to the film. A reviewer
watches the motion directly instead of inferring it from stills; a cutting failure is recorded
into the pack as a single `motion_clips_error` entry, never raised, and the field is an empty
list — never omitted — when no pack was requested or too few poses were rendered to measure.
