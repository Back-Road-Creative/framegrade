## Trail film: the day's photos and resume retry (`pipeline/drive_phases/trail_film.py`)

Follow-up to the in-drive trail film in `16-route-story-film.md`.

### The film gets only the day's photos

`film.main` refuses a `--photos-dir` holding media it cannot place on the route, and a drive's photo
folder spans every day. The pool is the profile's `photos-dir` (else `state.source_dir`). The photos whose
zoned EXIF capture instant (`extract_exif_capture_metadata(...).capture_utc`, the clock `film.main` places
a photo by) falls inside the day's GPX span are symlinked (copied if the filesystem refuses) into
`day_<date>/trail/photos_<date>`, and that directory is the film's `--photos-dir` (it overrides the
profile). The folder is rebuilt on every attempt, so a retry never inherits stale links.

- The span is the day's first to last fix as `film.main` selects it (`gpx.select_gpx_day`), widened by
  `film.BOOKEND_MAX_OUTSIDE_S` on both sides so the same-day bookend photos still reach the film
  (before-recording photos are then excluded by the film itself unless `--pre-trip-photos`).
- A photo with no zoned capture time cannot be placed on the day's clock and is left out, with a warning
  naming it. A missing pool, or a GPX with no complete segment for the day, is a recorded trail skip.
- `TrailSteps.gpx_span` and `TrailSteps.capture_utc` are the injectable seams (real by default).

### Resume

A resumed day is rebuilt from disk, which knows nothing of its trail outcome. `_resume_trail_films`
(`phase_bc_day_first.py`, called right after the resume partition) restores the outcome persisted in the
day's `deliverables` record onto it: films whose file still exists, and the recorded skips. Before this
a resume rewrote the record without them.

`retry_trail_skip` then re-runs only the trail step (never the day) when the day has recorded skips, no
film, `--no-trail` was not passed, and every input a film needs now exists: the day's GPX, a rendered LF,
a prepared imagery bundle and `--trail-profile`. While any is missing the recorded skip stands and nothing
renders. A retry replaces the old skips with its own outcome and the record is saved at once, before later
days encode. A day whose inputs all exist but whose proof or film failed is retried on every resume.

Tests: `tests/test_trail_film.py` (`test_trail_film_gets_only_that_days_photos`,
`test_trail_skip_retried_on_resume`).
