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

### The film as the LF's opening

When a film is rendered for a day, its satellite opening plus first map leg replaces the peak montage at
0 s of the LF the film was placed from (`pipeline/trail_opening.py`). A day with no film keeps its montage.
The film exists only after that LF rendered and its intermediates were purged, so the swap is a post-film
step on the FINAL LF:

- At burn time `run_day_pipeline` writes `lf_NN.lead.json` (the montage's probed length, the OST text,
  crossfade and logo, whether a clip insert is pinned, the earliest insert time). The film receipt gains
  `opening` (`fps`, `width`, `height`, `end_frame`), the frame where the opening card and first map leg end
  (`FilmTimeline.mark_opening_end`).
- `apply_trail_opening` cuts that span from the film (capped at `config.drive.montage_max_duration_sec`,
  scaled to the LF, silent), burns it with the same audited OST text via `apply_ost(..., website_cta=False)`
  (the closing CTA belongs to the LF's tail), re-encodes the seam to the first keyframe at or after the lead,
  stream-copies the rest of the LF, and joins them. Inserts already in the LF travel with the body, so stills
  keep landing on their footage.
- `lf_NN.trail_opening.json` is `pending` before the atomic replace and `done` after; a crash between them
  finishes on resume without redoing when the LF has the swapped file's length AND byte size (the pending
  marker records both: an opening as long as what it replaces leaves the length unchanged), else the swap reruns. Its key is folded into `lf_NN.ost.sha256` (`expected_fingerprint`), so a
  resume does not re-burn a swapped LF. A re-burned LF (`reset`) has its montage back and gets its opening
  through the normal film step of that run; `_resume_trail_films` covers only finished days. A re-rendered
  film (a different receipt hash) on an already-swapped LF replaces the earlier opening in place at once,
  cutting at that opening's recorded length (`opening_s` in the marker): the film step runs just before the
  day is recorded finished, and a resume never re-burns a finished day. The join is checked on the format duration and on
  the first video stream's own duration, so a video-only timestamp collapse fails the swap loudly and the
  original LF stays.
- A receipt with no usable opening, an opening under 1.5 s, an LF with no recorded lead (rendered before the trail opening existed),
  a pinned `--insert-clip` (its `@mm:ss` cannot move), an insert inside the lead, or a failed burn or join is
  a recorded `trail_skips` entry; the LF keeps what it had at 0 s, its montage or, for a re-rendered film,
  the earlier opening (the skip names which). Swapped LFs are listed in `Deliverables.trail_openings`.

- The opening is cut at the LF's exact frame rate (a rational such as 30000/1001, never a rounded 29.97003:
  a different timebase makes the stream-copy join collapse the body's timestamps), and burned with the brand's
  media profile. Each swap record carries `insert_delta_s` (the opening's length minus the planned montage), which
  Phase D adds to the recorded still-insert times so the insert chapter markers follow the new lead.

Known limit: the film itself still embeds the LF as rendered with its montage. Tests:
`tests/test_trail_opening.py`.
