## Trail film: the LF's label and the coverage gate (`pipeline/drive_phases/trail_film.py`)

Follow-up to the in-drive trail film in `16-route-story-film.md`.

The film is titled with the label of the LF it places as its source clip. `_lf_label` reads
`state.per_lf_ost_text[lf_index]`, the same persisted text the LF's YouTube title, burned text and thumbnail
use, and hands it to `film.main` as `--title`. `resolve_opening_title` uses an operator title verbatim, so the
opening card reads exactly the LF's label. The title overrides a `title` key in the `--trail-profile`. The media
step never resolves or edits a place name.

A day is skipped (recorded in `Deliverables.trail_skips`, logged as a warning) when:

- the LF has no persisted label, or
- its `ResolvedLocation` carries an `ost_signature` that differs from the label (the model was derived against
  other text), or
- it has no stamped `ResolvedLocation` (absent, or `ost_signature` unset), so nothing can vouch for the label.

Why the label comes from `state.per_lf_ost_text` and not from `RunSpec`: the drive does build a `RunSpec` per day
(`day_pipeline.py`), and it carries `per_lf_ost_text` and the state, so the label could travel that way. It does
not need to. `attach_trail_film` runs from the Phase B/C `_on_day_complete` callback, after `render_day` has
returned, so the `RunSpec` is already spent; reading the same persisted text from `state` gives the identical
string the LF's title and burn were built from, with no new field on `RunSpec` or `render_day`. The persisted
`per_lf_resolved_location` is used only for the signature check above; the coverage vote is a fresh one over the
film's own GPX span, because the stored model describes the LF's footage, not the whole track.

`film.py` and `opening.py` are untouched: `film.main` already takes `--title` and `resolve_opening_title` uses
an operator title verbatim, so passing the label needs no change inside the renderer.

Phase Q coverage runs on the film's GPX span, not the LF's footage. The track can cross counties the LF did
not, so `make_span_location(geocoder)` votes a `ResolvedLocation` over the day's selected GPX points
(`gps_enrichment.build_resolved_location`, the drive's shared geocoder). `check_trail_film_coverage`
(`validation/content_gates.py`) applies the existing `_coverage_failures` rule: every county holding at least
`LOCATION_MATERIAL_SHARE` of the vote must be named by the label. A miss is a skip, decided before the proof
renders. The gate fails closed: if no span vote can be computed (no geocoder wired, no result, unreadable track,
an error) the day is skipped with `coverage gate could not run`, and `TrailSteps.span_location` is a required
field so a step set cannot be built without it.

`TrailFilm.label` is stored in the day's `deliverables` record, so a resume shows what each film was titled.

Tests: `tests/test_trail_film.py` (`test_trail_film_title_equals_lf_label`). The drive wiring is covered by
`tests/test_phase_bc_day_first.py::TestTrailFilmDriveWiring`: `_on_day_complete` hands `attach_trail_film`, and
`_resume_trail_films` hands `retry_trail_skip`, a `span_location` that votes over the GPX with the drive's own
geocoder (pass `None` instead and both tests fail).
