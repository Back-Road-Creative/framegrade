### Drives (`console/drives.py`, `GET/POST /drives`, `GET/POST /drives/gates`, `GET /drives/preview`)

`GET /drives`: launch form (brand, source dir, output dir, region, publish day, LF day, dry-run
checkbox default ON, plus a "Pipeline flags" fieldset -- `--lf-mode`, `--lf-target-duration`,
`--superlong`, `--shorts-target`, `--shorts-gap` -- each labeled with its CLI flag).
`POST /drives/run` calls `drives.build_drive_argv()` -- the exact `gomoveshift drive <directory>
[--region R] --output-dir O --no-review [--dry-run] [...]` argv an operator would type -- via
`console/jobs.py` (`"drive"` added to `ALLOWED_SUBCOMMANDS`), redirecting to the job detail page.
Blank curated flags are omitted; an invalid numeric field or an `lf_mode` outside
`drives.LF_MODE_CHOICES` is a 400, never a 500. `brand` selects a
`console/settings.py::known_brands()` slug (default `current_brand()`, 400 if unknown) and picks
`env=brand_launch_env(slug)` for the launched job -- not itself a CLI flag.

**A refusal is an inline correction, not a page.** The 400 is still a 400, but it renders
`drives.html` again with every submitted value in place rather than a separate error page: the
form has thirteen inputs, a browser does not reliably restore a POSTed form, and a mistyped digit
used to cost the other twelve. `drives_run` collects the raw fields into one `form` dict and
`_refuse(message)` re-renders the tab context with it plus `error`; the template reads every
`value`/`selected`/`checked` from `form` when it is present. **Dry run is the one that cannot be
defaulted on re-render** -- it starts ticked on a fresh form, so re-ticking one the operator
deliberately cleared would turn a corrected form into a live run; the template ticks it only when
there is no `form` at all or when `form.dry_run` is set. An unknown `brand` takes the same path,
because the brand is a field on this form like any other. Numeric fields go through `_number`,
which names the field in the message: `int("seven")` says "invalid literal for int() with base 10",
which tells an operator nothing about which of thirteen inputs to fix.

`GET /drives/gates?output_dir=&date_key=` reads the matching `_gates/gate_cd_<date_key>.question.json`;
`POST /drives/gates` validates `decision` against the pending question's `choices` (400 via
`HTTPException` on a mismatch) and writes the `.answer` per the gate-answer contract in
`pipeline/review_gate.py`, so a headless drive run resumes without a TTY. `write_gate_answer` then
retires the question file to `.question.answered` -- the pipeline's answer lookup only ever reads
the `.answer` path, so this is safe, and `list_pending_gates`'s `*.question.json` glob naturally
stops listing an answered gate.

`list_pending_gates` also globs one level down (`<output_dir>/*/_gates/` -- the PROJECT
dir `_consume_or_ask` actually writes under) and every gate id, not only `gate_cd_*`; `date_key`
stays populated only for `gate_cd_<date>` ids, for the existing answer route.

**The gate page shows the footage it is asking about, not just the question text.**
`drives.gate_video_paths(resolved_output_dir, question)` reads the pending question's own
`artifacts` list (`review_gate._consume_or_ask` already writes `lf_paths + json_files` there) and
keeps only entries whose suffix is one of `drives.SUPPORTED_VIDEO_SUFFIXES` (this service's
supported video formats) AND whose **resolved** path is `is_relative_to()` the already-validated
`resolved_output_dir` -- a foreign or `..`-escaping artifact is silently dropped, never surfaced.
`drives.gate_media_items()` turns that list into one row per video (`index`, `name`, best-effort
`title` paired off the matching `<project>_lf<NN>_youtube.json` metadata artifact by the shared
`lf<NN>` token, and `readable`), which `GET /drives/gates` renders as a still frame plus title, or
an honest "could not be read" line when the file is missing/empty -- never a broken image icon or
a silently empty list. `GET /drives/gates/frame?output_dir=&date_key=&index=` serves that frame:
the request supplies only `index` (a position into the gate's own `gate_video_paths()` result),
never a filesystem path, and the route reuses `web.app._read_frame_sync` (the same ffmpeg
timestamp-accurate single-frame grab `/api/frame` uses) rather than reimplementing frame decode.
`output_dir`/`date_key` on this route go through the identical `resolve_allowed_output_dir` /
`is_valid_gate_date_key` checks as `GET /drives/gates` above -- the frame route is not a second,
looser path to the same bytes.

**The gate asks whether you PLAYED it, so the page plays it.** Gate C/D's question is
"Played each LF and reviewed all metadata for <date>?", and a still frame cannot answer it; the
approval it records is what stops a delivery reaching the queue
(`cli/handlers/enqueue_plan.py` compares `compute_gate_cd_hash` against
`state.gate_cd_confirmation_hash`). `GET /drives/gates/video?output_dir=&date_key=&index=` serves
the file itself through `drives.resolve_gate_video_media`, which goes through `gate_frame_path`
rather than repeating its lookup -- so playback is contained by exactly the rule the frame route
already is, and a path that resolves outside the validated output dir, or whose suffix is not a
supported video, never appears in the list an `index` can reach. `drives.GATE_VIDEO_CONTENT_TYPES`
is the one suffix table and `SUPPORTED_VIDEO_SUFFIXES` is derived from it, so a suffix that passes
the allowlist can never lack a content type (a `KeyError`, i.e. a 500, on a legitimate video).
`FileResponse` answers Range requests, so the reviewer can seek rather than download the whole file
to move. The still frame is not removed: it becomes the `poster=` shown before playback
starts, and an unreadable artifact keeps its "could not be read" row rather than rendering a broken
player.

**`output_dir`/`date_key` on these routes and `GET /drives/preview` are never trusted
straight from the request.** All three (`GET/POST /drives/gates`, `GET /drives/preview`) resolve
`output_dir` through `drives.resolve_allowed_output_dir(output_dir, known_output_dirs)`, where
`known_output_dirs` is exactly `drives.output_dirs_from_job_argvs()` over *this console's own*
`jobs.list_jobs()` rows -- the same trustworthy, tenant-scoped source `/drives/gates/pending`
already used, extended one level down (`_gates_dirs`'s "PROJECT dir" rule) so both the run's
`--output-dir` and its per-day project subdir resolve. The comparison is on `Path.resolve()` on
*both* sides, never a string compare -- a `..`-laden or symlinked `output_dir` that resolves to an
allowed dir passes, and a string-identical-looking one that resolves elsewhere is refused; a naive
`str(a) == str(b)` check would get either case backwards. A non-matching `output_dir` is a 404 whose
body never echoes the submitted path (previously `{output_dir!r}` in the detail message) -- echoing
it would let an attacker use the error page itself as an oracle for which paths exist. `date_key` is
independently checked by `drives.is_valid_gate_date_key()` against `^[A-Za-z0-9_-]+$` -- the only
shapes real gate ids use (`YYYY-MM-DD` or the literal `final` from `drive_orchestrator.py`'s
`_pending_gate_cd`) -- so a traversing `date_key` (`../..`) is refused even when `output_dir` itself
is allowed; that's a second, independent barrier, not a substitute for the `output_dir` check. The
hidden `output_dir`/`date_key` fields in `drives_gates.html`'s POST form still say *which* run is
being answered, but are never authority -- the server re-validates both server-side.

**The same check covers `GET /shorts/csv` and `GET /shorts/csv/download`**, which an earlier
revision missed -- both now resolve `output_dir` through the identical `resolve_allowed_output_dir` call
before calling `shorts.find_shorts_csv()`, use the *resolved* path for that lookup, and 404 with a
fixed, non-echoing message on both "unknown output_dir" and "no shorts CSV found" (the second used
to leak just as much, since it confirmed a directory exists and is readable).

`GET /drives/preview?output_dir=` is read-only: `drives.collect_metadata_previews()` globs
the same `*_youtube.json` files `review_gate.compute_gate_cd_hash` hashes, loads each LF's generated
title/description, and runs the real content gates (`check_metadata_quality` -- includes the
grammar check, `check_label_quality`, `check_description_redundancy`) against the metadata dict --
never re-implements gate logic. 404s when `output_dir` isn't one this console has ever launched a job
against or has no generated metadata yet.
