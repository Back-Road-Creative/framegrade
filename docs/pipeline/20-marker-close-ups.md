### Marker close-ups (`review_pack.marker_closeup` / `marker_sheet`, written by `run_preflight`)

`marker-sheet.jpg` is a fourth sheet: one 320 px crop centred on the traveller marker per sampled
preflight frame, captioned with that frame's route clock and its measured dot-to-line gap, the
over-ceiling ones filled in the warning colour. It exists because the defect it catches was found
by eye on a finished film and by nothing else — partway through, the marker began trailing behind
the end of its own bright travelled line, and a four-pixel gap is invisible in a 320 px contact-sheet
tile of a whole 1920 px frame while being obvious in a crop.

Two things keep it honest, both by construction rather than by discipline:

- The tiles are crops of the **preflight walk's own frames**, and their gaps come from calling the
  same `frame_qa.line_end_to_marker_px` that `check_frame` calls, on those same frames. There is no
  second render at second sample times, so the sheet costs nothing extra and cannot show a clean
  marker on a film the gate then refuses. `MARKER_GAP_CEILING_PX` is read off
  `FrameQaThresholds().max_line_end_to_marker_px` rather than restated.
- It is written **before** the gates raise, on the failing path too (`preflight._write_marker_sheet`
  is called in the `FrameQaError` handler as well as on the way through) — the failing run is the
  one an operator most needs the close-ups for.

The gap is measured in the writer rather than read out of a `FrameQaResult` precisely so the failing
path, where no result exists yet, can still produce a sheet. A frame whose renderer reported no
`last_marker_xy` is tiled whole and captioned `gap —`: dropping it would silently shorten the walk,
and printing `0.0` would read as a perfect result on a frame nothing was measured on. `run_preflight`
takes the sheet's path as `marker_sheet=` and returns it as `PreflightResult.marker_sheet`; `film.main`
passes `<pack dir>/marker-sheet.jpg` only on a `--preflight-only` run — the close-ups are a thing a
person reads, and there is no person on a render run. `review-pack.json` records it as a full path
under `marker_sheet`, like `preflight_sheet`, because the JSON is written before either file exists.

The pack is written **before** the frame checks run, on every `--preflight-only` run with a
usable track — so a run the frame checks refuse still leaves the pack behind; the refused run is
the one a person most needs to look at. It needs no `--receipt`. The pack never fails the run for
what it is showing: a photo that cannot be decoded (a RAW with no usable preview, a truncated file)
gets a warning-coloured tile naming the file and "COULD NOT BE READ"; a trip with no kept photos or
no media rows still gets a sheet that says so. Originals open through Pillow first, then the
camera preview, then the film's own still loader, so a NEF folder works. In the table only "not
placed" is warning red; "before the trail"/"after the trail" rows are amber — a notice, since those
photos are expected and play as bookends. Known limit: a source video with no resolved anchor
always reads "not placed", never before/after.


**The film command measures what it delivered** (`delivery.check_delivery`). Preflight judges
sample frames *before* the long render; nothing read the finished file back, so the command
printed a path and exited 0 for a film at the wrong size or frame rate, with 96 kHz audio, short
of frames, or with a black stretch in it — each of which was found by hand on early renders.
After the final encode the command probes its own output against the requested
`--width/--height/--fps`, the shared colour-tag contract (`color_tag_issues`), one 48 kHz stereo
AAC stream, and a frame count that matches the duration, then scans the **whole** picture (a film
is minutes long; the long-form validator's three sampled windows are for multi-hour finals) for
black runs of 0.5 s and frozen runs of 2 s. The report lands in the receipt as `delivery_check`.
Any spec departure, an unreadable file, a scan that did not run, or a black run is an `issue`:
each is printed to stderr and the command **exits 2** with the film and receipt still written.
A frozen run is recorded and is not an issue — a held card is a choice for a reader to judge.
Tests fake the render and encode with placeholder bytes, so `tests/conftest.py` stubs the
command's check to a clean report; `tests/test_route_story_delivery.py` installs its own.
