## Close-up subject (photos)

- **What it is** (`gomoveshift/pipeline/photos/subject.py`): a `SubjectCategory.CLOSE_UP` ("close_up") reading, for frames that are "not landscape (within feet) or macro". It is a subject label only; it never changes a pixel.
- **Rule.** Runs right after the operator, bracket and astro rules and before wildlife, so a night astro frame focused near stays astro, a bracket member stays a bracket, and a 250 mm frame at 2 m is a close-up, not wildlife.
  - Focus distance finite, above 0 and at most `CLOSE_UP_FOCUS_M_MAX` (3.0 m, about 10 ft): close-up at `CONFIDENCE_MODERATE` (0.65), evidence `focus distance 1.2 m <= 3.0 m`.
  - A macro signal on top (lens model containing "macro", or implied magnification `f/(d-f)` of at least `MACRO_MAGNIFICATION_MIN`, 0.25) lifts it to `min(0.65 + 0.1, 0.95)` = 0.75.
  - A macro lens name with no focus distance at all reads close-up at `CONFIDENCE_WEAK` (0.45). A macro name with a known far focus (and magnification under 0.25) is NOT a close-up: a macro lens shoots scenery too.
- **Inputs.** `CaptureOptics` gained `focus_distance_m` and `lens_model` (both default `None`). `read_capture_optics` still reads only focal length and aperture, since Pillow cannot read focus distance; callers pass `recognise_photos(paths, optics={path: CaptureOptics(...)})` (the exiftool records supply it) and a mapped path wins over `read_capture_optics`.
- **Refusals.** Focus 0, negative, NaN, infinite or missing gives no focus-based close-up. Operator `.subject.json` sidecars accept `close_up` (the enum validates them); an operator correction always wins.
- **Look routing.** `styles.SUBJECT_STYLE_ROUTING` is left without a `CLOSE_UP` entry on purpose: distance and magnification say nothing about light quality or dynamic range, the same reason wildlife and portrait are unrouted. A close-up gets the default scene-ranked styles.
- **Tests:** `tests/test_photo_close_up_subject.py` (synthetic optics and tmp_path files only).
