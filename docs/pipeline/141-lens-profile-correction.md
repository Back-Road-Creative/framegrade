## Lens-profile correction (`lens_correct.py`) <a id="lens-profile-correction"></a>

`DevelopOptions.lens_correct` (default `False`) is a run-level opt-in: a `lensfun`-backed
distortion + vignetting + transverse-CA correction keyed on the photo's own EXIF camera/lens
identity, run in `develop_variants` before `_straighten`/`_crop` so geometry measures the
corrected frame. Off by default, it is a bit-identical no-op — `lens_correct.correct` is never
even called — so existing develop output is unchanged.

- The `lensfunpy` binding is an OPTIONAL extra: `pip install -e '.[lens]'`. Absent it, this stage
  is a clear, NAMED no-op rather than a crash or an import error at CLI start.
- `lens_correct.correct(rgb, path)` returns a `LensCorrectionResult` (`rgb`, `applied`, `reason`).
  `reason` is always one of three named no-op cases, or a success line, never a bare `False`:
  - `"lensfun extra not installed"` — the optional extra is absent.
  - `"no camera EXIF to match a lens profile"` — the photo carries no camera make/model EXIF.
  - `"no lensfun profile for <make> <model> / <lens or 'unknown lens'>"` — EXIF present, no
    matching entry in the lensfun DB.
  - On a match: `"corrected: <make> <model> / <lens.model>"`.
- The decision threads into `DevelopedPhoto.lens_correct_applied`/`lens_correct_reason` the same
  way `denoise_applied` does, and both are recorded in the grade sidecar's `measured` block
  (`grade_sidecar_payload`) — a reviewer can see whether a profile matched without re-running the
  develop.
- Flag: `photos develop --lens-correct` (no value; unset = off, bit-identical). In
  `config.PHOTO_KNOBS` it is a RUN-owned row (`run=True`, like `lut`/`straighten`/`crop`): it runs
  before geometry, so a look naming it would vary the EXIF-matched correction per variant of one
  photo, and it is refused in a look by name at config load.
- EXIF reading (`read_lens_exif`) and the lensfun DB lookup (`_find_profile`)/remap (`_remap`) are
  isolated in their own functions so tests exercise the no-op ladder and the applied path without
  the real package installed.
