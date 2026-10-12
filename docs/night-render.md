# Private linear-stack rendering

`night_render.render_artifact(source, destination, recipe_name="natural")` accepts
an existing verified `.linear` artifact and an empty, exclusively allocated attempt
directory. It never decodes RAW, registers, stacks, or changes the original sidecar.
It returns the receipt written after both `master.tif` and `preview.jpg` succeed.
The caller owns resource admission and atomic publication of the attempt.

Both exports apply the same conventional recipe, then one sRGB transfer; the JPEG
is not stretched again. The master is genuinely 16-bit with an sRGB ICC profile.
No source EXIF is copied. Recipe values, producing module hashes, source manifest
and payload hashes, output hashes and dimensions are bound in `receipt.json`.
`validate_render` accepts reuse only when that complete receipt still matches the
source, current recipe/build, and both outputs. A TIFF alone is incomplete; preview
failure retains it and raises, with no success receipt. A fsynced
`preview-pending.json` records its verified source, recipe/build and master identity. Changed
source content or a new recipe/build requires a new attempt, never overwrite.

Completed means exported, not aesthetically accepted: `quality_review_required`
is always true. This version reconstructs the recorded horizontal horizon and
undoes the saved sky contribution quality weight. It does not yet reconstruct
foreground silhouettes, so tree edges can retain stack alignment ghosts. It does
not calibrate white balance or remove spatial color noise. These limitations are
recorded in each receipt; no automatic selection or publication follows a render.

The natural default is a baseline preview and may look dark on faint sky data;
completion does not imply aesthetic acceptance. Validation decodes both image
payloads and checks actual dimensions, master bit depth and ICC presence, as well
as hashes. The build is captured before processing and must remain unchanged.
The receipt is flushed, fsynced and atomically renamed before directory sync.

`recover_preview(source, destination, recipe_name)` recovers a matching partial
master under the caller's admission and exclusive worker lock. It validates source
payloads and the TIFF's hash, decoded dimensions, bit depth and profile against
`preview-pending.json`, then derives the JPEG directly from that already encoded
16-bit master. No sky recomposition, luminance rendering or RAW work is repeated.
The same quantization is used for initial and recovered JPEGs. A final receipt is
published only after valid JPEG export; the helper returns true only when full
render validation passes. Corrupt or mismatched partials remain untouched and
return false. Missing partial receipts and changed recipes/builds require a new
attempt. Temporary receipt files are unique, fsynced, atomically renamed and the
directory synced; abandoned partial JPEGs can be replaced on a validated retry.

The initial export decodes and validates the TIFF before publishing even the
preview-pending receipt. A corrupt master cannot be reported as recoverable or complete.
