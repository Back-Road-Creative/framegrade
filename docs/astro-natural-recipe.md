# Conventional astro recipes

`gomoveshift.pipeline.photos.astro_recipe` provides versioned Natural, Accent,
and Bold presets for existing scene-linear RGB. This module is a rendering
foundation, not a RAW decoder, stack aligner, color calibration tool, or publishing
path. Call `recipe_for("natural")`, then `render_linear(rgb, recipe, crop=...)`.
The return value is float32 **display-linear** RGB in [0, 1]; an export adapter
must apply the sRGB transfer function exactly once and embed the matching profile.
Inputs must already use linear sRGB primaries and camera/selected white balance.

Natural uses exposure multiplier 1, contrast 1, and black point 0. Accent uses
exposure 2 and contrast 1.15; Bold uses exposure 3 and contrast 1.3. Every preset
uses a luminance curve around linear 0.18 followed by a bounded shoulder. This
is a display interpretation, not a photometric measurement. All presets leave
saturation, white balance, and spatial detail alone. They cannot recover absent
color or remove noise. There is no denoising, background fit, percentile clipping,
masking, generative model, or star/sky replacement.

Per-pixel RGB ratios are preserved before export encoding, including highlight
compression. Neutral inputs remain neutral. Inputs are never modified. Negative,
nonfinite, empty, and malformed arrays fail before rendering. Optional crop is
explicit `(left, top, right, bottom)` with exclusive bounds; invalid or out-of-frame
coordinates fail. Recipe version and finite parameter bounds are validated.
A manually selected nonzero black point can erase faint signal and is never a
preset default. Crop and recipe values belong in the derivative receipt.

`derivative_receipt(source, output, recipe, crop=...)` hashes both files and the
canonical recipe/crop, returning JSON-compatible metadata without modifying any
file. The caller owns atomic persistence and must prevent concurrent replacement
of these files. A source may be an already validated linear-artifact manifest;
this helper does not independently verify its referenced payloads. Export encoding,
software/build identity, alignment, source selection, and mask provenance belong
in the caller's encompassing receipt. No sky mask from a particular photograph is
silently reused. More advanced contrast/color treatments require separate recipe
versions, explicit parameters, and quality checks before adoption.

The private console preview endpoint checks completed worker provenance, current
source-manifest identity, and preview-byte integrity with bounded streaming I/O.
It does not decode or revalidate full stack payloads or the master on every view;
that resource-admitted validation remains the worker's responsibility. Metadata
is capped at 1 MiB per file and served JPEGs at 64 MiB. Modified source payloads
with an unchanged manifest are detected on worker reuse, not on preview viewing.
