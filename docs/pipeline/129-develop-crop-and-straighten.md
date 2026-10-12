## Crop, straighten, and horizon-level (`develop.py`) <a id="develop-crop-and-straighten"></a>

`DevelopOptions` gained two run-level geometry fields, `straighten_angle` (degrees, positive =
counter-clockwise) and `crop_rect` (`(left, top, right, bottom)` fractions of the frame in
`[0, 1]`), both no-ops at their defaults (`0.0`, `()`) so existing develop output is byte-identical
when unset.

- Geometry is applied to the DECODED frame, once, in `develop_variants`, before the shared
  `_denoise`/measurement call every variant grades from — a run-level property of the photo, never
  a per-look creative knob. A look-specific crop would change a variant's shape and make it
  incomparable to its siblings, the same reasoning `LOOK_VARIANT_RUN_KNOBS` already applies to
  `lut_strength`/`lut_over`/`lut_auto`/`reference_lab`.
- `_straighten` rotates about the frame centre via `cv2.getRotationMatrix2D`/`warpAffine`,
  `borderMode=cv2.BORDER_REPLICATE` so corners exposed by the turn are filled by replicating the
  edge rather than a hard black border; canvas size is unchanged. `angle == 0.0` short-circuits to
  a bare copy — no `warpAffine` call at all — so it is bit-identical to the source.
- `_crop` trims to the requested fraction AFTER straightening, so the rect reads against the
  levelled frame. An empty rect is a no-op with no bounds arithmetic at all.
- Replay: both fields are plain `DevelopOptions` fields, so `--from-grade` should carry them
  through the generic `_from_sidecar` helper `develop_options_from_sidecar` uses. No test covers
  that round trip yet.
- Flags: `photos develop --straighten DEG` (range -45..45) and `--crop LEFT TOP RIGHT BOTTOM`
  (each in `[0, 1]`, left < right, top < bottom), both gated before any decode. In
  `config.PHOTO_KNOBS` they are RUN-owned rows (`run=True`, like `lut`). A crop or tilt fits one
  photo, never a fleet, so a look naming either is refused at config load by name.
