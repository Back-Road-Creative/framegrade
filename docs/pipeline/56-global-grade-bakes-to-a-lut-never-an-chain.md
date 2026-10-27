## Global grade bakes to a LUT, never an `eq` chain

`_build_single_filter` ends in `eq` (saturation + gamma), and `eq` **negotiates the whole stream to
8-bit `yuv420p` mid-chain**: `yuv420p10le` through `eq` comes out
`yuv420p`, while `curves` and `lut3d` stay `gbrp10le`. So every uniform-lighting day master was
graded through an 8-bit 4:2:0 bottleneck ahead of its 10-bit CQ18 encode, which cannot recover the
banding baked in there.

The per-span adaptive path already avoided this by baking with `params_to_lut3d_cube` and grading
through `lut3d`. `_single_span_filter` points the single-span path at the same machinery, so every
grade reaching an encode is now a baked LUT:

- `build_filter_chain([params], lut_paths=[cube])` emits `lut3d=<cube>` plus the spatial tail —
  `hqdn3d`/`unsharp` are spatial and cannot live in a 3D LUT, so `_spatial_tail` (now the one
  emitter for both paths) still appends them after it.
- `_bake_global_filter` does the bake and returns the temp dir the existing `process` finally-block
  already removes. An identity grade bakes nothing, stays `"null"`, and leaves no temp dir.
- A failed bake falls back to the legacy string chain — a **degraded 8-bit mode**, not the normal
  one, but better than aborting a multi-hour master encode.

LUT↔string parity is pinned by the golden test (<2 LSB), so this changes the grade's numeric
precision, not its look.

The fallback is no longer silent. `SinglePassResult.grade_fallback` is `None` for a LUT grade and `"eq_filter_chain"` when the bake failed and the 8-bit chain was used; it is in `to_dict()` and in the `<final stem>.tonemap.json` sidecar (an SDR master gets that sidecar only when a fallback happened). `drive.strict_grade` (default `false`) refuses the fallback instead: the bake failure raises `ValueError("GRADE-FALLBACK-REFUSED: ...")` before any encode.
