### Fixed

- HDR tonemap and wide-gamut conversion chains now end in 10-bit `yuv420p10le` (one shared `TONEMAP_OUTPUT_FORMAT`) instead of 8-bit `yuv420p`, which put an 8-bit hop (at most 256 levels per channel) ahead of the 16-bit grade. A 10-bit ramp through the chain keeps its levels.

### Added

- `grade_fallback` on the single-pass result and in the `.tonemap.json` sidecar records when the global LUT bake failed and the grade fell back to the 8-bit `eq` chain. `drive.strict_grade` (default off) refuses that fallback instead.
