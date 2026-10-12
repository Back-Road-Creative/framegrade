## Colour range domain — one probed answer, three consumers

GoPro footage is full-range (`pc`) and frequently arrives **untagged**. Every YUV→RGB hop here runs
through an ffmpeg *auto-inserted* converter, and that converter assumes **limited** range on
untagged input — so full-range input is expanded as if 16–235 were the legal span, clipping true
shadows and highlights irrecoverably and shifting everything between into the wrong domain.

`color_analysis.scale_range_token` is the single mapper (`pc`/`jpeg`/unknown/empty → `full`,
`tv`/`mpeg` → `limited`; the GoPro default is what makes untagged safe). It lives there rather than
in `single_pass` so `analysis/` can import it without the encoder; `single_pass._scale_range_token`
re-exports the same object so the two can never drift. Three consumers must agree:

- **Application** — `_build_video_filter` and `_build_blur_filter_complex` emit `setrange=<probed>`
  as the FIRST filter, ahead of the RGB grading filters the converter precedes. `setrange` writes
  metadata only. The `scale=in_range=<probed>:out_range=limited` squeeze is unchanged.
- **Measurement** — `sample_frames` / `_sample_frames_sequential` take `source_range` and tag ahead
  of the `rgb24` hop (the `-s WxH` resize folded into `-vf` so the scale is explicit).
  `_analyze_color` reuses the `_get_video_info` probe, so this costs no extra ffprobe; the
  batch→sequential fallback forwards the value, since dropping it resurrects the bug on that path.
- **Selection** — `_extract_frames_bgr` reads `color_range` in the ffprobe it already runs for the
  timeout budget (`_parse_frame_probe`; unreadable → 600 s and `full`) and tags both the GPU and CPU
  commands. These frames feed `drive_scores.npz`, so the bias reached footage selection itself.

**The look changes on untagged sources only.** A tagged clip (`color_range=pc`, or the `yuvj420p`
alias) yields a byte-identical grade string — `setrange` is a no-op when the tag already agrees.
Fixing one consumer alone is worse than fixing none: today's error is self-consistent, the analysis
over-reading black point, white point and saturation by what the application then mis-applies.
