## Grade working depth + the depth-regression gate

Nothing pinned a working format ahead of the grade, so ffmpeg negotiated one from the SOURCE depth.
A standard 8-bit GoPro file therefore put the whole grade — curve bend, gamma stretch, range squeeze
— through `rgb24`, quantising every intermediate before the 10-bit CQ18 encode ever saw the frame
(verified: `curves` and `lut3d` both negotiate `rgb24` on an 8-bit input, `gbrp16le` once pinned).

`GRADE_WORKING_FORMAT` (`gbrp16le`) is emitted immediately before the colour filter in both
`_build_video_filter` and `_build_blur_filter_complex`. Grading at ≥10 bits and letting the encoder
do one dithered downconvert is the standard anti-banding fix; the cost is CPU in the filter chain.

- **Pinned only when a grade actually runs.** A 16-bit planar RGB working copy of a 4K frame is not
  free, and the fade/drawtext stages do not need it.
- **It does not disturb the full→limited squeeze** — a regression test pins the squeezed output at
  exactly 16..235 with the working format in place. (A working format that silently neutered the
  squeeze would ship a full-range upload, so this interaction is gated, not assumed.)

`tests/test_grade_chain_depth.py` is the regression gate, and it is **not a string check**: it runs
the real emitted chain through ffmpeg at `-loglevel debug` and reads back every format the graph
converts *into*, asserting all are ≥10-bit. Only conversion **destinations** are read — the source's
own depth is a given, while every format the graph converts into is one we chose — and the probe
pins a 10-bit output so the tail convert cannot mask a collapse. A mid-chain collapse that later
widens back is invisible in the chain's final format, which is exactly how the `eq` stage hid for so
long; one test feeds the detector a known-collapsing `eq` chain so the gate cannot pass vacuously.
