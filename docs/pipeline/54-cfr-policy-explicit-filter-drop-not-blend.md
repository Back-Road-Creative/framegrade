## CFR policy — explicit filter, drop-not-blend

Everything publishes at NTSC 29.97 (`color_filters.OUTPUT_FPS`), so 59.94/60 fps sources are
decimated at encode time. **Frames are DROPPED or DUPLICATED, never blended** — for an exact 59.94
source that is a clean 2:1 decimation and the correct cadence. The policy is written down so nobody
"improves" it into a blend, which would smear motion on every pan.

The conversion is `color_filters.cfr_filter()` (`fps=30000/1001`), appended to the video chain — it
must **not** be left to the output-side `-r`, which hands frame selection to the muxer. Measured on
this build, a 10.000 s / 600-frame 60 fps source through `-r 30000/1001` yields **301 frames /
10.043 s** instead of the correct 300 / 10.010 s, and picks different frames than the filter
(~26 dB PSNR apart). `-fps_mode cfr` and `-vsync cfr` do not change that. `get_encoding_args` still
stamps `-r`, which is a no-op once the stream already carries the rate.

**Placement is the chain tail, and that is load-bearing on the blur path**: the privacy mask is
materialised one frame per SOURCE frame and `alphamerge` pairs them 1:1, so decimating before the
composite would desync the mask from the footage. The filter therefore rides the overlay tail,
after the composite and the text overlays.
