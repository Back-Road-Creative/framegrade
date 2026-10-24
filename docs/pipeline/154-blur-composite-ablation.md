## Blur composite ablation (Framegrade B1) and quarter-resolution blur branch (B2) -- measured 2026-10-07

Question: which single step of the privacy-blur composite in `single_pass._build_blur_filter_complex`
is the wall (G1, doc 159: composite chain 2.66 fps against grade-only 14.49 fps), and does running
only the blur branch at 1/4 resolution remove it without changing a pixel outside the mask?

**Verdict: PASS with one stated deviation. The wall is `boxblur` run at full 3840x2160 (variant 2:
grade + a discarded full-frame boxblur = 2.20 fps, about as slow as the whole composite at 2.94 fps).
Running that blur at 1/4 resolution (variant 7) measured 11.44 fps, 3.89x the full composite (gate 2x);
the masked region is as blurred (Laplacian variance ratio 0.987, 11 of 12 frames at or below today's)
and pixels more than 3 px from the mask are identical (PSNR inf). Deviation: PSNR of the strict
`mask == 0` region is 96.1-97.5 dB, not inf, because both chains alter the same 6600-pixel fringe
(<= 3 px from the mask edge, 4:2:0 chroma-alpha); against the unblurred base, v5 and v7 change the
identical pixels. All timings ran on a box at load average ~41-53 on 12 cores (other sessions' CI), so the
absolute fps are depressed and the grade-only reference did not reproduce (7.03 fps, not ~14.5).**

Every number comes from a run on 2026-10-07; arithmetic from them is marked "derived". Probe
scripts were throwaway and are not kept.

### Setup

- Graph = `SinglePassProcessor._build_blur_filter_complex` called with the real helpers (as G1
  did): the G1 LUT (`global.cube`), colour filter without `unsharp`, `blur_strength=20` so the
  radius is `boxblur=40:2` at 3840 (G1's value; the config default 12 gives 24), `skip_ost=True`,
  `source_range="pc"`, `duration=15`. `hwdownload,format=p010le,` is prepended to `[0:v]` as in G1.
- Clip: `GX020052.MP4`, `-ss 120 -t 15` stream copy (450 frames, 3840x2160 HEVC yuv420p10le,
  29.97 fps). Same NVENC arguments as G1 doc 159 chain A. GPU lock
  (`orchestrator_lock.acquire_gpu_lock`) held for every timed run; `nvidia-smi` sampled each second.
- fps = 450 / ffmpeg `-benchmark` rtime; wall = `/usr/bin/time -v`; CPU % from the same (of 1200 %).
  VRAM peak is the sampled `memory.used` maximum and includes ~2.5 GB held by Ollama.
- Mask: the B3 mask for this clip is not available, so a **synthetic** gray mask video was
  generated (`lavfi`): black 3840x2160 at 30000/1001 with two moving white rectangles (500x350
  moving right at 150 px/s, 600x400 moving left at 120 px/s), 5.0 % of the frame (measured
  `mask_frac` 0.0500 on all 12 frames).
- One run per variant. Box load: `/proc/loadavg` 1-minute sampled every 10 s during the runs;
  mean over each run window 42-53 (v1 ran before the sampler started; `uptime` then read 48.9).
  Treat the fps as ordering evidence, not as clean capacity numbers.

### Variants (graphs in `graphs.json`, built from the service's own string)

1. grade-only: base chain, `fps` -> `[vout]`.
2. grade + `split` + full-frame `boxblur=40:2` on `[b2]`, branch discarded by `nullsink`; `[b1]` out.
3. grade + `split` + `alphamerge` + `overlay`, blur replaced by `null`.
4. grade + `format=yuv420p10le,format=yuva420p10le,format=yuv420p10`, no split.
5. full composite (`[b2]boxblur=40:2`), chain A minus `unsharp`.
7. as 5 with `[b2]scale=iw/4:ih/4:flags=bilinear,boxblur=10:2,scale=iw*4:ih*4:flags=bilinear[bl]`.
`t` suffix = `-filter_threads 12 -filter_complex_threads 12`.

### Measurement table

| variant | fps | wall | CPU % | VRAM peak MiB | rtime s |
|---|---|---|---|---|---|
| v1 grade-only (reference) | 7.03 | 1:14.72 | 113 | 4997 | 64.03 |
| v2 grade + full-frame boxblur, discarded | 2.20 | 3:53.27 | 67 | 5170 | 204.18 |
| v3 grade + split/alphamerge/overlay, null blur | 4.65 | 1:52.36 | 110 | 5033 | 96.69 |
| v4 grade + format churn only | 9.68 | 0:54.15 | 182 | 4898 | 46.49 |
| v5 full composite | 2.94 | 2:54.09 | 109 | 5002 | 152.98 |
| v2t (v2 + filter threads 12) | 3.74 | 2:17.42 | 100 | 4974 | 120.27 |
| v3t (v3 + filter threads 12) | 6.68 | 1:17.51 | 145 | 4836 | 67.40 |
| v5t (v5 + filter threads 12) | 4.79 | 1:47.00 | 164 | 4840 | 93.91 |
| v7 quarter-res blur branch | 11.44 | 0:49.39 | 265 | 4927 | 39.34 |
| v7t (v7 + filter threads 12) | 11.90 | 0:44.13 | 290 | 4891 | 37.81 |

Derived ratios: v7 / v5 = 3.89x (v7t / v5t = 2.48x); v3 / v5 = 1.58x (the composite minus blur is
faster by that factor); v2 / v5 = 0.75 (the blur alone is slower than the whole composite, because
the discarded branch still forces the full-frame blur, while v5 also decodes and merges).

### Which step is the wall

Of the steps ablated, the full-frame `boxblur` is the wall. Evidence: v2, which carries only the
blur on top of the grade, runs at 2.20 fps, no faster than v5 (2.94); the same composite with the
blur replaced by `null` (v3) runs 4.65 fps; the format churn alone (v4) costs less than
the grade itself (9.68 vs 7.03 fps, within load noise); and shrinking only the blur's pixel count by 16x (v7)
lifts the whole composite to 11.44 fps, above the grade-only reference run under the same box load.
The hypothesis "single-threaded" is only partly supported: filter threads helped (v2 2.20 -> 3.74,
v5 2.94 -> 4.79) yet CPU % stayed at 100-164 % of 1200 %, but that CPU figure is capped by the
box's ~4x oversubscription, so thread behaviour is not resolved here. Second-order cost: the
alphamerge/overlay/format steps with no blur (v3) still cost more than the grade alone (4.65 vs 7.03).

### Fidelity of v7 against v5 (12 lossless 16-bit frames, n = 20, 60 ... 420)

Frames taken before the encode (`select` then `scale=in_range=limited:in_color_matrix=bt709:out_range=full,format=rgb48le`
-> PNG), mask frames from the same generator.

- PSNR outside the mask, strict `mask == 0`: 96.08-97.53 dB per frame (not inf). The differing
  pixels number 3520-4330 per frame, all within 3.00 px of the mask edge.
- PSNR outside the mask dilated by 4 px: inf on all 12 frames.
- Cause: the composite is 4:2:0, so an alpha edge blends neighbouring chroma. Against the unblurred
  base (variant 1) both v5 and v7 change the same 6600 pixels per frame (max distance 3.00 px) on
  frames 1 and 6 and 6596 vs 6600 on frame 12, so the fringe is a property of today's composite, not of B2.
- Blur inside the mask (Laplacian variance of the green plane, 16-bit scale, 33 px erosion
  of the mask): unblurred mean 84,521,739; v5 591,970; v7 584,249; v7 / v5 mean 0.9870, v7 at or
  below v5 on 11 of 12 frames (frame 3: 1.0059). v7 is at least as blurred (gate: v7 <= v5).
- The all-black mask no-op proof stays in `tests/test_blur_pixel_proof.py` /
  `tests/test_blur_composite_depth.py` (inf).

Gates: fps >= 2x v5 pass (3.89x); blur inside >= today's pass; PSNR outside the mask inf: fails
strict by the fringe above, passes at 4 px. Phase 2 (B2) proceeds on the reading that the gate's
intent is "no new change to unmasked pixels", which the 6600-pixel identical band supports. This reading is
the author's, not a pre-agreed one.

### Limits

- Not run: any quiet-box timing (the box stayed at load ~20-60 throughout), a second run of any variant
  (each number is one run), the real B3 mask, the radius 24 default, non-multiple-of-4 frame sizes
  (the code falls back to the full-resolution blur for those), `unsharp`.
- The grade-only reference (7.03 fps) did not reproduce G1's 14.49 fps; the ordering of variants is
  the evidence, not their absolute values.
- Bilinear down/up gives a smoother-than-boxblur look at the very edge of the blurred area; the
  masked pixels are anonymised either way (variance ratio above), but no face/plate detector
  recheck was made on the quarter-resolution blur.

### Commands (exact, `$O` = a scratch output directory)

```
ffmpeg -y -hide_banner -benchmark -stats [-filter_threads 12 -filter_complex_threads 12] \
 -hwaccel cuda -hwaccel_output_format cuda -i $O/slice.mp4 \
 -f lavfi -i "color=black:s=3840x2160:r=30000/1001:d=15[bg];color=white:s=500x350:r=30000/1001[w1];[bg][w1]overlay=x='200+t*150':y=400:shortest=1[o1];color=white:s=600x400:r=30000/1001[w2];[o1][w2]overlay=x='3000-t*120':y=1200:shortest=1,format=gray" \
 -t 15 -filter_complex "<graph>" -map "[vout]" -r 30000/1001 -c:v hevc_nvenc -preset p7 -tune hq \
 -rc vbr -cq 18 -profile:v main10 -pix_fmt p010le -rc-lookahead 32 -spatial-aq 1 -temporal-aq 1 \
 -b_ref_mode 2 -bf 4 -multipass 2 -g 60 -color_range tv -colorspace bt709 -color_primaries bt709 \
 -color_trc bt709 -b:v 40000k -maxrate 80000k -bufsize 120000k -an -write_tmcd 0 -movflags +faststart $O/<name>.mp4
```

v7 graph: `[0:v]hwdownload,format=p010le,setrange=pc,format=gbrp16le,lut3d=$O/global.cube,scale=in_range=pc:out_range=limited,fade=t=in:st=0:d=1.0:color=black,fade=t=out:st=14.0:d=1.0:color=black,format=yuv420p10le[base];[base]split[b1][b2];[b2]scale=iw/4:ih/4:flags=bilinear,boxblur=10:2,scale=iw*4:ih*4:flags=bilinear[bl];[1:v]format=gray[m];[bl][m]alphamerge,format=yuva420p10le[bla];[b1][bla]overlay=format=yuv420p10[ov];[ov]fps=30000/1001[vout]`
