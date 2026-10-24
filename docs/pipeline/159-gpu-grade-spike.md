## GPU grade chain spike (Framegrade G1) -- measured 2026-10-07

Question: can the day-master grade chain (`gbrp16le` grade + `lut3d` + blur overlay,
`single_pass.py`) run on the GPU through `hwupload` -> `libplacebo` -> `hevc_nvenc` at
>= 3x the CPU chain's fps, with max dE2000 <= 2.0 and SSIM >= 0.98?

**Verdict: LOSS: closes doc-only. This WSL2 box exposes no NVIDIA Vulkan device (only the
software `llvmpipe` CPU rasteriser), so `libplacebo` cannot run on the GPU at all; on
`llvmpipe` it measured 3.54 fps against 3.03 fps for the CPU chain (1.17x, gate is 3x), and
this build has no Vulkan equivalent of the `unsharp` or the blur/alphamerge/overlay steps.**

Every number below comes from a run on 2026-10-07; arithmetic derived from them is marked
"derived". Probe scripts were throwaway and are not kept; the exact ffmpeg
lines are in the Commands section.

### What the CPU chain is (read from source, not assumed)

- The "CPU chain" already encodes on the GPU: `hevc_nvenc` p7, `-tune hq -rc vbr -cq 18`,
  `-profile:v main10 -pix_fmt p010le`, lookahead 32, spatial+temporal AQ, `-bf 4`,
  `-multipass 2`, `-g 60`, `-b:v 40000k -maxrate 80000k -bufsize 120000k`
  (`color_filters.get_encoding_args`). NVDEC feeds it (`-hwaccel cuda -hwaccel_output_format
  cuda`, then `hwdownload,format=p010le`, `single_pass._run_encode_step`). Only the filters
  run on the CPU.
- Filter graph = `SinglePassProcessor._build_blur_filter_complex` called with the real
  helpers: `params_to_lut3d_cube` (64^3 `.cube`) + `build_filter_chain` ->
  `lut3d=<cube>,unsharp=5:5:0.50:5:5:0`, wrapped as `setrange=pc,format=gbrp16le,<grade>,
  scale=in_range=pc:out_range=limited,fade,fade,format=yuv420p10le`, then
  `split` / `boxblur=40:2` / `alphamerge` / `overlay` / `fps=30000/1001`.
- No `.cube` ships with the service (`config.py:772-774`: "The operator supplies the .cube
  files; none are bundled"). The test LUT is the service's own bake from a stubbed
  `CorrectionParams(shadow_lift=6, highlight_pull=5, saturation_mult=1.12,
  contrast_curve=0.08, white_balance_r=0.02, white_balance_b=-0.02, gamma=1.10,
  sharpen_amount=0.5)`. The B3 privacy mask is synthetic (`lavfi` black frame with one white
  800x500 box), same size and rate as the source.
- GPU lock: `orchestrator_lock.acquire_gpu_lock()` / `release_gpu_lock()` (an `fcntl.flock`
  on the shared lock file). Every timed run below called it from the service venv; no wait
  was needed. No pipeline run and no other encode process was present (`pgrep`,
  `nvidia-smi --query-compute-apps` empty), so no retry window was used.

### Setup

- Source: `GX020052.MP4`, 3840x2160 HEVC `yuv420p10le`, full range (`pc`), 29.97 fps.
  Slice: `-ss 120 -t 60 -c copy` (684 MB, 91 Mbps). Timed runs use its first **15 s** (450
  frames), not 60 s: the software-Vulkan chain is too slow for 60 s. fps = 450 frames /
  ffmpeg's own `-benchmark` rtime; wall = `/usr/bin/time -v` elapsed (adds the `+faststart`
  rewrite, ~5-18 s).
- Box: 12 logical cores, RTX 4070 Laptop 8 GB, driver 596.36 under WSL2, ffmpeg
  git-2026-04-10 (`--enable-libplacebo --enable-vulkan`). `nvidia-smi` before each run
  showed 2950-3363 MiB already used (local Ollama model); VRAM peak is the sampled maximum
  of `memory.used` at 1 s, so it includes that resident ~3.1 GB.

### Measurement table

| chain | fps | wall s | CPU % | VRAM peak MiB | out Mbps | mean dE | max dE | PSNR dB (avg) | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| A. CPU full (grade+unsharp+blur composite) -> nvenc | 3.03 | 166.6 | 128 | 5440 | 80.8 | ref | ref | ref | ref |
| B. CPU grade-only (setrange, gbrp16le, lut3d, range squeeze) -> nvenc | 14.49 | 36.5 | 190 | 5379 | 81.0 | ref (B) | ref (B) | ref (B) | ref (B) |
| C. libplacebo grade-only on Vulkan (llvmpipe = CPU) -> nvenc | 3.54 | 144.7 | 177 | 5573 | 81.0 | 2.15 vs B | 47.77 vs B | 32.94 vs B | 0.9670 vs B |
| D. CPU grade+unsharp (no blur) -> nvenc | 8.36 | 64.0 | 218 | 5736 | 81.0 | n/a | n/a | n/a | n/a |
| E. CPU grade+blur (no unsharp) -> nvenc | 2.66 | 190.4 | 107 | 5322 | 81.0 | n/a | n/a | n/a | n/a |
| F. NVDEC -> nvenc, no filters (ceiling) | 37.28 | 13.3 | 30 | 5292 | 81.0 | n/a | n/a | n/a | n/a |

Pre-encode fidelity of C against B (lossless, 12 frames at t = 1,2,3,4,5,6,7,8,9,10,12,14 s,
both chains ended at limited-range `yuv420p10le`, converted identically to 16-bit RGB):
**mean dE2000 0.0029, max 0.658** (per-frame max 0.52-0.66; per-frame mean 0.0028-0.0030).
dE2000 is `skimage.color.deltaE_ciede2000` (skimage 0.26 is in the service venv) on
full-resolution frames.

Reading the dE/SSIM columns honestly: C vs B on the *encoded* files is a poor test.
All six encodes land at 80.8-81.0 Mbps, the 80 Mbps `-maxrate` x 14.88 s (output sizes within
0.3 % of each other, `q=35` against `-cq 18`), i.e. NVENC is rate-starved on this noisy 4K
GoPro source and any 1-bit input difference sends it down a different path. So the encoded
dE 2.15 / 47.77 and SSIM 0.967 mostly measure NVENC divergence, not the grade. The lossless
0.0029 / 0.658 is the grade's own error. Strict gate arithmetic still reads: max dE 47.77 >
2.0 and SSIM 0.967 < 0.98 on the encoded outputs, fail.

Chain C ran with **no NVIDIA GPU involved in the grade**: its VRAM and GPU-util curves are
NVDEC + NVENC only (peak 5573 MiB, same envelope as B at 5379).

### Vulkan in WSL2: what happened

- `vulkaninfo --summary` lists exactly one device: `llvmpipe (LLVM 20.1.2, 256 bits)`,
  `PHYSICAL_DEVICE_TYPE_CPU`, Mesa 25.2.8. ffmpeg's `-init_hw_device vulkan` log says
  `GPU listing: 0: llvmpipe ... (software)` / `Device 0 selected: llvmpipe`.
- `/usr/share/vulkan/icd.d` has asahi, gfxstream, intel_hasvk, intel, lvp, nouveau, radeon,
  virtio; no NVIDIA ICD and no `dzn` (the D3D12-on-Vulkan layer that could reach `/dev/dxg`).
  `/usr/lib/wsl/lib` carries CUDA, NVENC/NVDEC and D3D12 libraries but no Vulkan or GLX
  library. `find / -name '*dzn*' -o -name '*nvidia_icd*'` found nothing.
- Exact errors:
  - `-init_hw_device vulkan=vk:NVIDIA` -> `[Vulkan] Unable to find device "NVIDIA"!` /
    `Device creation failed: -19.`
  - `-init_hw_device vulkan=vk:1` -> `Device creation failed: -19.`
  - CUDA fallback `-init_hw_device cuda=cu -filter_hw_device cu ... hwupload_cuda,libplacebo`
    -> `[Parsed_libplacebo_1] Failed creating Vulkan device!` /
    `Error initializing filters`. libplacebo needs Vulkan; the CUDA device cannot stand in.
    `hwupload_cuda` itself works, but this build has no CUDA 3D-LUT, unsharp or blur filter
    (`-filters` shows only `hwupload_cuda` for cuda; no `*_npp`, no `scale_cuda`).
- Not tried (needs host changes, out of scope for a spike): installing Mesa `dzn` or an
  NVIDIA Vulkan ICD. Whether either would expose the RTX 4070 here is **unverified**.

### What libplacebo can and cannot do in this build

- 3D LUT: works, option `lut=<.cube>` with `lut_type=native`. It must be fed an **RGB**
  surface: `format=gbrp16le,hwupload,libplacebo=lut=...:format=rgba64le`. Fed `yuv420p10le`
  the LUT is applied in YUV space and the result is wrong (mean dE ~13.6); `lut_type=conversion`
  is worse (mean dE ~31.7); with `format=gbrp16le` upload both modes match the CPU `lut3d`.
  This forces a 4K `yuv->gbrp16le` swscale on the CPU before upload, the same conversion the
  CPU chain already pays.
- Unsharp: no option in `ffmpeg -h filter=libplacebo` (only a `custom_shader` route, not
  tried). Blur composite: the only Vulkan filters are `avgblur_vulkan`, `bwdif_vulkan`,
  `v360_vulkan`; there is no `overlay_vulkan` or `alphamerge`, so the masked blur cannot stay
  on the GPU. A GPU chain would be grade only, with unsharp and blur still on the CPU.
- Upscale: option `upscaler` exists (default spline36); not measured, the source is already
  4K and the service's U1 upscale is off for it.
- Planar YUV `hwdownload` from `llvmpipe` returned garbage (green, chroma uncorrelated) for
  `nv12`, `yuv420p`, `p010le`, `yuv420p10le`; `rgba64le` is correct. Chain C therefore
  downloads RGBA64 and converts back on the CPU. This is a llvmpipe observation; it was not
  checked on a real GPU because none was reachable.

### LUT precision probe (10-bit gradient, one 3840x2160 frame)

`gradients` (4 colour stops, `seed=1`) -> `yuv420p10le` -> `gbrp16le` -> the same cube,
CPU `lut3d` vs `libplacebo` (software Vulkan), both written as 16-bit RGB:
**max |diff| = 1 code of 65535, mean 0.59 codes, max dE2000 0.0046, mean 0.00045.**
Distinct red levels 20647 (CPU) vs 21205 (libplacebo): no banding added by the 16-bit path.
Caveat: this is `llvmpipe`'s float path, so it says nothing about the NVIDIA driver's
precision. An earlier attempt without a fixed seed compared two different random gradients
(max dE 62.8) and was discarded.

### Where the time actually goes (derived from the table)

Per-frame cost = 1/fps: F ceiling 26.8 ms; B grade-only 69.0 ms; D adds unsharp +50.6 ms;
E adds blur composite +307 ms; A full 330 ms. The grade itself is ~42 ms above the
NVDEC->NVENC floor; the blur composite alone (E, 376 ms/frame) costs more than the whole of A (330 ms; stages
overlap, so the added costs are not additive), and
CPU use stays at 107-218 % of 1200 %, so the filters are largely single-threaded.
A perfect GPU grade could not lift chain A past the blur composite, which has no GPU form
in this build. Even with all filters free the box tops out at 37.3 fps (F).

### Hours per 45-minute 4K day master (derived: 2700 s x 30000/1001 = 80,919 frames / fps)

| chain | fps | hours |
|---|---|---|
| A. CPU full (today's chain) | 3.03 | 7.41 |
| E. CPU grade+blur, no unsharp | 2.66 | 8.46 |
| D. CPU grade+unsharp, no blur | 8.36 | 2.69 |
| B. CPU grade-only | 14.49 | 1.55 |
| C. libplacebo on llvmpipe, grade-only (blur and unsharp would add CPU time) | 3.54 | 6.35 |
| F. NVDEC->NVENC floor, no filters | 37.28 | 0.60 |

A's 7.41 h agrees with `single_pass.py:422-423` ("3-4 fps, ~0.11x wall").

### Commands (exact, 15 s slice; `$O` = a scratch output directory)

A: CPU full chain (`chain.json` holds the generated graph):

```
ffmpeg -y -hide_banner -benchmark -stats -hwaccel cuda -hwaccel_output_format cuda -i $O/slice.mp4 \
 -f lavfi -i "color=black:s=3840x2160:r=30000/1001:d=60,drawbox=x=1200:y=800:w=800:h=500:color=white:t=fill,format=gray" \
 -t 15 -filter_complex "[0:v]hwdownload,format=p010le,setrange=pc,format=gbrp16le,lut3d=$O/global.cube,unsharp=5:5:0.50:5:5:0,scale=in_range=pc:out_range=limited,fade=t=in:st=0:d=1.0:color=black,fade=t=out:st=59.0:d=1.0:color=black,format=yuv420p10le[base];[base]split[b1][b2];[b2]boxblur=40:2[bl];[1:v]format=gray[m];[bl][m]alphamerge,format=yuva420p10le[bla];[b1][bla]overlay=format=yuv420p10[ov];[ov]fps=30000/1001[vout]" \
 -map "[vout]" -r 30000/1001 -c:v hevc_nvenc -preset p7 -tune hq -rc vbr -cq 18 -profile:v main10 -pix_fmt p010le \
 -rc-lookahead 32 -spatial-aq 1 -temporal-aq 1 -b_ref_mode 2 -bf 4 -multipass 2 -g 60 \
 -color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 -b:v 40000k -maxrate 80000k -bufsize 120000k \
 -an -write_tmcd 0 -movflags +faststart $O/cpu15.mp4
```

C: GPU chain (Vulkan device = llvmpipe; same encoder arguments as A):

```
ffmpeg -y -hide_banner -benchmark -stats -init_hw_device vulkan=vk -filter_hw_device vk \
 -hwaccel cuda -hwaccel_output_format cuda -i $O/slice.mp4 -t 15 \
 -vf "hwdownload,format=p010le,setrange=pc,format=gbrp16le,hwupload,libplacebo=lut=$O/global.cube:lut_type=native:format=rgba64le,hwdownload,format=rgba64le,format=gbrp16le,scale=in_range=pc:out_range=limited,format=yuv420p10le,fps=30000/1001" \
 -r 30000/1001 -c:v hevc_nvenc -preset p7 -tune hq -rc vbr -cq 18 -profile:v main10 -pix_fmt p010le \
 -rc-lookahead 32 -spatial-aq 1 -temporal-aq 1 -b_ref_mode 2 -bf 4 -multipass 2 -g 60 \
 -color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 -b:v 40000k -maxrate 80000k -bufsize 120000k \
 -an -write_tmcd 0 -movflags +faststart $O/gpu-grade.mp4
```

B, D and E are A with the `unsharp` and/or blur segments removed; F is
`-hwaccel cuda -hwaccel_output_format cuda -i slice.mp4 -t 15 <encoder args without -r,
-pix_fmt and the colour tags>` (those insert a scaler that cannot take CUDA frames,
`error -38`).

### Retracted run and limits

- A first 60 s run of chain A, made without `-t`, produced a frozen dark frame for the whole
  file (constant luma 9.9) and a bogus 25 fps. It was discarded; chain A was re-run with
  `-t 15` and reproduces the 3-4 fps in `single_pass.py`. The cause of the freeze (the
  lavfi mask input plus no `-t`) was not investigated.
- Not run: any 60 s timing, any NVIDIA-device Vulkan run, any CUDA-side LUT, the
  `custom_shader` unsharp, the 8-bit/`yuv420p` source variant, and a second pass to check
  run-to-run fps variance (each number is one run).
- The test grade is one baked LUT, not the adaptive multi-span `xfade` graph.

### Follow-on

None for a GPU grade: there is no reachable device, and even a working one is bounded by the
CPU-only blur composite. If the cost of the day master matters, the measured lever is the blur
composite (E: +307 ms/frame) and `unsharp` (D: +50.6 ms/frame), both CPU and both outside
`quality.py`'s LUT. That is a separate CPU-side question and not proposed or tested here.

(Superseded in part by "Native Windows, NVIDIA Vulkan" below: a device is reachable from native Windows, and the verdict stays LOSS.)

## Native Windows, NVIDIA Vulkan (measured 2026-10-08)

Question: the WSL2 spike had no NVIDIA Vulkan device. Does the same chain clear the 3x gate when
run by a native Windows ffmpeg whose Vulkan device is the RTX 4070?

**Verdict: LOSS, again. The device works and the GPU blur composite works, but the best GPU chain
(G, grade + blur composite on the GPU) runs 12.28 fps against 6.47 fps for the CPU full chain A
run by the same Windows ffmpeg: 1.90x, gate is 3x. Against the like-for-like CPU chain without
unsharp (E, 7.19 fps) it is 1.71x. GPU grade-only (C 15.97 fps, H 20.62 fps) is slower than CPU
grade-only (B 26.68 fps).**

### Build and device

- Windows ffmpeg run from WSL via interop, cwd a scratch directory (relative file names, so no
  drive-letter colon in filter strings). Outputs in a `win` subdirectory of it.
- Used build: BtbN `ffmpeg-n8.1-latest-win64-gpl-8.1.zip`, `ffmpeg version n8.1.3-14-g330caae0c1-20261008`,
  sha256 `9609e32834f5fa9ed09741c73cb660e47d59f99be7981f910e33aac532cbe921` (matches the release's
  `checksums.sha256`). Extracted under the scratch directory.
- First tried `ffmpeg-master-latest-win64-gpl.zip`, sha256
  `7c2f2b099bf7799e55b16b8214f608b26064980cf4d8d19022b3995f49732319` (matches). Its Vulkan works but
  `hevc_nvenc` refuses to open: `Driver does not support the required nvenc API version. Required:
  13.1 Found: 13.0 ... minimum required Nvidia driver for nvenc is 610.00 or newer` (driver here is
  596.36). Master is unusable for encodes on this box until the driver moves; the n8.1 build opens
  NVENC fine.
- `-filters` lists `libplacebo`, `avgblur_vulkan`, `gblur_vulkan`, `overlay_vulkan`, `blend_vulkan`,
  `scale_vulkan`, `xfade_vulkan`, `nlmeans_vulkan` (no `unsharp` equivalent).
- `-init_hw_device vulkan=vk -v verbose`: `GPU listing: 0: NVIDIA GeForce RTX 4070 Laptop GPU
  (discrete) (0x2860)` / `Device 0 selected: NVIDIA GeForce RTX 4070 Laptop GPU`. So the device is
  the NVIDIA GPU; timing proceeded.
- Slice recreated per Setup: source `GX020052.MP4` (ffprobe: hevc 3840x2160 yuv420p10le, pc,
  30000/1001), `-ss 120 -t 60 -c copy`, 683,948,595 bytes (the earlier slice was 684 MB). Same
  `global.cube`. Same 15 s / 450 frames, fps = 450 / `-benchmark` rtime, one run each.

### Chains

A-F are the doc's Commands with relative paths (same encoder arguments; B/D/E keep the fades).
New GPU chains, all fed by NVDEC and ended by `hevc_nvenc`:

- C: as before (CPU `gbrp16le` upload, `libplacebo` LUT, RGBA64 download, CPU range squeeze).
- G: C's grade, then `split`; blur branch `libplacebo` crop to the mask box plus a 64 px margin
  (1136,736 928x628) -> `gblur_vulkan=sigma=33:size=127` -> `libplacebo` crop back to the box
  (800x500) -> `overlay_vulkan=x=1200:y=800` on the graded frame. The margin exceeds the blur
  support (+-63 px), so this equals blur-the-frame-then-mask for the synthetic rectangular mask.
  `sigma=33` is the std dev of two `boxblur` r=40 passes (sqrt(2 * 546.75)). A free-form
  B3 mask would need `blend_vulkan` or a mask plane; not tried. No `unsharp` and no fades.
- H: C without the CPU `yuv->gbrp16le` swscale: `hwdownload,format=p010le,setrange=pc,hwupload,
  libplacebo=format=rgba64le,libplacebo=lut=...:lut_type=native:format=p010le:colorspace=bt470bg:
  range=tv,hwdownload,format=p010le` (YUV to RGB, LUT and RGB to YUV all in `libplacebo`; `bt470bg`
  because swscale's RGB->YUV default is BT.601).
- I: H's colour path plus G's GPU blur composite.

### Measurement table

| chain | fps | rtime s | wall s | CPU % (utime+stime / rtime, of 1200) | VRAM peak MiB | resident before MiB | out Mbps |
|---|---|---|---|---|---|---|---|
| A. CPU full | 6.47 | 69.54 | 70.6 | 318 | 5961 | 3443 | 81.5 |
| B. CPU grade-only | 26.68 | 16.87 | 16.4 | 508 | 5834 | 3458 | 81.5 |
| C. libplacebo grade on NVIDIA Vulkan | 15.97 | 28.18 | 30.6 | 227 | 7023 | 4368 | 81.7 |
| D. CPU grade+unsharp | 18.18 | 24.75 | 26.5 | 600 | 5816 | 3495 | 81.5 |
| E. CPU grade+blur | 7.19 | 62.57 | 64.2 | 258 | 5985 | 3481 | 81.6 |
| F. NVDEC -> NVENC (ceiling) | 34.35 | 13.10 | 15.7 | 14 | 6695 | 4325 | 81.8 |
| G. GPU grade + GPU blur composite | 12.28 | 36.64 | 38.1 | 188 | 7411 | 3272 | 81.7 |
| H. GPU grade, GPU colour conversions | 20.62 | 21.83 | 21.6 | 428 | 5840 | 2970 | 81.7 |
| I. H + GPU blur composite | 10.68 | 42.14 | 43.3 | 226 | 7442 | 2970 | 81.7 |

Ratios (derived): G/A 1.90x, G/E 1.71x, I/A 1.65x, C/B 0.60x, H/B 0.77x. Hours per 45-minute
day master (80,919 frames / fps): A 3.47, E 3.13, B 0.84, C 1.41, H 1.09, G 1.83, F 0.65.
VRAM peak is the sampled `memory.used` maximum and includes the resident Ollama model, which
varied between runs (2970-4368 MiB before the run), so peaks are not comparable across rows.

Other facts from this session: the same CPU chain A is 2.1x faster under native Windows (6.47 fps)
than under WSL2 (3.03 fps above); B is 1.8x faster (26.68 vs 14.49). A first G attempt that blurred the
full 4K RGBA64 frame died with `Task finished with error code: -12 (Cannot allocate memory)`
at 27 frames (VRAM peak 7494 of 8188 MiB with ~3.3 GB resident); the cropped-blur form above
fits. Windows interop from WSL intermittently drops all ffmpeg output with `UtilAcceptVsock:271:
accept4 failed 110`; the runner retried until ffmpeg printed its `bench:` line (no timed run
needed a retry).

### Correctness

- Frame counts: ffprobe `-count_frames` returned 450 for all nine outputs A-I.
- Pre-encode (lossless, ffv1 `yuv420p10le`, one frame each at t = 5 s and 10 s, no fades): C vs B
  PSNR 82.64 / 82.69 dB, SSIM 0.999999; G vs E 56.38 / 56.86 dB, SSIM 0.9998; G vs A (A has
  unsharp, G does not) 55.78 / 56.17 dB, SSIM 0.99974; H vs B 53.58 / 53.47 dB, SSIM 0.998; I vs E
  51.86 / 51.92 dB, SSIM 0.998. For scale, E vs A (unsharp alone) is 63.65 / 63.67 dB. H and I
  sit below G in chroma (U/V PSNR ~50-51 dB) because libplacebo, not swscale, does their RGB to
  YUV. dE2000 was not recomputed; PSNR/SSIM only.
- Encoded files (all encodes sit at 81.5-81.8 Mbps, the rate cap, so NVENC diverges as in
  the WSL2 run), frames from t >= 2 s: C vs B PSNR avg 32.85 dB, SSIM 0.9660; G vs E 32.94 dB,
  0.9663; G vs A 32.89 dB, 0.9660; I vs E 32.64 dB, 0.9640; H vs B 31.37 dB, 0.9536. The same test
  on CPU-only pairs gives E vs A 33.42 dB / 0.9697 and D vs B 32.19 dB / 0.9611, so the GPU rows
  sit inside the CPU-vs-CPU NVENC noise. Strict gate arithmetic on encoded SSIM (< 0.98) still
  fails for every pair, CPU-only included, so that column does not discriminate here.
- Fidelity is not what loses: G is visually equivalent to E before encoding. Speed is.

### Verdict against the 3x gate

Decisive ratio: G 12.28 fps / A 6.47 fps = **1.90x < 3x: LOSS**. G still omits `unsharp` (no GPU
form exists), so the real GPU chain would be slower than 12.28 fps and the ratio lower; against
E the ratio is 1.71x. The GPU blur composite costs 19 ms/frame (G 81.4 ms vs C 62.6 ms), against
~102 ms/frame for the CPU composite (E 139.1 ms vs B 37.5 ms), so the composite is no longer the
bottleneck. The bottleneck moved to the CPU round trip around the GPU: C and H are slower than
B though the LUT itself is cheap, and the GPU chains still burn 188-428 % CPU (utime 56-87 s per
15 s clip). The cause is not isolated here (candidates: 4K `hwdownload` from CUDA, `hwupload` to
Vulkan, `hwdownload` from Vulkan, no Vulkan<->CUDA/NVENC hand-off in ffmpeg); that is unverified.
Even the NVDEC -> NVENC ceiling (F) is 34.35 fps, so 3x over A needs 19.4 fps, and G reaches 12.28.

### Follow-on (supersedes the WSL2 follow-on above)

Do not build a Vulkan grade stage in this form. Two measured levers remain, both outside this
spike's gate: (1) the CPU chain is 2.1x faster under native Windows ffmpeg than under WSL2 on this
box (A 6.47 vs 3.03 fps, B 26.68 vs 14.49), so where the day master is graded matters more than
which device grades it; (2) a GPU blur composite is viable (19 ms/frame) if a future change
removes the CPU hwdownload/hwupload round trip (needs a Vulkan-fed encoder or a CUDA-side
grade). Neither is proposed or tested here.
