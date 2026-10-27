## Stabilise / deflicker pass (`pipeline/stabilize.py`) <a id="stabilise-and-deflicker"></a>

Wired into the drive day master by `--stabilize` / `--deflicker` (see "Wiring" below). It owns
three independent pieces: a capability probe, argv builders for two ffmpeg passes, and a quality
gate that decides whether either pass's output is worth keeping.

**Capability probe.** `probe_ffmpeg_filters` runs `ffmpeg -filters` once and parses the registered
filter names the same way `cli/handlers/doctor.py::check_ffmpeg_filters` does (filter name is the
second whitespace-delimited token of each row), but returns the full `FfmpegFilterCapabilities`
set rather than a pass/fail against a hand-listed subset -- this module needs to know WHICH of
`deflicker` / `vidstabdetect` / `vidstabtransform` it has.

**Deflicker.** Single ffmpeg pass, `build_deflicker_argv` builds `deflicker=mode=<mode>:size=<size>`.
Present in a typical ffmpeg build (check with `ffmpeg -filters`).

**Stabilise.** Two-pass `vidstabdetect` (motion analysis, writes a `.trf` transforms file) then
`vidstabtransform` (applies it), via libvidstab. **Absent from some ffmpeg builds** (check with
`ffmpeg -filters`). `build_vidstabdetect_argv` / `build_vidstabtransform_argv` both
call `require_vidstab` first, which raises `StabilizeUnavailableError("ffmpeg built without
libvidstab")` when the capability probe doesn't show both filters -- there is no code path that
silently hands back an argv that would just fail later at ffmpeg's own filter-graph parse, and no
path that silently skips stabilisation and calls it done. A caller that wants "stabilise if
possible, else fall back to deflicker-only" must catch `StabilizeUnavailableError` explicitly.

**Intermediate encoding.** Neither output-writing pass (`build_deflicker_argv`, `build_vidstabtransform_argv`) leaves the intermediate to ffmpeg's defaults any more (an `.mp4` defaulted to h264 at crf 23, whatever the input was). `intermediate_video_args` writes `-c:v libx265 -preset <LONGFORM_PRESET> -crf <LONGFORM_CRF>` (fallback encoder) from `encoding_constants`, a `-pix_fmt` that keeps the input's own depth (10-bit stays 10-bit; an input libx265 cannot take falls back to `LONGFORM_PIX_FMT`, never lower than 10-bit), and the colour range, matrix, primaries and transfer copied from an ffprobe of the input (`probe_stream_spec`). A tag the input does not carry is left unwritten, not invented; an unprobeable input logs a warning and gets the 10-bit default with no tags. This build of libx265 writes only matrix and range from `-colorspace` / `-color_range`, so primaries and transfer also go in `-x265-params` (`colorprim` / `transfer` / `colormatrix`). `vidstabdetect` writes no video (`-f null`), so it is unchanged. The encoder follows the pipeline's usual selection (`color_filters.get_encoding_args`): `hevc_nvenc` (`LONGFORM_NVENC_PRESET` / `LONGFORM_NVENC_CQ`, main10 `p010le`, bitrate floors by height, no `-r` so the input's timing is kept) when `stabilize.nvenc_available()` finds a working GPU, otherwise libx265 as above. `day_master_encoder` passes that decision in (`has_nvenc`), so tests and callers inject it and need no GPU. x265 `slow` on a long 4K day master is the slow fallback only.

**Quality gate.** `measure_luma_jitter` runs `ffmpeg -vf signalstats,metadata=print -f null -` and
parses the `lavfi.signalstats.YAVG=<value>` lines ffmpeg emits to stderr, one per frame, then
returns the population std of the frame-to-frame deltas -- a flickering clip has large deltas and
a high std, a stable one doesn't. `gate_on_jitter(input_jitter, output_jitter)` keeps the output
only when its jitter measurably dropped below the input's; otherwise it reports why via
`JitterGateResult.reason`. `run_deflicker` runs the deflicker pass end to end and applies the gate
before the output ever replaces anything -- on a failed gate it `unlink`s the output file itself,
so a caller can never mistake a jitter-worsening encode for a kept result. (The two-pass vidstab
flow doesn't yet have an equivalent single-call wrapper -- callers currently compose
`build_vidstabdetect_argv` / `build_vidstabtransform_argv` and `measure_luma_jitter` /
`gate_on_jitter` themselves; adding one is part of the wiring follow-up below, since the shape of
that wrapper depends on where in the render path the two ffmpeg calls actually run.)

Every function here takes its `run` collaborator (default `subprocess.run`) as a parameter, so
`tests/test_stabilize.py` exercises the probe, the argv builders and the refusal against a fake
`run` and never depends on a real ffmpeg build for those cases. One test
(`test_deflicker_reduces_measured_jitter_on_real_ffmpeg`) runs the real deflicker pass against a
tiny synthetic alternating-brightness clip built from a numpy array piped into
`ffmpeg -f rawvideo`, and asserts the gate's measured output jitter is lower than its input
jitter; it's skipped, not xfailed, when ffmpeg is genuinely absent from the box.

### Wiring

* **Flags.** `gomoveshift drive --deflicker` and `--stabilize` (opt-in, both default `None` --
  an unset argparse sentinel -- so a flag already on `config.drive.stabilize` / `.deflicker` is
  never cleared by a run that omits it; `cli/handlers/drive.py::apply_picture_flags`). Neither is
  persisted in drive state: a resumed run that must keep them re-passes the flags.
* **`--fast`.** A plain `store_true` (never an opt-in). It forces `no_sharpen`, clears
  `stabilize` / `deflicker` even when saved on the config, and records each dropped step in
  `config.drive.fast_skipped`; the preflight plan prints one `fast mode: skipped ...` line. The
  honest-resolution upscale rule and every gate are untouched.
* **Preflight refusal.** `preflight.check_picture_filters` runs in `cmd_drive` before any work:
  `--stabilize` without libvidstab exits 2 with `--stabilize refused: ffmpeg built without
  libvidstab` (`VIDSTAB_MISSING_REASON`); `--deflicker` needs the `deflicker` filter. No flag set
  means nothing is probed.
* **Where it runs.** `encode_day_master`, on the raw concat before the grade, blur mask and
  encode: `apply_picture_steps` deflickers, then stabilises (vidstabdetect, vidstabtransform).
  Intermediates sit next to the raw concat under the day's output dir and are removed afterwards.
  The colour cache is keyed on the raw concat, so it is neither seeded nor saved for a day whose
  picture steps were kept (the grade re-analyses the changed frames).
* **The gate.** Each step is measured: `measure_luma_jitter` on input and output, kept only when
  jitter dropped. A discarded or failed step deletes its output, logs both numbers at WARNING and
  the chain continues with its input; it never raises.
* **Record.** `day_<date>_picture_steps.json` beside the master holds `deflicker_applied`,
  `stabilize_applied` (bool) and per-step `input_jitter` / `output_jitter` / `kept`.
* **Phase A.** `quality.flicker_score(frames)` (mean absolute frame-to-frame luma-mean change)
  exists, but Phase A does not record `source_flicker`: the Phase A sharpness loop samples one frame
  at evenly spaced timestamps, not consecutive frames, so reusing it would measure scene change,
  not flicker, and a consecutive-frame sample needs its own decode. A `shake_score` needs
  `vidstabdetect` and is out of scope until a build has it.
* **Host dependency.** This box's ffmpeg (git-2026-04-10) has `deflicker` but not libvidstab, so
  `--stabilize` refuses here. The rebuild is a `bin/host-reconcile.sh` step for a later PR.
