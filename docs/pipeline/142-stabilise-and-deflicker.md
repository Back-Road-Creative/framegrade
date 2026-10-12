## Stabilise / deflicker pass (`pipeline/stabilize.py`) <a id="stabilise-and-deflicker"></a>

Standalone module, not yet wired into the render path or CLI (named follow-up below). It owns
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

**Follow-up, named and not yet done:** wiring either pass into the drive pipeline's render path
(`pipeline/single_pass.py`) and/or a `gomoveshift` CLI flag is out of scope for this module --
`pipeline/color_correction.py`, `pipeline/color_filters.py`, `cli/parser.py` and `cli/handlers/*`
are all owned by other in-flight work and this change does not touch them.
