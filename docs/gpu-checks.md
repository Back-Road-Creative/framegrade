# GPU checks

Hosted CI (`.github/workflows/ci.yml`) runs on a standard runner with no GPU. It
proves the code installs, lints, type-checks and passes its test suite on the
CPU build. It does **not** prove anything on this page. These checks need an
NVIDIA GPU and are run by hand on a machine that has one, before a release and
after any change to the dependencies, the encoder ladder or the privacy blur.

No GPU runner is attached to this repository. A pull request never reaches a
self-hosted machine.

## What hosted CI installs instead

`ci/install-cpu.sh` builds the CPU variant of the same pinned dependency set:

- `ci/cpu_pins.py` reads `pyproject.toml`, drops `onnxruntime-gpu`, pins
  `onnxruntime` to the same version, and swaps the `gpu` extra of `redactcam`
  for `cpu` at the same revision.
- The script then fails if the resulting environment lists
  `CUDAExecutionProvider`, so a "CPU" install cannot quietly carry the GPU build.

`pyproject.toml` itself is unchanged: `pip install -e .` on a GPU host gets the
GPU build, which is the product default.

## What the test suite does not exercise

The suite is hermetic by design (`tests/conftest.py`):

- The detector models are not downloaded and no real inference session is
  built. Tests that reach the blur path run on a stubbed, degraded path. The
  opt-out marker `real_blur_models` exists, and no test in the suite uses it.
- Network access is blocked unless a test carries `@pytest.mark.allow_network`.
- No test encodes with a hardware encoder. Tests that need `ffmpeg` use
  software encoding and skip when `ffmpeg` is missing.
- One test runs a real clip through the whole grading path
  (`tests/test_grading_path_real_clip.py`): frame sampling, analysis, the
  correction, the baked LUT and a libx265 Main 10 encode, then reads the
  encoded pixels back and requires the fixture's blue cast to be cut. It does
  not skip in CI: with `CI` set, a missing `ffmpeg`, `ffprobe` or `libx265`
  fails it. Only the sampling decode differs from a GPU host (software instead
  of CUDA), and the encode is libx265, not NVENC.
- The NVENC availability probe is stubbed: the exact `ffmpeg` probe command
  that `detect_nvenc` runs is answered as "available" without launching it, and
  every other subprocess runs for real. Tests that patch `subprocess.run` or
  `detect_nvenc` themselves still win. The two tests that check detection on
  real hardware carry `@pytest.mark.real_nvenc`; they skip unless
  `GMS_GPU_TESTS=1` is set (see step 3).

So a green CI run says nothing about CUDA loading, hardware encoding, or blur
coverage on real footage.

## Checks to run on a GPU host

Run these in order. Each one names what it proves and what failure looks like.

### 1. The install carries the GPU inference build

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -c "import onnxruntime as ort; ort.preload_dlls(); print(ort.get_available_providers())"
```

Pass: `CUDAExecutionProvider` is in the list.

Fail: only `CPUExecutionProvider` is listed. The usual cause is a later install
that pulled the CPU `onnxruntime` wheel over the GPU one (both wheels own the
same package directory; the last install wins). Repair with
`tools/enable_gpu_detection.sh [VENV_DIR]`, then repeat this check.

### 2. CUDA actually loads, not merely lists

A provider can be listed and still fail to load (for example a CUDA 12 build
beside CUDA 13 runtime libraries). The pipeline's own preflight opens a real
inference session and requires CUDA to be the first bound provider:

```bash
.venv/bin/python -c "from gomoveshift.pipeline.preflight import check_blur_inference_provider as c; c(); print('ok')"
```

Pass: prints `ok`. Fail: raises `RuntimeError` naming the repair.

### 3. Hardware encoding is available

```bash
.venv/bin/gomoveshift doctor
```

Pass: the `gpu_nvenc` line reports `NVENC HEVC encode available`, and the
`encoders` line lists a hardware backend.

`gpu_nvenc` at `WARN` means the drive pipeline will refuse to run on this
machine: it requires NVENC and does not fall back to CPU encoding. `encoders`
at `WARN` means only the CPU floor (`libx265`) encodes here; at `FAIL`, nothing
does. The other hardware backends in the ladder (`gomoveshift/pipeline/encoders.py`)
are selectable but are not what the drive pipeline requires.

Then run the hardware-detection tests the hosted suite skips:

```bash
GMS_GPU_TESTS=1 .venv/bin/python -m pytest -m real_nvenc -o addopts=
```

Pass: every `real_nvenc` test passes and none is skipped.

### 4. The blur covers what it should on real footage

This needs a short clip you are entitled to use, and its rendered output:

```bash
.venv/bin/python tools/verify_blur_coverage.py ORIGINAL.mp4 BLURRED.mp4 --vehicle-model PATH
```

Exit 0: every detected vehicle cabin was blurred to the configured coverage.
Exit 1: at least one was left exposed. Treat exit 1 as a release blocker.

## Recording a run

A GPU check has no CI record, so write one down when you run it: the commit,
the GPU model and driver version, the output of step 1, and pass or fail for
each step. Put it in the release notes of the version it covers.
