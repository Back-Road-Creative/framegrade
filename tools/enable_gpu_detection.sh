#!/usr/bin/env bash
# Enable GPU privacy detection: swap the pipeline venv from the CPU onnxruntime
# build to onnxruntime-gpu + the CUDA-12 runtime wheels, then verify CUDA loads.
#
# onnxruntime-gpu==1.22.0 is a CORE dependency of this project
# (pyproject.toml), so a fresh `pip install -e .` lands the GPU build by itself;
# its [cuda,cudnn] extras also pull the nvidia-*-cu12 runtime
# wheels it links against, so a venv whose torch ships CUDA-13 wheels still loads
# CUDAExecutionProvider.
# This script is the REPAIR for a venv where the CPU wheel got in anyway (any
# `pip install` of something that lists plain `onnxruntime`): both wheels own the
# onnxruntime/ package directory and the last install wins, and after the CPU
# wheel is uninstalled pip still sees onnxruntime-gpu's metadata as satisfied and
# rewrites nothing — hence --force-reinstall --no-deps below. Preflight
# (check_blur_inference_provider) refuses to start a blur run in that state and
# points here.
#
# Version pin: onnxruntime-gpu 1.27 needs CUDA 13 (no pip wheels yet), so we pin
# the CUDA-12 build (1.22.0) — the same pin as pyproject.toml; keep them equal.
# The CUDA build runs the plate YOLO forward pass much faster than the CPU build.
#
# Run on the pipeline host while no batch is running.
#
# Usage: tools/enable_gpu_detection.sh [VENV_DIR]   (default: ./.venv)
set -euo pipefail

VENV="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv}"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
[ -x "$PY" ] || { echo "ERROR: no venv python at $PY" >&2; exit 1; }

echo ">> Target venv: $VENV"
echo ">> Removing the CPU onnxruntime build..."
"$PIP" uninstall -y onnxruntime >/dev/null 2>&1 || true

echo ">> Reinstalling onnxruntime-gpu (CUDA 12) — forced, so a gutted package dir is rewritten..."
"$PIP" install --force-reinstall --no-deps "onnxruntime-gpu==1.22.0"
echo ">> Installing the CUDA/cuDNN runtime wheels..."
"$PIP" install \
  "nvidia-cuda-runtime-cu12" "nvidia-cudnn-cu12" "nvidia-cublas-cu12" \
  "nvidia-cufft-cu12" "nvidia-curand-cu12"

echo ">> Verifying CUDAExecutionProvider loads (mirrors the runtime code path)..."
"$PY" - <<'PY'
import onnxruntime as ort
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()  # the runtime calls this too, so no LD_LIBRARY_PATH needed
provs = ort.get_available_providers()
print("   available providers:", provs)
assert "CUDAExecutionProvider" in provs, (
    "CUDAExecutionProvider missing — check the NVIDIA driver (nvidia-smi) and that "
    "the nvidia-*-cu12 wheels installed; if ORT still can't find the libs, export "
    "LD_LIBRARY_PATH to <venv>/lib/python*/site-packages/nvidia/*/lib"
)
print("   OK: GPU privacy detection is active (the blur stage auto-uses CUDA).")
PY

echo ">> Done. The blur stage now runs detection on the GPU."
echo "   Never install the CPU 'onnxruntime' wheel into this venv again; it clobbers this."
