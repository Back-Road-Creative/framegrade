#!/usr/bin/env bash
# Build .venv with the CPU inference wheels: what hosted CI and a machine
# without CUDA install. pyproject.toml is untouched; the GPU build stays the
# product default.
#
#   requirements.lock   pins everything (used as constraints)
#   ci/cpu_pins.py      derives the two CPU swaps from pyproject.toml
#   ci/siblings.txt     the waypost and gps-journey revisions to install.
#                       SIBLINGS_FILE=<file> selects another list, e.g. one
#                       holding `-e ../waypost` and `-e ../gps-journey` when
#                       the three repos sit side by side.
set -euo pipefail
cd "$(dirname "$0")/.."

siblings="${SIBLINGS_FILE:-ci/siblings.txt}"
if grep -qE '@(OWNER|SHA)@' "$siblings"; then
  echo "install-cpu: $siblings still holds placeholder pins" >&2
  exit 1
fi

scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
python3 ci/cpu_pins.py --overrides "$scratch/overrides.txt" --constraints "$scratch/constraints.txt"
grep -v '^-e ' requirements.lock >> "$scratch/constraints.txt"

uv venv .venv --python "${PYTHON:-python3}" -q
uv pip install --python .venv/bin/python -q \
  --override "$scratch/overrides.txt" \
  --constraint "$scratch/constraints.txt" \
  -r "$siblings" -e '.[dev]'

.venv/bin/python - <<'PY'
import onnxruntime as ort

providers = ort.get_available_providers()
if "CUDAExecutionProvider" in providers:
    raise SystemExit(f"CPU install carries a CUDA provider: {providers}")
print("onnxruntime", ort.__version__, providers)
PY
