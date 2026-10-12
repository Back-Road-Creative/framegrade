#!/usr/bin/env python3
"""Derive the CPU-install pin files from pyproject.toml.

pyproject.toml declares the GPU inference build (the product default). Hosted
CI and machines without CUDA install the CPU build of the same two
requirements at the same versions. Both files are derived here, so a version
bump in pyproject.toml cannot leave the CPU install on an older pin.

  --overrides    uv override file: drops onnxruntime-gpu, swaps redactcam's
                 ``gpu`` extra for ``cpu`` at the same revision
  --constraints  one line pinning onnxruntime to the onnxruntime-gpu version
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

_ORT = re.compile(r"onnxruntime-gpu(?:\[[^\]]*\])?==(?P<version>[0-9][0-9A-Za-z.]*)")
_CAM = re.compile(r"redactcam\[gpu\]\s*@\s*(?P<source>\S+)")


def derive(dependencies: list[str]) -> tuple[str, str]:
    """Return (overrides text, constraints text) for the CPU install."""
    ort = [m for d in dependencies if (m := _ORT.fullmatch(d.strip()))]
    cam = [m for d in dependencies if (m := _CAM.fullmatch(d.strip()))]
    if len(ort) != 1 or len(cam) != 1:
        raise ValueError(
            "expected exactly one pinned onnxruntime-gpu and one redactcam[gpu] "
            f"requirement, found {len(ort)} and {len(cam)}"
        )
    overrides = (
        "onnxruntime-gpu ; sys_platform == 'never'\n"
        f"redactcam[cpu] @ {cam[0]['source']}\n"
    )
    constraints = f"onnxruntime=={ort[0]['version']}\n"
    return overrides, constraints


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--constraints", type=Path, required=True)
    args = parser.parse_args(argv)
    with args.pyproject.open("rb") as fh:
        dependencies = tomllib.load(fh)["project"]["dependencies"]
    try:
        overrides, constraints = derive(dependencies)
    except ValueError as exc:
        print(f"cpu_pins: {exc}", file=sys.stderr)
        return 1
    args.overrides.write_text(overrides)
    args.constraints.write_text(constraints)
    return 0


if __name__ == "__main__":
    sys.exit(main())
