"""Which GL implementation the capture browser draws the scene with.

The scene was pinned to SwiftShader -- Chromium's CPU rasteriser -- on the
belief that the export host had no GPU a headless process could reach (it has
``/dev/dxg`` but no ``/dev/dri``). It can: ANGLE's GL-EGL backend gets there
through Mesa's d3d12 driver. On such a box software GL is the single largest
cost in a film: every settled frame is rasterised on cores the capture workers
are already fighting other processes for.

Nothing here imports Playwright or touches a browser. The capture hands in a
``probe`` that launches one and reports the WebGL renderer string; everything
about *deciding* lives here so it is testable without either.

A backend is judged by what the browser REPORTS it draws with, never by whether
it accepted the launch flags: Chromium takes ``--use-angle=gl`` without
complaint and then draws on SwiftShader, and ``--use-angle=vulkan`` resolves to
Mesa's llvmpipe -- a second software rasteriser that looks like a success.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)

ENV_VAR = "GMS_ROUTE_STORY_GL_BACKEND"

AUTO = "auto"
GPU = "gpu"
SWIFTSHADER = "swiftshader"
_REQUESTABLE = (AUTO, GPU, SWIFTSHADER)

_LAUNCH_ARGS: dict[str, tuple[str, ...]] = {
    SWIFTSHADER: (
        "--use-angle=swiftshader",
        "--enable-unsafe-swiftshader",
        "--ignore-gpu-blocklist",
    ),
    # ANGLE's GL backend over EGL is the one that reaches the GPU from a
    # headless Chromium (on WSL2: Mesa's d3d12 gallium driver, through /dev/dxg).
    GPU: (
        "--use-angle=gl-egl",
        "--ignore-gpu-blocklist",
        "--enable-gpu-rasterization",
    ),
}

# Only set when the operator has not already chosen: on a box where Mesa reaches
# the GPU natively these are inert, and on WSL2 they are what selects the
# discrete adapter over the integrated one.
_GPU_ENV_DEFAULTS = {
    "GALLIUM_DRIVER": "d3d12",
    "MESA_D3D12_DEFAULT_ADAPTER_NAME": "NVIDIA",
}

_SOFTWARE_MARKERS = ("swiftshader", "llvmpipe", "softpipe", "software")

# Evaluated in the page: the driver-reported renderer, or "" with no WebGL.
RENDERER_PROBE_JS = """() => {
  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
  if (!gl) return "";
  const info = gl.getExtension("WEBGL_debug_renderer_info");
  return String(gl.getParameter(info ? info.UNMASKED_RENDERER_WEBGL : gl.RENDERER) || "");
}"""


class GlBackendError(RuntimeError):
    """The demanded GL backend is not what the browser actually draws with."""


def _validated(backend: str) -> str:
    if backend not in _REQUESTABLE:
        raise ValueError(f"{ENV_VAR} must be one of {_REQUESTABLE!r}, got {backend!r}")
    return backend


def requested_backend() -> str:
    """The operator's choice. A typo is refused, never read as a default.

    Unset means software. This used to say the one-browser-at-a-time limit was
    a GPU-path quirk and that "a film opens one per worker" regardless -- that
    was wrong: measured, two SwiftShader
    browsers started together never reach ready either, in one process or in
    two (see ``capture.max_concurrent_captures``). Both backends hold exactly
    one browser, so the choice here is about SPEED, not about parallelism:
    software runs at roughly half the GPU path's capture rate.
    """
    return _validated(os.environ.get(ENV_VAR, SWIFTSHADER).strip().lower() or SWIFTSHADER)


def candidates(requested: str) -> tuple[str, ...]:
    """The concrete backends to try, in order."""
    if _validated(requested) == AUTO:
        return (GPU, SWIFTSHADER)
    return (requested,)


def launch_args(backend: str) -> tuple[str, ...]:
    return _LAUNCH_ARGS[backend]


def launch_env(backend: str, base: Mapping[str, str]) -> dict[str, str]:
    """The browser's whole environment (Playwright replaces, it does not merge)."""
    env = dict(base)
    if backend == GPU:
        for name, value in _GPU_ENV_DEFAULTS.items():
            env.setdefault(name, value)
    return env


def is_hardware_renderer(renderer: str) -> bool:
    lowered = renderer.strip().lower()
    return bool(lowered) and not any(marker in lowered for marker in _SOFTWARE_MARKERS)


def accepts(candidate: str, renderer: str) -> bool:
    """Software only has to draw at all; the GPU candidate has to prove itself.

    An empty renderer is a browser with no WebGL -- nothing can render on it.
    """
    if candidate == GPU:
        return is_hardware_renderer(renderer)
    return bool(renderer.strip())


_resolved: dict[str, str] = {}
_resolve_lock = threading.Lock()


def resolve(probe: Callable[[str], str], requested: str | None = None) -> str:
    """Decide the backend ONCE per process, however many captures ask.

    A film's frames are strided across capture workers, so two workers that
    decided differently would interleave two rasterisers frame by frame. The
    first caller probes under a lock; every later caller gets its answer.
    ``probe(candidate)`` returns the renderer string a browser launched with
    that candidate reports, or raises if it cannot launch at all.
    """
    requested = requested_backend() if requested is None else _validated(requested)
    with _resolve_lock:
        if requested in _resolved:
            return _resolved[requested]
        tried: list[str] = []
        for candidate in candidates(requested):
            try:
                renderer = probe(candidate)
            except Exception as exc:  # a backend that cannot launch is just not available
                renderer = ""
                tried.append(f"{candidate}: launch failed ({exc})")
            else:
                tried.append(f"{candidate}: {renderer or 'no WebGL'}")
            if accepts(candidate, renderer):
                if candidate != candidates(requested)[0]:
                    logger.warning(
                        "route-story scene: no hardware GL (%s); rendering on %s -- "
                        "expect roughly half the capture speed",
                        "; ".join(tried[:-1]),
                        candidate,
                    )
                logger.info("route-story scene GL backend: %s (%s)", candidate, renderer)
                _resolved[requested] = candidate
                return candidate
        raise GlBackendError(
            f"{ENV_VAR}={requested} but the browser does not draw with it: " + "; ".join(tried)
        )


def forget_resolution() -> None:
    """Drop the per-process decision (tests; a long-lived host that gained a GPU)."""
    with _resolve_lock:
        _resolved.clear()
