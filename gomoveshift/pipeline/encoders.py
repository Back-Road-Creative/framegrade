"""Encoder ladder: the registry, probe and selection of HEVC encode backends.

The public product runs on whatever the operator owns, so encode backends form
a ladder in preference order: ``hevc_nvenc`` (NVIDIA) -> ``hevc_videotoolbox``
(Apple) -> ``hevc_qsv`` / ``hevc_vaapi`` (Intel) -> ``hevc_amf`` (AMD on
Windows) -> ``libx265`` (CPU floor). The quality contract is the picture gate
plus a quality floor, not a codec name.

GMS's own drive pipeline keeps its policy ("NVENC required, CPU fallback
not permitted", ``color_filters.detect_nvenc``). This module does not change
that: it only ADDS the selectable ladder. ``select_encoder(required=...)``
expresses the GMS policy in ladder terms (probe one named backend, never fall
back); ``select_encoder()`` is the public "best available" walk.

The vendor backends (videotoolbox, qsv, vaapi, amf) each add one
:class:`EncoderBackend` constant and one slot in :data:`LADDER`. Their flags come
from the ffmpeg encoder documentation and are not yet run on real hardware; the
calibration run on that hardware is the gate (``pipeline/calibration.py``).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable

_PROBE_TIMEOUT_S = 10


def _nvenc_args(output_fps: str, gop_size: int) -> list[str]:
    from .color_filters import _nvenc_args as build

    return build(output_fps, gop_size)


def _x265_args(output_fps: str, gop_size: int) -> list[str]:
    from .color_filters import _x265_args as build

    return build(output_fps, gop_size)


def _videotoolbox_args(output_fps: str, gop_size: int) -> list[str]:
    """hevc_videotoolbox (Apple), main10.

    q:v is 1-100 quality; 65 is about the visually lossless tier, matching
    NVENC CQ 18 / x265 CRF 18. Flags are from the ffmpeg encoder documentation,
    not yet run on real hardware (calibration is the gate).
    """
    from . import encoding_constants as ec

    return [
        "-r",
        output_fps,
        "-c:v",
        "hevc_videotoolbox",
        "-profile:v",
        ec.LONGFORM_PROFILE,
        "-pix_fmt",
        "p010le",
        "-q:v",
        "65",
        "-tag:v",
        "hvc1",
        "-allow_sw",
        "0",
        "-g",
        str(gop_size),
    ]


def _qsv_args(output_fps: str, gop_size: int) -> list[str]:
    """hevc_qsv (Intel Quick Sync), main10, global_quality 18.

    Flags are from the ffmpeg encoder documentation, not yet run on real
    hardware (calibration is the gate).
    """
    from . import encoding_constants as ec

    return [
        "-r",
        output_fps,
        "-c:v",
        "hevc_qsv",
        "-preset",
        "veryslow",
        "-global_quality",
        "18",
        "-look_ahead",
        "1",
        "-profile:v",
        ec.LONGFORM_PROFILE,
        "-pix_fmt",
        "p010le",
        "-g",
        str(gop_size),
    ]


def _vaapi_args(output_fps: str, gop_size: int) -> list[str]:
    """hevc_vaapi (Intel/AMD on Linux), main10, CQP 18.

    VA-API also needs ``-vaapi_device /dev/dri/renderD128`` before the input and
    a ``format=p010,hwupload`` tail on the filter chain, which an arg builder
    cannot express. The wiring unit adds those; until then
    :func:`probe_encoder` reports vaapi unavailable, which is correct.
    Flags are from the ffmpeg encoder documentation, not yet run on real
    hardware (calibration is the gate).
    """
    from . import encoding_constants as ec

    return [
        "-r",
        output_fps,
        "-c:v",
        "hevc_vaapi",
        "-rc_mode",
        "CQP",
        "-qp",
        "18",
        "-profile:v",
        ec.LONGFORM_PROFILE,
        "-g",
        str(gop_size),
    ]


def _amf_args(output_fps: str, gop_size: int) -> list[str]:
    """hevc_amf (AMD on Windows), main10, CQP 18.

    Flags are from the ffmpeg encoder documentation, not yet run on real
    hardware (calibration is the gate).
    """
    from . import encoding_constants as ec

    return [
        "-r",
        output_fps,
        "-c:v",
        "hevc_amf",
        "-quality",
        "quality",
        "-rc",
        "cqp",
        "-qp_i",
        "18",
        "-qp_p",
        "18",
        "-profile:v",
        ec.LONGFORM_PROFILE,
        "-pix_fmt",
        "p010le",
        "-g",
        str(gop_size),
    ]


@dataclass(frozen=True)
class EncoderBackend:
    """One selectable HEVC encode backend."""

    name: str  # ffmpeg encoder name, e.g. "hevc_nvenc"
    vendor: str  # "nvidia", "cpu", ...
    hardware: bool
    args: Callable[[str, int], list[str]]  # (output_fps, gop_size) -> video args
    label: str = ""  # human name, e.g. "NVIDIA NVENC"

    def describe(self) -> str:
        label = self.label or self.vendor
        return f"{self.name} ({label}, {'hardware' if self.hardware else 'cpu'})"


NVENC = EncoderBackend("hevc_nvenc", "nvidia", True, _nvenc_args, "NVIDIA NVENC")
X265 = EncoderBackend("libx265", "cpu", False, _x265_args, "x265 CPU floor")

VIDEOTOOLBOX = EncoderBackend(
    "hevc_videotoolbox", "apple", True, _videotoolbox_args, "Apple VideoToolbox"
)
QSV = EncoderBackend("hevc_qsv", "intel", True, _qsv_args, "Intel Quick Sync")
VAAPI = EncoderBackend("hevc_vaapi", "intel", True, _vaapi_args, "VA-API")
AMF = EncoderBackend("hevc_amf", "amd", True, _amf_args, "AMD AMF")

#: Public preference order: NVENC, then Apple, Intel, AMD. X265 stays last: it
#: is the CPU floor.
LADDER: tuple[EncoderBackend, ...] = (NVENC, VIDEOTOOLBOX, QSV, VAAPI, AMF, X265)


@dataclass(frozen=True)
class ProbeResult:
    backend: EncoderBackend
    ok: bool
    reason: str


class EncoderUnavailableError(RuntimeError):
    """No usable encoder (or the required one is unavailable)."""


def probe_encoder(
    backend: EncoderBackend,
    *,
    run: Callable[..., object] = subprocess.run,
    timeout: float = _PROBE_TIMEOUT_S,
) -> ProbeResult:
    """Encode 1 s of 256x256 test video with ``backend``. Never raises.

    256x256 because hardware encoders reject frames below their minimum size.
    """
    cmd = [
        "ffmpeg",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1:size=256x256:rate=1",
        "-c:v",
        backend.name,
        "-f",
        "null",
        "-",
    ]
    try:
        result = run(cmd, capture_output=True, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - probe contract: never raise
        return ProbeResult(backend, False, f"{type(exc).__name__}: {exc}")
    if result.returncode != 0:
        stderr = result.stderr or b""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        tail = stderr.strip()[-300:] or f"ffmpeg exited {result.returncode}"
        return ProbeResult(backend, False, tail)
    return ProbeResult(backend, True, "")


def probe_ladder(*, run: Callable[..., object] = subprocess.run) -> list[ProbeResult]:
    """Probe every backend, in ladder order."""
    return [probe_encoder(b, run=run) for b in LADDER]


def select_encoder(
    *,
    required: str | None = None,
    run: Callable[..., object] = subprocess.run,
) -> EncoderBackend:
    """Pick the encode backend.

    ``required`` names one backend: only it is probed and failure raises (no
    fallback -- the GMS drive-pipeline policy). Otherwise the first working
    backend in :data:`LADDER` wins.
    """
    if required is not None:
        backend = next((b for b in LADDER if b.name == required), None)
        if backend is None:
            known = ", ".join(b.name for b in LADDER)
            raise EncoderUnavailableError(f"unknown encoder {required!r} (ladder: {known})")
        res = probe_encoder(backend, run=run)
        if not res.ok:
            raise EncoderUnavailableError(
                f"{required} is required and unavailable: {res.reason}. "
                "CPU fallback is not permitted for a required backend."
            )
        return backend
    results = probe_ladder(run=run)
    for res in results:
        if res.ok:
            return res.backend
    reasons = "; ".join(f"{r.backend.name}: {r.reason}" for r in results)
    raise EncoderUnavailableError(f"no HEVC encoder available ({reasons})")
