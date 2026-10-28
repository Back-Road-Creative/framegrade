"""Render a review set: paired source/output stills, excerpts and a provenance record.

Library only. Given a source file, an output file and (source_t, output_t) pairs it writes,
into an empty directory, one PNG still and one full-speed excerpt per side per pair plus
``provenance.json``. Nothing here runs unless called.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Optional, Sequence

from .. import __version__
from ..health.store import product_sha
from ._atomic_write import atomic_write_json
from .calibration import _ffmpeg_version

#: Wall-clock cap for one still or excerpt; a stuck decoder, not a slow one.
_RENDER_TIMEOUT_S = 300

#: Excerpt encode args, recorded verbatim in provenance. Re-encoded (never ``-c copy``) so the
#: excerpt starts on the requested frame rather than the preceding keyframe; video only.
EXCERPT_ENCODE_ARGS = [
    "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
]  # fmt: skip


def _still_argv(path: Path, t: float, dest: Path) -> list[str]:
    # Input-side -ss, as _read_frame_sync (gomoveshift/web/app.py): fast keyframe jump, then
    # decode-and-discard to the exact time. Copied rather than imported because web/app.py
    # pulls in the web framework; PNG (lossless) here instead of its piped JPEG.
    return ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{float(t):.6f}", "-i", str(path),
            "-frames:v", "1", str(dest)]  # fmt: skip


def _excerpt_argv(path: Path, t: float, seconds: float, dest: Path) -> list[str]:
    return ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{float(t):.6f}", "-i", str(path),
            "-t", f"{float(seconds):.6f}", *EXCERPT_ENCODE_ARGS, str(dest)]  # fmt: skip


def _run(argv: list[str]) -> None:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=_RENDER_TIMEOUT_S)
    if proc.returncode != 0 or not Path(argv[-1]).is_file():
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {(proc.stderr or '')[-500:]}")


def _describe(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return {"name": path.name, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def render_review_set(
    source: Path,
    output: Path,
    pairs: Sequence[tuple[float, float]],
    out_dir: Path,
    *,
    excerpt_seconds: float = 3.0,
    config: Optional[Any] = None,
    synthetic: bool = False,
) -> dict[str, Any]:
    """Write the review set under ``out_dir`` and return ``{"pairs": [...], "provenance": {...}}``.

    ``out_dir`` is created if missing and must be empty; a non-empty one raises
    ``FileExistsError`` before anything is written. ``config`` is a loaded ``Config`` (default:
    ``get_config()``); its ``drive`` section is recorded. ``synthetic`` is the caller's
    statement that the inputs are generated test media."""
    source, output, out_dir = Path(source), Path(output), Path(out_dir)
    if not pairs:
        raise ValueError("pairs must not be empty")
    if excerpt_seconds <= 0:
        raise ValueError("excerpt_seconds must be positive")
    for p in (source, output):
        if not p.is_file():
            raise FileNotFoundError(p)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        from ..config import get_config

        config = get_config()

    rows = []
    for i, (src_t, out_t) in enumerate(pairs):
        row: dict[str, Any] = {"index": i, "source_t": float(src_t), "output_t": float(out_t)}
        for side, path, t in (("source", source, src_t), ("output", output, out_t)):
            still, clip = f"pair_{i:03d}_{side}.png", f"pair_{i:03d}_{side}.mp4"
            _run(_still_argv(path, t, out_dir / still))
            _run(_excerpt_argv(path, t, excerpt_seconds, out_dir / clip))
            row[f"{side}_still"], row[f"{side}_excerpt"] = still, clip
        rows.append(row)

    provenance = {
        "version": __version__,
        "git_sha": product_sha(),  # None when not a checkout / recorded install
        "ffmpeg_version": _ffmpeg_version(),
        "config": config.drive.model_dump(mode="json"),
        "inputs": {"source": _describe(source), "output": _describe(output)},
        "pairs": [[float(s), float(o)] for s, o in pairs],
        "excerpt_seconds": float(excerpt_seconds),
        "excerpt_args": list(EXCERPT_ENCODE_ARGS),
        "synthetic": bool(synthetic),
    }
    atomic_write_json(out_dir / "provenance.json", provenance)
    return {"pairs": rows, "provenance": provenance}
