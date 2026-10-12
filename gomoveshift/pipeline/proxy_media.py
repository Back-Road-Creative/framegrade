"""Low-resolution review proxies.

The drives review gate asks a human to actually watch each long-form clip
before it can ship (`console/drives.py` gate C/D). Reviewing the full-weight
source over a slow link is needlessly heavy. ``build_review_proxy`` renders a
small, muted, fast-start MP4 the gate can prefer instead.

Proxies are strictly optional: a build failure WARNs and returns ``None`` so
the caller falls back to the original file. Nothing in the pipeline hard-fails
on a missing proxy. Proxies never carry audio (``-an``) -- the near-silent
contract must not gain a new audible surface.

Proxies always land under the caller-supplied staging/stages root, never
``/tmp`` and never ``--output-dir``.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 640px-wide keeps a review proxy small while remaining watchable; -2 keeps
# height even (required by most H.264 profiles) while preserving aspect
# ratio. CRF 28 is a visibly-soft-but-legible quality/size tradeoff for
# review-only playback, never a delivery encode.
PROXY_SCALE_WIDTH = 640
PROXY_CRF = 28


def build_review_proxy(source: "str | Path", stages_dir: "str | Path") -> Optional[Path]:
    """Render a small muted H.264 review proxy for ``source`` under
    ``stages_dir``, or ``None`` on any failure (WARN, never raise).

    ``stages_dir`` is the caller's staging/stages root (e.g.
    ``gms_intermediate_dir(...)``) -- never ``/tmp``, never the pipeline's
    ``--output-dir``. Proxies live at ``stages_dir/proxies/<stem>_proxy.mp4``.
    """
    source = Path(source)
    stages_dir = Path(stages_dir)
    proxy_dir = stages_dir / "proxies"
    proxy_path = proxy_dir / f"{source.stem}_proxy.mp4"

    try:
        proxy_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"scale={PROXY_SCALE_WIDTH}:-2",
            "-c:v",
            "libx264",
            "-crf",
            str(PROXY_CRF),
            "-an",
            "-movflags",
            "+faststart",
            str(proxy_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.warning(
                "Review proxy build failed for %s (rc=%s): %s -- review will use the original file.",
                source,
                result.returncode,
                result.stderr,
            )
            return None
        if not proxy_path.exists() or proxy_path.stat().st_size == 0:
            logger.warning(
                "Review proxy build for %s produced no/empty output -- "
                "review will use the original file.",
                source,
            )
            return None
        return proxy_path
    except Exception as e:
        logger.warning(
            "Review proxy build for %s raised %s -- review will use the original file.",
            source,
            e,
        )
        return None
