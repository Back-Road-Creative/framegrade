"""Production builder for ``MediaFingerprint`` — see docs/social-publishing-plan-builder.md.

``build_publish_plan()`` requires a ``media_by_short_index`` mapping of
``MediaFingerprint`` records but nothing in production built one; callers
had to hand-roll ffprobe/hashing themselves (or, in practice, only tests
did). This module is that one builder, reusing the repo's existing
``FFprobeClient`` (``gomoveshift/pipeline/validation/ffprobe_client.py``)
rather than a third inline ``subprocess.run(["ffprobe", ...])`` pattern.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from .models import MediaFingerprint, ShortPublishRecord

# Matches the DriveConfig default (``DriveConfig.ffprobe_timeout_sec``) so a caller
# that doesn't thread a run's configured value through still gets the same
# behavior as the rest of the pipeline.
DEFAULT_FFPROBE_TIMEOUT_SEC = 30

_HASH_CHUNK_SIZE = 1024 * 1024  # 1 MiB — never load a whole LF into memory


def fingerprint_media(
    path: Path, *, ffprobe_timeout_sec: int = DEFAULT_FFPROBE_TIMEOUT_SEC
) -> MediaFingerprint:
    """Build a ``MediaFingerprint`` for a rendered short/LF on local disk.

    Raises loudly, naming ``path``, on: a missing file, a non-zero ffprobe
    exit, a probe with no video stream, or an unparseable duration. Never
    returns a fingerprint with a zeroed/placeholder field — this value
    flows into the publish-plan hash and misidentifying media a post
    consented to is worse than failing the run.
    """
    if not path.exists():
        raise FileNotFoundError(f"fingerprint_media: no such file: {path}")

    size_bytes = path.stat().st_size
    sha256 = _sha256_streamed(path)
    duration_seconds, width, height, codec = _probe_video(path, timeout=ffprobe_timeout_sec)

    return MediaFingerprint(
        path=str(path),
        size_bytes=size_bytes,
        duration_seconds=duration_seconds,
        width=width,
        height=height,
        codec=codec,
        sha256=sha256,
    )


class MediaVerificationError(Exception):
    """``verify_media`` couldn't confirm ``record.media`` still matches disk."""


def verify_media(record: ShortPublishRecord) -> None:
    """Raise :class:`MediaVerificationError`, naming the path, unless
    ``record.media`` still matches what's on local disk.

    Cheapest checks first — existence, then the recorded ``size_bytes`` —
    so a missing/truncated file is caught without ever paying for a hash of
    (up to) a whole day master. Only a same-size candidate gets hashed,
    reusing ``_sha256_streamed`` rather than a second hashing pass.
    """
    path = Path(record.media.path)
    if not path.exists():
        raise MediaVerificationError(f"verify_media: no such file: {path}")

    actual_size = path.stat().st_size
    if actual_size != record.media.size_bytes:
        raise MediaVerificationError(
            f"verify_media: size mismatch for {path}: fingerprint says "
            f"{record.media.size_bytes} bytes, disk has {actual_size}"
        )

    actual_sha256 = _sha256_streamed(path)
    if actual_sha256 != record.media.sha256:
        raise MediaVerificationError(
            f"verify_media: sha256 mismatch for {path}: fingerprint says "
            f"{record.media.sha256}, disk has {actual_sha256}"
        )


def _probe_video(path: Path, *, timeout: int) -> tuple[float, int, int, str]:
    # Local import mirrors the established call sites (day_master_encoder.py,
    # pipeline_validator.py, ost_applier.py) — the client is imported inside
    # the function that needs it, not at module load.
    from ..pipeline.validation.ffprobe_client import FFprobeClient, FFprobeError

    client = FFprobeClient(subprocess, timeout=timeout)
    try:
        data = client.probe(path, show_format=True, show_streams=True)
    except FFprobeError as exc:
        raise RuntimeError(f"fingerprint_media: ffprobe failed for {path}: {exc}") from exc

    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"), None
    )
    if video_stream is None:
        raise RuntimeError(f"fingerprint_media: no video stream in {path}")

    try:
        duration_seconds = float(data.get("format", {}).get("duration"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"fingerprint_media: unparseable duration for {path}") from exc

    # Same loud-or-nothing rule as duration: a stream missing width/height/
    # codec_name would otherwise raise a bare TypeError naming neither the
    # field nor the file, and int(None) is the likeliest real-world shape of
    # a malformed probe.
    try:
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        codec = str(video_stream["codec_name"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"fingerprint_media: incomplete video stream for {path}: {exc}") from exc

    return duration_seconds, width, height, codec


def _sha256_streamed(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()
