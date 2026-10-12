"""TikTok's upload needs H.264; the other three platforms are happy with the
HEVC master.

A large HEVC short can upload cleanly to the account inbox
(`status/fetch` → `SEND_TO_USER_INBOX`) while its notification will not open —
tapping it does nothing. The same footage re-encoded as a much smaller H.264
file opens the composer and publishes (`PUBLISH_COMPLETE`).

Nothing rejects the HEVC file: TikTok's Media Transfer Guide lists H.265 as
supported and the limits are 4 GB / 10 minutes / 4096 px, all of which a
short of that size is far inside. But H.264 is what that guide
*recommends*, and the app has to pull the whole file down before the composer
renders. The delivery therefore looks completely successful — accepted,
delivered, parked awaiting a human — while being impossible to publish. That is
the worst shape a failure can take, and it is why this is fixed by construction
here rather than documented as a caveat.

The evidence is limited: re-encoding changes codec, pixel format,
`+faststart` and file size together, so it does not isolate which of them the
app chokes on. It shows only that an H.264 rendition of the same footage
publishes where the HEVC master did not — enough to fix by construction, not
enough to name a single cause.

YouTube and Facebook transcode server-side and keep the master. Instagram does
NOT tolerate an over-cap master: Meta's Reels spec says "Video bitrate: VBR,
25Mbps maximum", and an over-cap 10-bit HEVC master can publish late or stall
IN_PROGRESS, while Facebook and YouTube transcode the same file.
``ensure_instagram_rendition`` gives Instagram an 8-bit H.264 rendition
whenever the master is over the cap or its bitrate is unknown.
"""

from __future__ import annotations

import logging
import subprocess  # noqa: S404 — fixed argv, no shell
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

#: Suffix of the sibling rendition. Sits next to the short so it survives with
#: it and is cleaned up with it.
RENDITION_SUFFIX = ".tiktok-h264.mp4"
#: 5 Mb/s over 1080x1920 gives a small file for a short -- comfortable
#: for a phone to stream before the composer renders, and visually unchanged.
TARGET_BITRATE = "5M"
MAX_BITRATE = "6M"
BUFSIZE = "10M"


class TikTokRenditionError(Exception):
    """The rendition could not be produced. Terminal for this attempt: an
    upload of a missing or empty file lands another unopenable draft."""


def rendition_path(source: Path) -> Path:
    return source.with_name(source.stem + RENDITION_SUFFIX)


def _ffmpeg_argv(source: Path, target: Path) -> List[str]:
    return [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-b:v",
        TARGET_BITRATE,
        "-maxrate",
        MAX_BITRATE,
        "-bufsize",
        BUFSIZE,
        # 10-bit HEVC masters would otherwise produce a yuv420p10le H.264 that
        # the app cannot decode -- the same silent failure in a new costume.
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-level",
        "4.1",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        # Moves the moov atom to the front: the app streams the file rather
        # than downloading it whole before it can show anything.
        "-movflags",
        "+faststart",
        str(target),
    ]


def _run_ffmpeg(argv: List[str]) -> None:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=1800, check=False)  # noqa: S603
    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()[-3:]
        raise TikTokRenditionError(f"ffmpeg failed (rc={result.returncode}): {' | '.join(tail)}")


def ensure_h264_rendition(
    source: Path,
    *,
    codec: str,
    run: Callable[[List[str]], None] = _run_ffmpeg,
) -> Path:
    """Return the path TikTok should upload for ``source``.

    ``codec`` is the master's codec as already recorded on the delivery's
    ``MediaFingerprint`` — no probe is needed, the sidecar knows. An H.264
    master is returned untouched; anything else is transcoded once into a
    sibling file and reused on later passes.
    """
    source = Path(source)
    if not source.is_file():
        raise TikTokRenditionError(f"short not found: {source}")
    if codec.lower() in ("h264", "avc", "avc1"):
        return source

    target = rendition_path(source)
    if target.is_file() and target.stat().st_size > 0:
        # Rebuild when the short has been re-rendered since: uploading the
        # previous run's video would be worse than a slow drain pass.
        if target.stat().st_mtime >= source.stat().st_mtime:
            return target
        logger.info("tiktok rendition older than its source; rebuilding %s", target)

    logger.info("transcoding %s to H.264 for TikTok (master is %s)", source.name, codec)
    run(_ffmpeg_argv(source, target))

    if not target.is_file() or target.stat().st_size == 0:
        raise TikTokRenditionError(f"H.264 rendition is missing or empty after transcode: {target}")
    return target


#: Meta's Reels limit is 25 Mbps; 24 leaves headroom for container overhead.
INSTAGRAM_MAX_BITRATE_BPS = 24_000_000
INSTAGRAM_RENDITION_SUFFIX = ".instagram-h264.mp4"
_IG_TARGET_BITRATE = "16M"
_IG_MAX_BITRATE = "20M"
_IG_BUFSIZE = "40M"


def instagram_rendition_path(source: Path) -> Path:
    return source.with_name(source.stem + INSTAGRAM_RENDITION_SUFFIX)


def _instagram_argv(source: Path, target: Path) -> List[str]:
    argv = _ffmpeg_argv(source, target)
    for flag, value in (
        ("-b:v", _IG_TARGET_BITRATE),
        ("-maxrate", _IG_MAX_BITRATE),
        ("-bufsize", _IG_BUFSIZE),
    ):
        argv[argv.index(flag) + 1] = value
    return argv


def _probe_bitrate(path: Path) -> Optional[int]:
    """Container bitrate in bits/s via ffprobe, or None when unknowable."""
    try:
        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=bit_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return int(result.stdout.strip()) if result.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def ensure_instagram_rendition(
    source: Path,
    *,
    bitrate_bps: Optional[int] = None,
    probe: Callable[[Path], Optional[int]] = _probe_bitrate,
    run: Callable[[List[str]], None] = _run_ffmpeg,
) -> Path:
    """Return the path Instagram should upload for ``source``.

    A master at or under ``INSTAGRAM_MAX_BITRATE_BPS`` is returned untouched.
    Over the cap, or with a bitrate that cannot be probed (fail safe toward
    Meta's spec), it is transcoded once into a sibling file and reused.

    The cap is on the AVERAGE bitrate. ``bitrate_bps`` is the caller's
    already-known average (the delivery fingerprint's size over duration);
    only when it is absent does ``probe`` run. Short masters are VBR with
    ``SHORTS_MAXRATE`` 30 Mbps (``pipeline/encoding_constants.py``), so an
    under-cap master can still peak above 25 Mbps; the cap is judged on
    AVERAGE bitrate.
    """
    source = Path(source)
    if not source.is_file():
        raise TikTokRenditionError(f"short not found: {source}")
    bitrate = bitrate_bps if bitrate_bps is not None else probe(source)
    if bitrate is not None and bitrate <= INSTAGRAM_MAX_BITRATE_BPS:
        return source

    target = instagram_rendition_path(source)
    if target.is_file() and target.stat().st_size > 0:
        if target.stat().st_mtime >= source.stat().st_mtime:
            return target
        logger.info("instagram rendition older than its source; rebuilding %s", target)

    logger.info("transcoding %s for Instagram (master bitrate %s bps)", source.name, bitrate)
    run(_instagram_argv(source, target))

    if not target.is_file() or target.stat().st_size == 0:
        raise TikTokRenditionError(
            f"Instagram rendition is missing or empty after transcode: {target}"
        )
    return target
