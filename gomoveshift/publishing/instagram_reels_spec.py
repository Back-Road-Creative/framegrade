"""An in-spec Instagram Reels delivery file, derived from a rendered Short.

Meta's IG User media reference (Reels specifications) asks for "MOV or MP4, no
edit lists, moov atom at the front", "HEVC or H264, progressive scan, closed
GOP, 4:2:0", "VBR, 25Mbps maximum", AAC at 48 kHz or less, 23-60 fps and at
most 300 MB. Our rendered Shorts are HEVC Main 10 with an ``elst`` edit list at
27-30 Mbps, and 11 of 16 rows on 2026-11-16 died with rupload
``ProcessingFailedError`` on all three tries (#399). So Instagram gets its own
file: ``make_in_spec_bytes`` transcodes, gates the result with ffprobe plus a
box walk, and only then hands back bytes. The YouTube file is never touched.
"""

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path
from typing import Callable

from gomoveshift.pipeline.ffmpeg_runner import run_ffmpeg_quick

MAX_BITRATE_BPS = 25_000_000
MAX_SIZE_BYTES = 300 * 1024 * 1024
MIN_FPS, MAX_FPS = 23.0, 60.0
MAX_AUDIO_RATE_HZ = 48_000
#: ``-maxrate`` sits under Meta's 25 Mbps with room for the audio track.
VIDEO_MAXRATE = "20M"
VIDEO_BUFSIZE = "40M"
TRANSCODE_TIMEOUT_S = 900
_ACCEPTED_VIDEO = {"h264": "H.264", "hevc": "HEVC"}


class ReelsSpecError(Exception):
    """The transcode failed, or its output breaks a Reels rule."""


def build_ffmpeg_argv(src: Path, dst: Path, ffmpeg: str = "ffmpeg") -> list[str]:
    """H.264 High 8-bit 4:2:0, closed GOP, capped VBR, faststart, no edit list,
    AAC 48 kHz stereo; resolution and frame rate are left as rendered."""
    return [
        ffmpeg, "-hide_banner", "-nostdin", "-y",
        "-i", str(src),
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-crf", "18", "-maxrate", VIDEO_MAXRATE, "-bufsize", VIDEO_BUFSIZE,
        "-x264-params", "open-gop=0",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", "-use_editlist", "0",
        "-f", "mp4", str(dst),
    ]  # fmt: skip


def _boxes(data: bytes, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack(">I4s", data[pos : pos + 8])
        header = 8
        if size == 1:
            size = struct.unpack(">Q", data[pos + 8 : pos + 16])[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header:
            return
        yield kind.decode("latin-1"), pos + header, min(pos + size, end)
        pos += size


def mp4_layout(path: Path) -> tuple[bool, bool]:
    """``(moov_before_mdat, has_edit_list)`` from the file's boxes."""
    data = Path(path).read_bytes()
    order = [kind for kind, _, _ in _boxes(data, 0, len(data))]
    moov_first = "moov" in order and (
        "mdat" not in order or order.index("moov") < order.index("mdat")
    )
    has_elst = False
    for kind, s, e in _boxes(data, 0, len(data)):
        if kind != "moov":
            continue
        for k2, s2, e2 in _boxes(data, s, e):
            if k2 != "trak":
                continue
            for k3, s3, e3 in _boxes(data, s2, e2):
                if k3 == "edts" and any(k4 == "elst" for k4, _, _ in _boxes(data, s3, e3)):
                    has_elst = True
    return moov_first, has_elst


def _fps(rate: str) -> float:
    num, _, den = (rate or "0/1").partition("/")
    try:
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def check_spec(probe: dict, *, moov_first: bool, has_elst: bool, size_bytes: int) -> list[str]:
    """Every Reels rule this file breaks, one line each; empty means in spec."""
    fails: list[str] = []
    streams = probe.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        return ["no video stream"]
    codec = video.get("codec_name")
    if codec not in _ACCEPTED_VIDEO:
        fails.append(f"video codec {codec!r} is not H.264 or HEVC")
    if video.get("pix_fmt") != "yuv420p":
        fails.append(f"pix_fmt {video.get('pix_fmt')!r} is not 8-bit 4:2:0 (yuv420p)")
    if video.get("field_order", "progressive") not in ("progressive", "unknown"):
        fails.append(f"field_order {video.get('field_order')!r} is not progressive")
    fps = _fps(video.get("avg_frame_rate", ""))
    if not MIN_FPS <= fps <= MAX_FPS:
        fails.append(f"frame rate {fps:.2f} fps is outside {MIN_FPS:g}-{MAX_FPS:g}")
    bitrate = int((probe.get("format") or {}).get("bit_rate") or 0)
    if bitrate > MAX_BITRATE_BPS:
        fails.append(
            f"bitrate {bitrate / 1e6:.1f} Mbps exceeds {MAX_BITRATE_BPS // 1_000_000} Mbps"
        )
    if size_bytes > MAX_SIZE_BYTES:
        fails.append(f"size {size_bytes / 1048576:.0f} MB exceeds 300 MB")
    if has_elst:
        fails.append("an edit list (elst) is present")
    if not moov_first:
        fails.append("moov atom is not before mdat")
    if audio is not None:
        if audio.get("codec_name") != "aac":
            fails.append(f"audio codec {audio.get('codec_name')!r} is not AAC")
        if int(audio.get("sample_rate") or 0) > MAX_AUDIO_RATE_HZ:
            fails.append(f"audio sample rate {audio.get('sample_rate')} Hz exceeds 48000")
    return fails


def probe_file(path: Path, run: Callable = run_ffmpeg_quick) -> dict:
    result = run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
         str(path)],
        timeout=60,
        description="ffprobe reels delivery file",
    )  # fmt: skip
    return json.loads(result.stdout)


def gate_file(path: Path, run: Callable = run_ffmpeg_quick) -> None:
    """Raise ``ReelsSpecError`` naming each broken rule; return when in spec."""
    path = Path(path)
    moov_first, has_elst = mp4_layout(path)
    fails = check_spec(
        probe_file(path, run),
        moov_first=moov_first,
        has_elst=has_elst,
        size_bytes=path.stat().st_size,
    )
    if fails:
        raise ReelsSpecError("delivery file breaks the Reels spec: " + "; ".join(fails))


def make_in_spec_bytes(
    video_bytes: bytes,
    *,
    run: Callable = run_ffmpeg_quick,
    scratch_dir: str | None = None,
) -> bytes:
    """Transcode ``video_bytes`` to an in-spec MP4 in a scratch dir this call
    owns, gate it, return its bytes, and delete the scratch dir."""
    with tempfile.TemporaryDirectory(prefix="ig-reels-", dir=scratch_dir) as tmp:
        src, dst = Path(tmp) / "source.mp4", Path(tmp) / "reels.mp4"
        src.write_bytes(video_bytes)
        try:
            run(
                build_ffmpeg_argv(src, dst),
                output_path=dst,
                timeout=TRANSCODE_TIMEOUT_S,
                description="instagram reels transcode",
            )
        except RuntimeError as exc:
            raise ReelsSpecError(f"in-spec transcode failed: {exc}") from exc
        gate_file(dst, run)
        return dst.read_bytes()
