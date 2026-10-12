"""Video property detection via ffprobe.

Extracts resolution, FPS, duration, codec, and bitrate from video files
using ffprobe. Provides the VideoProperties dataclass used throughout
the pipeline.
"""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from gomoveshift.pipeline._json_safe import safe_json_loads

logger = logging.getLogger(__name__)


@dataclass
class VideoProperties:
    """Detected properties of the source video."""

    width: int
    height: int
    fps: float
    duration: float
    codec: str
    audio_codec: str
    file_size_bytes: int
    bitrate_kbps: int = 0

    @property
    def resolution_label(self) -> str:
        if self.height >= 2160:
            return "4K"
        elif self.height >= 1440:
            return "1440p"
        elif self.height >= 1080:
            return "1080p"
        elif self.height >= 720:
            return "720p"
        else:
            return f"{self.height}p"


def detect_video_properties(video_path: Path) -> VideoProperties:
    """Use ffprobe to detect video properties."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    probe = safe_json_loads(result.stdout, "ffprobe video properties")

    # Find video stream
    video_stream = None
    audio_stream = None
    for stream in probe.get("streams", []):
        if stream["codec_type"] == "video" and video_stream is None:
            video_stream = stream
        elif stream["codec_type"] == "audio" and audio_stream is None:
            audio_stream = stream

    if not video_stream:
        raise RuntimeError("No video stream found in file")

    # Parse FPS from r_frame_rate (e.g., "30000/1001" or "30/1")
    fps_str = video_stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 30.0
    else:
        fps = float(fps_str)

    fmt = probe.get("format", {})
    duration = float(fmt.get("duration", 0))
    file_size = int(fmt.get("size", 0))
    bitrate_kbps = int(fmt.get("bit_rate", 0)) // 1000

    return VideoProperties(
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        fps=round(fps, 3),
        duration=duration,
        codec=video_stream.get("codec_name", "unknown"),
        audio_codec=audio_stream.get("codec_name", "unknown") if audio_stream else "none",
        file_size_bytes=file_size,
        bitrate_kbps=bitrate_kbps,
    )
