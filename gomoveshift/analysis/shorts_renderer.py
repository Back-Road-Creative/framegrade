"""FFmpeg-based clip rendering and thumbnail extraction for shorts."""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class ShortsRenderer:
    """Renders short clips and extracts thumbnails using FFmpeg."""

    def render_clip(
        self,
        source: Path,
        output: Path,
        start_time: float,
        end_time: float,
    ) -> bool:
        """Render a clip using ffmpeg."""
        duration = end_time - start_time

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_time),
            "-i",
            str(source),
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(output),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            return result.returncode == 0 and output.exists()
        except Exception as e:
            print(f"Error rendering clip: {e}")
            return False

    def extract_thumbnail(
        self,
        source: Path,
        output: Path,
        timestamp: float,
    ) -> bool:
        """Extract a thumbnail frame."""
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(timestamp),
            "-i",
            str(source),
            "-vframes",
            "1",
            "-q:v",
            "2",
            str(output),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0 and output.exists()
        except Exception as e:
            print(f"Error extracting thumbnail: {e}")
            return False
