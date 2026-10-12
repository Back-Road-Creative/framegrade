"""Video encoding utilities using FFmpeg."""

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..pipeline import encoding_constants as ec

logger = logging.getLogger(__name__)

# A start within this of a keyframe counts as landing on it — comfortably
# under one frame at any cadence the pipeline ingests (59.94 fps = 16.7 ms),
# so it can never swallow a whole frame of footage.
KEYFRAME_TOLERANCE_SEC = 0.01


@dataclass
class VideoSegment:
    """Defines a segment to extract from source video."""

    start_time: float  # seconds
    end_time: float  # seconds
    source_path: Path


@dataclass
class Chapter:
    """Chapter marker for output video."""

    start_time: float  # seconds
    title: str


class VideoEncoder:
    """
    Handles video encoding operations using FFmpeg.

    Optimized for YouTube 4K quality output.

    Every re-encode carries the same frozen output contract the day-first
    path stamps (``encoding_constants.LONGFORM_*``): 10-bit pix_fmt, main10
    profile, the four BT.709 color tags, and a resolution-based bitrate
    floor. Emitting only codec/CRF/preset — as this class used to — left the
    web ``/api/approve`` and CLI ``analyze``/``process`` renders 8-bit and
    untagged while ``config.output.pix_fmt``/``profile`` (both marked
    "Frozen") were read by nothing at all.
    """

    def __init__(
        self,
        codec: str = "libx265",
        crf: int = 18,
        preset: str = "slow",
        audio_codec: str = "aac",
        audio_bitrate: str = "320k",
        pix_fmt: str = ec.LONGFORM_PIX_FMT,
        profile: str = ec.LONGFORM_PROFILE,
    ):
        self.codec = codec
        self.crf = crf
        self.preset = preset
        self.audio_codec = audio_codec
        self.audio_bitrate = audio_bitrate
        self.pix_fmt = pix_fmt
        self.profile = profile
        self._height_cache: dict[str, Optional[int]] = {}

    def _source_height(self, source: Path) -> Optional[int]:
        """Pixel height of *source*'s first video stream, ffprobed once per path."""
        key = str(source)
        if key not in self._height_cache:
            height = None
            for stream in self.get_video_info(source).get("streams", []):
                candidate = stream.get("height")
                if isinstance(candidate, int) and candidate > 0:
                    height = candidate
                    break
            self._height_cache[key] = height
        return self._height_cache[key]

    def _quality_args(self, source: Path) -> list[str]:
        """Frozen-constant video args for a re-encode.

        Mirrors the floor selection in ``color_filters.get_encoding_args``
        (the day-first twin) — same constants, same thresholds. A source
        whose height cannot be probed gets no floor rather than a guessed
        one; CRF still governs quality.
        """
        args = [
            "-profile:v",
            self.profile,
            "-pix_fmt",
            self.pix_fmt,
            "-color_range",
            ec.LONGFORM_COLOR_RANGE,
            "-colorspace",
            ec.LONGFORM_COLORSPACE,
            "-color_primaries",
            ec.LONGFORM_COLOR_PRIMARIES,
            "-color_trc",
            ec.LONGFORM_COLOR_TRC,
        ]
        height = self._source_height(source)
        if height is None:
            logger.warning(
                "Could not probe height of %s — encoding without a bitrate floor", source
            )
            return args
        if height >= 2160:
            floor = ec.LONGFORM_MIN_BITRATE_4K
        elif height >= 1080:
            floor = ec.LONGFORM_MIN_BITRATE_1080P
        else:
            floor = 5000
        args.extend(
            [
                "-b:v",
                f"{floor}k",
                "-maxrate",
                f"{int(floor * ec.LONGFORM_MAXRATE_RATIO)}k",
                "-bufsize",
                f"{int(floor * ec.LONGFORM_BUFSIZE_RATIO)}k",
            ]
        )
        return args

    def _starts_land_on_keyframes(self, source: Path, starts: list[float]) -> bool:
        """True when EVERY start is a keyframe, so the whole cut can stream-copy.

        All-or-nothing on purpose: :meth:`concatenate` joins with ``-c copy``,
        which needs identical stream parameters, so a mix of copied source
        packets and freshly re-encoded ones would produce a broken file.
        Unknown (probe unavailable or failed) means re-encode — the lossy
        answer is always safe, the lossless one is not.
        """
        try:
            from keycut import probe_keyframes
        except ImportError:  # pragma: no cover - keycut is a hard dependency
            return False
        try:
            keyframes = probe_keyframes(source)
        except Exception as e:
            # WARNING (not DEBUG): a probe failure here silently downgrades
            # the whole cut to a re-encode (see ``trim_and_encode``'s INFO
            # line below) — an operator running at default log level must
            # see WHY, not just that it happened.
            logger.warning("Keyframe probe failed for %s: %s", source, e)
            return False
        if not keyframes:
            return False
        return all(any(abs(k - s) <= KEYFRAME_TOLERANCE_SEC for k in keyframes) for s in starts)

    def extract_segment(
        self,
        source: Path,
        output: Path,
        start_time: float,
        end_time: float,
        reencode: bool = False,
    ) -> bool:
        """
        Extract a segment from source video.

        Args:
            source: Source video path
            output: Output video path
            start_time: Start time in seconds
            end_time: End time in seconds
            reencode: If True, re-encode; if False, try stream copy

        Returns:
            True if successful
        """
        duration = end_time - start_time

        if reencode:
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
                self.codec,
                "-crf",
                str(self.crf),
                "-preset",
                self.preset,
                *self._quality_args(source),
                "-c:a",
                self.audio_codec,
                "-b:a",
                self.audio_bitrate,
                str(output),
            ]
        else:
            # Stream copy (fast, no quality loss, but may have issues at cut points)
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start_time),
                "-i",
                str(source),
                "-t",
                str(duration),
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                str(output),
            ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except Exception as e:
            logger.error("FFmpeg error: %s", e, exc_info=True)
            return False

    def concatenate(
        self,
        segments: list[Path],
        output: Path,
        chapters: Optional[list[Chapter]] = None,
    ) -> bool:
        """
        Concatenate multiple video segments into one.

        Uses FFmpeg concat demuxer for stream copy when possible.
        """
        if not segments:
            return False

        # Create concat file
        concat_file = output.parent / f"{output.stem}_concat.txt"
        try:
            with open(concat_file, "w") as f:
                for seg in segments:
                    # Escape single quotes in path
                    escaped_path = str(seg).replace("'", "'\\''")
                    f.write(f"file '{escaped_path}'\n")

            # Build FFmpeg command
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
            ]

            # Add chapters if provided
            if chapters:
                metadata_file = output.parent / f"{output.stem}_chapters.txt"
                self._write_chapters_metadata(chapters, metadata_file)
                cmd.extend(["-i", str(metadata_file), "-map_metadata", "1"])

            cmd.append(str(output))

            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0

        finally:
            # Cleanup temp files
            if concat_file.exists():
                concat_file.unlink()
            metadata_file = output.parent / f"{output.stem}_chapters.txt"
            if metadata_file.exists():
                metadata_file.unlink()

    def _write_chapters_metadata(self, chapters: list[Chapter], output_path: Path) -> None:
        """Write FFmpeg metadata file for chapters."""
        with open(output_path, "w") as f:
            f.write(";FFMETADATA1\n")
            for i, chapter in enumerate(chapters):
                # Calculate end time (next chapter start or video end)
                if i + 1 < len(chapters):
                    end_time = chapters[i + 1].start_time
                else:
                    end_time = chapter.start_time + 3600  # Arbitrary large value

                # FFmpeg uses milliseconds
                start_ms = int(chapter.start_time * 1000)
                end_ms = int(end_time * 1000)

                f.write("\n[CHAPTER]\n")
                f.write("TIMEBASE=1/1000\n")
                f.write(f"START={start_ms}\n")
                f.write(f"END={end_ms}\n")
                f.write(f"title={chapter.title}\n")

    def get_video_info(self, video_path: Path) -> dict:
        """Get video metadata using ffprobe."""
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

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return json.loads(result.stdout)
        except Exception as e:
            logger.error("ffprobe error: %s", e, exc_info=True)

        return {}

    def trim_and_encode(
        self,
        source: Path,
        output: Path,
        segments_to_keep: list[tuple[float, float]],
        chapters: Optional[list[Chapter]] = None,
    ) -> bool:
        """
        Create output video keeping only specified time ranges.

        This is the main method for creating cleaned videos with
        dead spots removed.

        Args:
            source: Source video path
            output: Output video path
            segments_to_keep: List of (start, end) tuples in seconds
            chapters: Optional chapter markers
        """
        if not segments_to_keep:
            return False

        # Bit-exact when every cut already sits on a keyframe: the documented
        # stream-copy branch used to be unreachable because both call sites
        # hardcoded reencode=True, so every trim paid a full generation of
        # loss even when nothing needed re-encoding.
        lossless = self._starts_land_on_keyframes(source, [s for s, _ in segments_to_keep])
        logger.info(
            "Assembly trim: %s (%d segment(s))",
            "keyframe-aligned stream copy — no generation loss" if lossless else "re-encoding",
            len(segments_to_keep),
        )
        reencode = not lossless

        # If only one segment, just extract it
        if len(segments_to_keep) == 1:
            start, end = segments_to_keep[0]
            return self.extract_segment(source, output, start, end, reencode=reencode)

        # Multiple segments - extract each, then concatenate
        temp_dir = output.parent / f"{output.stem}_temp"
        temp_dir.mkdir(exist_ok=True)

        temp_segments = []
        try:
            for i, (start, end) in enumerate(segments_to_keep):
                temp_path = temp_dir / f"segment_{i:03d}.mp4"
                if not self.extract_segment(source, temp_path, start, end, reencode=reencode):
                    return False
                temp_segments.append(temp_path)

            # Concatenate all segments
            return self.concatenate(temp_segments, output, chapters)

        finally:
            # Cleanup temp files
            for temp_seg in temp_segments:
                if temp_seg.exists():
                    temp_seg.unlink()
            if temp_dir.exists():
                temp_dir.rmdir()
