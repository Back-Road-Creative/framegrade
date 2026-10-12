"""Measure the finished film against the export that was asked for.

The film command used to print a path and exit 0 whatever it had encoded; the
only way to learn a film was off-spec, black, or short of frames was to probe
it by hand. This reads the delivered file back -- stream format, colour tags,
audio format, frame count -- and scans the WHOLE picture for black and frozen
stretches. A film is minutes long, not hours, so unlike the long-form
validator's three sampled windows this never samples.
"""

from __future__ import annotations

import json
import re
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

from ..validation.longform_validator import color_tag_issues

VIDEO_CODEC = "h264"
PIXEL_FORMAT = "yuv420p"
# What ffprobe reports for the 10-bit (p010le) HEVC delivery encode.
HEVC_PIXEL_FORMAT = "yuv420p10le"
AUDIO_CODEC = "aac"
AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNELS = 2
# A black run this long is never a cut; nothing in a route-story film is black.
BLACK_MIN_S = 0.5
# A still held this long is worth a reader's eye (a title card, or a stuck map).
FROZEN_MIN_S = 2.0
_SCAN_FILTER = f"blackdetect=d={BLACK_MIN_S}:pix_th=0.10,freezedetect=n=0.003:d={FROZEN_MIN_S}"
_PROBE_TIMEOUT_S = 120
_SCAN_TIMEOUT_S = 1800

_BLACK_RE = re.compile(r"black_start:\s*([0-9.]+).*?black_duration:\s*([0-9.]+)")
_FREEZE_START_RE = re.compile(r"freeze_start:\s*([0-9.]+)")
_FREEZE_DURATION_RE = re.compile(r"freeze_duration:\s*([0-9.]+)")

Runner = Callable[..., Any]


def check_delivery(
    path: Path,
    *,
    width: int,
    height: int,
    fps: int,
    runner: Runner = subprocess.run,
    expected_codec: str = VIDEO_CODEC,
    expected_pix_fmt: str = PIXEL_FORMAT,
) -> dict[str, Any]:
    """Return what the delivered film measures as, and every way it misses the spec.

    ``issues`` is empty for a film that is what was asked for. A black stretch
    is an issue; a frozen one is recorded for the reader and is not, because a
    held card is a choice. A file that will not probe is an issue, never a pass.
    """
    report: dict[str, Any] = {"file": Path(path).name, "issues": [], "black": [], "frozen": []}
    issues: list[str] = report["issues"]
    probe = _probe(Path(path), runner)
    if probe is None:
        issues.append("ffprobe could not read the delivered file")
        return report

    streams = probe.get("streams", [])
    duration = float(probe.get("format", {}).get("duration", 0) or 0)
    report["duration_s"] = duration
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        issues.append("the delivered file has no video stream")
    else:
        issues.extend(
            _video_issues(
                video, report, width, height, fps, duration, expected_codec, expected_pix_fmt
            )
        )
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if len(audio) != 1:
        issues.append(f"expected one audio stream, found {len(audio)}")
    else:
        issues.extend(_audio_issues(audio[0], report))

    scanned = _scan(Path(path), duration, runner)
    if scanned is None:
        issues.append("the black/frozen picture scan did not run")
        return report
    report["black"], report["frozen"] = scanned
    for run in report["black"]:
        issues.append(f"black picture for {run['duration_s']:g} s from {run['start_s']:g} s")
    return report


def _probe(path: Path, runner: Runner) -> dict[str, Any] | None:
    cmd = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
    try:
        result = runner(cmd, capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S)
        if result.returncode != 0:
            return None
        parsed: dict[str, Any] = json.loads(result.stdout)
        return parsed
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _video_issues(
    video: dict[str, Any],
    report: dict[str, Any],
    width: int,
    height: int,
    fps: int,
    duration: float,
    expected_codec: str = VIDEO_CODEC,
    expected_pix_fmt: str = PIXEL_FORMAT,
) -> list[str]:
    issues: list[str] = []
    try:
        rate = float(Fraction(str(video.get("r_frame_rate") or "0")))
    except (ValueError, ZeroDivisionError):
        rate = 0.0
    frames = int(video.get("nb_frames") or 0)
    report["video"] = {
        "codec": video.get("codec_name"),
        "width": video.get("width"),
        "height": video.get("height"),
        "pix_fmt": video.get("pix_fmt"),
        "fps": rate,
        "frames": frames,
    }
    if video.get("codec_name") != expected_codec:
        issues.append(f"video codec is {video.get('codec_name')}, expected {expected_codec}")
    if (video.get("width"), video.get("height")) != (width, height):
        issues.append(
            f"video is {video.get('width')}x{video.get('height')}, expected {width}x{height}"
        )
    if abs(rate - fps) > 1e-6:
        issues.append(f"frame rate is {rate:g} fps, expected {fps}")
    if video.get("pix_fmt") != expected_pix_fmt:
        issues.append(f"pixel format is {video.get('pix_fmt')}, expected {expected_pix_fmt}")
    issues.extend(color_tag_issues(video))
    expected_frames = round(duration * fps)
    if abs(frames - expected_frames) > 1:
        issues.append(
            f"{frames} frames does not match {duration:g} s at {fps} fps ({expected_frames} frames)"
        )
    return issues


def _audio_issues(audio: dict[str, Any], report: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    sample_rate = int(audio.get("sample_rate") or 0)
    channels = int(audio.get("channels") or 0)
    report["audio"] = {
        "codec": audio.get("codec_name"),
        "sample_rate": sample_rate,
        "channels": channels,
    }
    if audio.get("codec_name") != AUDIO_CODEC:
        issues.append(f"audio codec is {audio.get('codec_name')}, expected {AUDIO_CODEC}")
    if sample_rate != AUDIO_SAMPLE_RATE:
        issues.append(f"audio sample rate is {sample_rate}, expected {AUDIO_SAMPLE_RATE}")
    if channels != AUDIO_CHANNELS:
        issues.append(f"audio has {channels} channels, expected {AUDIO_CHANNELS}")
    return issues


def _scan(
    path: Path, duration: float, runner: Runner
) -> tuple[list[dict[str, float]], list[dict[str, float]]] | None:
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path)]
    cmd += ["-an", "-vf", _SCAN_FILTER, "-f", "null", "-"]
    try:
        result = runner(cmd, capture_output=True, text=True, timeout=_SCAN_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    stderr = result.stderr or ""
    black = [
        {"start_s": float(start), "duration_s": float(length)}
        for start, length in _BLACK_RE.findall(stderr)
    ]
    frozen: list[dict[str, float]] = []
    for line in stderr.splitlines():
        started = _FREEZE_START_RE.search(line)
        if started:
            # Open until its duration line; a freeze still running at the end
            # of the film never gets one, so it is closed against the duration.
            start = float(started.group(1))
            frozen.append({"start_s": start, "duration_s": round(max(0.0, duration - start), 3)})
            continue
        length = _FREEZE_DURATION_RE.search(line)
        if length and frozen:
            frozen[-1]["duration_s"] = float(length.group(1))
    return black, frozen
