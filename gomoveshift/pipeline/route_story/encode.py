"""Route Story final encode: single-pass delivery quality and delivery sidecars.

This module turns the concatenated Route Story cut (``assembly``/``film``)
into a delivery-ready file -- single-pass libx264 at delivery quality with
BT.709 colour tags and attenuate-only audio -- plus the ``publish.json``
sidecar. It does not decide clip order, timing, or chapter titles: the
chapter list comes from ``beats.chapter_rows`` (which also feeds
``beats.write_chapters_vtt``), and this module mirrors it.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..audio_profile import AudioProfile, LicenceLedger, resolve_bed
from ..color_filters import NVENCUnavailableError, detect_nvenc
from ..encoding_constants import RENDITIONS
from ..journal_publisher import ffprobe_duration_seconds
from ..loudnorm_measure import (
    build_attenuate_filter,
    build_loudnorm_measure_cmd,
    parse_loudnorm_stats,
)
from .assembly import _AUDIO_CODEC_ARGS

logger = logging.getLogger(__name__)

__all__ = ["encode_final", "encode_rendition", "write_publish_json", "RenditionCeilingError"]

# Logical codec name -> ffmpeg encoder. Rendition.codec is the delivery
# family ("hevc"); the NVENC-vs-CPU-fallback choice each caller makes
# (color_filters.py, shorts_extractor.py) is out of scope for this CPU-only
# command builder, so it always takes the libx265 leg.
_CODEC_ENCODERS = {"hevc": "libx265"}


class RenditionCeilingError(Exception):
    """A rendition's own ceiling (e.g. max_seconds) can't be met by this master."""


_BT709_TAGS = [
    "-color_range",
    "tv",
    "-colorspace",
    "bt709",
    "-color_primaries",
    "bt709",
    "-color_trc",
    "bt709",
]


def _hevc_video_args() -> list[str]:
    """10-bit HEVC delivery args: hevc_nvenc, or libx265 when the GPU probe fails."""
    try:
        has_nvenc = detect_nvenc()
    except NVENCUnavailableError:
        has_nvenc = False
    if has_nvenc:
        head = ["-c:v", "hevc_nvenc", "-preset", "p7", "-rc", "vbr", "-cq", "18"]
    else:
        logger.warning("NVENC unavailable; full4k film falls back to libx265 (CPU)")
        head = ["-c:v", "libx265", "-preset", "slow", "-crf", "18"]
    return [*head, "-pix_fmt", "p010le", "-profile:v", "main10", *_BT709_TAGS]


def _attenuate_filter(source: Path) -> str:
    """The service's attenuate-only gain for this cut: down to the ceiling, never up.

    A bare ``loudnorm`` normalises TOWARD a target, so it lifted a -51 LUFS
    phone clip to -46 -- a boost, which no published video may carry. The
    measurement pass is the long-form one; when it fails, the builder falls
    back to a fixed attenuation rather than passing the audio through.
    """
    measured = None
    try:
        result = subprocess.run(
            build_loudnorm_measure_cmd(source), capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            measured = parse_loudnorm_stats(result.stderr or "")
    except (OSError, ValueError) as exc:
        logger.warning("film loudness measurement failed (%s); using the fixed attenuation", exc)
    if measured is None:
        logger.warning("no loudness measurement for %s; using the fixed attenuation", source.name)
    audio_filter = build_attenuate_filter(measured)
    logger.info("film audio filter: %s", audio_filter)
    return audio_filter


def _music_bed_receipt(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "file": entry.get("file"),
        "ledger_entry": dict(entry),
        "source": entry.get("source"),
        "licence": entry.get("licence"),
    }


def encode_final(
    source: Path,
    output: Path,
    *,
    audio_profile: AudioProfile | None = None,
    ledger: LicenceLedger | None = None,
    rendition: str | None = None,
) -> dict[str, Any] | None:
    """Single-pass delivery encode: crf 18, preset slow, BT.709 tags, attenuate-only audio.

    Runs once, on the fully-assembled cut -- never per clip.

    ``audio_profile``, when given and enabled with a ``bed_path``, mixes a
    licensed music bed under the attenuated source: the bed is
    looped (``-stream_loop -1``), gained by ``audio_profile.gain_db``, and
    faded out over its last ``audio_profile.fade_out_s`` before being
    mixed under the source at ``duration=first`` -- so the mixed track
    never runs longer than the source. ``resolve_bed`` (called before any
    ffmpeg invocation) raises ``UnlicensedBedError`` when the bed file has
    no matching entry in the ledger -- an unlicensed bed is never burned.
    Returns the music-bed receipt (``{file, ledger_entry, source,
    licence}``) for the caller to pass to :func:`write_publish_json`, or
    ``None`` when no bed was mixed in -- in which case the command is
    byte-identical to the no-profile path.

    ``rendition`` names a ``RENDITIONS`` row; when its codec is ``hevc`` the
    video leg is hevc_nvenc (libx265 when ``detect_nvenc`` finds no GPU)
    instead of libx264. ``None`` (full, social) keeps the libx264 command
    byte-identical to the pre-preset one.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Resolve the bed's licence BEFORE any ffmpeg invocation (including the
    # loudness measurement pass) -- an unlicensed bed must never trigger
    # even a read-only ffmpeg call against the source.
    music_bed: dict[str, Any] | None = None
    bed_path: Path | None = None
    if audio_profile is not None and audio_profile.enabled and audio_profile.bed_path is not None:
        resolved_ledger = (
            ledger if ledger is not None else LicenceLedger.load(audio_profile.ledger_path)
        )
        entry = resolve_bed(audio_profile, resolved_ledger)
        music_bed = _music_bed_receipt(entry)
        bed_path = audio_profile.bed_path

    audio_filter = _attenuate_filter(Path(source))

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
    ]
    if bed_path is not None:
        duration = ffprobe_duration_seconds(Path(source))
        fade_start = max(0.0, duration - audio_profile.fade_out_s)
        filter_complex = (
            f"[1:a]volume={audio_profile.gain_db}dB,"
            f"afade=t=out:st={fade_start:.3f}:d={audio_profile.fade_out_s:.3f}[bed];"
            f"[0:a]{audio_filter}[src];"
            "[src][bed]amix=inputs=2:duration=first:dropout_transition=0[aout]"
        )
        cmd += [
            "-stream_loop",
            "-1",
            "-i",
            str(bed_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v",
            "-map",
            "[aout]",
        ]
    if rendition is not None and RENDITIONS[rendition].codec == "hevc":
        cmd += _hevc_video_args()
    else:
        cmd += [
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "slow",
            "-pix_fmt",
            "yuv420p",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            # libx264 needs the VUI flags spelled out via -x264-params too --
            # the top-level -color_* options alone tag the container but
            # leave color_primaries/color_transfer "unknown" in the encoded
            # stream itself, which is what any downstream probe reads.
            "-x264-params",
            "colorprim=bt709:transfer=bt709:colormatrix=bt709",
        ]
    if bed_path is None:
        cmd += ["-af", audio_filter]
    cmd += [
        # The clips going in are 48 kHz stereo (``assembly._AUDIO_CODEC_ARGS``);
        # the delivery file is pinned there whatever the filter hands on.
        *_AUDIO_CODEC_ARGS,
        str(output),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return music_bed


def encode_rendition(master: Path, platform: str, output: Path) -> Path:
    """Encode ``master`` to ``platform``'s spec in ``encoding_constants.RENDITIONS``.

    Refuses -- before any ffmpeg invocation -- when the master's duration
    exceeds the rendition's ``max_seconds`` ceiling. A ceiling left ``None``
    (no source constant for it) is never enforced; guessed platform ceilings
    belong in the table, not in this function.
    """
    master = Path(master)
    output = Path(output)
    rendition = RENDITIONS[platform]

    if rendition.max_seconds is not None:
        duration = ffprobe_duration_seconds(master)
        if duration > rendition.max_seconds:
            raise RenditionCeilingError(
                f"{platform}: master duration {duration:.2f}s exceeds the "
                f"{rendition.max_seconds:.2f}s rendition ceiling"
            )

    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y", "-i", str(master)]
    if rendition.width is not None and rendition.height is not None:
        cmd += ["-vf", f"scale={rendition.width}:{rendition.height}"]
    cmd += ["-c:v", _CODEC_ENCODERS.get(rendition.codec, rendition.codec)]
    if rendition.faststart:
        cmd += ["-movflags", "+faststart"]
    if not rendition.audio:
        cmd += ["-an"]
    cmd += [str(output)]

    subprocess.run(cmd, check=True, capture_output=True)
    return output


def write_publish_json(
    chapters: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    film_duration_s: float,
    privacy: Mapping[str, Any] | None = None,
    music_bed: Mapping[str, Any] | None = None,
) -> None:
    """Write the publish sidecar: ``beats.chapter_rows`` verbatim, machine-readable.

    The rows are the same list ``beats.write_chapters_vtt`` writes, so the two
    sidecars cannot disagree; this module never builds a chapter list itself.
    ``privacy``, when given, is ``route_story.privacy.PrivacyResult.attestation``
    verbatim -- the same block the film receipt carries -- so a downstream
    consumer of ``publish.json`` alone can still evaluate
    ``publishing.dispatch.privacy_hold_reason`` without the full receipt.
    ``music_bed`` is ``encode_final``'s returned receipt verbatim,
    or ``None`` when no bed was mixed in -- always written (unlike
    ``privacy``) so a published film can always prove whether it carries
    licensed music.
    """
    if film_duration_s <= 0:
        raise ValueError("film_duration_s must be positive")
    if not chapters:
        raise ValueError("needs at least one chapter")

    payload: dict[str, Any] = {
        "duration_s": film_duration_s,
        "chapters": [dict(row) for row in chapters],
        "music_bed": dict(music_bed) if music_bed is not None else None,
    }
    if privacy is not None:
        payload["privacy"] = dict(privacy)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
