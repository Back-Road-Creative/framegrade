"""Route Story film assembly: frame sequencing, encoding, and video evidence.

This module turns rendered map/photo frames into files on disk — appending a
map leg or a photo hold to the frame sequence, saving individual frames,
invoking ffmpeg/ffprobe to encode and concatenate clips, and reading optional
creation-time/GPS tags from a source video. It does not decide what a frame
looks like (``layers``, ``hud``) or how a whole film is structured
(``film``).
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

from PIL import Image, ImageFilter

from .anchors import MediaAnchor, MediaEvidence, captured_at_route_seconds
from .hud import (
    HIGH_POINT_CALLOUT_SECONDS,
    UNPLACED_PHOTO_CAPTION,
    UNPLACED_VIDEO_SUBTITLE,
    _draw_photo_label,
    _fit_cover,
    _load_still,
    _unlocated_video_card,
    draw_high_point,
    draw_watermark,
    prepare_still,
    still_card_frame,
)
from .layers import MapRenderer
from .metrics import RoutePoint

if TYPE_CHECKING:
    from .film import PhotoMoment

logger = logging.getLogger(__name__)

# A film render runs for an hour or more and used to print nothing, so nobody
# could read an ETA off a live run. Every this-many map frames a leg says how
# far along it is and what a frame is costing.
PROGRESS_EVERY_FRAMES = 50


@dataclass
class FilmTimeline:
    """Planned cuts/holds for ``continuity.analyze_rendered_continuity``,
    recorded in per-segment-local frame numbers as frames are appended --
    the same code that writes a hold's frames is what records its span, so
    the two can never disagree.

    A "segment" is one clip in final concatenation order (each main frame
    sequence, the placed source video if any, and the recap); the final
    concatenation is itself a hard cut, so ``resolve()`` adds one at every
    segment's own start once every segment's frame count is known -- callers
    never record that boundary cut by hand.
    """

    _segment_lengths: list[int | None] = field(default_factory=list)
    _local_cuts: list[list[int]] = field(default_factory=list)
    _local_holds: list[list[tuple[int, int]]] = field(default_factory=list)
    #: Frame (in the film's first segment) where the opening card and the first map leg
    #: end -- the span an LF can borrow as its opening. ``None`` until the leg is rendered.
    opening_end_frame: int | None = None

    def mark_opening_end(self, frame: int) -> None:
        """Record the first map leg's end; only the first call counts."""
        if self.opening_end_frame is None:
            self.opening_end_frame = frame

    def new_segment(self) -> int:
        """Open a new segment, returning the id later calls address it by."""
        self._segment_lengths.append(None)
        self._local_cuts.append([])
        self._local_holds.append([])
        return len(self._segment_lengths) - 1

    def close_segment(self, segment: int, frame_count: int) -> None:
        """Record how many frames ``segment`` ended up with, once known."""
        self._segment_lengths[segment] = frame_count

    def record_cut(self, segment: int, local_frame: int) -> None:
        self._local_cuts[segment].append(local_frame)

    def record_hold(self, segment: int, local_frame: int, length: int) -> None:
        self._local_holds[segment].append((local_frame, length))

    def resolve(self) -> tuple[list[int], list[tuple[int, int]]]:
        """Absolute ``(cut_indices, hold_spans)`` once every segment is closed."""
        if any(length is None for length in self._segment_lengths):
            raise ValueError("every FilmTimeline segment must be closed before resolve()")
        offsets: list[int] = []
        total = 0
        for length in self._segment_lengths:
            offsets.append(total)
            total += length  # type: ignore[operator]
        cuts = set(offsets)
        holds: list[tuple[int, int]] = []
        for segment, base in enumerate(offsets):
            for local in self._local_cuts[segment]:
                cuts.add(base + local)
            for local, length in self._local_holds[segment]:
                holds.append((base + local, length))
        return sorted(cuts), holds


def _append_map_leg(
    frames: Path,
    frame: int,
    segments: Sequence[Sequence[RoutePoint]],
    renderer: MapRenderer,
    start_time: float,
    end_time: float,
    *,
    frame_count: int,
    width: int,
    height: int,
    callout: tuple[float, str] | None = None,
    watermark: str | None = None,
    dissolve_from: Image.Image | None = None,
    fps: int | None = None,
) -> tuple[int, Image.Image | None]:
    """Returns ``(next_frame_number, last_rendered_image)``.

    ``last_rendered_image`` is the exact frame the leg's own renderer last
    produced -- untouched by any callout badge drawn on top of it, and
    carrying whatever bearing the renderer's rate limiting settled on. A
    caller that needs to hold on the leg's ending view (the settle-before-a-
    photo hold, the video-arrival hold) must reuse this image rather than
    asking the renderer for a second, un-rate-limited frame at the same
    time -- that second call is a camera snap the viewer sees as a jump.
    ``dissolve_from`` (the still that just played) fades into the leg's first
    frames on the SAVED copy only; the rendered image is never blended.
    """
    if dissolve_from is not None and fps is None:
        raise ValueError("a map leg dissolve needs the film's fps")
    times = [
        start_time + (end_time - start_time) * (index + 1) / frame_count
        for index in range(frame_count)
    ]
    started = time.monotonic()
    last_image: Image.Image | None = None
    for done, image in enumerate(renderer.render_leg(segments, times, width, height), start=1):
        last_image = image
        # A landmark badge belongs to its own moment on the route. Drawn on
        # every frame of the leg it would stop being a landmark and become
        # permanent chrome, so it rides only the seconds either side of the
        # fix it names -- and a leg that never passes that fix shows nothing.
        if callout is not None and abs(times[done - 1] - callout[0]) <= HIGH_POINT_CALLOUT_SECONDS:
            draw_high_point(image, callout[1], width, height)
        saved = image
        if dissolve_from is not None and fps is not None:
            # Blend into a copy: ``last_image`` must stay the renderer's own
            # pixels for the settle hold that reuses it.
            saved = _dissolved(image.copy(), dissolve_from, done - 1, fps)
        _save_frame(saved, frames, frame, watermark=watermark)
        frame += 1
        if done % PROGRESS_EVERY_FRAMES == 0 or done == frame_count:
            per_frame = (time.monotonic() - started) / done
            logger.info(
                "map leg %d/%d frames, %.2f s/frame, ~%.0f s left in this leg (film frame %d)",
                done,
                frame_count,
                per_frame,
                per_frame * (frame_count - done),
                frame,
            )
    return frame, last_image


# How long the map holds still on a beat's own route moment before the media
# plays. A placed video announces itself with an arrival badge and needs a beat
# to be read; a still needs only enough stillness to land on, so equal lengths
# would make every photo feel like a video that failed to play.
MAP_SETTLE_SECONDS = 0.6
VIDEO_ARRIVAL_HOLD_SECONDS = 1.2
# Map -> still and still -> map cross-fade. Spent inside the frames each scene
# already has, so no film ever gets longer or re-paced by it.
DISSOLVE_SECONDS = 0.5


def _dissolved(
    image: Image.Image, dissolve_from: Image.Image | None, index: int, fps: int
) -> Image.Image:
    """``image`` blended from ``dissolve_from`` over the first DISSOLVE_SECONDS.

    Returns ``image`` itself (never a copy) once the dissolve is over or when
    there is nothing to dissolve from.
    """
    steps = max(1, round(fps * DISSOLVE_SECONDS))
    if dissolve_from is None or index >= steps:
        return image
    return Image.blend(
        dissolve_from.convert("RGB"), image.convert("RGB"), (index + 1) / (steps + 1)
    )


def _append_hold(
    frames: Path,
    frame: int,
    image: Image.Image,
    fps: int,
    seconds: float,
    *,
    timeline: FilmTimeline | None = None,
    timeline_segment: int | None = None,
) -> int:
    """Write one already-rendered image for ``seconds``, returning the next frame.

    One render call, reused for every frame: re-rendering pixels that never
    change costs real minutes under the scene renderer. Always at least one
    frame -- a settle rounded away to nothing at a low fps would silently put
    the cut back exactly where it was.

    When ``timeline`` and ``timeline_segment`` are both given, the exact hold
    just written is recorded on it -- the frozen span this hold produces is
    always the one the film's own render loop planned, never a second guess
    computed from the outside.
    """
    count = max(1, round(fps * seconds))
    if timeline is not None and timeline_segment is not None:
        timeline.record_hold(timeline_segment, frame, count)
    for _ in range(count):
        _save_frame(image, frames, frame)
        frame += 1
    return frame


def _append_photo(
    frames: Path,
    frame: int,
    photo: "PhotoMoment",
    fps: int,
    seconds: float,
    width: int,
    height: int,
    *,
    mapped: bool,
    still_source: str = "developed",
    caption: str | None = None,
    watermark: str | None = None,
    dissolve_from: Image.Image | None = None,
) -> int:
    # Decode once and bound it to the pixels the card samples; every frame of
    # the push-in then resamples that one source.
    still = prepare_still(_load_still(photo.source, source=still_source), width, height)
    count = max(1, round(fps * seconds))
    for index in range(count):
        progress = index / (count - 1) if count > 1 else 1.0
        image = still_card_frame(still, width, height, progress)
        if not mapped:
            _draw_photo_label(image, caption if caption is not None else UNPLACED_PHOTO_CAPTION)
        _save_frame(
            _dissolved(image, dissolve_from, index, fps), frames, frame, watermark=watermark
        )
        frame += 1
    return frame


def map_leg_seconds(
    start_time: float, end_time: float, *, minimum: float = 3.0, maximum: float = 14.0
) -> float:
    """The default per-leg film seconds ``plan_map_legs`` sums for its default total.

    No longer decides an individual leg's on-screen duration directly — see
    ``plan_map_legs``, which applies one global compression factor across every
    leg so marker motion stays proportional to real speed.
    """
    if end_time < start_time:
        raise ValueError("map leg times must be ordered")
    if minimum <= 0 or maximum < minimum:
        raise ValueError("map leg duration bounds are invalid")
    import math

    scaled = minimum + math.log1p(end_time - start_time) * 1.2
    return min(maximum, max(minimum, scaled))


def plan_map_legs(
    leg_bounds: Sequence[tuple[float, float]],
    fps: int,
    *,
    map_seconds: float | None = None,
) -> tuple[float, list[int]]:
    """Plan one global route-clock time-compression factor for every map leg.

    Marker motion is proportional to real speed everywhere a film shows the
    route: a stop stays a stop and a fast descent moves fast. ``factor`` is
    the ratio of total route seconds across ``leg_bounds`` to the total film
    seconds they get; every leg's film seconds is its own route seconds
    divided by that one factor. ``map_seconds`` overrides the total film
    seconds directly; left ``None``, the default total is the sum
    ``map_leg_seconds`` would have produced for the same legs, so the film's
    length does not change by default.

    Returns ``(factor, frame_counts)`` where ``frame_counts[i]`` is the frame
    count for ``leg_bounds[i]``, each at least 1. A zero-length leg
    (``start == end``) always gets exactly 1 frame and never divides by zero.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if map_seconds is not None and map_seconds <= 0:
        raise ValueError("map_seconds override must be positive")
    route_seconds = [max(0.0, end - start) for start, end in leg_bounds]
    total_route_seconds = sum(route_seconds)
    if map_seconds is not None:
        total_film_seconds = map_seconds
    else:
        total_film_seconds = sum(map_leg_seconds(start, end) for start, end in leg_bounds)
    factor = total_route_seconds / total_film_seconds if total_route_seconds > 0 else 1.0
    frame_counts = [
        max(1, round(route / factor * fps)) if route > 0 else 1 for route in route_seconds
    ]
    return factor, frame_counts


def _save_frame(
    image: Image.Image, frames: Path, number: int, *, watermark: str | None = None
) -> None:
    """Save a rendered frame, burning ``watermark`` on it first when given.

    The single choke point every composited frame passes through -- a map
    leg, a photo, the opening, the recap -- so the maker's mark never needs
    drawing at each of those call sites individually. A held frame (the
    settle before a photo, the video-arrival hold) reuses an already-saved,
    already-marked image and is never re-marked here.
    """
    if watermark:
        width, height = image.size
        draw_watermark(image, watermark, width, height)
    image.save(frames / f"{number:06d}.jpg", "JPEG", quality=95, subsampling=0)


# Every clip fed to ``_concat`` must carry an audio stream of identical
# params (codec, sample rate, channel layout) -- the concat demuxer walks
# streams positionally, so a clip missing audio desyncs everything after it.
# Map/photo legs have no source audio, so they get silence: padding to keep
# the timeline aligned, never generated sound content.
_AUDIO_CODEC_ARGS = ["-c:a", "aac", "-ar", "48000", "-ac", "2"]

# Every part of the film (frame sequences, the placed source clip, the recap)
# is encoded with this ONE video spec so ``_concat`` can stream-copy them:
# joining parts whose codec, profile, pixel format, colour tags or timescale
# differ is only valid by re-encoding. crf 8 is visually lossless -- the parts
# are scratch files, and ``encode_final`` is the film's one lossy delivery
# generation, not the third.
_PART_VIDEO_ARGS = [
    "-c:v",
    "libx264",
    "-crf",
    "8",
    "-preset",
    "veryfast",
    "-profile:v",
    "high",
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
    "-video_track_timescale",
    "15360",
]


def _encode_frames(frames: Path, output: Path, fps: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames / "%06d.jpg"),
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-shortest",
            *_PART_VIDEO_ARGS,
            "-r",
            str(fps),
            *_AUDIO_CODEC_ARGS,
            str(output),
        ],
        check=True,
        capture_output=True,
    )


# How much of a backdrop's brightness survives behind the picture laid over it.
_BACKDROP_DIM = 0.55


def _encode_source_video(
    source: Path,
    output: Path,
    fps: int,
    width: int,
    height: int,
    *,
    watermark: str | None = None,
) -> None:
    """Re-encode a clip in place, keeping its own audio (never generating sound content).

    ``watermark`` burns the same maker's mark ``_save_frame`` puts on every
    drawn frame: ``hud.draw_watermark`` onto a transparent card, overlaid, so
    the clip's mark is pixel-for-pixel the map's.
    """
    # A clip that does not fill the frame (a portrait phone clip) sits on a
    # blurred, dimmed copy of itself -- the same treatment a photo hold gets --
    # rather than between two flat bars. A clip that fills the frame covers it.
    fit = f"scale={width}:{height}:in_range=pc:out_range=tv:force_original_aspect_ratio"
    filtergraph = (
        f"split[backdrop_in][clip_in];"
        f"[backdrop_in]{fit}=increase,crop={width}:{height},"
        f"gblur=sigma={max(2.0, width / 60):g},"
        f"colorchannelmixer=rr={_BACKDROP_DIM}:gg={_BACKDROP_DIM}:bb={_BACKDROP_DIM}[backdrop];"
        f"[clip_in]{fit}=decrease[clip];"
        f"[backdrop][clip]overlay=(W-w)/2:(H-h)/2"
    )
    mark_inputs: list[str] = []
    mark_card: Path | None = None
    if watermark:
        mark_card = output.with_name(f".{output.stem}.watermark.png")
        card = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw_watermark(card, watermark, width, height)
        card.save(mark_card)
        mark_inputs = ["-i", str(mark_card)]
        filtergraph += "[framed];[framed][1:v]overlay=0:0"
    filtergraph += ",format=yuv420p"
    try:
        _run_source_encode(source, output, fps, filtergraph, mark_inputs)
    finally:
        if mark_card is not None:
            mark_card.unlink(missing_ok=True)


def _run_source_encode(
    source: Path, output: Path, fps: int, filtergraph: str, extra_inputs: list[str]
) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            *extra_inputs,
            "-filter_complex",
            filtergraph,
            "-map",
            "0:a?",
            *_PART_VIDEO_ARGS,
            "-r",
            str(fps),
            *_AUDIO_CODEC_ARGS,
            str(output),
        ],
        check=True,
        capture_output=True,
    )


def _video_evidence(path: Path, source_started_at) -> MediaEvidence:
    """Read optional creation time/GPS tags without treating either as trusted alone."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format_tags=creation_time,location",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    tags = json.loads(result.stdout).get("format", {}).get("tags", {})
    capture = tags.get("creation_time")
    captured_at_s = None
    if capture:
        from datetime import datetime

        captured_at_s = captured_at_route_seconds(
            datetime.fromisoformat(capture.replace("Z", "+00:00")), source_started_at
        )
    latitude = longitude = None
    match = re.fullmatch(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)/?", tags.get("location", ""))
    if match:
        latitude, longitude = float(match.group(1)), float(match.group(2))
    return MediaEvidence(path.name, captured_at_s, latitude, longitude)


def _concat(clips: Iterable[Path], output: Path) -> None:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", prefix=".route-story-concat-", dir=output.parent, delete=False
    ) as manifest:
        manifest.write("".join(f"file '{clip}'\n" for clip in clips))
        manifest.flush()
        manifest_path = Path(manifest.name)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(manifest_path),
                # Video is copied: every part shares _PART_VIDEO_ARGS. Audio is
                # re-encoded (cheap) so AAC priming never clicks at a join.
                "-c:v",
                "copy",
                *_AUDIO_CODEC_ARGS,
                str(output),
            ],
            check=True,
            capture_output=True,
        )
    finally:
        manifest_path.unlink(missing_ok=True)


def _interlude_subtitle(anchor: MediaAnchor | None) -> str:
    """What the card ahead of the clip tells the VIEWER.

    A clip whose own coordinates disagree with where its clock puts it on the
    track is, for the viewer, simply not placed on the map. Which of the two is
    wrong is a question for whoever runs the render (the log) and whoever reads
    the receipt (``placement: review_required``) -- never for the screen.
    """
    if anchor is not None and anchor.status != "coordinate_time_conflict":
        if anchor.track_time_s is not None:
            return "VIDEO ARRIVAL • ROUTE ANCHORED"
    return UNPLACED_VIDEO_SUBTITLE


def _video_backdrop(source: Path, workdir: Path, width: int, height: int) -> Image.Image | None:
    """The clip's first frame, filling the frame, blurred and dimmed like a photo hold's."""
    still = Path(workdir) / "interlude-backdrop.png"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-frames:v", "1", str(still)],
            check=True,
            capture_output=True,
        )
        with Image.open(still) as opened:
            frame = opened.convert("RGB")
    except (OSError, subprocess.CalledProcessError):
        logger.warning("no interlude backdrop from %s; using the flat card", Path(source).name)
        return None
    blurred = _fit_cover(frame, width, height).filter(
        ImageFilter.GaussianBlur(radius=max(8, width // 80))
    )
    return Image.eval(blurred, lambda value: int(value * _BACKDROP_DIM))


def _interlude_card(
    source: Path, anchor: MediaAnchor | None, workdir: Path, width: int, height: int
) -> Image.Image:
    if anchor is not None and anchor.status == "coordinate_time_conflict":
        logger.warning(
            "%s: its embedded coordinates disagree with where its clock puts it on the "
            "track -- shown unplaced; review the clip's GPS tag and clock",
            Path(source).name,
        )
    return _unlocated_video_card(
        width,
        height,
        _interlude_subtitle(anchor),
        backdrop=_video_backdrop(source, workdir, width, height),
    )
