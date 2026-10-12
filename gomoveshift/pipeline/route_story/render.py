"""Deterministic local Route Story map-frame rendering."""

import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from .frame_state import frame_state_at
from .metrics import RoutePoint
from .opening import SatelliteOpening
from .satellite import SatelliteRaster
from .timeline import StoryTimeline


def render_video(
    timeline: StoryTimeline,
    segments: Sequence[Sequence[RoutePoint]],
    output: Path,
    *,
    width: int = 1920,
    height: int = 1080,
    opening: SatelliteOpening | None = None,
    satellite: SatelliteRaster | None = None,
) -> None:
    """Encode every deterministic map frame into a local H.264 MP4."""
    with tempfile.TemporaryDirectory(prefix="route-story-") as temp:
        frames = Path(temp) / "frames"
        for frame in range(timeline.frame_count):
            render_map_frame(
                timeline,
                frame,
                segments,
                frames / f"{frame:06d}.png",
                width=width,
                height=height,
                opening=opening,
                satellite=satellite,
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(timeline.fps),
                "-i",
                str(frames / "%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output),
            ],
            check=True,
            capture_output=True,
        )


def render_map_frame(
    timeline: StoryTimeline,
    frame: int,
    segments: Sequence[Sequence[RoutePoint]],
    output: Path,
    *,
    width: int = 1920,
    height: int = 1080,
    opening: SatelliteOpening | None = None,
    satellite: SatelliteRaster | None = None,
) -> None:
    if width < 320 or height < 180:
        raise ValueError("frame is too small for readable Route Story controls")
    points = [point for segment in segments for point in segment]
    if not points:
        raise ValueError("Route Story needs at least one route point")
    image = Image.new("RGB", (width, height), "#10212a")
    if opening and satellite and opening.start_frame <= frame <= opening.end_frame:
        image = satellite.view(opening.pose_at_frame(frame), width, height)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    min_lat, max_lat = min(p.latitude for p in points), max(p.latitude for p in points)
    min_lon, max_lon = min(p.longitude for p in points), max(p.longitude for p in points)
    lat_span, lon_span = max(max_lat - min_lat, 0.001), max(max_lon - min_lon, 0.001)
    pad = 56

    def xy(point: RoutePoint):
        return (
            pad + (point.longitude - min_lon) / lon_span * (width - 2 * pad),
            height - pad - (point.latitude - min_lat) / lat_span * (height - 2 * pad),
        )

    for segment in segments:
        if len(segment) > 1:
            draw.line([xy(p) for p in segment], fill="#6b8490", width=5, joint="curve")
    state = frame_state_at(timeline, frame, segments)
    if state.data_available:
        for segment in segments:
            traveled = [point for point in segment if point.time_s <= state.recording_time_s]
            if len(traveled) > 1:
                draw.line([xy(point) for point in traveled], fill="#ffd166", width=6, joint="curve")
    if state.data_available:
        marker = RoutePoint(state.recording_time_s, state.latitude or 0, state.longitude or 0)
        x, y = xy(marker)
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill="#ffd166", outline="white", width=2)
    hud = (
        "GPS gap"
        if not state.data_available
        else f"{state.speed_mps or 0:.1f} m/s   {state.elevation_m if state.elevation_m is not None else '—'} m   {state.grade_percent if state.grade_percent is not None else '—'}%"
    )
    draw.rounded_rectangle((24, 20, min(width - 24, 470), 62), radius=12, fill="#071219")
    draw.text((40, 34), hud, fill="white", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "PNG")
