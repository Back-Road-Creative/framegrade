"""Terrain-aware camera clearance, shot-plan serialization, and film relief overlays.

Three jobs, all renderer-independent:

* :class:`TerrainSampler` reads a bundle's Terrain-RGB tiles back into metre heights, so the
  planner can ask "how high is the ground under this camera?" without a renderer.
* :func:`clear_terrain` rewrites a :class:`~gomoveshift.pipeline.route_story.planner.ShotPlan`
  so no frame puts the camera inside a hillside. MapLibre has no collision handling: a pose
  whose camera altitude falls below the terrain renders the inside of the mesh, which reads as
  a full-frame smear of dirt. Clearance is applied by pulling zoom out (the only altitude
  control a MapLibre pose has), through a Lipschitz lower envelope so the correction inherits
  the plan's own smoothness limits instead of stepping frame to frame.
* :class:`TerrainGrid` builds a small, deterministic relief overlay from an approved 3DEP
  sample grid for the route story film's satellite view.

:func:`plan_to_json` is the handoff the scene capture harness consumes: one pose per frame,
in MapLibre units, with the shot boundaries kept for debugging.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .bundle import TILE_SIZE, decode_terrain_rgb, lonlat_to_tile
from .opening import CameraPose
from .planner import DEFAULT_MOTION_LIMITS, Shot, ShotPlan

PLAN_SCHEMA = "route-story-shot-plan/1"

#: MapLibre's default vertical field of view, in degrees.
DEFAULT_FOV_DEG = 36.87

#: Web-Mercator metres per pixel at zoom 0 on the equator, for 256px tiles.
_EQUATOR_M_PER_PX_Z0 = 156543.03392804097

_EARTH_RADIUS_M = 6_371_008.8


# ---------------------------------------------------------------------------
# Terrain sampling
# ---------------------------------------------------------------------------


@dataclass
class TerrainSampler:
    """Bilinear elevation lookups against a bundle's Terrain-RGB tile pyramid."""

    root: Path
    zoom: int
    exaggeration: float = 1.0

    def __post_init__(self) -> None:
        if self.exaggeration <= 0:
            raise ValueError("exaggeration must be positive")
        self._tiles: dict[tuple[int, int], np.ndarray] = {}

    @classmethod
    def from_bundle(cls, root: Path, *, exaggeration: float | None = None) -> TerrainSampler:
        """Build a sampler from ``<root>/manifest.json``, using its highest terrain zoom.

        The manifest's own ``terrain.exaggeration`` is used unless the caller overrides it, so
        clearance is measured against the heights the scene actually renders.
        """
        root = Path(root)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"bundle has no manifest.json: {root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        terrain = manifest.get("terrain")
        if not isinstance(terrain, dict) or "maxzoom" not in terrain:
            raise ValueError(f"bundle manifest declares no terrain: {manifest_path}")
        if exaggeration is None:
            exaggeration = float(terrain.get("exaggeration", 1.0))
        return cls(root=root, zoom=int(terrain["maxzoom"]), exaggeration=float(exaggeration))

    def _tile(self, x: int, y: int) -> np.ndarray:
        cached = self._tiles.get((x, y))
        if cached is not None:
            return cached
        path = self.root / "terrain" / str(self.zoom) / str(x) / f"{y}.png"
        if not path.is_file():
            raise ValueError(f"no terrain tile for {self.zoom}/{x}/{y}: {path}")
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        heights = decode_terrain_rgb(rgb)
        self._tiles[(x, y)] = heights
        return heights

    def elevation_at(self, longitude: float, latitude: float) -> float:
        """Return the exaggerated ground height in metres under a lon/lat.

        Raises ``ValueError`` when the point falls outside the bundled tiles -- a caller that
        can carry on without an answer (clearance does) catches it rather than guessing a
        height, because a guessed zero would fly a camera straight into a real mountain.
        """
        tile_x, tile_y = lonlat_to_tile(longitude, latitude, self.zoom)
        heights = self._tile(int(math.floor(tile_x)), int(math.floor(tile_y)))
        # Pixel centres sit half a pixel in from the tile edge.
        col = (tile_x - math.floor(tile_x)) * TILE_SIZE - 0.5
        row = (tile_y - math.floor(tile_y)) * TILE_SIZE - 0.5
        col = min(max(col, 0.0), TILE_SIZE - 1.0)
        row = min(max(row, 0.0), TILE_SIZE - 1.0)
        col0, row0 = int(math.floor(col)), int(math.floor(row))
        col1 = min(col0 + 1, TILE_SIZE - 1)
        row1 = min(row0 + 1, TILE_SIZE - 1)
        fx, fy = col - col0, row - row0
        top = heights[row0, col0] * (1.0 - fx) + heights[row0, col1] * fx
        bottom = heights[row1, col0] * (1.0 - fx) + heights[row1, col1] * fx
        return float((top * (1.0 - fy) + bottom * fy) * self.exaggeration)


# ---------------------------------------------------------------------------
# Camera geometry
# ---------------------------------------------------------------------------


def metres_per_pixel(latitude: float, zoom: float) -> float:
    """Web-Mercator ground resolution at a latitude and zoom."""
    return _EQUATOR_M_PER_PX_Z0 * math.cos(math.radians(latitude)) / (2.0**zoom)


def camera_distance_m(
    pose: CameraPose, *, viewport_height_px: int, fov_deg: float = DEFAULT_FOV_DEG
) -> float:
    """Distance from the camera to the pose's centre point, in metres."""
    half_height = viewport_height_px / 2.0
    return (
        half_height
        * metres_per_pixel(pose.latitude, pose.zoom)
        / math.tan(math.radians(fov_deg) / 2.0)
    )


def camera_altitude_m(
    pose: CameraPose, *, viewport_height_px: int, fov_deg: float = DEFAULT_FOV_DEG
) -> float:
    """Height of the camera above the map plane, in metres."""
    distance = camera_distance_m(pose, viewport_height_px=viewport_height_px, fov_deg=fov_deg)
    return distance * math.cos(math.radians(pose.pitch))


def camera_ground_point(
    pose: CameraPose, *, viewport_height_px: int, fov_deg: float = DEFAULT_FOV_DEG
) -> tuple[float, float]:
    """The lon/lat the camera itself stands over -- behind the centre when pitched."""
    distance = camera_distance_m(pose, viewport_height_px=viewport_height_px, fov_deg=fov_deg)
    back_m = distance * math.sin(math.radians(pose.pitch))
    if back_m == 0.0:
        return pose.longitude, pose.latitude
    bearing = math.radians((pose.bearing + 180.0) % 360.0)
    lat_rad = math.radians(pose.latitude)
    delta_lat = back_m * math.cos(bearing) / _EARTH_RADIUS_M
    delta_lon = back_m * math.sin(bearing) / (_EARTH_RADIUS_M * math.cos(lat_rad))
    return pose.longitude + math.degrees(delta_lon), pose.latitude + math.degrees(delta_lat)


def zoom_for_altitude(
    pose: CameraPose,
    altitude_m: float,
    *,
    viewport_height_px: int,
    fov_deg: float = DEFAULT_FOV_DEG,
) -> float:
    """The zoom that puts this pose's camera at ``altitude_m`` above the map plane."""
    if altitude_m <= 0:
        raise ValueError("altitude_m must be positive")
    pitch_cos = math.cos(math.radians(pose.pitch))
    distance = altitude_m / pitch_cos if pitch_cos > 1e-9 else altitude_m
    numerator = (
        (viewport_height_px / 2.0)
        * _EQUATOR_M_PER_PX_Z0
        * math.cos(math.radians(pose.latitude))
        / math.tan(math.radians(fov_deg) / 2.0)
    )
    return math.log2(numerator / distance)


# ---------------------------------------------------------------------------
# Clearance
# ---------------------------------------------------------------------------


def _lipschitz_lower_envelope(values: np.ndarray, rate: float) -> np.ndarray:
    """The greatest function <= ``values`` whose frame-to-frame step is at most ``rate``."""
    out = values.astype(np.float64).copy()
    for index in range(1, out.size):
        out[index] = min(out[index], out[index - 1] + rate)
    for index in range(out.size - 2, -1, -1):
        out[index] = min(out[index], out[index + 1] + rate)
    return out


def clear_terrain(
    plan: ShotPlan,
    sampler: TerrainSampler,
    *,
    fps: float,
    viewport_height_px: int,
    margin_m: float = 150.0,
    min_zoom: float = 0.0,
    fov_deg: float = DEFAULT_FOV_DEG,
    zoom_velocity_per_s: float | None = None,
) -> ShotPlan:
    """Return a plan whose every frame keeps ``margin_m`` of air under the camera.

    Only zoom changes: longitude, latitude, pitch and bearing are the story, and moving them
    to dodge a hill would rewrite the shot. Frames the bundle has no terrain for keep their
    planned pose -- a missing tile is not evidence of flat ground, and the fetch layer's own
    coverage check is what owns that failure.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if margin_m < 0:
        raise ValueError("margin_m must not be negative")
    frame_count = plan.frame_count
    planned = [plan.pose_at(frame) for frame in range(frame_count)]
    caps = np.empty(frame_count, dtype=np.float64)
    for index, pose in enumerate(planned):
        caps[index] = pose.zoom
        try:
            longitude, latitude = camera_ground_point(
                pose, viewport_height_px=viewport_height_px, fov_deg=fov_deg
            )
            ground = sampler.elevation_at(longitude, latitude)
        except ValueError:
            continue
        needed = ground + margin_m
        if needed <= 0:
            continue
        if (
            camera_altitude_m(pose, viewport_height_px=viewport_height_px, fov_deg=fov_deg)
            >= needed
        ):
            continue
        caps[index] = min(
            pose.zoom,
            zoom_for_altitude(pose, needed, viewport_height_px=viewport_height_px, fov_deg=fov_deg),
        )

    if zoom_velocity_per_s is None:
        zoom_velocity_per_s = DEFAULT_MOTION_LIMITS["zoom_velocity_per_s"]
    zooms = _lipschitz_lower_envelope(caps, zoom_velocity_per_s / fps)
    zooms = np.maximum(zooms, min_zoom)

    poses = [
        pose
        if math.isclose(zoom, pose.zoom, rel_tol=0.0, abs_tol=1e-12)
        else CameraPose(pose.longitude, pose.latitude, float(zoom), pose.pitch, pose.bearing)
        for pose, zoom in zip(planned, zooms)
    ]

    def pose_fn_for(shot: Shot):
        def pose_at(frame: int) -> CameraPose:
            return poses[min(frame, frame_count - 1)]

        return pose_at

    return ShotPlan(
        tuple(
            Shot(shot.kind, shot.start_frame, shot.end_frame, pose_fn_for(shot))
            for shot in plan.shots
        )
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def plan_to_json(plan: ShotPlan, *, fps: float) -> dict:
    """Serialize a shot plan as one pose per frame, in MapLibre units."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    poses = []
    for frame in range(plan.frame_count):
        pose = plan.pose_at(frame)
        poses.append(
            {
                "frame": frame,
                "longitude": pose.longitude,
                "latitude": pose.latitude,
                "zoom": pose.zoom,
                "pitch": pose.pitch,
                "bearing": pose.bearing,
            }
        )
    return {
        "schema": PLAN_SCHEMA,
        "fps": float(fps),
        "frame_count": plan.frame_count,
        "shots": [
            {"kind": shot.kind, "start_frame": shot.start_frame, "end_frame": shot.end_frame}
            for shot in plan.shots
        ],
        "poses": poses,
    }


def write_plan_json(plan: ShotPlan, *, fps: float, path: Path) -> Path:
    """Write :func:`plan_to_json` to ``path`` (temp-then-rename) and return it."""
    payload = plan_to_json(plan, fps=fps)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path


# ---------------------------------------------------------------------------
# Film relief overlays
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TerrainGrid:
    west: float
    south: float
    east: float
    north: float
    elevations_m: np.ndarray
    _shade: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        gradient_y, gradient_x = np.gradient(self.elevations_m)
        slope = np.pi / 2.0 - np.arctan(np.hypot(gradient_x, gradient_y))
        aspect = np.arctan2(-gradient_x, gradient_y)
        shade = np.sin(np.deg2rad(45.0)) * np.sin(slope) + np.cos(np.deg2rad(45.0)) * np.cos(
            slope
        ) * np.cos(np.deg2rad(315.0) - aspect)
        normalized = ((shade - shade.min()) / max(1e-6, shade.max() - shade.min()) * 255).astype(
            np.uint8
        )
        object.__setattr__(self, "_shade", normalized)

    @classmethod
    def from_json(cls, source: Path) -> "TerrainGrid":
        samples = json.loads(Path(source).read_text(encoding="utf-8"))["samples"]
        if not samples:
            raise ValueError("terrain sample grid is empty")
        xs = sorted({float(item["location"]["x"]) for item in samples})
        ys = sorted({float(item["location"]["y"]) for item in samples})
        values = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
        x_index = {value: index for index, value in enumerate(xs)}
        y_index = {value: index for index, value in enumerate(ys)}
        for item in samples:
            location = item["location"]
            values[y_index[float(location["y"])], x_index[float(location["x"])]] = float(
                item["value"]
            )
        if np.isnan(values).any():
            raise ValueError("terrain sample grid has holes")
        return cls(xs[0], ys[0], xs[-1], ys[-1], values)

    def overlay(
        self, west: float, south: float, east: float, north: float, width: int, height: int
    ) -> Image.Image:
        """Return a transparent, real-elevation hillshade for a map viewport."""
        base = Image.fromarray(self._shade, mode="L").resize(
            (width, height), Image.Resampling.BICUBIC
        )
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        pixel = np.asarray(base)
        # Neutral, low-opacity relief keeps the source satellite colors intact.
        # Strong chroma here reads as a camouflage texture rather than terrain.
        rgba[:, :, :3] = np.where(pixel[:, :, None] > 128, 235, 18)
        rgba[:, :, 3] = np.abs(pixel.astype(np.int16) - 128).clip(0, 28).astype(np.uint8)
        return Image.fromarray(rgba, mode="RGBA")
