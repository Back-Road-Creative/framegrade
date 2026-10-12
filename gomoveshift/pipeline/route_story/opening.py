"""Deterministic camera poses for a Route Story satellite opening.

This is deliberately independent of the map renderer. Preview and final export
consume the same frame-addressable poses while imagery remains a local,
versioned asset selected by the project recipe.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraPose:
    """A north-up geographic camera pose, expressed in MapLibre units."""

    longitude: float
    latitude: float
    zoom: float
    pitch: float = 0.0
    bearing: float = 0.0

    def __post_init__(self) -> None:
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError("longitude must be in [-180, 180]")
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError("latitude must be in [-90, 90]")
        if self.zoom < 0:
            raise ValueError("zoom must be non-negative")


def _smootherstep(value: float) -> float:
    """A zero-velocity and zero-acceleration endpoint easing function."""
    return value * value * value * (value * (value * 6 - 15) + 10)


def _shortest_longitude_delta(start: float, end: float) -> float:
    delta = (end - start + 180.0) % 360.0 - 180.0
    return 180.0 if delta == -180.0 else delta


def _national_pose(latitude: float, longitude: float) -> CameraPose:
    """Choose a truthful broad-USA view without moving Alaska or Hawaii insets."""
    if 51.0 <= latitude <= 72.0 and -180.0 <= longitude <= -130.0:
        return CameraPose(-124.0, 46.0, 2.2)
    if 18.0 <= latitude <= 24.0 and -162.0 <= longitude <= -153.0:
        return CameraPose(-125.0, 33.0, 2.2)
    return CameraPose(-98.5795, 39.8283, 3.35)


@dataclass(frozen=True)
class SatelliteOpening:
    """A complete, deterministic U.S.-to-route camera move."""

    start_frame: int
    end_frame: int
    start: CameraPose
    end: CameraPose

    @classmethod
    def for_route_start(
        cls,
        *,
        latitude: float,
        longitude: float,
        start_frame: int,
        end_frame: int,
        final_zoom: float,
    ) -> SatelliteOpening:
        """Build an overview whose endpoint is the approved visible-route start."""
        end = CameraPose(longitude, latitude, final_zoom)
        if start_frame < 0:
            raise ValueError("opening start frame must be non-negative")
        if end_frame <= start_frame:
            raise ValueError("opening frame range must be positive")
        return cls(start_frame, end_frame, _national_pose(latitude, longitude), end)

    def pose_at_frame(self, frame: int) -> CameraPose:
        """Resolve a camera pose at an opening boundary, including its endpoint."""
        if not self.start_frame <= frame <= self.end_frame:
            raise ValueError("frame is outside the satellite opening")
        if frame == self.start_frame:
            return self.start
        if frame == self.end_frame:
            return self.end
        fraction = (frame - self.start_frame) / (self.end_frame - self.start_frame)
        eased = _smootherstep(fraction)
        longitude = (
            self.start.longitude
            + _shortest_longitude_delta(self.start.longitude, self.end.longitude) * eased
        )
        if longitude > 180.0:
            longitude -= 360.0
        if longitude < -180.0:
            longitude += 360.0
        return CameraPose(
            longitude,
            self.start.latitude + (self.end.latitude - self.start.latitude) * eased,
            self.start.zoom + (self.end.zoom - self.start.zoom) * eased,
            self.start.pitch + (self.end.pitch - self.start.pitch) * eased,
            self.start.bearing + (self.end.bearing - self.start.bearing) * eased,
        )
