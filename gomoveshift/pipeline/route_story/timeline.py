"""A reproducible mapping from presentation frames to recording time.

The map renderer may move quickly while the HUD remains grounded in the source
recording.  This small pure model is the shared contract for preview and final
render workers; neither is allowed to derive route time from browser wall time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

SceneKind = Literal[
    "overview",
    "route_follow",
    "approach",
    "photo_hold",
    "source_video",
    "grouped_stop",
    "gap_transition",
    "day_transition",
    "recap",
]

_FROZEN_KINDS = {
    "overview",
    "photo_hold",
    "grouped_stop",
    "gap_transition",
    "day_transition",
    "recap",
}


@dataclass(frozen=True)
class Scene:
    """One half-open presentation frame range and its recording-time span."""

    kind: SceneKind
    start_frame: int
    end_frame: int
    route_start_s: float
    route_end_s: float

    def __post_init__(self) -> None:
        if self.start_frame < 0:
            raise ValueError("scene start frame must be non-negative")
        if self.end_frame <= self.start_frame:
            raise ValueError("scene frame range must be positive")
        if self.route_end_s < self.route_start_s:
            raise ValueError("recording time cannot run backward")
        if self.kind in _FROZEN_KINDS and self.route_end_s != self.route_start_s:
            raise ValueError(f"{self.kind} must freeze recording time")


@dataclass(frozen=True)
class StoryTimeline:
    """Validated scenes that deterministically map every output frame.

    Scene boundaries are half-open: ``end_frame`` is the first frame of the
    next scene.  Querying that final boundary returns the last recording time,
    which makes recap metadata and FFmpeg duration checks unambiguous without
    assigning one output frame to two scenes.
    """

    fps: int
    scenes: Sequence[Scene]

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if not self.scenes:
            raise ValueError("timeline needs at least one scene")
        if self.scenes[0].start_frame != 0:
            raise ValueError("first scene must begin at frame zero")
        for previous, current in zip(self.scenes, self.scenes[1:]):
            if previous.end_frame != current.start_frame:
                raise ValueError("scenes must be contiguous without overlap or frame holes")
            if previous.route_end_s != current.route_start_s:
                raise ValueError("scenes must preserve recording-time continuity")

    @property
    def frame_count(self) -> int:
        return self.scenes[-1].end_frame

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps

    def route_time_at_frame(self, frame: int) -> float:
        """Return original recording time for an output-frame boundary."""
        if frame < 0 or frame > self.frame_count:
            raise ValueError("frame is outside the storyboard")
        if frame == self.frame_count:
            return self.scenes[-1].route_end_s
        for scene in self.scenes:
            if scene.start_frame <= frame < scene.end_frame:
                fraction = (frame - scene.start_frame) / (scene.end_frame - scene.start_frame)
                return scene.route_start_s + (scene.route_end_s - scene.route_start_s) * fraction
        raise AssertionError("validated timeline omitted a frame")
