"""GPS-to-video time mapping for edited videos with stop removal."""

from dataclasses import dataclass
from typing import Optional


def map_from_edited_time(
    edited_time: float,
    keep_segments: Optional[list[tuple[float, float]]],
    crossfade: float = 0.0,
) -> float:
    """Inverse of _map_to_edited_time: edited video time -> original video time.

    When stop removal stitched the kept segments with a ``crossfade``-second
    dissolve at each seam (``stop_detection.crossfade_duration > 0``), every seam
    overlaps by ``crossfade``, so each non-final segment advances the EDITED clock
    by only ``duration - crossfade``. Without this correction the mapping drifts
    later and later by ``crossfade`` per crossed seam (the edited timeline is
    ``Σ durations - (N-1)*crossfade``, not ``Σ``). ``crossfade=0`` (hard cuts)
    reproduces the original behavior exactly.
    """
    if not keep_segments:
        return edited_time
    if edited_time < 0:
        return keep_segments[0][0] + edited_time
    # Imported here: gomoveshift.pipeline's package init imports gomoveshift.analysis.
    from gomoveshift.pipeline.edit_timeline import planned_timeline

    # Inside a seam overlap the edited clock already belongs to the incoming
    # segment (the last position), which is what this mapper has always returned.
    positions = planned_timeline(keep_segments, crossfade).output_to_source(edited_time)
    if positions:
        return positions[-1].time
    return keep_segments[-1][1]  # Past all segments (or exactly at end)


@dataclass
class VideoToGpsTimeMapper:
    """Maps edited video timestamps to GPS timestamps.

    Handles the chain: edited video time -> original video time -> GPS time.
    """

    gps_offset: float = 0.0
    keep_segments: Optional[list[tuple[float, float]]] = None
    gps_duration: float = 0.0
    original_duration: float = 0.0
    crossfade_duration: float = 0.0  # seam overlap baked in by stop removal
    # Stamped GPX: GPS seconds ARE footage seconds (scale exactly 1.0, origin =
    # footage second 0). Only the stop-removal inversion applies.
    footage_clock: bool = False

    def video_to_gps_time(self, video_time: float) -> float:
        """Convert edited video timestamp to GPS timestamp using piecewise mapping."""
        if self.footage_clock:
            if self.keep_segments:
                video_time = map_from_edited_time(
                    video_time, self.keep_segments, self.crossfade_duration
                )
            return video_time + self.gps_offset
        if self.keep_segments and self.gps_duration > 0 and self.original_duration > 0:
            # Step 1: edited video time -> original video time (invert stop removal)
            original_time = map_from_edited_time(
                video_time, self.keep_segments, self.crossfade_duration
            )
            # Step 2: original video time -> GPS time (linear scale)
            gps_time = original_time * (self.gps_duration / self.original_duration)
            # Step 3: add GPS start offset
            return gps_time + self.gps_offset
        # Fallback: simple linear offset (backward compatible)
        return video_time + self.gps_offset
