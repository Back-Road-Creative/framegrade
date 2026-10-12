"""GPS-based scenic scoring for video segments."""

import logging
import math

from .shorts_models import GPSData, _haversine_m

logger = logging.getLogger(__name__)

# Minimum ground distance (metres) between two GPS fixes for the bearing between
# them to count as a real heading change. Fixes closer than this are within the
# consumer-GPS noise floor, so their bearing is jitter — see get_turn_intensity.
_MIN_MOVE_M = 5.0


class ScenicScorer:
    """
    Scores GPS segments for scenic interest.

    Scenic factors:
    - Elevation gain rate (climbing = interesting)
    - High absolute elevation (mountaintops)
    - Turn intensity (winding roads, hairpins)
    """

    def __init__(self, gps_data: GPSData):
        self.gps_data = gps_data

    def get_elevation_gain_rate(self, start_time: float, end_time: float) -> float:
        """
        Calculate elevation gain rate in meters per minute.

        Args:
            start_time: Segment start (GPS timestamp)
            end_time: Segment end (GPS timestamp)

        Returns:
            Elevation gain rate (m/min), only counting uphill
        """
        points = self.gps_data.get_points_in_range(start_time, end_time)

        if len(points) < 2:
            return 0.0

        # Calculate total elevation gain (only count uphill)
        total_gain = 0.0
        for i in range(1, len(points)):
            if points[i].elevation is not None and points[i - 1].elevation is not None:
                diff = points[i].elevation - points[i - 1].elevation
                if diff > 0:
                    total_gain += diff

        # Duration in minutes
        duration_min = (end_time - start_time) / 60.0
        if duration_min <= 0:
            return 0.0

        return total_gain / duration_min

    def get_turn_intensity(self, start_time: float, end_time: float) -> float:
        """
        Calculate turn intensity (how winding the road is).

        Uses bearing changes between consecutive points.
        Higher value = more turns/curves.

        Args:
            start_time: Segment start (GPS timestamp)
            end_time: Segment end (GPS timestamp)

        Returns:
            Turn intensity score (0-100)
        """
        points = self.gps_data.get_points_in_range(start_time, end_time)

        if len(points) < 3:
            return 0.0

        # Drop near-stationary fixes before measuring bearings. A slow crawl or a
        # parked idle emits many fixes that barely move, and the bearing between
        # two sub-noise-floor points is GPS jitter, not a real turn — counting it
        # inflates turn intensity and mislabels stopped/creeping footage as
        # "winding" (also skewing scenic segment selection). Keep only the
        # subsequence that actually advanced >= _MIN_MOVE_M from the last kept
        # fix, then measure heading change along that real path.
        moved = [points[0]]
        for p in points[1:]:
            if _haversine_m(moved[-1].lat, moved[-1].lon, p.lat, p.lon) >= _MIN_MOVE_M:
                moved.append(p)

        if len(moved) < 3:
            return 0.0

        # Calculate bearing changes
        bearings = []
        for i in range(1, len(moved)):
            lat1 = math.radians(moved[i - 1].lat)
            lon1 = math.radians(moved[i - 1].lon)
            lat2 = math.radians(moved[i].lat)
            lon2 = math.radians(moved[i].lon)

            dlon = lon2 - lon1
            x = math.sin(dlon) * math.cos(lat2)
            y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
            bearing = math.degrees(math.atan2(x, y))
            bearings.append(bearing)

        if len(bearings) < 2:
            return 0.0

        # Calculate total absolute bearing change
        total_change = 0.0
        for i in range(1, len(bearings)):
            diff = abs(bearings[i] - bearings[i - 1])
            # Handle wrap-around (e.g., 350° to 10°)
            if diff > 180:
                diff = 360 - diff
            total_change += diff

        # Normalize to per-minute rate
        duration_min = (end_time - start_time) / 60.0
        if duration_min <= 0:
            return 0.0

        change_per_min = total_change / duration_min

        # Scale to 0-100 (assume 200°/min = 100)
        return min(change_per_min / 2.0, 100.0)

    def get_speed_variance(self, start_time: float, end_time: float) -> float:
        """Calculate GPS speed standard deviation over a time window.

        High variance = curves, towns, interesting terrain.
        Low variance = constant highway speed = boring.

        Args:
            start_time: Segment start (GPS timestamp).
            end_time: Segment end (GPS timestamp).

        Returns:
            Speed standard deviation in km/h.
        """
        points = self.gps_data.get_points_in_range(start_time, end_time)

        if len(points) < 3:
            return 0.0

        speeds = []
        for i in range(1, len(points)):
            dt = points[i].timestamp - points[i - 1].timestamp
            if dt <= 0:
                continue
            dist_m = _haversine_m(
                points[i - 1].lat,
                points[i - 1].lon,
                points[i].lat,
                points[i].lon,
            )
            speed_kmh = (dist_m / dt) * 3.6
            speeds.append(speed_kmh)

        if len(speeds) < 2:
            return 0.0

        import numpy as np

        return float(np.std(speeds))

    def set_poi_timestamps(self, poi_timestamps: list[float]) -> None:
        """Set POI timestamps (in GPS time) for proximity scoring."""
        self._poi_timestamps = poi_timestamps

    def score_segment(self, start_time: float, end_time: float) -> float:
        """
        Score a segment for scenic interest.

        Args:
            start_time: Segment start (GPS timestamp)
            end_time: Segment end (GPS timestamp)

        Returns:
            Scenic score (0-100)
        """
        points = self.gps_data.get_points_in_range(start_time, end_time)

        if len(points) < 2:
            return 50.0  # Neutral score if no data

        # Elevation gain rate score (40m/min = 100)
        gain_rate = self.get_elevation_gain_rate(start_time, end_time)
        elevation_score = min(gain_rate / 0.4, 100.0)

        # Turn intensity score
        turn_score = self.get_turn_intensity(start_time, end_time)

        # Absolute elevation score (higher = better, up to 500m)
        elevations = [p.elevation for p in points if p.elevation is not None]
        if elevations:
            avg_elevation = sum(elevations) / len(elevations)
            elevation_height_score = min(avg_elevation / 5.0, 100.0)
        else:
            elevation_height_score = 50.0

        # POI proximity bonus: +25 if a POI timestamp falls within this segment
        poi_bonus = 0.0
        poi_timestamps = getattr(self, "_poi_timestamps", None)
        if poi_timestamps:
            if any(start_time <= pt <= end_time for pt in poi_timestamps):
                poi_bonus = 25.0

        # Normalize POI bonus to 0-100 scale
        poi_bonus_normalized = min(poi_bonus * 4.0, 100.0)

        # Weighted combination
        scenic_score = (
            0.35 * elevation_score
            + 0.30 * turn_score
            + 0.20 * elevation_height_score
            + 0.15 * poi_bonus_normalized
        )

        return scenic_score
