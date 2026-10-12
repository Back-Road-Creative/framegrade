"""Journey loading and filtering utilities."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from gps_journey.core.gpx_parser import GPXParser
from gps_journey.core.models import Journey, Waypoint, local_day

from ...analysis.shorts_models import GPX_MEDIA_S_TAG

logger = logging.getLogger(__name__)


def load_and_filter_journey(
    gpx_path: Path,
    gps_date: Optional[str],
    gps_range: Optional[str],
    _mock_journey: Optional[Journey],
    gpx_parser: Optional[GPXParser] = None,
) -> Optional[Journey]:
    """Parse GPX file and apply date/range filters.

    Returns the filtered Journey, or None if parsing fails or yields
    no usable waypoints (caller should fall back to basic chapters).
    """
    return load_journey_and_origin(gpx_path, gps_date, gps_range, _mock_journey, gpx_parser)[0]


def _read_stamped_journey(gpx_path: Path) -> Optional[tuple[Journey, datetime]]:
    """Footage-clock journey from a GPX whose fixes all carry gms_media_s.

    Returns None when NO point is stamped (caller uses the wall-clock parser).
    Raises ValueError when only SOME points are stamped (mixed clocks).
    """
    import gpxpy

    with open(gpx_path, encoding="utf-8") as fh:
        gpx = gpxpy.parse(fh)
    points = [pt for trk in gpx.tracks for seg in trk.segments for pt in seg.points if pt.time]
    stamped = []
    for pt in points:
        media_s = None
        for ext in pt.extensions:
            if str(ext.tag).rsplit("}", 1)[-1] == GPX_MEDIA_S_TAG and ext.text:
                media_s = float(ext.text)
                break
        stamped.append((media_s, pt))
    n_stamped = sum(1 for m, _ in stamped if m is not None)
    if n_stamped == 0:
        return None
    if n_stamped != len(stamped):
        raise ValueError(
            f"GPX {gpx_path.name} is partially stamped with {GPX_MEDIA_S_TAG} "
            f"({n_stamped}/{len(stamped)} points)"
        )
    stamped.sort(key=lambda mp: mp[0])
    first_media, first_pt = stamped[0]
    w0 = first_pt.time - timedelta(seconds=first_media)
    wps = tuple(
        Waypoint(
            latitude=pt.latitude,
            longitude=pt.longitude,
            timestamp=w0 + timedelta(seconds=m),
            elevation=pt.elevation,
        )
        for m, pt in stamped
    )
    return Journey(start_date=wps[0].timestamp, end_date=wps[-1].timestamp, waypoints=wps), w0


def load_journey_and_origin(
    gpx_path: Path,
    gps_date: Optional[str],
    gps_range: Optional[str],
    _mock_journey: Optional[Journey],
    gpx_parser: Optional[GPXParser] = None,
) -> tuple[Optional[Journey], Optional[datetime]]:
    """Like load_and_filter_journey but also returns the footage origin.

    When every trackpoint carries gms_media_s the journey is re-timed onto
    the footage clock and the origin is the wall-clock instant of footage
    second 0; otherwise the origin is None and the wall-clock path is used.
    """
    origin: Optional[datetime] = None
    try:
        if _mock_journey:
            journey = _mock_journey
        elif gpx_path.exists():
            logger.info(f"Parsing GPS file: {gpx_path}")
            try:
                stamped = _read_stamped_journey(gpx_path)
            except ValueError as e:
                logger.error(f"Failed to parse GPS file: {e}")
                return None, None
            if stamped is not None:
                journey, origin = stamped
            else:
                if gpx_parser is None:
                    gpx_parser = GPXParser()
                journey = gpx_parser.parse(gpx_path)
        else:
            logger.warning(f"GPS file not found: {gpx_path}")
            return None, None
    except Exception as e:
        logger.error(f"Failed to parse GPS file: {e}")
        return None, None

    # Filter to specific date if gps_date provided
    if gps_date is not None and journey.waypoints:
        journey = _filter_by_date(journey, gps_date)

    # Apply GPS range filter if specified
    if gps_range is not None:
        journey = apply_gps_range(journey, gps_range)
        if not journey.waypoints:
            logger.warning(f"No waypoints in GPS range '{gps_range}'")
            return None, None

    return journey, origin


def _filter_by_date(journey: Journey, gps_date: str) -> Journey:
    """Filter journey waypoints to a specific LOCAL calendar day.

    Waypoints are bucketed by longitude-local day (see ``local_day``) so a
    drive crossing UTC midnight isn't truncated at the wrong boundary.

    Hard-fails (``ValueError``) on a malformed date OR a zero-match instead
    of silently returning the whole, possibly multi-day journey: on
    camera-clock skew the unfiltered fallback samples OTHER days' waypoints,
    corrupting chapters and the end location. Raising here surfaces the
    mismatch upstream of any encode/upload so the operator can supply the
    correct ``--gps-date`` or fix the camera clock.
    """
    try:
        target_date = datetime.strptime(gps_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Invalid gps_date format '{gps_date}', expected YYYY-MM-DD") from exc
    filtered_wps = [
        wp for wp in journey.waypoints if local_day(wp.timestamp, wp.longitude) == target_date
    ]
    if not filtered_wps:
        raise ValueError(
            f"gps_date '{gps_date}' matched no waypoints "
            f"(track spans {journey.start_date.date()} to {journey.end_date.date()}); "
            f"supply --gps-date for the correct local day or fix the camera clock"
        )
    logger.info(
        f"GPS date filter '{gps_date}': {len(filtered_wps)}/{len(journey.waypoints)} waypoints"
    )
    return Journey(
        start_date=filtered_wps[0].timestamp,
        end_date=filtered_wps[-1].timestamp,
        waypoints=tuple(filtered_wps),
        statistics=journey.statistics,
    )


def apply_gps_range(journey: Journey, gps_range: str) -> Journey:
    """Slice journey waypoints to a specific time window.

    Hard-fails (``ValueError``) on a malformed ``gps_range`` — one that doesn't
    split into exactly ``START-END`` or whose bounds aren't numeric — mirroring
    ``_filter_by_date``: the operator explicitly asked for a slice, so silently
    proceeding with the WHOLE journey would burn the wrong GPS bounds. A
    valid-but-zero-match range keeps the warn-and-return-empty behaviour (the
    caller then falls back to basic chapters).

    Args:
        journey: Full journey parsed from GPX
        gps_range: Time range in minutes as "START-END" (e.g. "0-60", "60-120")

    Returns:
        New Journey with only the waypoints within the specified range
    """
    parts = gps_range.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid gps_range format '{gps_range}', expected 'START-END'")
    try:
        start_min = float(parts[0])
        end_min = float(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"Invalid gps_range values '{gps_range}', expected numeric START-END"
        ) from exc

    range_start = journey.start_date + timedelta(minutes=start_min)
    range_end = journey.start_date + timedelta(minutes=end_min)
    filtered = [wp for wp in journey.waypoints if range_start <= wp.timestamp < range_end]

    if not filtered:
        logger.warning(
            f"GPS range {gps_range} ({range_start} to {range_end}) "
            f"yielded no waypoints from {len(journey.waypoints)} total"
        )
        return Journey(
            start_date=range_start,
            end_date=range_end,
            waypoints=(),
            statistics=journey.statistics,
        )

    logger.info(
        f"GPS range '{gps_range}': {len(filtered)}/{len(journey.waypoints)} waypoints, "
        f"{start_min:.0f}-{end_min:.0f} min"
    )
    return Journey(
        start_date=filtered[0].timestamp,
        end_date=filtered[-1].timestamp,
        waypoints=tuple(filtered),
        statistics=journey.statistics,
    )
