"""GPX file parsing for shorts extraction."""

import logging
from pathlib import Path

from .shorts_models import GPX_MEDIA_S_TAG, GPSData, GPSError, GPSTrackpoint

logger = logging.getLogger(__name__)


def parse_gpx_file(gpx_path: Path) -> GPSData:
    """
    Parse a GPX file and extract trackpoints.

    Args:
        gpx_path: Path to GPX file

    Returns:
        GPSData with parsed trackpoints

    Raises:
        GPSError: If file not found or invalid
    """
    from datetime import datetime, timezone

    # GPX is operator/GoPro-supplied untrusted input. defusedxml.ElementTree is a
    # drop-in for .parse that forbids entity expansion (billion-laughs DoS) and
    # external-entity resolution; it re-exports ParseError so malformed-XML
    # handling below is unchanged.
    import defusedxml.ElementTree as ET  # noqa: N817  drop-in for stdlib ET, keep name
    from defusedxml.common import DefusedXmlException

    gpx_path = Path(gpx_path)

    if not gpx_path.exists():
        raise GPSError(f"GPX file not found: {gpx_path}")

    try:
        tree = ET.parse(str(gpx_path))
        root = tree.getroot()
    except ET.ParseError as e:
        raise GPSError(f"Invalid GPX XML: {e}")
    except DefusedXmlException as e:
        raise GPSError(f"Unsafe GPX XML (entity-expansion/external-entity blocked): {e}")

    # Handle namespace
    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}

    trackpoints = []

    # Try with namespace first, then without
    trkpts = root.findall(".//gpx:trkpt", ns)
    if not trkpts:
        trkpts = root.findall(".//{http://www.topografix.com/GPX/1/1}trkpt")
    if not trkpts:
        trkpts = root.findall(".//trkpt")

    for trkpt in trkpts:
        try:
            lat = float(trkpt.get("lat"))
            lon = float(trkpt.get("lon"))
        except (TypeError, ValueError):
            logger.warning(
                "Skipping malformed trackpoint: lat=%s, lon=%s",
                trkpt.get("lat"),
                trkpt.get("lon"),
            )
            continue

        # Elevation (optional)
        ele_elem = trkpt.find("gpx:ele", ns)
        if ele_elem is None:
            ele_elem = trkpt.find("{http://www.topografix.com/GPX/1/1}ele")
        if ele_elem is None:
            ele_elem = trkpt.find("ele")

        elevation = float(ele_elem.text) if ele_elem is not None else None

        # Time
        time_elem = trkpt.find("gpx:time", ns)
        if time_elem is None:
            time_elem = trkpt.find("{http://www.topografix.com/GPX/1/1}time")
        if time_elem is None:
            time_elem = trkpt.find("time")

        if time_elem is None:
            # Timeless trackpoint: DROP it rather than defaulting to 0.0. A 0.0
            # timestamp sorts to the front and scrambles both chronological
            # order and the GPS↔video offset (matching the GPXParser
            # behaviour). Warn so a systematically timeless GPX is still visible.
            logger.warning(
                "GPX trackpoint has no <time> element — dropping it (lat=%s, lon=%s).",
                trkpt.get("lat"),
                trkpt.get("lon"),
            )
            continue

        # Parse ISO format timestamp. Handle both Z suffix and +00:00.
        time_str = time_elem.text
        if time_str.endswith("Z"):
            time_str = time_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(time_str)
        # A <time> without a Z/offset parses naive; .timestamp() would then read
        # it in the machine's local tz and skew the epoch by whole hours. Treat
        # a naive timestamp as UTC (existing offsets are preserved).
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        timestamp = dt.timestamp()

        media_s = None
        for ext in trkpt.iter():
            if ext.tag in (GPX_MEDIA_S_TAG, f"{{{ns['gpx']}}}{GPX_MEDIA_S_TAG}"):
                try:
                    media_s = float(ext.text)
                except (TypeError, ValueError):
                    media_s = None
                break

        trackpoints.append(GPSTrackpoint(lat, lon, elevation, timestamp, media_s))

    stamped = [tp for tp in trackpoints if tp.media_s is not None]
    if stamped and len(stamped) != len(trackpoints):
        # A half-stamped file has no single clock; refusing it (callers treat a
        # parse failure as no-GPS) is safer than naming a wrong place.
        raise GPSError(
            f"GPX {gpx_path.name} is partially stamped with {GPX_MEDIA_S_TAG} "
            f"({len(stamped)}/{len(trackpoints)} trackpoints)"
        )
    if stamped:
        # Footage clock: timestamp = t0 + media_s, with t0 anchored so the first
        # point keeps its own wall clock. Within one recording this is ~wall
        # clock; across recordings the stop gap the concat video skipped is gone.
        trackpoints.sort(key=lambda tp: tp.media_s)
        w0 = trackpoints[0].timestamp - trackpoints[0].media_s
        for tp in trackpoints:
            tp.timestamp = w0 + tp.media_s
    else:
        # Sort by timestamp
        trackpoints.sort(key=lambda tp: tp.timestamp)

    logger.debug(f"Parsed {len(trackpoints)} trackpoints from {gpx_path.name}")

    return GPSData(trackpoints=trackpoints)
