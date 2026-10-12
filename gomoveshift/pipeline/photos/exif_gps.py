"""Photo location from EXIF GPS, via the shared geocode ladder.

Photos carry their fix in EXIF; every other GPS source here is GPX-based. So
this mirrors ``pipeline/gopro_gps.py`` rather than adding a second geocoder:
extract the coordinates, emit a synthetic GPX, hand it to
:func:`derive_location_from_gpx`. That keeps the label **voted** over the
dominant county (never a bare centroid, which lands on the wrong side of a
county line), keeps Nominatim the only geocoding host with the shared cache,
and lets a degraded geocoder yield *no* location rather than a fabricated one
that could flip a non-US set to a US label.

No GPS EXIF is the common case, not an error: it returns an explicit
unresolved :class:`PhotoLocation` the caller must act on, never a default.
Coordinates only derive a label — the published JPEG still has its GPS IFD
stripped by the resizer, and the synthetic GPX is deleted after use.
"""

import datetime
import logging
import re
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from sun_track import TrackIndex, load_track_index, solar_elevation_deg

from ..ingest.geocoding import derive_location_from_gpx

logger = logging.getLogger(__name__)

# How far (minutes) a capture time may sit from the nearest track point and
# still take a position. Upstream this was ``GPX_MATCH_MAX_GAP_MIN`` inside the
# in-repo gpx_position module — a module constant with no override, so the
# value was structurally fixed. sun-track makes it the per-call ``max_gap_min``
# argument, whose default this pipeline must NOT inherit: a future change to
# the package default would silently widen or tighten photo matching on an
# unattended run. Pinned here and passed explicitly at the one call site;
# asserted by tests/test_photo_gpx_position.py.
GPX_MATCH_MAX_GAP_MIN = 90.0

GPS_IFD_TAG = 0x8825
_GPS_LAT_REF, _GPS_LAT, _GPS_LON_REF, _GPS_LON = 1, 2, 3, 4

# Reasons: no fix in any photo (caller must supply a label); fixes present but
# the geocoder gave nothing usable; resolved from EXIF; supplied by the
# operator, which is authoritative and bypasses the geocode entirely.
NO_EXIF_GPS = "no_exif_gps"
NO_GEOCODE = "geocode_unavailable"
RESOLVED = "resolved"
OPERATOR = "operator"

# The GPX parser drops track points without a time, and capture times don't
# affect a reverse-geocode — one monotonic second per photo satisfies both.
_GPX_EPOCH = datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc)

# Cap the reverse-geocode budget on a large card. Each fix costs one
# rate-limited Nominatim request (~1 s), so a 2000-photo card would spend
# ~33 min resolving a single voted label and hammer the geocoder. Sampling a
# fixed number of evenly-spaced anchor photos bounds that to this many distinct
# requests regardless of card size. Even index spacing makes each anchor stand
# in for an equal-size run of neighbouring photos (nearest-anchor voting), so
# the vote still reflects where the day was spent; ~3x the video path's five
# samples, because a photo set spans more distinct places than one drive.
_MAX_GEOCODE_ANCHORS = 16


@dataclass(frozen=True)
class PhotoLocation:
    """One location for a set of photos. Check :attr:`resolved`, not the string."""

    location: str = ""
    reason: str = NO_EXIF_GPS
    coords: tuple[tuple[float, float], ...] = field(default_factory=tuple)

    @property
    def resolved(self) -> bool:
        return self.reason in (RESOLVED, OPERATOR) and bool(self.location)


def _to_degrees(dms, ref: str) -> Optional[float]:
    """Convert an EXIF (degrees, minutes, seconds) rational triple to a float."""
    try:
        deg, minutes, seconds = (float(v) for v in dms)
    except (TypeError, ValueError):
        return None
    value = deg + minutes / 60.0 + seconds / 3600.0
    if str(ref).strip().upper() in ("S", "W"):
        value = -value
    return value


def extract_exif_gps(photo_path) -> Optional[tuple[float, float]]:
    """Return ``(lat, lon)`` from a photo's GPS EXIF, or ``None``.

    ``None`` covers every "no usable fix" case — absent GPS IFD, malformed
    rationals, unreadable file. Never raises: one bad photo can't abort a batch.
    """
    from PIL import Image

    try:
        with Image.open(photo_path) as img:
            gps = img.getexif().get_ifd(GPS_IFD_TAG)
    except Exception as exc:  # unreadable/corrupt file — degrade, never raise
        logger.debug("EXIF read failed for %s: %s", photo_path, exc)
        return None

    if not gps:
        return None

    lat = _to_degrees(gps.get(_GPS_LAT), gps.get(_GPS_LAT_REF, "N"))
    lon = _to_degrees(gps.get(_GPS_LON), gps.get(_GPS_LON_REF, "E"))
    if lat is None or lon is None:
        return None
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        logger.debug("Discarding out-of-range EXIF fix (%s, %s)", lat, lon)
        return None
    return (lat, lon)


# Capture metadata lives in the Exif sub-IFD, not IFD0.
_EXIF_IFD_TAG = 0x8769
_DATETIME_ORIGINAL = 0x9003
_OFFSET_TIME_ORIGINAL = 0x9011
_ISO_SPEED = 0x8827
_EXPOSURE_TIME = 0x829A
_F_NUMBER = 0x829D
_FOCAL_LENGTH = 0x920A
_PIXEL_X_DIMENSION = 0xA002
_PIXEL_Y_DIMENSION = 0xA003
_IMAGE_WIDTH = 0x0100
_IMAGE_HEIGHT = 0x0101
_SUB_IFDS = 0x014A

# EXIF 2.31's OffsetTimeOriginal is exactly "±HH:MM" — the UTC offset the
# camera clock carried when DateTimeOriginal was recorded. Anything else
# ("", "Z", bare hours) reads as absent, never guessed at.
_UTC_OFFSET = re.compile(r"([+-])(\d{2}):(\d{2})")


# A camera RAW is a TIFF container. Pillow refuses to *open* one (it cannot
# decode the mosaic), but the Exif sub-IFD sits in front of that mosaic as
# ordinary TIFF metadata, so Pillow's own IFD parser reads it from the file
# object directly — header seeks only, never a demosaic. Ingest calls this per
# file before any decode, so cost matters more than completeness.
_TIFF_BYTE_ORDER = {b"II\x2a\x00": "<", b"MM\x00\x2a": ">"}


def _pillow_sub_ifd(photo_path) -> dict:
    """The Exif sub-IFD as Pillow's decoders see it — JPEG/HEIC/TIFF/DNG."""
    from PIL import Image

    with Image.open(photo_path) as img:
        return img.getexif().get_ifd(_EXIF_IFD_TAG)


def _tiff_sub_ifd(photo_path) -> dict:
    """The Exif sub-IFD walked straight out of a TIFF container — NEF/CR2/ARW.

    ``{}`` for anything that isn't TIFF-headed, so a JPEG/PNG costs 8 bytes.
    """
    from PIL import TiffImagePlugin

    with open(photo_path, "rb") as fh:
        header = fh.read(8)
        byte_order = _TIFF_BYTE_ORDER.get(header[:4])
        if byte_order is None:
            return {}
        fh.seek(struct.unpack(byte_order + "I", header[4:8])[0])
        ifd0 = TiffImagePlugin.ImageFileDirectory_v2(header)
        ifd0.load(fh)
        offset = ifd0.get(_EXIF_IFD_TAG)
        if not offset:
            return {}
        fh.seek(int(offset[0] if isinstance(offset, tuple) else offset))
        sub = TiffImagePlugin.ImageFileDirectory_v2(header)
        sub.load(fh)
        return dict(sub)


def _exif_sub_ifd(photo_path) -> dict:
    """The photo's Exif sub-IFD, or ``{}`` — the one reader for every tag below.

    Tries Pillow's decoders first, so every format Pillow opens parses exactly
    as it always did, then falls back to the raw TIFF walk that a Nikon
    ``.NEF`` (and every other RAW container) needs. Never raises: an
    unreadable, corrupt or truncated file degrades to no tags through both
    readers, so one bad photo can't abort a batch.
    """
    for reader in (_pillow_sub_ifd, _tiff_sub_ifd):
        try:
            tags = reader(photo_path)
        except Exception as exc:  # unreadable/corrupt file — degrade, never raise
            logger.debug("EXIF read failed for %s via %s: %s", photo_path, reader.__name__, exc)
            continue
        if tags:
            return tags
    return {}


def _as_float(value) -> Optional[float]:
    """An EXIF rational/int as a float; anything unusable reads ``None``."""
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _capture_datetime(value) -> Optional[datetime.datetime]:
    """A ``DateTimeOriginal`` value parsed whole, or ``None``.

    This is camera-LOCAL wall time — EXIF carries no timezone — so it is the
    clock the photographer stood in: a look wants the hour, a bracket the seconds.
    """
    if not value:
        return None
    try:
        return datetime.datetime.strptime(str(value).strip()[:19], "%Y:%m:%d %H:%M:%S")
    except ValueError:
        logger.debug("Malformed EXIF DateTimeOriginal %r", value)
        return None


def _capture_hour(value) -> Optional[int]:
    """Hour-of-day from a ``DateTimeOriginal`` value, or ``None``."""
    stamp = _capture_datetime(value)
    return stamp.hour if stamp else None


def _capture_utc(datetime_value, offset_value) -> Optional[datetime.datetime]:
    """``DateTimeOriginal`` anchored by ``OffsetTimeOriginal`` as a UTC instant.

    The wall-time tag alone cannot place a capture on the UTC timeline — solar
    geometry needs an absolute instant, not the camera-local clock the hour
    reader keeps. ``None`` whenever either tag is absent or malformed, so a
    card without offsets degrades to exactly the wall-time-only behaviour and
    a bad offset never poisons the local readings. Never raises.
    """
    stamp = _capture_datetime(datetime_value)
    if stamp is None or not offset_value:
        return None
    match = _UTC_OFFSET.fullmatch(str(offset_value).strip())
    if match is None:
        logger.debug("Malformed EXIF OffsetTimeOriginal %r", offset_value)
        return None
    sign, hours, minutes = match.group(1), int(match.group(2)), int(match.group(3))
    if hours > 23 or minutes > 59:
        logger.debug("Out-of-range EXIF OffsetTimeOriginal %r", offset_value)
        return None
    delta = datetime.timedelta(hours=hours, minutes=minutes)
    zone = datetime.timezone(-delta if sign == "-" else delta)
    return stamp.replace(tzinfo=zone).astimezone(datetime.timezone.utc)


# The GPX track index is a lazy module-level singleton KEYED BY the configured
# ``photos.gpx_dir`` value: the day's tracks parse once per run rather than once
# per photo, and a repointed dir (tests use tmp_path) reloads instead of
# replaying the old track. "" — the shipped default — means no index at all.
_track_index_cache: Optional[tuple[str, TrackIndex]] = None


def _reset_gpx_cache() -> None:
    """Test hook: drop the cached track index so a repointed config reloads."""
    global _track_index_cache
    _track_index_cache = None


def _configured_track_index() -> Optional[TrackIndex]:
    global _track_index_cache
    from ...config import get_config  # local: mirrors the sibling modules' accessor

    gpx_dir = get_config().photos.gpx_dir
    if not gpx_dir:
        return None
    if _track_index_cache is None or _track_index_cache[0] != gpx_dir:
        _track_index_cache = (gpx_dir, load_track_index(Path(gpx_dir)))
    return _track_index_cache[1]


def _solar_elevation_at_capture(capture_utc: Optional[datetime.datetime]) -> Optional[float]:
    """Apparent solar elevation at the capture instant, or ``None``.

    Computed only when BOTH halves exist: a UTC anchor (``capture_utc``) and a
    track position for it. Everything else — no ``photos.gpx_dir`` configured,
    no track point within the match gap, no offset tag — reads ``None``, so the
    selector's wall-clock fallback behaves exactly as before this field existed.
    """
    if capture_utc is None:
        return None
    index = _configured_track_index()
    if index is None:
        return None
    position = index.position_at(capture_utc, max_gap_min=GPX_MATCH_MAX_GAP_MIN)
    if position is None:
        return None
    return solar_elevation_deg(position[0], position[1], capture_utc)


def extract_exif_capture_timestamp(photo_path) -> Optional[datetime.datetime]:
    """Return the full capture time from ``DateTimeOriginal``, or ``None``.

    Seconds resolution, camera-local, naive — the ordering bracket detection
    needs and the hour-only/date-only readers throw away. Same never-raise
    contract as its siblings: absent, malformed and unreadable all read ``None``.
    """
    return _capture_datetime(_exif_sub_ifd(photo_path).get(_DATETIME_ORIGINAL))


def extract_exif_aperture(photo_path) -> Optional[float]:
    """Return the f-number from ``FNumber``, or ``None``.

    Bracket detection's tag, and deliberately NOT a ``PhotoExif`` field: nothing
    in scene profiling meters aperture, so widening that record would ripple
    through every profile call site to serve one grouping rule. Same never-raise
    contract as its siblings — absent, malformed, non-positive and unreadable
    all read ``None`` — and a card whose files carry no ``FNumber`` reads
    ``None`` on every frame, which is what it did before this tag was read.
    """
    value = _as_float(_exif_sub_ifd(photo_path).get(_F_NUMBER))
    return value if value and value > 0 else None


def extract_exif_focal_length(photo_path) -> Optional[float]:
    """Return the focal length in mm from ``FocalLength``, or ``None``.

    Subject recognition's tag (``subject.py``: a 400 mm frame is not a scenic
    and an 85 mm at f/1.8 is not wildlife), and deliberately NOT a ``PhotoExif``
    field for the same reason ``extract_exif_aperture`` is not — nothing in
    scene profiling meters a lens. Same never-raise contract as its siblings:
    absent, malformed, non-positive and unreadable all read ``None``, so a card
    whose files carry no ``FocalLength`` recognises on the other evidence.
    """
    value = _as_float(_exif_sub_ifd(photo_path).get(_FOCAL_LENGTH))
    return value if value and value > 0 else None


def _dimension_pair(width_value, height_value) -> Optional[tuple[int, int]]:
    width = _as_float(width_value)
    height = _as_float(height_value)
    if (
        width is None
        or height is None
        or width <= 0
        or height <= 0
        or not width.is_integer()
        or not height.is_integer()
    ):
        return None
    return int(width), int(height)


def _tiff_dimensions(photo_path) -> Optional[tuple[int, int]]:
    """Largest recorded TIFF/SubIFD geometry, using header seeks only."""
    from PIL import TiffImagePlugin

    with open(photo_path, "rb") as fh:
        header = fh.read(8)
        byte_order = _TIFF_BYTE_ORDER.get(header[:4])
        if byte_order is None:
            return None
        fh.seek(struct.unpack(byte_order + "I", header[4:8])[0])
        root = TiffImagePlugin.ImageFileDirectory_v2(header)
        root.load(fh)
        candidates = [_dimension_pair(root.get(_IMAGE_WIDTH), root.get(_IMAGE_HEIGHT))]
        offsets = root.get(_SUB_IFDS, ())
        if isinstance(offsets, int):
            offsets = (offsets,)
        for offset in offsets:
            fh.seek(int(offset))
            child = TiffImagePlugin.ImageFileDirectory_v2(header)
            child.load(fh)
            candidates.append(_dimension_pair(child.get(_IMAGE_WIDTH), child.get(_IMAGE_HEIGHT)))
    known = [pair for pair in candidates if pair is not None]
    return max(known, key=lambda pair: pair[0] * pair[1], default=None)


def extract_exif_dimensions(photo_path) -> Optional[tuple[int, int]]:
    """Return recorded ``(width, height)`` without decoding image pixels.

    JPEG/DNG commonly record PixelX/YDimension in Exif. Nikon RAW records its
    full mosaic geometry in TIFF SubIFDs alongside smaller previews, so the
    largest valid header pair wins there. Partial or malformed metadata stays
    unknown; no image payload is decoded to infer a size.
    """
    ifd = _exif_sub_ifd(photo_path)
    dimensions = _dimension_pair(ifd.get(_PIXEL_X_DIMENSION), ifd.get(_PIXEL_Y_DIMENSION))
    if dimensions is not None:
        return dimensions
    try:
        return _tiff_dimensions(photo_path)
    except Exception as exc:
        logger.debug("TIFF dimension read failed for %s: %s", photo_path, exc)
        return None


def extract_exif_capture_metadata(photo_path):
    """Return this photo's capture metadata as a ``PhotoExif`` for scene profiling.

    Same never-raise contract as the date reader below, per field: an absent or
    malformed tag yields ``None`` for that field alone. A source with no EXIF
    at all — a PNG, a synthesised frame — yields an all-``None`` record, which
    profiles exactly as passing no EXIF at all. A RAW carries real tags.
    """
    from .scene import PhotoExif  # local: scene imports develop, develop imports this

    ifd = _exif_sub_ifd(photo_path)
    capture_utc = _capture_utc(ifd.get(_DATETIME_ORIGINAL), ifd.get(_OFFSET_TIME_ORIGINAL))
    return PhotoExif(
        iso=_as_float(ifd.get(_ISO_SPEED)),
        exposure_s=_as_float(ifd.get(_EXPOSURE_TIME)),
        capture_hour=_capture_hour(ifd.get(_DATETIME_ORIGINAL)),
        capture_utc=capture_utc,
        solar_elevation_deg=_solar_elevation_at_capture(capture_utc),
    )


def extract_exif_capture_date(photo_path) -> Optional[str]:
    """Return the capture date (``YYYY-MM-DD``) from ``DateTimeOriginal``, or ``None``.

    ``None`` covers every "no readable date" case — absent tag, malformed
    value, unreadable file. Never raises: one bad photo can't abort a batch,
    and a dateless legacy card must stay ingestable under an explicit day.
    """
    value = _exif_sub_ifd(photo_path).get(_DATETIME_ORIGINAL)
    if not value:
        return None
    try:
        # EXIF format: "YYYY:MM:DD HH:MM:SS" — the date is the first 10 chars.
        return datetime.datetime.strptime(str(value).strip()[:10], "%Y:%m:%d").date().isoformat()
    except ValueError:
        logger.debug("Malformed EXIF DateTimeOriginal %r in %s", value, photo_path)
        return None


def write_synthetic_gpx(coords: Sequence[tuple[float, float]], gpx_path: Path) -> Path:
    """Write ``coords`` as a single-track GPX the shared ladder can parse."""
    import gpxpy.gpx

    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(track)
    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)
    for i, (lat, lon) in enumerate(coords):
        segment.points.append(
            gpxpy.gpx.GPXTrackPoint(lat, lon, time=_GPX_EPOCH + datetime.timedelta(seconds=i))
        )
    gpx_path.write_text(gpx.to_xml(), encoding="utf-8")
    return gpx_path


def _sample_anchor_coords(coords):
    """Evenly-spaced anchor coordinates, true endpoints pinned, identical GPS
    deduped — the only path from photo fixes to the geocoder.

    At or below the cap every fix is an anchor, so the label is byte-for-byte
    what the full set would elect. Above it, exactly ``_MAX_GEOCODE_ANCHORS``
    indices are sampled with ``round(i * (n - 1) / (k - 1))``, which pins the
    first (i=0) and last (i=k-1) photo — a naive ``range(0, n, step)`` skips the
    true last one, losing the end-of-trip location. Even spacing means each
    anchor represents an equal-size run of neighbours, so at most that many
    distinct coordinates ever reach the geocoder.
    """
    n = len(coords)
    if n <= _MAX_GEOCODE_ANCHORS:
        return list(coords)
    k = _MAX_GEOCODE_ANCHORS
    anchors, seen = [], set()
    for i in range(k):
        coord = coords[round(i * (n - 1) / (k - 1))]
        if coord not in seen:
            seen.add(coord)
            anchors.append(coord)
    return anchors


def derive_location_from_photos(photo_paths, work_dir=None) -> PhotoLocation:
    """Derive ONE voted location for a set of photos from their EXIF GPS.

    Compute once per set and persist it — re-deriving from raw coordinates at a
    second site is the documented cause of far-road label bleed.
    """
    coords = []
    for photo in photo_paths or []:
        fix = extract_exif_gps(photo)
        if fix is not None:
            coords.append(fix)

    if not coords:
        logger.info("  No GPS EXIF in %d photo(s) — location unresolved", len(photo_paths or []))
        return PhotoLocation(reason=NO_EXIF_GPS)

    frozen = tuple(coords)
    anchors = _sample_anchor_coords(frozen)
    base = Path(work_dir) if work_dir else None
    with tempfile.TemporaryDirectory(dir=base) as tmp:
        gpx_path = write_synthetic_gpx(anchors, Path(tmp) / "photos_exif.gpx")
        # photo_mode: every sampled anchor votes, endpoints included.
        location = derive_location_from_gpx(gpx_path, photo_mode=True)

    if not location:
        logger.warning(
            "  Geocoder returned no label for %d photo fix(es) — location unresolved", len(frozen)
        )
        return PhotoLocation(reason=NO_GEOCODE, coords=frozen)

    logger.info(
        "  Photo location (voted over %d of %d fix(es)): %s", len(anchors), len(frozen), location
    )
    return PhotoLocation(location=location, reason=RESOLVED, coords=frozen)
