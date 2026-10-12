"""Photo EXIF-GPS → shared geocode ladder."""

import pytest
from PIL import Image

from gomoveshift.pipeline.ingest.geocoding import derive_location_from_gpx
from gomoveshift.pipeline.photos import exif_gps
from gomoveshift.pipeline.photos.exif_gps import (
    NO_EXIF_GPS,
    NO_GEOCODE,
    derive_location_from_photos,
    extract_exif_gps,
)

GPS_IFD = 0x8825

# Two Irish counties, 3-2 in Clare's favour. CENTROID is the arithmetic mean
# of the five points and is mapped to a county of its own, so a centroid-based
# lookup returns a label the majority vote never can.
CLARE = [(52.90, -9.50), (52.95, -9.45), (53.00, -9.40)]
GALWAY = [(53.40, -9.05), (53.45, -9.00)]
CENTROID = (53.14, -9.28)


class _Loc:
    def __init__(self, state, country):
        self.state = state
        self.county = state
        self.country = country


def _fake_client(mapping, default=None):
    """NominatimRoadsClient stand-in: coordinate → _Loc, no network."""

    class _Client:
        def __init__(self, *a, **k):
            pass

        def reverse_geocode(self, lat, lon):
            return mapping.get((round(lat, 4), round(lon, 4)), default)

    return _Client


@pytest.fixture
def isolate_geocoder(monkeypatch, tmp_path):
    """Point the shared cache at tmp_path — no real cache or network."""
    monkeypatch.setenv("GMS_GPS_CACHE_DIR", str(tmp_path / "geocache"))


def _photo(path, coords=None):
    """Write a small JPEG, optionally carrying GPS EXIF at ``coords``."""
    exif = Image.Exif()
    if coords is not None:
        lat, lon = coords
        exif[GPS_IFD] = {
            1: "N" if lat >= 0 else "S",
            2: _dms(abs(lat)),
            3: "E" if lon >= 0 else "W",
            4: _dms(abs(lon)),
        }
    Image.new("RGB", (48, 32), (70, 70, 70)).save(path, "JPEG", exif=exif)
    return path


def _flat(pairs):
    """Flatten coordinate pairs — pytest.approx cannot compare nested tuples."""
    return [v for pair in pairs for v in pair]


def _dms(value):
    deg = int(value)
    minutes = int((value - deg) * 60)
    seconds = round((value - deg - minutes / 60) * 3600, 4)
    return (float(deg), float(minutes), seconds)


class TestExtractExifGps:
    def test_reads_dms_with_hemisphere_signs(self, tmp_path):
        path = _photo(tmp_path / "clare.jpg", (52.925, -9.5042))
        lat, lon = extract_exif_gps(path)
        assert lat == pytest.approx(52.925, abs=1e-4)
        assert lon == pytest.approx(-9.5042, abs=1e-4)

    def test_photo_without_gps_returns_none(self, tmp_path):
        assert extract_exif_gps(_photo(tmp_path / "bare.jpg")) is None

    def test_unreadable_file_returns_none(self, tmp_path):
        broken = tmp_path / "broken.jpg"
        broken.write_bytes(b"not a jpeg")
        assert extract_exif_gps(broken) is None


class TestSyntheticGpxPath:
    def test_reaches_shared_derive_location_from_gpx(self, tmp_path, monkeypatch):
        """Coordinates must arrive as a GPX at the shared ladder, not a bespoke lookup."""
        seen = {}

        def _spy(gpx_path, **kwargs):
            from gps_journey.core.gpx_parser import GPXParser

            journey = GPXParser().parse(gpx_path)
            seen["coords"] = [(w.latitude, w.longitude) for w in journey.waypoints]
            seen["kwargs"] = kwargs
            return "County Clare, Ireland"

        monkeypatch.setattr(exif_gps, "derive_location_from_gpx", _spy)
        photos = [_photo(tmp_path / f"p{i}.jpg", c) for i, c in enumerate(CLARE)]
        result = derive_location_from_photos(photos, work_dir=tmp_path)
        assert result.resolved
        assert result.location == "County Clare, Ireland"
        assert _flat(seen["coords"]) == pytest.approx(_flat(CLARE), abs=1e-4)
        # Photos reach the ladder in photo mode, so no fix is skipped inward.
        assert seen["kwargs"] == {"photo_mode": True}


class TestVoting:
    def test_dominant_county_wins_over_centroid(self, tmp_path, monkeypatch, isolate_geocoder):
        mapping = {CENTROID: _Loc("Centroid County", "Ireland")}
        for group, name in ((CLARE, "County Clare"), (GALWAY, "County Galway")):
            for lat, lon in group:
                mapping[(round(lat, 4), round(lon, 4))] = _Loc(name, "Ireland")
        monkeypatch.setattr(
            "gps_journey.geocoding.nominatim_roads.NominatimRoadsClient",
            _fake_client(mapping),
        )
        photos = [_photo(tmp_path / f"p{i}.jpg", c) for i, c in enumerate(CLARE + GALWAY)]
        result = derive_location_from_photos(photos, work_dir=tmp_path)
        assert result.location == "County Clare, Ireland"
        # A bare-centroid lookup would have produced this instead.
        assert "Centroid County" not in result.location


#: Seven fixes along a line. Kerry holds the two endpoints plus the two the
#: video sampler steps over; Clare holds exactly the indices it samples. Kerry
#: can only win if the first and last fix are allowed to vote.
SEVEN = [(52.0 + i * 0.1, -9.0 - i * 0.1) for i in range(7)]
KERRY_AT = (0, 2, 4, 6)
#: Indices the video sampler visits for n=7 (anchors skipped 1 inward, and the
#: quarter/three-quarter marks land on the anchors again).
VIDEO_SAMPLE_IDX = (1, 1, 3, 5, 5)


def _rounded(points):
    return [(round(lat, 4), round(lon, 4)) for lat, lon in points]


class TestPhotoModeLetsEveryFixVote:
    """The inward anchor skip dodges stale GoPro clip-boundary fixes — a video
    failure mode. A photo fix is a deliberate shutter press at a real place, so
    photo mode disables the skip. The GPX/video path must not move."""

    @pytest.fixture
    def gpx(self, tmp_path):
        return exif_gps.write_synthetic_gpx(SEVEN, tmp_path / "seven.gpx")

    def _derive(self, monkeypatch, gpx, **kwargs):
        mapping = {
            coord: _Loc("County Kerry" if i in KERRY_AT else "County Clare", "Ireland")
            for i, coord in enumerate(_rounded(SEVEN))
        }
        calls = []
        base = _fake_client(mapping)

        class _Recording(base):
            def reverse_geocode(self, lat, lon):
                calls.append((round(lat, 4), round(lon, 4)))
                return super().reverse_geocode(lat, lon)

        monkeypatch.setattr(
            "gps_journey.geocoding.nominatim_roads.NominatimRoadsClient", _Recording
        )
        return derive_location_from_gpx(gpx, **kwargs), calls

    def test_photo_mode_votes_every_fix_including_the_endpoints(
        self, monkeypatch, isolate_geocoder, gpx
    ):
        label, calls = self._derive(monkeypatch, gpx, photo_mode=True)
        assert calls == _rounded(SEVEN)
        assert label == "County Kerry, Ireland"

    def test_gpx_video_path_keeps_its_inward_skip(self, monkeypatch, isolate_geocoder, gpx):
        """Shared with the drive pipeline: anchor selection stays byte-for-byte."""
        label, calls = self._derive(monkeypatch, gpx)
        assert calls == _rounded([SEVEN[i] for i in VIDEO_SAMPLE_IDX])
        assert label == "County Clare, Ireland"


class TestGeocodeBudgetIsCapped:
    """A card can carry thousands of photos; each fix costs one rate-limited
    Nominatim request. The sampler must bound the distinct coordinates that
    ever reach the geocoder, while always voting the true first and last fix."""

    def _recording_client(self, monkeypatch, coords):
        mapping = {c: _Loc("County Clare", "Ireland") for c in _rounded(coords)}
        calls: list[tuple[float, float]] = []
        base = _fake_client(mapping)

        class _Recording(base):
            def reverse_geocode(self, lat, lon):
                calls.append((round(lat, 4), round(lon, 4)))
                return super().reverse_geocode(lat, lon)

        monkeypatch.setattr(
            "gps_journey.geocoding.nominatim_roads.NominatimRoadsClient", _Recording
        )
        return calls

    def test_large_batch_caps_distinct_calls_and_pins_endpoints(
        self, tmp_path, monkeypatch, isolate_geocoder
    ):
        n = 200
        # Every photo a distinct fix, so we can read off exactly which reach
        # the geocoder. A naive range(0, n, step) would skip the true last one.
        coords = [(52.0 + i * 0.003, -9.0 - i * 0.003) for i in range(n)]
        calls = self._recording_client(monkeypatch, coords)
        photos = [_photo(tmp_path / f"p{i}.jpg", c) for i, c in enumerate(coords)]

        result = derive_location_from_photos(photos, work_dir=tmp_path)

        assert result.resolved
        distinct = set(calls)
        assert len(distinct) <= exif_gps._MAX_GEOCODE_ANCHORS
        # The crux: both true endpoints vote, never skipped inward.
        assert _rounded([coords[0]])[0] in distinct
        assert _rounded([coords[-1]])[0] in distinct
        # No coordinate is geocoded twice — the sampler dedupes by construction.
        assert len(calls) == len(distinct)
        # All fixes still surface for the caller, only the geocode is sampled.
        assert len(result.coords) == n

    def test_batch_at_cap_geocodes_every_photo_unchanged(
        self, tmp_path, monkeypatch, isolate_geocoder
    ):
        """At or below the cap the label is byte-for-byte the full-set vote."""
        n = exif_gps._MAX_GEOCODE_ANCHORS
        coords = [(52.0 + i * 0.01, -9.0 - i * 0.01) for i in range(n)]
        seen = {}

        def _spy(gpx_path, **kwargs):
            from gps_journey.core.gpx_parser import GPXParser

            journey = GPXParser().parse(gpx_path)
            seen["coords"] = [(w.latitude, w.longitude) for w in journey.waypoints]
            return "County Clare, Ireland"

        monkeypatch.setattr(exif_gps, "derive_location_from_gpx", _spy)
        photos = [_photo(tmp_path / f"p{i}.jpg", c) for i, c in enumerate(coords)]

        result = derive_location_from_photos(photos, work_dir=tmp_path)

        assert result.resolved
        # No sampling at the cap: every fix reaches the ladder, in order.
        assert _flat(seen["coords"]) == pytest.approx(_flat(coords), abs=1e-4)


class TestNoLocationIsExplicit:
    def test_photos_without_gps_return_no_exif_gps(self, tmp_path):
        photos = [_photo(tmp_path / f"p{i}.jpg") for i in range(3)]
        result = derive_location_from_photos(photos, work_dir=tmp_path)
        assert not result.resolved
        assert result.location == ""
        assert result.reason == NO_EXIF_GPS
        assert result.coords == ()

    def test_empty_input_returns_no_exif_gps(self, tmp_path):
        assert derive_location_from_photos([], work_dir=tmp_path).reason == NO_EXIF_GPS

    def test_degraded_geocoder_never_flips_non_us_to_us(
        self, tmp_path, monkeypatch, isolate_geocoder
    ):
        """A dead geocoder yields no location — never a fabricated US label."""
        monkeypatch.setattr(
            "gps_journey.geocoding.nominatim_roads.NominatimRoadsClient",
            _fake_client({}, default=None),
        )
        photos = [_photo(tmp_path / f"p{i}.jpg", c) for i, c in enumerate(CLARE)]
        result = derive_location_from_photos(photos, work_dir=tmp_path)
        assert not result.resolved
        assert result.reason == NO_GEOCODE
        assert "United States" not in result.location
        assert "USA" not in result.location
        # Coordinates still surface so a caller can prompt the operator.
        assert _flat(result.coords) == pytest.approx(_flat(CLARE), abs=1e-4)
