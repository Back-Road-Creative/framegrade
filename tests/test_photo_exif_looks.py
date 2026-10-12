"""Capture-time EXIF actually steers the look.

``compute_scene_profile`` has always accepted EXIF, but no call site populated
it — ISO-aware noise handling was dead in practice and ``capture_hour`` was
read by nobody. Both call sites now extract it, and a golden-hour capture time
favors ``golden`` on its own. Absent EXIF (every path before this) must leave
automatic selection bit-identical, which the pinned rankings below assert.
"""

import datetime

import pytest
from PIL import Image

from gomoveshift.pipeline.color_correction import CorrectionParams
from gomoveshift.pipeline.photos import develop as develop_mod
from gomoveshift.pipeline.photos import exif_gps
from gomoveshift.pipeline.photos import ingest as ingest_mod
from gomoveshift.pipeline.photos import scene as scene_mod
from gomoveshift.pipeline.photos import styles
from gomoveshift.pipeline.photos.exif_gps import (
    extract_exif_capture_metadata,
    extract_exif_dimensions,
)
from gomoveshift.pipeline.photos.scene import SceneProfile

FULL = 6  # natural + the five styles select_styles ranks over

#: Rankings measured on the code BEFORE golden-hour favoring existed.
PINNED = {
    "neutral": ["natural", "vivid", "moody", "golden", "mono", "soft"],
    "misty_ridgeline": ["natural", "moody", "mono", "soft", "golden", "vivid"],
    "golden_sunset": ["natural", "golden", "vivid", "soft", "moody", "mono"],
    "bright_meadow": ["natural", "vivid", "soft", "moody", "golden", "mono"],
    "muted_overcast": ["natural", "soft", "moody", "golden", "mono", "vivid"],
    "low_key": ["natural", "moody", "vivid", "golden", "mono", "soft"],
}
SCENES = {
    "neutral": {},
    "misty_ridgeline": {"haze": "foggy", "sat_profile": "muted", "dynamic_range": "flat"},
    "golden_sunset": {"warmth": "warm", "sat_profile": "rich"},
    "bright_meadow": {"exposure_class": "bright", "chroma_headroom": 0.8},
    "muted_overcast": {"sat_profile": "muted"},
    "low_key": {"exposure_class": "low_key"},
}


def _profile(**overrides) -> SceneProfile:
    fields = {
        "exposure_class": "normal",
        "warmth": "neutral",
        "sat_profile": "normal",
        "haze": "clear",
        "dynamic_range": "normal",
        "is_near_monochrome": False,
    }
    return SceneProfile(**(fields | overrides))


def _jpeg(path, *, when=None, iso=None, exposure=None, offset=None, rgb=None):
    """A real JPEG, optionally carrying capture EXIF in the Exif sub-IFD."""
    exif = Image.Exif()
    sub = {}
    if when is not None:
        sub[0x9003] = when
    if iso is not None:
        sub[0x8827] = iso
    if exposure is not None:
        sub[0x829A] = exposure
    if offset is not None:
        sub[0x9011] = offset
    if sub:
        exif[0x8769] = sub
    img = Image.fromarray(rgb) if rgb is not None else Image.new("RGB", (48, 36), (200, 150, 100))
    img.save(path, "JPEG", exif=exif)
    return path


class TestSelectionUnchangedWithoutExif:
    """capture_hour is None on every pre-existing path — the score must not move."""

    @pytest.mark.parametrize("name", sorted(SCENES))
    def test_full_ranking_matches_the_pinned_baseline(self, name):
        assert styles.select_styles(_profile(**SCENES[name]), count=FULL) == PINNED[name]

    def test_capture_hour_defaults_to_none(self):
        assert _profile().capture_hour is None


class TestExtractExifCaptureMetadata:
    def test_reads_hour_iso_and_exposure(self, tmp_path):
        meta = extract_exif_capture_metadata(
            _jpeg(tmp_path / "p.jpg", when="2026:07:20 19:05:26", iso=1250, exposure=1 / 60)
        )
        assert meta.capture_hour == 19
        assert meta.iso == 1250
        assert meta.exposure_s == pytest.approx(1 / 60, rel=1e-3)

    def test_malformed_absent_or_unreadable_yields_none_without_raising(self, tmp_path):
        assert extract_exif_capture_metadata(_jpeg(tmp_path / "bad.jpg", when="nope")) == (
            scene_mod.PhotoExif()
        )
        assert extract_exif_capture_metadata(_jpeg(tmp_path / "bare.jpg")) == scene_mod.PhotoExif()
        broken = tmp_path / "broken.jpg"
        broken.write_bytes(b"not a jpeg")
        assert extract_exif_capture_metadata(broken) == scene_mod.PhotoExif()
        assert extract_exif_capture_metadata(tmp_path / "gone.jpg") == scene_mod.PhotoExif()

    def test_partial_exif_keeps_the_readable_fields(self, tmp_path):
        meta = extract_exif_capture_metadata(_jpeg(tmp_path / "iso.jpg", iso=100))
        assert (meta.iso, meta.capture_hour, meta.exposure_s) == (100, None, None)


class TestCaptureUtcAnchor:
    """``DateTimeOriginal`` + ``OffsetTimeOriginal`` combine into one aware UTC instant."""

    def test_negative_offset_shifts_forward_to_utc(self, tmp_path):
        meta = extract_exif_capture_metadata(
            _jpeg(tmp_path / "p.jpg", when="2026:07:31 08:15:29", offset="-04:00")
        )
        assert meta.capture_utc == datetime.datetime(
            2026, 7, 31, 12, 15, 29, tzinfo=datetime.timezone.utc
        )
        assert meta.capture_hour == 8  # the camera-local hour must not move

    def test_positive_offset_shifts_back_to_utc(self, tmp_path):
        meta = extract_exif_capture_metadata(
            _jpeg(tmp_path / "p.jpg", when="2026:07:31 08:15:29", offset="+05:30")
        )
        assert meta.capture_utc == datetime.datetime(
            2026, 7, 31, 2, 45, 29, tzinfo=datetime.timezone.utc
        )

    def test_missing_offset_reads_none_and_keeps_local_fields(self, tmp_path):
        meta = extract_exif_capture_metadata(_jpeg(tmp_path / "p.jpg", when="2026:07:31 08:15:29"))
        assert meta.capture_utc is None
        assert meta.capture_hour == 8

    def test_missing_datetime_reads_none(self, tmp_path):
        meta = extract_exif_capture_metadata(_jpeg(tmp_path / "p.jpg", offset="-04:00"))
        assert meta.capture_utc is None
        assert meta.capture_hour is None

    @pytest.mark.parametrize("offset", ["", "Z", "UTC-4", "-4:00", "-0400", "+99:99", "junk"])
    def test_malformed_offset_reads_none_without_raising(self, tmp_path, offset):
        meta = extract_exif_capture_metadata(
            _jpeg(tmp_path / "p.jpg", when="2026:07:31 08:15:29", offset=offset)
        )
        assert meta.capture_utc is None
        assert meta.capture_hour == 8  # a bad offset never poisons the local reading

    def test_malformed_datetime_reads_none(self, tmp_path):
        meta = extract_exif_capture_metadata(
            _jpeg(tmp_path / "p.jpg", when="nope", offset="-04:00")
        )
        assert meta.capture_utc is None

    def test_capture_utc_defaults_to_none(self):
        assert scene_mod.PhotoExif().capture_utc is None


def test_exif_dimensions_require_two_positive_integral_tags(monkeypatch):
    monkeypatch.setattr(exif_gps, "_exif_sub_ifd", lambda _path: {0xA002: 6000, 0xA003: 4000})
    assert extract_exif_dimensions("frame.nef") == (6000, 4000)

    monkeypatch.setattr(exif_gps, "_exif_sub_ifd", lambda _path: {0xA002: 6000})
    monkeypatch.setattr(exif_gps, "_tiff_dimensions", lambda _path: None)
    assert extract_exif_dimensions("frame.nef") is None

    monkeypatch.setattr(exif_gps, "_exif_sub_ifd", lambda _path: {0xA002: 6000.5, 0xA003: 4000})
    assert extract_exif_dimensions("frame.nef") is None


def test_exif_dimensions_fall_back_to_raw_tiff_subifd_geometry(monkeypatch):
    monkeypatch.setattr(exif_gps, "_exif_sub_ifd", lambda _path: {})
    monkeypatch.setattr(exif_gps, "_tiff_dimensions", lambda _path: (5600, 3728))
    assert extract_exif_dimensions("frame.nef") == (5600, 3728)


class TestGoldenHourFavoring:
    def test_a_golden_hour_capture_favors_golden_where_midday_does_not(self):
        midday = _profile(capture_hour=12)
        assert styles.select_styles(midday, count=FULL) == PINNED["neutral"]
        for hour in (7, 19):
            assert styles.select_styles(_profile(capture_hour=hour)) == [
                "natural",
                "golden",
                "vivid",
            ]

    def test_window_edges_are_start_inclusive_end_exclusive(self):
        for hour in (styles.GOLDEN_HOUR_MORNING_START, styles.GOLDEN_HOUR_EVENING_START):
            assert styles._is_golden_hour(hour)
        for hour in (styles.GOLDEN_HOUR_MORNING_END, styles.GOLDEN_HOUR_EVENING_END, None):
            assert not styles._is_golden_hour(hour)


class TestBothCallSitesPopulateExif:
    def test_develop_plan_reads_the_capture_hour(self, tmp_path):
        plan = develop_mod.plan_variants(_jpeg(tmp_path / "g.jpg", when="2000:01:01 07:54:30"))
        assert plan.scene.capture_hour == 7

    def test_develop_plan_still_works_without_exif(self, tmp_path):
        Image.new("RGB", (48, 36), (120, 130, 140)).save(tmp_path / "n.png")
        plan = develop_mod.plan_variants(tmp_path / "n.png")
        assert plan.scene.capture_hour is None
        assert plan.styled

    def test_ingest_develop_styled_passes_the_capture_exif(self, tmp_path, monkeypatch):
        seen = []
        real = scene_mod.compute_scene_profile

        def spy(rgb, exif=None):
            seen.append(exif)
            return real(rgb, exif)

        monkeypatch.setattr(scene_mod, "compute_scene_profile", spy)
        source = _jpeg(tmp_path / "c.jpg", when="2026:07:20 19:05:26", iso=1250)
        ingest_mod._develop_styled(source, CorrectionParams(), 0, "golden", 1.0)
        Image.new("RGB", (48, 36), (120, 130, 140)).save(tmp_path / "d.png")
        ingest_mod._develop_styled(tmp_path / "d.png", CorrectionParams(), 0, "golden", 1.0)

        assert (seen[0].capture_hour, seen[0].iso) == (19, 1250)
        assert seen[1] == scene_mod.PhotoExif()
