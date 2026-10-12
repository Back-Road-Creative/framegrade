"""Tests for the photo scene profiler (gomoveshift.pipeline.photos.scene).

SceneProfile is a pure function of a decoded frame plus optional EXIF; the
archetypes below are synthetic frames engineered to land in one class each.
"""

import datetime
from dataclasses import fields

import numpy as np

from gomoveshift.pipeline.photos import scene


def _frame(r, g, b, *, width=160, height=100, spread=25):
    """Landscape ramp frame around the given channel centers."""
    ramp = np.linspace(-spread, spread, width, dtype=np.float32)
    plane = np.repeat(ramp[None, :], height, axis=0)
    frame = np.stack([plane + r, plane + g, plane + b], axis=2)
    return np.clip(frame, 0, 255).astype(np.uint8)


def test_scene_profile_carries_exactly_the_consumed_traits():
    """The zero-flip traits are deleted, not wired — the field set IS the contract."""
    assert tuple(f.name for f in fields(scene.SceneProfile)) == (
        "exposure_class",
        "warmth",
        "sat_profile",
        "haze",
        "dynamic_range",
        "is_near_monochrome",
        "chroma_headroom",
        "warm_fraction",
        "capture_hour",
        "solar_elevation_deg",
    )


def test_misty_ridgeline_profiles_foggy_and_flat():
    """Lifted blacks + compressed spread + muted sat: the mist/fog scene class."""
    profile = scene.compute_scene_profile(_frame(128, 120, 116, spread=24))

    assert profile.haze == "foggy"
    assert profile.dynamic_range == "flat"
    assert profile.sat_profile == "muted"
    assert profile.is_near_monochrome is False


def test_golden_hour_sunset_profiles_warm_and_rich():
    profile = scene.compute_scene_profile(_frame(230, 140, 60))

    assert profile.warmth == "warm"
    assert profile.sat_profile == "rich"
    # saturated warm light is the scene, never an atmospheric veil
    assert profile.haze == "clear"


def test_deliberate_silhouette_profiles_low_key():
    """Blacks at true black, intact highlights — low_key, not underexposed-normal."""
    sil = np.full((100, 160, 3), 3, dtype=np.uint8)
    sil[:40] = 225  # bright sky above a black foreground

    profile = scene.compute_scene_profile(sil)

    assert profile.exposure_class == "low_key"
    assert profile.dynamic_range == "wide"


def test_a_dark_frame_without_intact_highlights_is_not_low_key():
    profile = scene.compute_scene_profile(_frame(20, 18, 16))

    assert profile.exposure_class == "normal"


def test_bright_meadow_profiles_bright_not_overexposed():
    profile = scene.compute_scene_profile(_frame(150, 200, 80))

    assert profile.exposure_class == "bright"
    assert profile.sat_profile in ("normal", "rich")
    assert profile.haze == "clear"


def test_near_monochrome_frame_is_flagged():
    profile = scene.compute_scene_profile(_frame(128, 128, 128))

    assert profile.is_near_monochrome is True
    assert profile.warmth == "neutral"


def test_a_plain_dict_exif_reads_identically_to_the_dataclass():
    frame = _frame(230, 140, 60)

    assert scene.compute_scene_profile(frame, {"capture_hour": 19}) == scene.compute_scene_profile(
        frame, scene.PhotoExif(capture_hour=19)
    )


def test_photo_exif_carries_the_utc_anchor_with_a_none_default():
    assert tuple(f.name for f in fields(scene.PhotoExif)) == (
        "iso",
        "exposure_s",
        "capture_hour",
        "capture_utc",
        "solar_elevation_deg",
    )
    assert scene.PhotoExif().capture_utc is None


def test_a_plain_dict_exif_carries_the_utc_anchor_through(monkeypatch):
    """Manifest/fixture EXIF arrives as a dict — the anchor must not drop there."""
    seen = {}
    real = scene.PhotoExif

    def spy(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(scene, "PhotoExif", spy)
    anchor = datetime.datetime(2026, 7, 31, 12, 15, 29, tzinfo=datetime.timezone.utc)
    scene.compute_scene_profile(_frame(230, 140, 60), {"capture_hour": 8, "capture_utc": anchor})
    assert seen["capture_utc"] == anchor


def test_same_frame_profiles_identically_twice():
    frame = _frame(150, 200, 80)

    assert scene.compute_scene_profile(frame) == scene.compute_scene_profile(frame)


def test_uint16_input_profiles_like_its_uint8_source():
    frame = _frame(230, 140, 60)
    wide = frame.astype(np.uint16) * 257  # exact 8->16-bit lift

    assert scene.compute_scene_profile(wide) == scene.compute_scene_profile(frame)


def test_non_rgb_or_float_input_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        scene.compute_scene_profile(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        scene.compute_scene_profile(np.zeros((10, 10, 3), dtype=np.float32))


def test_warm_sunset_reads_a_dominant_warm_fraction():
    assert scene.compute_scene_profile(_frame(230, 140, 60)).warm_fraction > 0.5


def test_gray_card_reads_full_headroom_and_no_warm_votes():
    """Zero chroma: gray pixels never vote, so the warm share stays 0."""
    profile = scene.compute_scene_profile(_frame(128, 128, 128))

    assert profile.chroma_headroom > 0.95
    assert profile.warm_fraction == 0.0


def test_fully_saturated_frame_has_no_chroma_headroom():
    assert scene.compute_scene_profile(_frame(255, 0, 0)).chroma_headroom < 0.1


def test_uint16_color_readings_match_the_uint8_view():
    frame = _frame(60, 180, 40)
    wide = frame.astype(np.uint16) * 257  # exact 8->16-bit lift

    assert scene.compute_scene_profile(wide) == scene.compute_scene_profile(frame)


def test_hues_outside_the_warm_band_never_vote():
    """Chroma living outside the warm band (cyans, violets) casts no warm vote."""
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    frame[:, :80] = (40, 200, 195)  # cyan, ~178 degrees
    frame[:, 80:] = (160, 40, 220)  # violet, ~280 degrees

    profile = scene.compute_scene_profile(frame)

    assert profile.is_near_monochrome is False
    assert profile.warm_fraction == 0.0


# ── solar elevation: the field and its one computation boundary ──────


def test_solar_elevation_defaults_none_on_every_no_track_path():
    """No GPX configured is the common case: the field exists and stays None."""
    assert scene.PhotoExif().solar_elevation_deg is None
    assert scene.compute_scene_profile(_frame(230, 140, 60)).solar_elevation_deg is None


def test_solar_elevation_flows_like_capture_hour_through_both_exif_branches():
    frame = _frame(230, 140, 60)
    via_dataclass = scene.compute_scene_profile(frame, scene.PhotoExif(solar_elevation_deg=5.0))
    via_mapping = scene.compute_scene_profile(frame, {"solar_elevation_deg": 5.0})
    assert via_dataclass.solar_elevation_deg == 5.0
    assert via_mapping == via_dataclass


def _solar_reader_env(monkeypatch, gpx_dir: str):
    """Point the config at ``gpx_dir`` and feed the reader a fixed Exif sub-IFD."""
    import gomoveshift.config as config_module
    from gomoveshift.config import Config, PhotosConfig
    from gomoveshift.pipeline.photos import exif_gps

    photos = PhotosConfig(gpx_dir=gpx_dir)
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))
    exif_gps._reset_gpx_cache()
    monkeypatch.setattr(
        exif_gps,
        "_exif_sub_ifd",
        lambda path: {
            exif_gps._DATETIME_ORIGINAL: "2026:07:04 06:10:00",
            exif_gps._OFFSET_TIME_ORIGINAL: "+00:00",
        },
    )
    return exif_gps


def _write_track(tmp_path):
    (tmp_path / "day.gpx").write_text(
        '<gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>'
        '<trkpt lat="53.0" lon="-8.0"><time>2026-07-04T05:55:00Z</time></trkpt>'
        '<trkpt lat="53.0" lon="-8.0"><time>2026-07-04T06:25:00Z</time></trkpt>'
        "</trkseg></trk></gpx>"
    )


def test_exif_reader_computes_solar_elevation_from_the_configured_track(tmp_path, monkeypatch):
    """UTC anchor + a GPX position → the apparent solar elevation, at the one
    boundary where PhotoExif is built."""
    import pytest

    from sun_track import solar_elevation_deg

    _write_track(tmp_path)
    exif_gps = _solar_reader_env(monkeypatch, str(tmp_path))
    meta = exif_gps.extract_exif_capture_metadata("synthetic.jpg")
    assert meta.capture_utc is not None
    assert meta.solar_elevation_deg == pytest.approx(
        solar_elevation_deg(53.0, -8.0, meta.capture_utc)
    )
    exif_gps._reset_gpx_cache()


def test_no_gpx_dir_leaves_solar_elevation_none_even_with_a_utc_anchor(monkeypatch):
    """The shipped default ("" = off) degrades bit-identically: anchor kept, no solar."""
    exif_gps = _solar_reader_env(monkeypatch, "")
    meta = exif_gps.extract_exif_capture_metadata("synthetic.jpg")
    assert meta.capture_utc is not None
    assert meta.solar_elevation_deg is None
