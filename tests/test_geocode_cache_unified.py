"""The geocode cache is ONE namespace per run.

Three phases once resolved the reverse-geocode cache to three
different directories:

* Phase A  — episode-location GPX derivation — ``GMS_GPS_CACHE_DIR`` or
  a shared default cache directory.
* Phase BC — per-LF centroid day labels — ``DriveConfig.gps_cache_dir``
  (``~/.cache/gomoveshift/gps``), which is NOT env-backed.
* Phase D  — chapters — ``geocoding_cache_dir`` / ``GMS_GEOCODING_CACHE_DIR``.

So the SAME coordinate could cache (and therefore resolve) differently
across the title, the day label, the ``Through:`` line, and the chapters
within a single run. These tests pin that all three consumers resolve to
ONE directory, routed through ``DriveConfig.resolved_geocode_cache_dir``.
"""

from pathlib import Path
from unittest.mock import patch

from gomoveshift import paths


def _cache_parent(mock_cache) -> Path:
    """Directory of the ``cache_path`` handed to GeocodingCache."""
    ca = mock_cache.call_args
    cache_path = ca.kwargs.get("cache_path")
    if cache_path is None and ca.args:
        cache_path = ca.args[0]
    return Path(cache_path).parent.resolve()


def _phase_a_cache_dir() -> Path:
    with (
        patch("gps_journey.geocoding.cache.GeocodingCache") as m,
        patch("gps_journey.geocoding.nominatim_roads.NominatimRoadsClient"),
    ):
        from gomoveshift.pipeline.ingest.geocoding import _get_shared_geocoder

        _get_shared_geocoder()
    assert m.called, "Phase A geocoder never built a GeocodingCache"
    return _cache_parent(m)


def _phase_bc_cache_dir() -> Path:
    # The no-arg build is the ``--list-lfs`` / fallback production path;
    # the fix routes its default through resolved_geocode_cache_dir().
    with (
        patch("gps_journey.geocoding.cache.GeocodingCache") as m,
        patch("gps_journey.geocoding.nominatim_roads.NominatimRoadsClient"),
    ):
        from gomoveshift.pipeline.gps_enrichment import _build_shared_geocoder

        _build_shared_geocoder()
    assert m.called, "Phase BC geocoder never built a GeocodingCache"
    return _cache_parent(m)


def _phase_d_cache_dir(config) -> Path:
    with (
        patch("gomoveshift.pipeline.chapters.generator.NominatimRoadsClient"),
        patch("gomoveshift.pipeline.chapters.generator.GeocodingCache") as m,
    ):
        from gomoveshift.pipeline.chapters.generator import ChapterGenerator

        ChapterGenerator(config=config)
    assert m.called, "Phase D geocoder never built a GeocodingCache"
    return _cache_parent(m)


def test_three_geocode_consumers_share_one_cache_dir(tmp_path, monkeypatch):
    """Phase A, Phase BC, and Phase D must resolve the reverse-geocode
    cache to the SAME directory for a given config/env."""
    from gomoveshift.config import Config

    shared = tmp_path / "geo-cache"
    shared.mkdir()
    # A single env var must steer ALL three consumers at one directory.
    monkeypatch.setenv("GMS_GPS_CACHE_DIR", str(shared))
    monkeypatch.delenv("GMS_GEOCODING_CACHE_DIR", raising=False)

    config = Config()

    phase_a = _phase_a_cache_dir()
    phase_bc = _phase_bc_cache_dir()
    phase_d = _phase_d_cache_dir(config)

    assert phase_a == phase_bc == phase_d == shared.resolve(), (
        f"geocode cache namespaces diverge: phase_a={phase_a} phase_bc={phase_bc} phase_d={phase_d}"
    )


def test_resolved_geocode_cache_dir_is_the_single_source(tmp_path, monkeypatch):
    """The canonical resolver honors the legacy env var and otherwise
    falls back to the config default (never the process CWD)."""
    from gomoveshift.config import DriveConfig

    custom = tmp_path / "env-dir"
    monkeypatch.setenv("GMS_GPS_CACHE_DIR", str(custom))
    monkeypatch.delenv("GMS_GEOCODING_CACHE_DIR", raising=False)
    assert DriveConfig().resolved_geocode_cache_dir() == custom

    monkeypatch.delenv("GMS_GPS_CACHE_DIR", raising=False)
    resolved = DriveConfig().resolved_geocode_cache_dir()
    assert resolved.is_absolute()
    assert resolved == paths.gms_geocoding_cache_dir()
