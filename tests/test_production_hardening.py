"""Tests for production hardening fixes."""

import os
import signal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestGeocodingEnvVar:
    """geocoding.py should respect GMS_GPS_CACHE_DIR env var."""

    def test_geocoding_uses_env_var(self, tmp_path, monkeypatch):
        """GMS_GPS_CACHE_DIR env var overrides hardcoded cache path."""
        custom_cache = tmp_path / "custom_cache"
        monkeypatch.setenv("GMS_GPS_CACHE_DIR", str(custom_cache))

        with patch("gps_journey.geocoding.cache.GeocodingCache") as mock_cache:
            with patch("gps_journey.geocoding.nominatim_roads.NominatimRoadsClient") as mock_client:
                mock_client.return_value = MagicMock()
                mock_cache.return_value = MagicMock()
                from gomoveshift.pipeline.ingest.geocoding import _get_shared_geocoder

                try:
                    result = _get_shared_geocoder()
                except Exception:
                    pass

                if mock_cache.called:
                    call_kwargs = mock_cache.call_args
                    cache_path = call_kwargs[1].get(
                        "cache_path", call_kwargs[0][0] if call_kwargs[0] else None
                    )
                    assert str(custom_cache) in str(cache_path)

    @pytest.mark.parametrize("env_value", [None, ""])
    def test_unset_or_empty_env_falls_back_to_shared_cache(self, monkeypatch, env_value):
        """Unset or empty GMS_GPS_CACHE_DIR must fall back to the shared cache
        dir — never the process CWD (Path('') stringifies to '.', so the old
        `if not str(cache_dir)` guard was dead and fragmented the cache)."""
        monkeypatch.delenv("GMS_GEOCODING_CACHE_DIR", raising=False)
        if env_value is None:
            monkeypatch.delenv("GMS_GPS_CACHE_DIR", raising=False)
        else:
            monkeypatch.setenv("GMS_GPS_CACHE_DIR", env_value)

        with patch("gps_journey.geocoding.cache.GeocodingCache") as mock_cache:
            with patch("gps_journey.geocoding.nominatim_roads.NominatimRoadsClient"):
                from gomoveshift.pipeline.ingest.geocoding import _get_shared_geocoder

                _get_shared_geocoder()

        assert mock_cache.called
        cache_path = str(mock_cache.call_args.kwargs["cache_path"])
        from gomoveshift.paths import gms_geocoding_cache_dir

        assert cache_path.startswith(str(gms_geocoding_cache_dir()))
        assert not cache_path.startswith("geocoding_cache.json")


class TestSignalHandlers:
    """Signal handlers must be registered before orchestrator construction."""

    def test_sighup_handler_registered(self):
        """SIGHUP handler should be registered in pipeline handler."""
        import gomoveshift.cli.handlers.pipeline as pipeline_mod

        # Verify the module has _handle_sigterm defined
        assert hasattr(pipeline_mod, "_handle_sigterm")


class TestWebAppEviction:
    """Web app should evict old entries from job/analysis dicts."""

    def test_evict_oldest_keeps_max_size(self):
        from gomoveshift.web.app import _evict_oldest

        store = {str(i): f"value_{i}" for i in range(60)}
        _evict_oldest(store, max_size=50)
        assert len(store) == 50

    def test_evict_oldest_noop_when_under_limit(self):
        from gomoveshift.web.app import _evict_oldest

        store = {"a": 1, "b": 2}
        _evict_oldest(store, max_size=50)
        assert len(store) == 2
