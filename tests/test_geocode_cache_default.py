"""The geocode cache default lives in the user cache home, not in a GPS checkout.

The single geocode-cache namespace used to default to a directory inside another
repository's checkout. The default is now ``~/.cache/gomoveshift/geocoding`` (the
same home as the overpass and route-story caches in ``gomoveshift.paths``).
While a cache file exists at an earlier location configured in
``paths._LEGACY_GEOCODING_CACHE_DIR`` it is still used, even if the new default
also holds one, so a box with a warm cache resolves every coordinate exactly as
before and no live data has to move.
"""

import pytest

from gomoveshift import paths

_CACHE_FILE = "geocoding_cache.json"


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    legacy = tmp_path / "legacy-gps-cache"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("GMS_GEOCODING_CACHE_DIR", raising=False)
    monkeypatch.delenv("GMS_GPS_CACHE_DIR", raising=False)
    monkeypatch.setattr(paths, "_LEGACY_GEOCODING_CACHE_DIR", str(legacy))
    return home / ".cache" / "gomoveshift" / "geocoding", legacy


def test_default_is_the_gomoveshift_user_cache(isolated):
    new, _legacy = isolated
    assert paths.gms_geocoding_cache_dir() == new


def test_default_stays_under_the_user_home(isolated):
    new, _legacy = isolated
    assert paths.gms_geocoding_cache_dir().is_relative_to(new.parents[2])


def test_a_legacy_cache_file_wins_while_it_exists(isolated):
    """A small stray cache at the new default must not cold-start the warm
    legacy cache mid-campaign: the legacy file wins until it is moved."""
    new, legacy = isolated
    legacy.mkdir()
    (legacy / _CACHE_FILE).write_text("{}", encoding="utf-8")
    assert paths.gms_geocoding_cache_dir() == legacy

    new.mkdir(parents=True)
    (new / _CACHE_FILE).write_text("{}", encoding="utf-8")
    assert paths.gms_geocoding_cache_dir() == legacy

    (legacy / _CACHE_FILE).unlink()
    assert paths.gms_geocoding_cache_dir() == new


def test_an_empty_legacy_dir_is_not_a_warm_cache(isolated):
    new, legacy = isolated
    legacy.mkdir()
    assert paths.gms_geocoding_cache_dir() == new


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_env_falls_back_to_the_default(isolated, monkeypatch, value):
    new, _legacy = isolated
    monkeypatch.setenv("GMS_GEOCODING_CACHE_DIR", value)
    assert paths.gms_geocoding_cache_dir() == new


def test_env_wins(isolated, monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_GEOCODING_CACHE_DIR", str(tmp_path / "custom"))
    assert paths.gms_geocoding_cache_dir() == tmp_path / "custom"


def test_drive_config_resolves_through_the_helper(isolated):
    from gomoveshift.config import DriveConfig

    new, _legacy = isolated
    assert DriveConfig().resolved_geocode_cache_dir() == new
