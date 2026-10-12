"""Env overrides win for paths; defaults resolve under the per-user data,
cache and config homes; validation raises with an actionable message when
required dirs are missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gomoveshift import paths as gms_paths


def test_env_override_wins_for_staging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GMS_STAGING_DIR", str(tmp_path / "stage"))
    assert gms_paths.gms_staging_dir() == tmp_path / "stage"


def test_env_override_wins_for_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path / "out"))
    assert gms_paths.gms_output_dir() == tmp_path / "out"


def test_assert_required_paths_raises_on_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "does-not-exist"
    monkeypatch.setenv("GMS_STAGING_DIR", str(missing))
    monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path))  # exists
    with pytest.raises(FileNotFoundError) as exc:
        gms_paths.assert_required_paths_exist()
    msg = str(exc.value)
    assert "GMS_STAGING_DIR" in msg
    assert str(missing) in msg
    # Actionable: tells the operator what to do.
    assert "Set the env var or create the directory" in msg


def test_assert_required_paths_passes_when_all_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GMS_STAGING_DIR", str(tmp_path))
    monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path))
    gms_paths.assert_required_paths_exist()  # no raise


_HOME_VARS = (
    "GMS_DATA_HOME",
    "GMS_CACHE_HOME",
    "GMS_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "GMS_STAGING_DIR",
    "GMS_OUTPUT_DIR",
    "GMS_PHOTO_OUTPUT_DIR",
    "GMS_EXPORT_DIR",
    "GMS_INTERMEDIATE_DIR",
    "GMS_OVERPASS_CACHE_DIR",
    "GMS_ROUTE_STORY_CACHE_DIR",
)


@pytest.fixture
def clean_homes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Clear every path env var and point ``Path.home`` at a temp dir."""
    for var in _HOME_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


@pytest.mark.parametrize(
    ("helper", "gms_var", "xdg_var", "home_rel"),
    [
        ("gms_data_home", "GMS_DATA_HOME", "XDG_DATA_HOME", ".local/share"),
        ("gms_cache_home", "GMS_CACHE_HOME", "XDG_CACHE_HOME", ".cache"),
        ("gms_config_home", "GMS_CONFIG_HOME", "XDG_CONFIG_HOME", ".config"),
    ],
)
def test_home_helpers_precedence(
    monkeypatch: pytest.MonkeyPatch,
    clean_homes: Path,
    helper: str,
    gms_var: str,
    xdg_var: str,
    home_rel: str,
) -> None:
    fn = getattr(gms_paths, helper)
    # 1. Nothing set: under the invoking user's home.
    assert fn() == clean_homes / home_rel / "gomoveshift"
    # 2. XDG variable set: XDG root plus the app name.
    monkeypatch.setenv(xdg_var, str(clean_homes / "xdg"))
    assert fn() == clean_homes / "xdg" / "gomoveshift"
    # 3. The GMS variable wins over XDG and is used as given.
    monkeypatch.setenv(gms_var, str(clean_homes / "explicit"))
    assert fn() == clean_homes / "explicit"


def test_defaults_resolve_under_data_cache_homes(clean_homes: Path) -> None:
    data = clean_homes / ".local/share/gomoveshift"
    cache = clean_homes / ".cache/gomoveshift"
    assert gms_paths.gms_staging_dir() == data / "input"
    assert gms_paths.gms_output_dir() == data / "output"
    assert gms_paths.gms_photo_output_dir() == data / "photos"
    assert gms_paths.gms_export_dir() == data / "output"
    assert gms_paths.gms_intermediate_base() == cache
    assert gms_paths.gms_overpass_cache_dir() == cache / "overpass"
    assert gms_paths.gms_route_story_cache_dir() == cache / "route-story"


def test_data_home_moves_every_data_default(
    monkeypatch: pytest.MonkeyPatch, clean_homes: Path
) -> None:
    monkeypatch.setenv("GMS_DATA_HOME", str(clean_homes / "d"))
    assert gms_paths.gms_staging_dir() == clean_homes / "d" / "input"
    assert gms_paths.gms_output_dir() == clean_homes / "d" / "output"
    assert gms_paths.gms_photo_output_dir() == clean_homes / "d" / "photos"
    assert gms_paths.gms_export_dir() == clean_homes / "d" / "output"


def test_specific_env_overrides_beat_the_homes(
    monkeypatch: pytest.MonkeyPatch, clean_homes: Path
) -> None:
    monkeypatch.setenv("GMS_DATA_HOME", str(clean_homes / "d"))
    monkeypatch.setenv("GMS_CACHE_HOME", str(clean_homes / "c"))
    monkeypatch.setenv("GMS_PHOTO_OUTPUT_DIR", str(clean_homes / "p"))
    monkeypatch.setenv("GMS_EXPORT_DIR", str(clean_homes / "e"))
    monkeypatch.setenv("GMS_INTERMEDIATE_DIR", str(clean_homes / "i"))
    monkeypatch.setenv("GMS_OVERPASS_CACHE_DIR", str(clean_homes / "o"))
    monkeypatch.setenv("GMS_ROUTE_STORY_CACHE_DIR", str(clean_homes / "r"))
    assert gms_paths.gms_photo_output_dir() == clean_homes / "p"
    assert gms_paths.gms_export_dir() == clean_homes / "e"
    assert gms_paths.gms_intermediate_base() == clean_homes / "i"
    assert gms_paths.gms_overpass_cache_dir() == clean_homes / "o"
    assert gms_paths.gms_route_story_cache_dir() == clean_homes / "r"


def test_empty_overpass_cache_env_means_no_cache(
    monkeypatch: pytest.MonkeyPatch, clean_homes: Path
) -> None:
    monkeypatch.setenv("GMS_OVERPASS_CACHE_DIR", "")
    assert gms_paths.gms_overpass_cache_dir() is None


def test_legacy_default_names_resolve_lazily(
    monkeypatch: pytest.MonkeyPatch, clean_homes: Path
) -> None:
    monkeypatch.setenv("GMS_CACHE_HOME", str(clean_homes / "c"))
    assert gms_paths._DEFAULT_INTERMEDIATE_DIR == str(clean_homes / "c")
    assert gms_paths._DEFAULT_OVERPASS_CACHE_DIR == str(clean_homes / "c" / "overpass")
    assert gms_paths._DEFAULT_ROUTE_STORY_CACHE_DIR == str(clean_homes / "c" / "route-story")
    with pytest.raises(AttributeError):
        gms_paths._NOT_A_DEFAULT  # noqa: B018
