"""The operator's inputs live in ``GMS_INPUTS_DIR``, outside the publishable tree."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift import paths
from gomoveshift.console import route_story
from gomoveshift.pipeline import playlist_reconciler as pr


def test_inputs_dir_defaults_to_operator_inputs_under_the_data_home(monkeypatch, tmp_path):
    monkeypatch.delenv("GMS_INPUTS_DIR", raising=False)
    monkeypatch.setenv("GMS_DATA_HOME", str(tmp_path))
    assert paths.gms_inputs_dir() == tmp_path / "operator-inputs"


def test_default_inputs_dir_does_not_stat_the_filesystem(monkeypatch, tmp_path):
    # Tests patch pathlib.Path.stat to fake a file's size; resolving the default
    # (a fresh install has no inputs dir yet) must not go through it.
    monkeypatch.delenv("GMS_INPUTS_DIR", raising=False)
    monkeypatch.setenv("GMS_DATA_HOME", str(tmp_path))
    with patch("pathlib.Path.stat", return_value=MagicMock(st_size=0)):
        assert paths.gms_inputs_dir() == tmp_path / "operator-inputs"


def test_inputs_dir_env_overrides_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_INPUTS_DIR", str(tmp_path))
    assert paths.gms_inputs_dir() == tmp_path


def test_inputs_dir_env_naming_a_missing_directory_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_INPUTS_DIR", str(tmp_path / "missing"))
    with pytest.raises(paths.InputsDirError, match="GMS_INPUTS_DIR"):
        paths.gms_inputs_dir()


def test_route_story_profiles_are_read_from_the_inputs_dir(monkeypatch, tmp_path):
    profiles = tmp_path / "route_story_profiles"
    profiles.mkdir()
    (profiles / "trip.yaml").write_text("{}\n")
    monkeypatch.setenv("GMS_INPUTS_DIR", str(tmp_path))
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "data"))
    assert route_story.list_profiles() == ["trip.yaml"]
    assert route_story.resolve_profile("trip.yaml") == profiles / "trip.yaml"


def test_relative_overrides_path_resolves_in_the_inputs_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_INPUTS_DIR", str(tmp_path))
    assert pr.resolve_overrides_path("overrides.json") == tmp_path / "overrides.json"
    assert pr.resolve_overrides_path("/abs/x.json") == Path("/abs/x.json")


def test_config_load_reads_the_inputs_dir_config(monkeypatch, tmp_path):
    """No ./config.yaml in the launch directory: the project tier is the inputs
    dir's config.yaml, so a run from any cwd loads the operator's config."""
    from gomoveshift.config import Config

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "config.yaml").write_text("drive:\n  best_moments_percentile: 71.0\n")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("GMS_INPUTS_DIR", str(inputs))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "nohome"))
    assert Config.load().drive.best_moments_percentile == 71.0


def test_config_load_prefers_a_launch_directory_config(monkeypatch, tmp_path):
    from gomoveshift.config import Config

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "config.yaml").write_text("drive:\n  best_moments_percentile: 71.0\n")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "config.yaml").write_text("drive:\n  best_moments_percentile: 72.0\n")
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("GMS_INPUTS_DIR", str(inputs))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "nohome"))
    assert Config.load().drive.best_moments_percentile == 72.0
