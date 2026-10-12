"""Guards against crashing on malformed ffprobe/YAML output.

Covers one malformed-input case per genuinely-unguarded call class:
- ffprobe garbage JSON via the shared ``safe_json_loads`` helper
  (``video_detect.detect_video_properties`` + ``color_analysis.get_video_properties``);
- truncated/invalid YAML in ``Config.load``.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.config import Config, ConfigError
from gomoveshift.pipeline._json_safe import safe_json_loads


class TestSafeJsonLoads:
    """The shared helper raises a contextual RuntimeError, not a bare decode error."""

    def test_parses_valid_json(self):
        assert safe_json_loads('{"a": 1}', "ctx") == {"a": 1}

    def test_malformed_raises_runtime_error_with_context(self):
        with pytest.raises(RuntimeError, match="ffprobe probe: malformed JSON"):
            safe_json_loads("not json at all }{", "ffprobe probe")


def _mock_run(stdout: str):
    result = MagicMock()
    result.returncode = 0
    result.stdout = stdout
    result.stderr = ""
    return result


class TestVideoDetectGuard:
    """detect_video_properties surfaces a contextual RuntimeError on ffprobe garbage."""

    def test_garbage_json_raises_runtime_error(self, tmp_path):
        from gomoveshift.pipeline import video_detect

        with patch.object(
            video_detect.subprocess, "run", return_value=_mock_run("<<garbage not json>>")
        ):
            with pytest.raises(RuntimeError, match="malformed JSON"):
                video_detect.detect_video_properties(tmp_path / "v.mp4")


class TestColorAnalysisGuard:
    """get_video_properties surfaces a contextual RuntimeError on ffprobe garbage."""

    def test_garbage_json_raises_runtime_error(self, tmp_path):
        from gomoveshift.pipeline import color_analysis

        with patch.object(
            color_analysis.subprocess, "run", return_value=_mock_run("totally not json")
        ):
            with pytest.raises(RuntimeError, match="malformed JSON"):
                color_analysis.get_video_properties(tmp_path / "v.mp4")


class TestConfigYamlGuard:
    """Config.load raises ConfigError (not a raw YAMLError) on truncated/invalid YAML."""

    def test_invalid_yaml_explicit_path_raises_config_error(self, tmp_path):
        bad = tmp_path / "config.yaml"
        # Unbalanced bracket — yaml.safe_load raises YAMLError.
        bad.write_text("drive: [unterminated\n  nested: : :\n")
        with pytest.raises(ConfigError, match="malformed YAML"):
            Config.load(config_path=bad)

    def test_invalid_yaml_search_path_raises_config_error(self, tmp_path, monkeypatch):
        bad = tmp_path / "config.yaml"
        bad.write_text(":\n: invalid\n\t- mixed\n")
        monkeypatch.chdir(tmp_path)
        # Force the user-path branch to be skipped so only the project file is read.
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "nonexistent_home")
        with pytest.raises(ConfigError, match="malformed YAML"):
            Config.load()
