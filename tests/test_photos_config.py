"""``PhotosConfig`` — de-hardcoded photo-pipeline paths.

The photo pipeline baked its site root, selection-state file and resized-output
base in as module-level absolute literals with no override, and read
``GMS_PHOTO_OUTPUT_DIR`` / ``GMS_EXPORT_DIR`` ad-hoc in violation of the
``gomoveshift.paths`` contract. These tests pin the replacement: defaults land
under the per-user data home, env/config overrides win, and no absolute
literal survives in the photo modules.
"""

import re
from pathlib import Path

import pytest

from gomoveshift import paths
from gomoveshift.config import Config, ConfigError, PhotosConfig, SEOConfig

# Defaults land under the per-user data home (``GMS_DATA_HOME``).
_ISOLATED_VARS = (
    "GMS_PHOTO_OUTPUT_DIR",
    "GMS_EXPORT_DIR",
    "GMS_DATA_HOME",
    "XDG_DATA_HOME",
)


@pytest.fixture
def data_home(monkeypatch, tmp_path):
    for var in _ISOLATED_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GMS_DATA_HOME", str(tmp_path / "data"))
    return tmp_path / "data"


class TestPhotosConfigDefaults:
    def test_config_exposes_photos_section(self):
        assert isinstance(Config().photos, PhotosConfig)

    def test_defaults_resolve_under_the_data_home(self, data_home):
        cfg = PhotosConfig()
        assert cfg.resolve_output_dir() == data_home / "photos"
        assert cfg.resolve_state_path() == data_home / "photos" / ".photo_selection_state.json"
        assert cfg.resolve_resized_base() == data_home / "photos" / "resized"

    def test_site_root_has_no_default(self):
        """The Hugo site checkout is deployment-specific: unset means
        "not configured", and the error names the key to set."""
        cfg = PhotosConfig()
        assert cfg.site_root == ""
        with pytest.raises(ConfigError, match="photos.site_root"):
            cfg.resolve_site_root()

    def test_site_root_and_resized_base_overrides_win(self, tmp_path):
        cfg = PhotosConfig(site_root=str(tmp_path / "site"), resized_base=str(tmp_path / "resized"))
        assert cfg.resolve_site_root() == tmp_path / "site"
        assert cfg.resolve_resized_base() == tmp_path / "resized"

    def test_default_encode_knobs(self):
        cfg = PhotosConfig()
        assert cfg.target_width == 1080
        assert cfg.jpeg_quality == 85

    def test_no_enhance_model_slots(self):
        """Reserved slots consumed by nothing were deleted outright,
        not left dead — re-add them when an enhance model actually ships."""
        cfg = PhotosConfig()
        assert not hasattr(cfg, "enhance_model_path")
        assert not hasattr(cfg, "enhance_model_url")
        assert not hasattr(cfg, "enhance_model_sha256")


class TestEnvOverrides:
    def test_paths_module_exposes_photo_accessors(self, data_home):
        assert paths.gms_photo_output_dir() == data_home / "photos"
        assert paths.gms_export_dir() == data_home / "output"

    def test_env_override_wins_over_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GMS_PHOTO_OUTPUT_DIR", str(tmp_path / "photos"))
        monkeypatch.setenv("GMS_EXPORT_DIR", str(tmp_path / "exports"))
        assert paths.gms_photo_output_dir() == tmp_path / "photos"
        assert paths.gms_export_dir() == tmp_path / "exports"
        cfg = PhotosConfig()
        assert cfg.resolve_output_dir() == tmp_path / "photos"
        # State file rides the photo output dir so the two can never diverge.
        assert cfg.resolve_state_path() == tmp_path / "photos" / ".photo_selection_state.json"

    def test_explicit_config_value_beats_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GMS_PHOTO_OUTPUT_DIR", str(tmp_path / "from-env"))
        cfg = PhotosConfig(output_dir=str(tmp_path / "from-config"))
        assert cfg.resolve_output_dir() == tmp_path / "from-config"


@pytest.mark.real_seo_validation
class TestStoragePathMustBeAbsolute:
    """The content ledger derives from ``seo.storage_path``; a relative value
    silently selects a different ledger per cwd, voiding slot-collision avoidance."""

    def test_relative_storage_path_rejected(self, tmp_path):
        cfg = SEOConfig(storage_path="./data/seo", manager_path=str(tmp_path))
        with pytest.raises(ConfigError, match="absolute"):
            cfg.validate_paths_exist()

    def test_absolute_storage_path_accepted(self, tmp_path):
        (tmp_path / "seo-storage").mkdir()
        cfg = SEOConfig(
            storage_path=str(tmp_path / "seo-storage" / "seo"),
            manager_path=str(tmp_path),
        )
        assert cfg.validate_paths_exist() is None


class TestSelectionStateUnknownField:
    """A renamed/removed field makes ``cls(**data)`` raise TypeError. The old
    behavior silently reset the posted-photos ledger; the load now fails
    closed instead — see TestSelectionStateFailClosed in test_photo_selector."""

    def test_load_fails_closed_on_unknown_field(self, tmp_path):
        from gomoveshift.pipeline.photos.selector import (
            PhotoSelectionState,
            PhotoSelectionStateError,
        )

        state_path = tmp_path / "state.json"
        state_path.write_text('{"posted_photos": ["a/photo-001.jpg"], "future_field": 1}')
        with pytest.raises(PhotoSelectionStateError, match="--reset-selection-state"):
            PhotoSelectionState.load(state_path)


class TestNoAbsoluteLiteralsRemain:
    """Regression guard: the same hardcoded workspace path has been removed from
    this pipeline twice already. Fail the build if it returns."""

    @pytest.mark.parametrize("module_name", ["selector", "resizer"])
    def test_photo_module_has_no_absolute_path_literal(self, module_name):
        import importlib

        module = importlib.import_module(f"gomoveshift.pipeline.photos.{module_name}")
        source = Path(module.__file__).read_text(encoding="utf-8")
        literals = re.findall(r"""["'](/[^"'\n]*)["']""", source)
        assert literals == [], f"absolute path literal(s) in {module_name}.py: {literals}"

    @pytest.mark.parametrize("module_name", ["selector", "resizer"])
    def test_photo_module_resolves_paths_from_config(self, monkeypatch, tmp_path, module_name):
        """No import-time path evaluation: a config swap after import wins."""
        import gomoveshift.config as config_module

        photos = PhotosConfig(
            site_root=str(tmp_path / "site"),
            resized_base=str(tmp_path / "resized"),
            target_width=640,
            jpeg_quality=70,
        )
        monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))
        if module_name == "selector":
            from gomoveshift.pipeline.photos.selector import PhotoSelector

            assert PhotoSelector().site_root == tmp_path / "site"
        else:
            from gomoveshift.pipeline.photos.resizer import PhotoResizer

            resizer = PhotoResizer()
            assert resizer.output_base == tmp_path / "resized"
            assert resizer.target_width == 640
            assert resizer.quality == 70
