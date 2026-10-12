"""``SEOConfig.validate_paths_exist``.

Contract: returns ``None`` when storage_path's parent dir + manager_path
both exist; raises ``ConfigError`` listing every missing path otherwise.
``credentials_dir`` is *not* validated — see ``SEOConfig.validate_paths_exist``
docstring for rationale. Honors ``seo.skip_validation``.
"""

from __future__ import annotations

import pytest

from gomoveshift.config import ConfigError, SEOConfig

# Opt out of the conftest no-op stub so we exercise the real validator.
pytestmark = pytest.mark.real_seo_validation


def _make_config(tmp_path, *, storage_present=True, creds_present=True, manager_present=True):
    """Build a SEOConfig pointed at tmp_path subdirectories.

    Each path's on-disk presence is independently controlled.
    """
    storage_parent = tmp_path / "seo-storage"
    creds_dir = tmp_path / "credentials"
    manager_dir = tmp_path / "manager"
    if storage_present:
        storage_parent.mkdir()
    if creds_present:
        creds_dir.mkdir()
    if manager_present:
        manager_dir.mkdir()
    return SEOConfig(
        storage_path=str(storage_parent / "seo"),  # file inside storage_parent
        credentials_dir=str(creds_dir),
        manager_path=str(manager_dir),
    )


class TestHappyPath:
    def test_all_paths_present_returns_none(self, tmp_path):
        assert _make_config(tmp_path).validate_paths_exist() is None

    def test_storage_file_itself_may_be_absent(self, tmp_path):
        """The validator checks the *parent* of storage_path, not the file.

        ``events.jsonl`` does not exist on a fresh install — only the parent
        directory needs to be present.
        """
        # storage_path points to ``tmp_path/seo-storage/seo`` which does NOT
        # exist as a file/dir, but its parent ``tmp_path/seo-storage`` does.
        assert _make_config(tmp_path).validate_paths_exist() is None


class TestMissingPaths:
    def test_missing_storage_parent_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="seo-storage"):
            _make_config(tmp_path, storage_present=False).validate_paths_exist()

    def test_missing_credentials_dir_does_not_raise(self, tmp_path):
        """``credentials_dir`` is no longer validated — the SOPS path uses
        ``creds_dir.parent`` and BrandCredentials self-guards. Requiring
        the dir on disk caused a 30-minute mid-launch interruption.
        """
        cfg = _make_config(tmp_path, creds_present=False)
        assert cfg.validate_paths_exist() is None

    def test_missing_manager_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="manager"):
            _make_config(tmp_path, manager_present=False).validate_paths_exist()

    def test_message_lists_every_missing_path(self, tmp_path):
        """Fail-fast value: operator sees the full repair list in one shot.

        Two validated paths now (credentials_dir was dropped): storage parent
        + manager_path.
        """
        cfg = _make_config(
            tmp_path, storage_present=False, creds_present=False, manager_present=False
        )
        with pytest.raises(ConfigError) as exc_info:
            cfg.validate_paths_exist()
        msg = str(exc_info.value)
        assert "seo-storage" in msg
        assert "manager" in msg
        assert "credentials" not in msg  # no longer validated
        # One line per missing path → at least 2 newlines (header + 2 paths).
        assert msg.count("\n") >= 2


class TestSkipValidation:
    def test_skip_validation_short_circuits(self, tmp_path):
        cfg = _make_config(
            tmp_path, storage_present=False, creds_present=False, manager_present=False
        )
        cfg.skip_validation = True
        # Despite every path being absent, the validator returns cleanly.
        assert cfg.validate_paths_exist() is None

    def test_skip_validation_default_is_false(self):
        assert SEOConfig().skip_validation is False
