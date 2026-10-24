"""``tools/release.py`` writes a release the CHANGELOG gate accepts.

The release bumps the version in ``pyproject.toml`` and
``gomoveshift/__init__.py``, gives ``[Unreleased]`` a dated heading and moves
the link references on, keeping each file's line endings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import release  # noqa: E402

URL = "https://github.com/Back-Road-Creative/framegrade"

CHANGELOG = f"""# Changelog

## [Unreleased]

### Changed

- something new

## [0.3.4] - 2026-10-23

### Fixed

- something old

[Unreleased]: {URL}/compare/v0.3.4...HEAD
[0.3.4]: {URL}/compare/v0.3.3...v0.3.4
"""


def _tree(tmp_path: Path, changelog: str = CHANGELOG, init_newline: str = "\r\n") -> Path:
    (tmp_path / "gomoveshift").mkdir()
    (tmp_path / "pyproject.toml").write_bytes(
        b'[project]\nname = "gomoveshift-video"\nversion = "0.3.4"\n'
        b'[tool.black]\ntarget-version = ["py312"]\n'
    )
    init = f'"""Doc."""{init_newline}{init_newline}__version__ = "0.3.4"{init_newline}'
    (tmp_path / "gomoveshift" / "__init__.py").write_bytes(init.encode())
    (tmp_path / "CHANGELOG.md").write_bytes(changelog.encode())
    return tmp_path


def test_release_bumps_both_version_strings(tmp_path):
    root = _tree(tmp_path)
    release.release(root, "0.3.5", "2026-10-24")
    assert b'version = "0.3.5"\n' in (root / "pyproject.toml").read_bytes()
    assert b'target-version = ["py312"]' in (root / "pyproject.toml").read_bytes()
    assert b'__version__ = "0.3.5"\r\n' in (root / "gomoveshift" / "__init__.py").read_bytes()


def test_release_keeps_crlf_line_endings(tmp_path):
    root = _tree(tmp_path)
    release.release(root, "0.3.5", "2026-10-24")
    init = (root / "gomoveshift" / "__init__.py").read_bytes()
    assert init.count(b"\r\n") == 3
    assert b"\n" not in init.replace(b"\r\n", b"")


def test_release_dates_the_unreleased_entries(tmp_path):
    root = _tree(tmp_path)
    release.release(root, "0.3.5", "2026-10-24")
    text = (root / "CHANGELOG.md").read_text()
    assert "## [Unreleased]\n\n## [0.3.5] - 2026-10-24\n\n### Changed\n\n- something new" in text


def test_release_moves_the_link_references(tmp_path):
    root = _tree(tmp_path)
    release.release(root, "0.3.5", "2026-10-24")
    text = (root / "CHANGELOG.md").read_text()
    assert f"[Unreleased]: {URL}/compare/v0.3.5...HEAD\n" in text
    assert f"[0.3.5]: {URL}/compare/v0.3.4...v0.3.5\n" in text
    assert f"[0.3.4]: {URL}/compare/v0.3.3...v0.3.4\n" in text


def test_release_refuses_an_empty_unreleased_section(tmp_path):
    empty = CHANGELOG.replace("### Changed\n\n- something new\n\n", "")
    root = _tree(tmp_path, changelog=empty)
    with pytest.raises(release.ReleaseError, match="nothing to release"):
        release.release(root, "0.3.5", "2026-10-24")
    assert (root / "CHANGELOG.md").read_text() == empty


@pytest.mark.parametrize("version", ["0.3.4", "0.3.3", "0.3", "v0.3.5"])
def test_release_refuses_a_version_that_does_not_move_forward(tmp_path, version):
    root = _tree(tmp_path)
    with pytest.raises(release.ReleaseError):
        release.release(root, version, "2026-10-24")
    assert b'version = "0.3.4"' in (root / "pyproject.toml").read_bytes()


def test_release_writes_nothing_when_a_file_is_off_pattern(tmp_path):
    root = _tree(tmp_path)
    (root / "gomoveshift" / "__init__.py").write_bytes(b'__version__ = "0.3.3"\n')
    with pytest.raises(release.ReleaseError, match="__init__"):
        release.release(root, "0.3.5", "2026-10-24")
    assert b'version = "0.3.4"' in (root / "pyproject.toml").read_bytes()
    assert (root / "CHANGELOG.md").read_text() == CHANGELOG


def test_the_released_tree_passes_the_changelog_gate(tmp_path, monkeypatch):
    import test_changelog as gate

    root = _tree(tmp_path)
    release.release(root, "0.3.5", "2026-10-24")
    monkeypatch.setattr(gate, "CHANGELOG", root / "CHANGELOG.md")
    monkeypatch.setattr(gate.gomoveshift, "__version__", "0.3.5")
    gate.test_current_version_has_a_heading()
    gate.test_every_heading_has_a_link_reference()
    gate.test_current_version_link_points_at_its_tag()
    gate.test_released_versions_link_to_the_previous_tag()
    gate.test_unreleased_compares_from_the_latest_tag()
    gate.test_headings_are_newest_first()


def test_main_reports_the_bump_and_refusals(tmp_path, capsys):
    root = _tree(tmp_path)
    assert release.main(["0.3.5", "--date", "2026-10-24", "--root", str(root)]) == 0
    assert capsys.readouterr().out == "0.3.4 -> 0.3.5\n"
    assert release.main(["0.3.5", "--root", str(root)]) == 1
    assert "not an X.Y.Z version after 0.3.5" in capsys.readouterr().err
