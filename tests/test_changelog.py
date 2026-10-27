"""CHANGELOG gate: every released version has a heading and a link reference."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import gomoveshift

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import release  # noqa: E402

CHANGELOG = REPO_ROOT / "CHANGELOG.md"
REPO_URL = "https://github.com/Back-Road-Creative/framegrade"


def _text() -> str:
    return CHANGELOG.read_text(encoding="utf-8")


def _headings(text: str) -> list[str]:
    return re.findall(r"^## \[([^\]]+)\]", text, flags=re.MULTILINE)


def _link_refs(text: str) -> dict[str, str]:
    return dict(re.findall(r"^\[([^\]]+)\]:\s*(\S+)\s*$", text, flags=re.MULTILINE))


def test_current_version_has_a_heading() -> None:
    assert gomoveshift.__version__ in _headings(_text())


def test_every_heading_has_a_link_reference() -> None:
    refs = _link_refs(_text())
    missing = [h for h in _headings(_text()) if h not in refs]
    assert not missing, f"CHANGELOG headings without a link reference: {missing}"


def test_current_version_link_points_at_its_tag() -> None:
    url = _link_refs(_text())[gomoveshift.__version__]
    assert url.startswith(f"{REPO_URL}/")
    assert url.endswith(f"v{gomoveshift.__version__}")


def test_released_versions_link_to_the_previous_tag() -> None:
    released = [h for h in _headings(_text()) if h != "Unreleased"]
    refs = _link_refs(_text())
    for newer, older in zip(released, released[1:]):
        assert refs[newer] == f"{REPO_URL}/compare/v{older}...v{newer}"


def test_unreleased_compares_from_the_latest_tag() -> None:
    latest = [h for h in _headings(_text()) if h != "Unreleased"][0]
    assert _link_refs(_text())["Unreleased"] == f"{REPO_URL}/compare/v{latest}...HEAD"


def test_headings_are_newest_first() -> None:
    released = [h for h in _headings(_text()) if h != "Unreleased"]
    keys = [tuple(int(p) for p in h.split(".")) for h in released]
    assert keys == sorted(keys, reverse=True)


def test_every_changelog_fragment_is_releasable() -> None:
    """``tools/release.py`` folds each ``changelog.d/*.md`` into the next release,
    so a fragment it would refuse fails here, in the PR that adds it."""
    for path in release.fragment_paths(REPO_ROOT):
        release.read_fragment(path)
