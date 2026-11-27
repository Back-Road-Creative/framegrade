"""CHANGELOG gate: every released version has a heading and a link reference."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import gomoveshift
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import release  # noqa: E402

CHANGELOG = REPO_ROOT / "CHANGELOG.md"
REPO_URL = "https://github.com/Back-Road-Creative/framegrade"
#: Name prefixes the workspaces merge robot holds for a human (its hard-deny
#: globs ``**/deploy*``, ``**/publish*``, ``**/upload*``). A release PR deletes
#: every fragment, so one so named stops the release itself: 0.3.116 (#445)
#: stopped on ``changelog.d/upload-sends-recording-facts.md``.
HELD_NAME_PREFIXES = ("deploy", "publish", "upload")


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


def _held_by_merge_robot(name: str) -> bool:
    return name.lower().startswith(HELD_NAME_PREFIXES)


@pytest.mark.parametrize(
    ("name", "held"),
    [
        ("upload-sends-recording-facts.md", True),
        ("publish-x.md", True),
        ("deploy-x.md", True),
        ("Upload-x.md", True),
        ("sends-upload-facts.md", False),
        ("heal-retry-by-fingerprint.md", False),
    ],
)
def test_held_name_matches_the_robots_globs_from_the_start(name: str, held: bool) -> None:
    assert _held_by_merge_robot(name) is held


def test_no_changelog_fragment_name_is_held_by_the_merge_robot() -> None:
    """Fails the PR that adds such a name, before it can stop a release PR."""
    held = [p.name for p in release.fragment_paths(REPO_ROOT) if _held_by_merge_robot(p.name)]
    assert held == [], f"rename these changelog.d fragments: {held}"


def _unreleased_body(text: str) -> str:
    match = re.search(r"^## \[Unreleased\]\n(.*?)(?=^## \[)", text, flags=re.MULTILINE | re.DOTALL)
    assert match, "CHANGELOG.md has no [Unreleased] section before a released version"
    return match.group(1)


def test_unreleased_holds_no_entries() -> None:
    """The release robot cuts only when ``changelog.d`` holds a fragment, so an
    entry written under ``[Unreleased]`` instead never starts a release: #687
    (day reel site URL) merged on 2026-11-27 and no release was cut for it."""
    body = _unreleased_body(_text()).strip()
    assert body == "", (
        "move the [Unreleased] entries in CHANGELOG.md to a changelog.d/<name>.md fragment"
    )


def test_unreleased_entries_are_found() -> None:
    text = "## [Unreleased]\n\n### Added\n\n- x\n\n## [0.0.1] - 2026-01-01\n"
    assert _unreleased_body(text).strip() == "### Added\n\n- x"
    assert _unreleased_body("## [Unreleased]\n\n## [0.0.1] - 2026-01-01\n").strip() == ""
