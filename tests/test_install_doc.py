"""docs/install.md: no unfilled placeholders, a pinned release tag, a measured minimum.

The guide told a reader to `git clone <this repo>`, to "fill in the owner and
revision placeholders" in ci/siblings.txt (already pinned), and gave no minimum
machine. Each claim is checked against the file or function it describes.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from gomoveshift import __version__
from gomoveshift.pipeline.preflight import run_preflight

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "install.md").read_text(encoding="utf-8")


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def test_clone_command_has_a_real_url() -> None:
    clones = re.findall(r"^git clone .*$", DOC, flags=re.MULTILINE)
    assert clones, "install.md has no git clone line"
    for line in clones:
        assert "<" not in line, f"placeholder left in: {line}"
        assert "https://github.com/" in line, f"clone line names no URL: {line}"


def test_clone_line_names_a_release_tag_not_newer_than_this_version() -> None:
    tags = re.findall(r"^git clone .*--branch (v\d+\.\d+\.\d+)\b", DOC, flags=re.MULTILINE)
    assert tags, "the clone line must name a release tag (--branch vX.Y.Z)"
    for tag in tags:
        assert _version_tuple(tag[1:]) <= _version_tuple(__version__), (
            f"{tag} is newer than the package version {__version__}"
        )


def test_placeholder_prose_matches_siblings_file() -> None:
    siblings = (ROOT / "ci" / "siblings.txt").read_text(encoding="utf-8")
    still_placeholders = bool(re.search(r"@(OWNER|SHA)@", siblings))
    if not still_placeholders:
        assert "placeholder" not in DOC.lower(), (
            "install.md tells the reader to fill placeholders, but ci/siblings.txt is pinned"
        )


def test_minimum_machine_section_matches_preflight_defaults() -> None:
    match = re.search(r"^## Minimum machine\n(.*?)(?=^## )", DOC, flags=re.MULTILINE | re.DOTALL)
    assert match, "install.md needs a '## Minimum machine' section"
    section = match.group(1)
    params = inspect.signature(run_preflight).parameters
    for name in ("min_free_gb", "min_staging_gb", "min_ram_gb"):
        value = params[name].default
        shown = f"{value:g}"
        assert re.search(rf"\b{re.escape(shown)} GB\b", section), (
            f"section does not state {shown} GB for {name}"
        )


def test_minimum_machine_names_only_the_command_that_runs_preflight() -> None:
    match = re.search(r"^## Minimum machine\n(.*?)(?=^## )", DOC, flags=re.MULTILINE | re.DOTALL)
    assert match
    section = match.group(1)
    assert "gomoveshift drive" in section
    assert "gomoveshift pipeline" not in section, (
        "`pipeline` never calls run_preflight; the section must not claim it does"
    )
    assert "naming the one that failed" not in section, "run_preflight lists every failure"
    assert "GPU below" not in section, "the NVENC section is above, not below"
