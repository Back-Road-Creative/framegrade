"""Every relative link in the repo's Markdown resolves to a file that exists.

A link is resolved against the file it sits in, the way GitHub renders it, so
`[install](install.md)` inside docs/pipeline/ points at docs/pipeline/install.md.
docs/pipeline/96-packaging.md linked the install guide that way and 404'd.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


def _markdown() -> list[Path]:
    return sorted([*ROOT.glob("*.md"), *(ROOT / "docs").rglob("*.md")])


def broken_links(files: list[Path], root: Path = ROOT) -> list[str]:
    out = []
    for md in files:
        for target in _LINK.findall(md.read_text(encoding="utf-8")):
            if re.match(r"[a-z][a-z0-9+.-]*:", target):  # http:, mailto:, ...
                continue
            if not (md.parent / target).exists():
                out.append(f"{md.relative_to(root)} -> {target}")
    return out


def test_relative_markdown_links_resolve():
    assert broken_links(_markdown()) == []


def test_a_link_resolves_against_its_own_directory(tmp_path):
    (tmp_path / "docs" / "pipeline").mkdir(parents=True)
    (tmp_path / "docs" / "install.md").write_text("# install\n")
    page = tmp_path / "docs" / "pipeline" / "p.md"
    page.write_text("[a](install.md) [b](../install.md) [c](https://x.test/y.md)\n")
    assert broken_links([page], tmp_path) == ["docs/pipeline/p.md -> install.md"]
