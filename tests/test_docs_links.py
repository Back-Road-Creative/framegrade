"""The front door: README.md and docs/README.md.

`tests/test_docs_relative_links.py` proves every Markdown link in the repo
resolves. This file pins what a first-time reader needs from the two entry
pages: an honest opening line, a pinned install line, a way into the docs, and
a docs index that does not silently miss a page.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DOCS = ROOT / "docs"
INDEX = DOCS / "README.md"
_LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")
_INSTALL = re.compile(
    r"git\+https://github\.com/Back-Road-Creative/framegrade@(v\d+\.\d+\.\d+)"
)


def _links(md: Path) -> list[str]:
    return [
        t
        for t in _LINK.findall(md.read_text(encoding="utf-8"))
        if not re.match(r"[a-z][a-z0-9+.-]*:", t)
    ]


def _released_versions() -> set[str]:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    return {f"v{v}" for v in re.findall(r"^## \[(\d+\.\d+\.\d+)\]", text, re.M)}


def test_docs_index_exists_and_its_links_resolve():
    assert INDEX.is_file(), "docs/README.md is the docs index"
    broken = [t for t in _links(INDEX) if not (INDEX.parent / t).exists()]
    assert broken == []


def test_readme_links_resolve():
    broken = [t for t in _links(README) if not (README.parent / t).exists()]
    assert broken == []


def test_readme_opens_with_the_product_not_the_sign_detector():
    lines = README.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# ")
    opening = "\n".join(lines[:12]).lower()
    assert "sign detection" not in opening
    assert "ocr" not in opening
    assert "easyocr" not in opening


def test_readme_install_line_pins_a_released_tag():
    tags = _INSTALL.findall(README.read_text(encoding="utf-8"))
    assert tags, "README needs a `pip install ... git+https://...@vX.Y.Z` line"
    assert set(tags) <= _released_versions()


def test_readme_leads_the_reader_to_install_and_the_docs_index():
    links = _links(README)
    assert "docs/install.md" in links
    assert "docs/README.md" in links


def test_readme_reaches_the_operator_pages_before_the_contributor_notes():
    text = README.read_text(encoding="utf-8")
    assert text.index("docs/README.md") < text.index("\n## Testing")


def test_readme_does_not_link_the_scratch_scripts_directory():
    assert not [t for t in _links(README) + _links(INDEX) if t.startswith("scripts/")]


def test_docs_index_lists_every_top_level_page():
    linked = {Path(t).name for t in _links(INDEX) if "/" not in t.rstrip("/")}
    pages = {p.name for p in DOCS.glob("*.md") if p.name != "README.md"}
    assert sorted(pages - linked) == []


def test_docs_index_points_into_the_numbered_pipeline_notes():
    assert "pipeline/" in _links(INDEX) or any(
        t.startswith("pipeline/") for t in _links(INDEX)
    )
