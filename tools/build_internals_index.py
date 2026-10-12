#!/usr/bin/env python3
"""Generate docs/pipeline-internals.md as an index over docs/pipeline/*.md.

Why: one file holding every module's implementation detail becomes the file
nearly every change collides on. The detail lives one file per module under
docs/pipeline/ (docs/pipeline/NN-slug.md, in reading order); this script
re-derives the index from those files' own headings, so
docs/pipeline-internals.md is generated, never hand-edited.

The index names no module. A line per module would put every change that adds
a module doc back on this one file. The module files' own names are the table
of contents; this script still refuses a module file whose first line is not
its heading.

Usage:
    python3 tools/build_internals_index.py            # write the index
    python3 tools/build_internals_index.py --check     # exit 1 if stale/edited by hand
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
MODULE_DIR = DOCS_DIR / "pipeline"
PREAMBLE_PATH = MODULE_DIR / "_preamble.md"
INDEX_PATH = DOCS_DIR / "pipeline-internals.md"

_FILE_RE = re.compile(r"^(\d+)-.*\.md$")
_HEADING_RE = re.compile(r"^(#{2,3}) (.*)$")

_NOTE = (
    "> **Generated.** This file is rebuilt by `tools/build_internals_index.py` from "
    "its preamble and checks the heading of every file under `docs/pipeline/`. Do "
    "not hand-edit it -- edit "
    "the module file instead and re-run the generator (a test checks this "
    "file matches its output). Module boundary: each `docs/pipeline/NN-*.md` holds "
    "one `##`/`###` section."
)

_POINTER = (
    "The sections are the files under [`docs/pipeline/`](pipeline/), one per "
    "module, numbered in reading order -- browse that directory or "
    "`grep -rl '<topic>' docs/pipeline/`. This index deliberately lists none of "
    "them, so adding a module never edits this file."
)


def module_files() -> list[Path]:
    """Every numbered module file, in reading order."""
    numbered = [(m, p) for p in MODULE_DIR.iterdir() if (m := _FILE_RE.match(p.name))]
    return [p for m, p in sorted(numbered, key=lambda mp: int(mp[0].group(1)))]


def render_index() -> str:
    if not PREAMBLE_PATH.is_file():
        raise FileNotFoundError(f"missing {PREAMBLE_PATH}")
    preamble = PREAMBLE_PATH.read_text(encoding="utf-8").rstrip("\n")

    for f in module_files():
        first_line = f.read_text(encoding="utf-8").split("\n", 1)[0]
        if not _HEADING_RE.match(first_line):
            raise ValueError(f"{f}: first line is not a '## '/'### ' heading: {first_line!r}")
    return "\n".join([preamble, "", _NOTE, "", _POINTER]) + "\n"


def main(argv: list[str]) -> int:
    check = "--check" in argv
    rendered = render_index()
    if check:
        current = INDEX_PATH.read_text(encoding="utf-8") if INDEX_PATH.is_file() else ""
        if current != rendered:
            print(
                f"{INDEX_PATH} is stale or hand-edited. Run:\n"
                f"  python3 tools/build_internals_index.py",
                file=sys.stderr,
            )
            return 1
        print(f"{INDEX_PATH} matches the generator.")
        return 0
    INDEX_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {INDEX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
