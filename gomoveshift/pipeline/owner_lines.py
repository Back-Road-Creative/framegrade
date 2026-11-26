"""Owner lines: one sentence in the channel owner's own voice on a post.

An optional ``<project_dir>/social/owner_lines.yaml`` beside ``titles.yaml``::

    shorts:
      3: Stopped here for chips, best in the county.
    longform:
      1: My favourite stretch of the whole trip.

``shorts`` is keyed by the Short's global index (the ``titles.yaml`` key) and
``longform`` by the LF index. A Short's line opens its Facebook and Instagram
captions; a long-form's line opens its YouTube description. The line is
read when the text is generated, so it must be in the file before the drive's
social text is built and approved: editing it afterwards changes nothing that
was already reviewed.

A missing file means no lines. A present-but-malformed file is a hard
``ValueError`` naming the path and the key, matching ``load_hand_titles``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

import yaml

from .text_format import close_sentence

OWNER_LINES_FILE = "owner_lines.yaml"
#: Longest line accepted: one sentence, short enough that Instagram's 280-char
#: caption fit never has to cut the composed place sentence after it.
OWNER_LINE_MAX_LEN = 100
_SECTIONS = ("shorts", "longform")
# A link or hashtag belongs to the pipeline's own text; the owner line is words.
_REFUSED = re.compile(r"https?://|www\.|#\w")


@dataclass(frozen=True)
class OwnerLines:
    """Lines by global Short index and by LF index; empty when none."""

    shorts: Mapping[int, str] = field(default_factory=dict)
    longform: Mapping[int, str] = field(default_factory=dict)


def owner_lines_path(titles_file: Path) -> Path:
    """The owner-lines file beside a drive's ``titles.yaml``."""
    return Path(titles_file).parent / OWNER_LINES_FILE


def _section(path: Path, raw: dict, name: str) -> dict[int, str]:
    block = raw.get(name) or {}
    if not isinstance(block, dict):
        raise ValueError(f"owner lines file {path}: '{name}' must be a mapping of index to line")
    lines: dict[int, str] = {}
    for key, value in block.items():
        try:
            idx = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"owner lines file {path}: {name} key {key!r} is not an int") from exc
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"owner lines file {path}: {name} {idx} must be a non-empty string")
        line = " ".join(value.split())
        if len(line) > OWNER_LINE_MAX_LEN:
            raise ValueError(
                f"owner lines file {path}: {name} {idx} is {len(line)} characters, "
                f"over {OWNER_LINE_MAX_LEN}"
            )
        if _REFUSED.search(line):
            raise ValueError(
                f"owner lines file {path}: {name} {idx} carries a link or hashtag; "
                "the line is words only"
            )
        lines[idx] = line
    return lines


def load_owner_lines(path: Optional[Path]) -> OwnerLines:
    """Load ``path``; ``OwnerLines()`` when ``path`` is ``None`` or absent."""
    if path is None or not Path(path).exists():
        return OwnerLines()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"owner lines file {path} must be a mapping")
    unknown = sorted(set(raw) - set(_SECTIONS))
    if unknown:
        raise ValueError(f"owner lines file {path}: unknown section {unknown[0]!r}")
    return OwnerLines(
        shorts=_section(Path(path), raw, "shorts"),
        longform=_section(Path(path), raw, "longform"),
    )


def with_owner_sentence(text: str, line: Optional[str]) -> str:
    """``line`` as the caption's opening sentence, then ``text``."""
    if not line:
        return text
    return f"{close_sentence(line)} {text}".rstrip() if text else close_sentence(line)


def with_owner_paragraph(description: str, line: Optional[str]) -> str:
    """``line`` as a description's first paragraph, then ``description``."""
    if not line:
        return description
    return f"{close_sentence(line)}\n\n{description}" if description else close_sentence(line)
