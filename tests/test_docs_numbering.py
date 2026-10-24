"""The numbered pipeline notes keep one number per note.

A number is how a note is cited ("doc 153"), so two notes sharing one make the
citation ambiguous. This pins unique numbers, the two ported measurement
spikes, and the range the docs index advertises.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "docs" / "pipeline"
INDEX = ROOT / "docs" / "README.md"
_NUMBERED = re.compile(r"^(\d+)-.+\.md$")


def _by_number() -> dict[int, list[str]]:
    found: dict[int, list[str]] = defaultdict(list)
    for p in sorted(NOTES.glob("*.md")):
        m = _NUMBERED.match(p.name)
        if m:
            found[int(m.group(1))].append(p.name)
    return found


def test_every_pipeline_note_number_is_used_once():
    dupes = {n: names for n, names in _by_number().items() if len(names) > 1}
    assert dupes == {}


def test_the_gpu_grade_and_blur_composite_spikes_are_ported():
    names = [n for group in _by_number().values() for n in group]
    assert any("gpu-grade-spike" in n for n in names)
    assert any("blur-composite-ablation" in n for n in names)


def test_ported_spikes_cite_each_other_by_a_live_note_number():
    gpu = next(NOTES.glob("*-gpu-grade-spike.md"))
    blur = next(NOTES.glob("*-blur-composite-ablation.md"))
    gpu_number = gpu.name.split("-", 1)[0]
    assert f"doc {gpu_number}" in blur.read_text(encoding="utf-8")


def test_ported_spikes_name_no_machine_paths():
    for p in list(NOTES.glob("*-gpu-grade-spike.md")) + list(
        NOTES.glob("*-blur-composite-ablation.md")
    ):
        text = p.read_text(encoding="utf-8")
        assert not re.search(r"/mnt/|/home/|\.data/|[A-Z]:\\\\", text), p.name


def test_docs_index_advertises_the_highest_note_number():
    highest = max(_by_number())
    assert f"({highest})" in INDEX.read_text(encoding="utf-8")
