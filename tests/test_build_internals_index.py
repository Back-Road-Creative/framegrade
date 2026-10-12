"""Regression tests for the pipeline-internals index generator
(tools/build_internals_index.py).

docs/pipeline-internals.md is a generated index over docs/pipeline/*.md (the
module files it would otherwise hold inline, which makes it the file every
change collides on). This pins two things a
hand edit or a stale generator run would otherwise miss silently:

1. The committed docs/pipeline-internals.md matches what the generator
   produces right now, from the module files as committed -- so a hand edit
   to the index is a red test, not a drift nobody notices.
2. No module file exceeds the line cap below -- the regrowth guard, so a
   module doc that keeps growing fails the suite instead of passing silently.
"""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "tools" / "build_internals_index.py"
_SPEC = importlib.util.spec_from_file_location("build_internals_index", _SCRIPT)
assert _SPEC and _SPEC.loader
gen = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gen)

# Maximum length of one docs/pipeline/NN-*.md module file.
MAX_MODULE_LINES = 500


def test_committed_index_matches_generator_output():
    committed = gen.INDEX_PATH.read_text(encoding="utf-8")
    rendered = gen.render_index()
    assert committed == rendered, (
        "docs/pipeline-internals.md is out of date or was hand-edited. It is "
        "generated -- run: python3 tools/build_internals_index.py"
    )


def test_no_module_file_exceeds_the_size_cap():
    over = [
        f"{f.name} ({len(f.read_text(encoding='utf-8').splitlines())} lines)"
        for f in gen.module_files()
        if len(f.read_text(encoding="utf-8").splitlines()) > MAX_MODULE_LINES
    ]
    assert not over, (
        f"module file(s) over the {MAX_MODULE_LINES}-line cap: {over}. "
        "docs/pipeline/ is one file per module so parallel PRs stop colliding "
        "on a single giant doc -- split the module further instead of letting "
        "it regrow."
    )


def test_module_files_found():
    """A glob/regex that stopped matching would make the other assertions vacuous."""
    assert len(gen.module_files()) > 100, gen.module_files()


def test_every_module_file_starts_with_its_own_heading():
    for f in gen.module_files():
        first_line = f.read_text(encoding="utf-8").split("\n", 1)[0]
        assert first_line.startswith("## ") or first_line.startswith("### "), (
            f"{f}: first line must be a '## '/'### ' heading, got {first_line!r}"
        )


def test_render_index_raises_when_preamble_missing(tmp_path, monkeypatch):
    """render_index() must refuse to fabricate a preamble section."""
    monkeypatch.setattr(gen, "PREAMBLE_PATH", tmp_path / "_preamble.md")
    with pytest.raises(FileNotFoundError, match="_preamble.md"):
        gen.render_index()


def test_render_index_raises_on_a_module_file_with_no_heading(tmp_path, monkeypatch):
    """A module file whose first line isn't '## '/'### ' would silently corrupt
    the generated index -- render_index() must refuse instead."""
    (tmp_path / "_preamble.md").write_text("Preamble text.\n", encoding="utf-8")
    bad = tmp_path / "01-bad.md"
    bad.write_text("not a heading at all\n", encoding="utf-8")
    monkeypatch.setattr(gen, "MODULE_DIR", tmp_path)
    monkeypatch.setattr(gen, "PREAMBLE_PATH", tmp_path / "_preamble.md")
    with pytest.raises(ValueError, match="not a '## '/'### ' heading"):
        gen.render_index()


def test_adding_a_module_file_does_not_change_the_index(tmp_path, monkeypatch):
    """The index lists no module by name, so a PR that adds docs/pipeline/NN-*.md
    never touches docs/pipeline-internals.md. A per-module line would make every
    change that adds a module collide on that one file."""
    (tmp_path / "_preamble.md").write_text("Preamble text.\n", encoding="utf-8")
    (tmp_path / "01-a.md").write_text("## A\n", encoding="utf-8")
    (tmp_path / "02-b.md").write_text("### B\n", encoding="utf-8")
    monkeypatch.setattr(gen, "MODULE_DIR", tmp_path)
    monkeypatch.setattr(gen, "PREAMBLE_PATH", tmp_path / "_preamble.md")
    before = gen.render_index()
    (tmp_path / "03-c.md").write_text("## C\n", encoding="utf-8")
    assert gen.render_index() == before
    assert "01-a.md" not in before
    assert "docs/pipeline/" in before


def test_main_check_mode_reports_stale_index_and_returns_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gen, "INDEX_PATH", tmp_path / "pipeline-internals.md")
    (tmp_path / "pipeline-internals.md").write_text("stale content\n", encoding="utf-8")
    rc = gen.main(["--check"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "is stale or hand-edited" in err
    assert "python3 tools/build_internals_index.py" in err


def test_main_check_mode_reports_missing_index_and_returns_1(tmp_path, monkeypatch, capsys):
    """No committed index at all is the same failure as a stale one -- covers
    the INDEX_PATH.is_file() branch of the --check comparison."""
    monkeypatch.setattr(gen, "INDEX_PATH", tmp_path / "does-not-exist.md")
    rc = gen.main(["--check"])
    assert rc == 1
    assert "is stale or hand-edited" in capsys.readouterr().err


def test_main_check_mode_matches_and_returns_0(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gen, "INDEX_PATH", tmp_path / "pipeline-internals.md")
    (tmp_path / "pipeline-internals.md").write_text(gen.render_index(), encoding="utf-8")
    rc = gen.main(["--check"])
    assert rc == 0
    assert "matches the generator." in capsys.readouterr().out


def test_main_write_mode_regenerates_the_index_and_returns_0(tmp_path, monkeypatch, capsys):
    target = tmp_path / "pipeline-internals.md"
    monkeypatch.setattr(gen, "INDEX_PATH", target)
    rc = gen.main([])
    assert rc == 0
    assert target.read_text(encoding="utf-8") == gen.render_index()
    assert f"wrote {target}" in capsys.readouterr().out


def test_dunder_main_entrypoint_dispatches_to_main(monkeypatch, capsys):
    """`python3 tools/build_internals_index.py --check` must actually hit the
    `if __name__ == "__main__":` dispatch line, not just the imported module
    object the rest of this file uses -- runpy executes the real script with
    __name__ set to "__main__", the same as invoking it from a shell."""
    monkeypatch.setattr(sys, "argv", [str(_SCRIPT), "--check"])
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(_SCRIPT), run_name="__main__")
    assert exc_info.value.code == 0
    assert "matches the generator." in capsys.readouterr().out
