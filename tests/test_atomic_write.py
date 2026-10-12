"""Tests for ``gomoveshift.pipeline._atomic_write``.

Validates the four behaviours that the helper guarantees:

* writes a payload to a fresh path,
* leaves an existing file untouched if ``json.dump`` raises mid-write,
* overwrites an existing file on success,
* creates missing parent directories.

A mid-write crash leaving a corrupt JSON sidecar was the bug class
the atomic-write helper eliminates.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gomoveshift.pipeline import _atomic_write
from gomoveshift.pipeline._atomic_write import atomic_write_json, atomic_write_text


def test_atomic_write_replaces_target(tmp_path: Path) -> None:
    """Writing a payload to a fresh path leaves the JSON on disk."""
    target = tmp_path / "out.json"
    payload = {"hello": "world", "n": 1}
    atomic_write_json(target, payload)
    assert target.exists()
    assert json.loads(target.read_text()) == payload


def test_atomic_write_leaves_existing_untouched_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mid-write ``json.dump`` failure preserves the prior file contents.

    Also verifies that no orphan ``.tmp`` file is left in the directory —
    the ``except`` branch cleans up the temp file before re-raising.
    """
    target = tmp_path / "state.json"
    atomic_write_json(target, {"old": 1})
    assert json.loads(target.read_text()) == {"old": 1}

    def boom(*args, **kwargs):
        raise RuntimeError("simulated mid-write crash")

    monkeypatch.setattr(_atomic_write.json, "dump", boom)

    with pytest.raises(RuntimeError, match="simulated mid-write crash"):
        atomic_write_json(target, {"new": 2})

    # Existing file untouched.
    assert json.loads(target.read_text()) == {"old": 1}
    # No orphan .tmp siblings.
    tmps = list(tmp_path.glob("*.tmp"))
    assert tmps == [], f"orphan tmp files left behind: {tmps}"


def test_atomic_write_overwrites_existing_on_success(tmp_path: Path) -> None:
    """Successive writes replace the prior payload."""
    target = tmp_path / "out.json"
    atomic_write_json(target, {"old": 1})
    atomic_write_json(target, {"new": 2})
    assert json.loads(target.read_text()) == {"new": 2}


def test_atomic_write_text_writes_verbatim(tmp_path: Path) -> None:
    """The text helper writes the caller's string byte-for-byte (utf-8)."""
    target = tmp_path / "deep" / "page.md"
    text = "---\ntitle: Cliffs of Móher\n# a comment\n---\nBody.\n"
    atomic_write_text(target, text)
    assert target.read_text(encoding="utf-8") == text


def test_atomic_write_text_leaves_existing_untouched_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure before the swap preserves the prior file and drops the temp."""
    target = tmp_path / "page.md"
    atomic_write_text(target, "old content\n")

    def boom(*args, **kwargs):
        raise RuntimeError("simulated mid-write crash")

    monkeypatch.setattr(_atomic_write.os, "replace", boom)

    with pytest.raises(RuntimeError, match="simulated mid-write crash"):
        atomic_write_text(target, "new content\n")

    assert target.read_text() == "old content\n"
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_creates_parent_dirs(tmp_path: Path) -> None:
    """Writing to a path under a missing parent succeeds (parent created)."""
    target = tmp_path / "deep" / "nested" / "dir" / "out.json"
    assert not target.parent.exists()
    atomic_write_json(target, {"ok": True})
    assert target.exists()
    assert json.loads(target.read_text()) == {"ok": True}
