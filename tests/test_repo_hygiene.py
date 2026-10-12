"""Repo-hygiene invariants.

These tests are construction-correctness guards: they assert that secret
material and ad-hoc backup residue cannot recur in the working tree without
the ignore list also being edited. A plaintext token file committed at
the repo root is the canonical failure these tests prevent from recurring.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _gitignore_patterns() -> list[str]:
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_gateway_token_in_gitignore() -> None:
    assert ".gateway_token" in _gitignore_patterns(), (
        ".gateway_token must be listed in .gitignore — plaintext secrets "
        "must never be committable from a clean checkout."
    )


def test_gateway_token_absent_from_working_tree() -> None:
    assert not (REPO_ROOT / ".gateway_token").exists(), (
        "Plaintext .gateway_token found at repo root. Remove it and rotate "
        "the secret and keep it encrypted with sops — never recommit the file."
    )


def test_bak_versioned_residue_ignored() -> None:
    assert "*.bak.[0-9]*" in _gitignore_patterns(), (
        "*.bak.[0-9]* glob must be in .gitignore — versioned backup residue "
        "(e.g. file.bak.20300101-000000) must not be committable."
    )


def test_drive_letter_residue_ignored() -> None:
    assert "D:*" in _gitignore_patterns(), (
        "D:* glob must be in .gitignore — Windows-style drive-letter paths "
        "created by mistake on Linux or WSL must not be committable."
    )


def test_tool_cache_dir_ignored() -> None:
    assert ".cache/" in _gitignore_patterns(), (
        ".cache/ must be in .gitignore — tool caches (for example a Hypothesis "
        "plugin's example database, redirected there by tests/conftest.py) "
        "must not be committable."
    )
