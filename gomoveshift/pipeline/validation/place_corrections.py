"""Corrections-ledger-fed place-name denylist/mapping (docs in
``docs/pipeline-internals.md``). Closes the loop the comments triage
report stops short of: ``title_builder.build_lf_title`` substitutes a mapped
wrong name or refuses a bare denylisted one; ``content_gates.
check_metadata_quality`` hard-fails a denylisted name that survives anyway.
A missing/empty/malformed ledger (``GMS_CORRECTIONS_PATH``) is a strict
no-op for both."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from ...paths import gms_data_home


def default_corrections_path() -> Path:
    """``$GMS_CORRECTIONS_PATH``, else ``<data home>/place_corrections.yaml``."""
    override = os.getenv("GMS_CORRECTIONS_PATH")
    if override and override.strip():
        return Path(override)
    return gms_data_home() / "place_corrections.yaml"


@dataclass(frozen=True)
class PlaceCorrections:
    """A loaded corrections ledger: wrong->correct mapping plus a denylist.

    Keys/entries are matched case-insensitively by callers; this dataclass
    stores them exactly as the ledger wrote them (correct case for display).
    """

    mapping: dict[str, str] = field(default_factory=dict)
    denylist: tuple[str, ...] = field(default_factory=tuple)


def load_place_corrections(path: Path | None = None) -> PlaceCorrections:
    """Load the corrections ledger. Missing/empty/malformed -> empty bundle."""
    path = path or default_corrections_path()
    if not path.exists():
        return PlaceCorrections()
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 -- a bad ledger must never break a run
        return PlaceCorrections()
    if not isinstance(data, dict):
        return PlaceCorrections()

    mapping: dict[str, str] = {}
    for entry in data.get("corrections") or []:
        if not isinstance(entry, dict):
            continue
        wrong = str(entry.get("wrong") or "").strip()
        correct = str(entry.get("correct") or "").strip()
        if wrong and correct:
            mapping[wrong] = correct

    denylist = tuple(
        str(name).strip()
        for name in (data.get("denylist") or [])
        if isinstance(name, str) and name.strip()
    )
    return PlaceCorrections(mapping=mapping, denylist=denylist)


def _word_re(name: str) -> re.Pattern:
    return re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)


def apply_corrections(text: str, corrections: PlaceCorrections) -> str:
    """Substitute every mapped wrong name in ``text`` with its correction."""
    if not text or not corrections.mapping:
        return text
    out = text
    for wrong, correct in corrections.mapping.items():
        out = _word_re(wrong).sub(correct, out)
    return out


def find_denylisted(text: str, corrections: PlaceCorrections) -> list[str]:
    """Return denylisted names present (case-insensitively) in ``text``."""
    if not text or not corrections.denylist:
        return []
    return [name for name in corrections.denylist if _word_re(name).search(text)]
