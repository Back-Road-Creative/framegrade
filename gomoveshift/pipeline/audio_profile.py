"""Per-brand music-bed audio profile and its licence ledger.

A Route Story film may carry a licensed music bed mixed under the
attenuated source audio (see ``route_story.encode``). A bed is only ever
burned when its exact file has a matching entry in the brand's licence
ledger -- so a published film's receipt can prove where its music came
from and under what licence.

GoMoveShift's own brand refuses a bed outright and unconditionally:
its audio contract is
near-silent and attenuate-only; a music bed is a different, disallowed,
thing regardless of anything a brand's ``brand.yaml`` might say.

The optional ``audio:`` block lives in ``brand.yaml`` beside the
``policy:`` block (``publishing.brand_policy``) and is read with the
same idiom: same shipped-brands directory, same "missing file/block/key
falls back to disabled" rule. ``brand_policy._SHIPPED_BRANDS_DIR`` is
imported (not duplicated) so the two loaders can never disagree about
where a brand's config lives, and so a test that monkeypatches it for
one loader also relocates the other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from ..publishing import brand_policy as _brand_policy

# Default music source: Pixabay Music (Pixabay Content
# License). The ledger's ``source`` field accepts any string -- this is
# only the seed default, not a constraint -- so a later library needs no
# code change, only a new ledger entry.
DEFAULT_SOURCE = "pixabay"

_GMS_SLUG = "gomoveshift"

_DEFAULT_GAIN_DB = -18.0
_DEFAULT_FADE_OUT_S = 2.0


class GmsAudioRefused(Exception):  # noqa: N818 -- public name is part of the audio contract
    """GoMoveShift's audio contract refuses a music bed -- no config can turn it on."""


class UnlicensedBedError(Exception):
    """The profile's configured bed file has no matching entry in its ledger."""


@dataclass(frozen=True)
class AudioProfile:
    """A brand's music-bed configuration, defaults matching "no bed"."""

    enabled: bool = False
    bed_path: Optional[Path] = None
    gain_db: float = _DEFAULT_GAIN_DB
    fade_out_s: float = _DEFAULT_FADE_OUT_S
    ledger_path: Optional[Path] = None


def _load_raw_audio_block(slug: str) -> dict:
    path = _brand_policy._SHIPPED_BRANDS_DIR / slug / "brand.yaml"
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    return raw.get("audio") or {}


def load_audio_profile(slug: str) -> AudioProfile:
    """Read the optional ``audio:`` block from ``brand.yaml`` for ``slug``.

    ``gomoveshift`` raises :class:`GmsAudioRefused` unconditionally, before
    any file is even read. Any other missing file/block/key falls back to
    :class:`AudioProfile`'s disabled defaults.
    """
    if slug == _GMS_SLUG:
        raise GmsAudioRefused(
            "GoMoveShift's near-silent, attenuate-only audio contract refuses any music bed"
        )

    slug_dir = _brand_policy._SHIPPED_BRANDS_DIR / slug
    block = _load_raw_audio_block(slug)
    if not block:
        return AudioProfile()

    bed_path = block.get("bed_path")
    ledger_path = block.get("ledger_path")

    return AudioProfile(
        enabled=bool(block.get("enabled", False)),
        bed_path=(slug_dir / bed_path) if bed_path else None,
        gain_db=float(block.get("gain_db", _DEFAULT_GAIN_DB)),
        fade_out_s=float(block.get("fade_out_s", _DEFAULT_FADE_OUT_S)),
        ledger_path=(slug_dir / ledger_path)
        if ledger_path
        else (slug_dir / "music" / "ledger.json"),
    )


@dataclass
class LicenceLedger:
    """A brand's licensed music-bed inventory, keyed by ``file``."""

    entries: list[dict]

    def find(self, file_name: str) -> Optional[Mapping[str, Any]]:
        """Return the ledger entry for ``file_name`` (basename match), or ``None``."""
        for entry in self.entries:
            if entry.get("file") == file_name:
                return entry
        return None

    @classmethod
    def load(cls, path: Path) -> "LicenceLedger":
        """Load ``path``'s ``{"entries": [...]}`` JSON. Missing file -> empty ledger."""
        path = Path(path)
        if not path.exists():
            return cls(entries=[])
        data = json.loads(path.read_text())
        return cls(entries=list(data.get("entries", [])))


def resolve_bed(profile: AudioProfile, ledger: "LicenceLedger") -> Mapping[str, Any]:
    """Return the ledger entry for ``profile.bed_path``, or raise ``UnlicensedBedError``.

    A bed file with no matching ledger entry must never be burned into a
    published film -- there would be nothing to point to if its licence
    were ever challenged.
    """
    file_name = profile.bed_path.name if profile.bed_path else None
    entry = ledger.find(file_name) if file_name else None
    if entry is None:
        raise UnlicensedBedError(
            f"music bed {profile.bed_path!r} has no matching entry in the licence ledger"
        )
    return entry
