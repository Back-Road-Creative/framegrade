"""Free-form operator tags on library items.

Tenant isolation is PATH SEPARATION -- one JSON file per brand under the
console data dir -- never a shared store keyed by a brand column
(``console/tenant.py``'s stamp guard documents why path separation, not an
in-schema check, is this console's isolation primitive). A tag file this
module has never written for a brand reads as "no tags yet" for that brand,
never another brand's file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union


def _tags_path(data_dir: Union[str, Path], brand: str) -> Path:
    return Path(data_dir) / "brands" / brand / "library_tags.json"


def load_tags(data_dir: Union[str, Path], brand: str) -> dict[str, list[str]]:
    """``{catalog_id: [tag, ...]}`` for ``brand``. A missing, damaged, or
    malformed file reads as "no tags recorded" rather than raising -- tags
    are supplementary metadata, never load-bearing for the library itself."""
    path = _tags_path(data_dir, brand)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    result: dict[str, list[str]] = {}
    for catalog_id, values in raw.items():
        if not isinstance(values, list):
            continue
        cleaned = sorted({str(v).strip() for v in values if str(v).strip()})
        if cleaned:
            result[str(catalog_id)] = cleaned
    return result


def _save_tags(data_dir: Union[str, Path], brand: str, tags: dict[str, list[str]]) -> None:
    path = _tags_path(data_dir, brand)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tags, sort_keys=True), encoding="utf-8")


def add_tag(data_dir: Union[str, Path], brand: str, catalog_id: str, tag: str) -> list[str]:
    """Add ``tag`` to ``catalog_id`` for ``brand``; returns the item's tags
    after the change. A blank tag is a no-op."""
    tag = tag.strip()
    tags = load_tags(data_dir, brand)
    if not tag:
        return tags.get(catalog_id, [])
    tags[catalog_id] = sorted({*tags.get(catalog_id, []), tag})
    _save_tags(data_dir, brand, tags)
    return tags[catalog_id]


def remove_tag(data_dir: Union[str, Path], brand: str, catalog_id: str, tag: str) -> list[str]:
    """Remove ``tag`` from ``catalog_id`` for ``brand``; returns the item's
    remaining tags. Removing a tag that was never there is a no-op."""
    tag = tag.strip()
    tags = load_tags(data_dir, brand)
    remaining = sorted(set(tags.get(catalog_id, [])) - {tag})
    if remaining:
        tags[catalog_id] = remaining
    else:
        tags.pop(catalog_id, None)
    _save_tags(data_dir, brand, tags)
    return remaining
