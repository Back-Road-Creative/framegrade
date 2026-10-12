"""One catalog of the photo effects, derived from the registries that render them.

Three registries already exist and are what ``develop`` actually reads:
``looks.BUILTIN_LOOKS`` (plus the operator's ``photos.looks``), ``masks.MASKS`` with
``masks.MASK_PARAMS`` derived from each mask's signature, and ``styles.STYLES``. The
console and CLI used to list some of these by hand and the rest not at all -- the
``styles`` control was a free-text field. This module is the single enumerable view
of all three, so a list an operator sees is the list the renderer accepts.

Nothing here probes dependencies. Every effect imports only core deps, so a
"capability" that could never be absent is not a capability, just a lie in waiting.
"""

from __future__ import annotations

from typing import Any

from .looks import BUILTIN_LOOKS
from .masks import MASK_PARAMS, MASKS
from .styles import STYLES


def style_names() -> list[str]:
    """Every registered style, in declaration order -- the order ``select_styles``
    breaks ties by, so the list reads the way the ranker thinks."""
    return list(STYLES)


def masks() -> dict[str, list[str]]:
    """Each mask with the shape params it accepts (``MASK_PARAMS``, from its
    signature), so a UI can show ``angle`` beside ``linear`` and not beside ``sky``."""
    return {name: sorted(MASK_PARAMS[name]) for name in MASKS}


def _masks_layered(recipe: dict[str, Any]) -> list[str]:
    return sorted({layer["mask"] for layer in recipe.get("layers", ()) if "mask" in layer})


def looks(configured: dict[str, dict] | None = None) -> dict[str, list[str]]:
    """Each look with the masks its layers name (``[]`` for a flat look).

    ``configured`` is the operator's ``photos.looks`` table; it is merged AFTER the
    built-ins so a configured name wins a collision -- the same precedence the CLI's
    ``--look`` resolution and the console's ``resolved_looks`` already give it.
    """
    table = {**BUILTIN_LOOKS, **(configured or {})}
    return {name: _masks_layered(recipe) for name, recipe in table.items()}


def invalid_looks(table: dict[str, dict]) -> dict[str, list[str]]:
    """Looks whose layers name a mask that is not registered, keyed by look name.

    A recipe can be wrong in exactly one structural place -- the mask it asks for --
    so this is the whole cross-effect check, and it says which name is wrong rather
    than letting ``mask_for`` raise on it at develop time.
    """
    bad = {}
    for name, recipe in table.items():
        unknown = [m for m in _masks_layered(recipe) if m not in MASKS]
        if unknown:
            bad[name] = unknown
    return bad


def effects_catalog(configured: dict[str, dict] | None = None) -> dict[str, Any]:
    """The whole catalog as one JSON-shaped document."""
    return {"looks": looks(configured), "masks": masks(), "styles": style_names()}
