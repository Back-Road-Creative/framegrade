"""Read-only media-catalog contract. See ``reader.py`` for the
mechanism; ``docs/pipeline-internals.md`` for the design note."""

from .reader import (
    CatalogEntry,
    CatalogResult,
    CatalogSourceUnavailableError,
    DamagedSource,
    read_catalog,
)

__all__ = [
    "CatalogEntry",
    "CatalogResult",
    "CatalogSourceUnavailableError",
    "DamagedSource",
    "read_catalog",
]
