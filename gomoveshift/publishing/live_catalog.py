"""Where the local live-catalog snapshot lives (a leaf: pipeline code must not import the CLI)."""

from pathlib import Path

LIVE_CATALOG_SNAPSHOT = "live_catalog.latest.json"


def live_catalog_snapshot_path(storage_path) -> Path:
    return Path(storage_path) / LIVE_CATALOG_SNAPSHOT
