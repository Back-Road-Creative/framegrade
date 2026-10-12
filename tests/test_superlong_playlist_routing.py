"""Superlong parts must resolve to their own playlist, distinct from the
region-only "Drives" playlists regular LFs land in — see
``config.yaml``'s ``drive.youtube_playlists.rules`` and
``gomoveshift.pipeline.playlist_resolver.resolve_playlist``.
"""

from __future__ import annotations

from gomoveshift.config import Config
from gomoveshift.pipeline.playlist_resolver import resolve_playlist


def _cfg() -> Config:
    return Config.model_validate(
        {
            "drive": {
                "youtube_playlists": {
                    "rules": [
                        {
                            "match": {"region": "ireland", "format": "highlights"},
                            "playlist_name": "Ireland Highlights",
                        },
                        {
                            "match": {"region": "usa", "format": "highlights"},
                            "playlist_name": "USA Highlights",
                        },
                        {
                            "match": {"region": "ireland", "format": "superlong"},
                            "playlist_name": "Ireland Superlong",
                        },
                        {
                            "match": {"region": "usa", "format": "superlong"},
                            "playlist_name": "USA Superlong",
                        },
                        {"match": {"region": "ireland"}, "playlist_name": "Ireland Drives"},
                        {"match": {"region": "usa"}, "playlist_name": "USA Drives"},
                    ],
                    "default": "Unsorted (fallback)",
                }
            }
        }
    )


def test_superlong_part_metadata_resolves_to_superlong_playlist():
    """Mirrors the metadata dict ``_upload_superlong_parts_to_youtube`` builds
    (external_systems_coordinator.py) once it stamps ``format: superlong``."""
    metadata = {
        "location": "Dingle, Kerry, Ireland",
        "region": "ireland",
        "recording_date": "2026-08-01",
        "format": "superlong",
    }
    assert resolve_playlist(metadata, _cfg()) == "Ireland Superlong"


def test_regular_lf_still_resolves_to_drives_playlist():
    """No regression: a regular full-drive LF (trim_ratio present, no
    explicit format) must still land in the region-only Drives playlist."""
    metadata = {
        "location": "Dingle, Kerry, Ireland",
        "region": "ireland",
        "trim_ratio": 0.95,
    }
    assert resolve_playlist(metadata, _cfg()) == "Ireland Drives"
