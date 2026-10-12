"""Verified-dead config knobs removed from ``gomoveshift.config`` (chore).

Two halves, TDD order:

1. A regression pin for the twelve ``DriveConfig`` model fields read ONLY
   through the dynamic ``getattr(drive, f"{prefix}_model_{suffix}")`` in
   ``gomoveshift.analysis._resolve_drive_model`` — those must survive any
   dead-config sweep, static-scan-invisible or not.
2. A red-first assertion that the sixteen verified-dead fields (plus the
   structural fallout: ``ClaudeConfig``, ``RoadSelectionConfig``,
   ``Settings``, ``get_settings``, and the duplicate
   ``montage_max_duration_sec`` declaration) are gone from the source.
"""

import inspect

from gomoveshift.config import DriveConfig

_PREFIXES = ("plate", "face", "vehicle", "person")
_SUFFIXES = ("path", "url", "sha256")


def test_dynamic_model_fields_survive():
    """The 12 {prefix}_model_{suffix} fields ``_resolve_drive_model`` reads via
    ``getattr`` must still exist on ``DriveConfig`` — they look dead to any
    static scan but are live."""
    field_names = set(DriveConfig.model_fields)
    for prefix in _PREFIXES:
        for suffix in _SUFFIXES:
            name = f"{prefix}_model_{suffix}"
            assert name in field_names, f"{name} must remain on DriveConfig"


_DEAD_MODULE_NAMES = [
    "SignDetectionConfig.min_confidence",
    "OCRConfig.paddle_confidence_threshold",
    "OCRConfig.claude_fallback_threshold",
    "SegmentationConfig.prefer_location_cuts",
    "OutputConfig.nvenc_preset",
    "ClaudeConfig",
    "GPSQualityConfig.max_hdop",
    "GPSQualityConfig.min_speed_mps",
    "RoadSelectionConfig",
    "DriveConfig.road_selection",
    "Settings",
    "get_settings",
]


def test_dead_config_fields_and_symbols_removed():
    import gomoveshift.config as config_mod

    for dotted in _DEAD_MODULE_NAMES:
        if "." in dotted:
            cls_name, field_name = dotted.split(".", 1)
            cls = getattr(config_mod, cls_name, None)
            if cls is None:
                continue  # class itself already removed (structural fallout)
            assert field_name not in getattr(cls, "model_fields", {}), (
                f"{dotted} should have been removed"
            )
        else:
            assert not hasattr(config_mod, dotted), f"{dotted} should have been removed"


def test_montage_max_duration_sec_declared_exactly_once():
    import gomoveshift.config as config_mod

    source = inspect.getsource(config_mod)
    lines = [
        line
        for line in source.splitlines()
        if line.strip().startswith("montage_max_duration_sec: float")
    ]
    assert len(lines) == 1, (
        f"expected exactly one montage_max_duration_sec declaration, found {len(lines)}"
    )
