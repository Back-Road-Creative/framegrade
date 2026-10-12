"""State-shape normalization at load boundary.

Asserts that ``PipelineState.from_dict`` coerces every int-keyed-dict
field declared in ``_INT_KEY_DICT_FIELDS`` from JSON-string keys back
to int, including the per-LF GPS fields and the
``lf_thumbnail_drive_urls`` map that were previously dropped because
they weren't declared on the dataclass.
"""

from __future__ import annotations

import pytest

from gomoveshift.pipeline.drive_state_model import (
    _INT_KEY_DICT_FIELDS,
    PipelineState,
    _assert_canonical_key_shape,
)
from gomoveshift.pipeline.uploader import UploadManifest


def test_int_key_dict_fields_round_trip_through_json_strings():
    """Every entry in ``_INT_KEY_DICT_FIELDS`` accepts str keys at load."""
    raw = {
        "per_lf_distance_km": {"1": 12.4, "2": 7.1},
        "per_lf_elevation_gain_m": {"1": 80.0, "2": 0.0},
        "per_lf_regions": {"1": ["Kerry"], "2": ["Portfable"]},
        "lf_thumbnail_drive_urls": {"1": "https://drive/x", "2": "https://drive/y"},
        "per_lf_recording_date": {"1": "2026-02-20"},
        "per_lf_trim_ratio": {"1": 0.42},
        "per_lf_gpx_path": {"1": "/tmp/a.gpx"},
        "per_lf_locations": {"1": "Fableport County, SC"},
    }

    state = PipelineState.from_dict(raw)

    assert state.per_lf_distance_km == {1: 12.4, 2: 7.1}
    assert state.per_lf_elevation_gain_m == {1: 80.0, 2: 0.0}
    assert state.per_lf_regions == {1: ["Kerry"], 2: ["Portfable"]}
    assert state.lf_thumbnail_drive_urls == {1: "https://drive/x", 2: "https://drive/y"}
    assert state.per_lf_recording_date == {1: "2026-02-20"}
    assert state.per_lf_trim_ratio == {1: 0.42}
    assert state.per_lf_gpx_path == {1: "/tmp/a.gpx"}
    assert state.per_lf_locations == {1: "Fableport County, SC"}


def test_upload_manifests_dict_shape_rehydrates_to_dataclass():
    """Dict-shaped ``upload_manifests`` entries become ``UploadManifest`` instances.

    Regression: resumed
    manifests left as plain dicts caused ``getattr(m, 'files', {})``
    to return empty everywhere.
    """
    raw = {
        "upload_manifests": {
            "1": {
                "project_name": "test",
                "folder_id": "abc",
                "files": {"short_01.mp4": "https://drive/link"},
                "completed": True,
            }
        }
    }

    state = PipelineState.from_dict(raw)

    assert 1 in state.upload_manifests
    manifest = state.upload_manifests[1]
    assert isinstance(manifest, UploadManifest)
    assert manifest.files == {"short_01.mp4": "https://drive/link"}
    assert manifest.completed is True


def test_canonical_key_shape_invariant_fires_on_non_int_keys():
    """Manually-injected str-keyed dict trips the invariant."""
    state = PipelineState()
    object.__setattr__(state, "per_lf_distance_km", {"1": 12.4, "2": 7.1})

    with pytest.raises(ValueError, match="non-int keys"):
        _assert_canonical_key_shape(state)


def test_canonical_key_shape_invariant_passes_on_clean_state():
    """Properly int-keyed state passes the invariant silently."""
    state = PipelineState()
    object.__setattr__(state, "per_lf_distance_km", {1: 12.4, 2: 7.1})
    object.__setattr__(state, "per_lf_regions", {1: ["Kerry"]})

    _assert_canonical_key_shape(state)  # no raise


def test_int_key_dict_fields_includes_new_wave_fields():
    """Contract: the per-LF GPS and Metricool fields are tracked."""
    for fname in (
        "per_lf_distance_km",
        "per_lf_elevation_gain_m",
        "per_lf_regions",
        "lf_thumbnail_drive_urls",
    ):
        assert fname in _INT_KEY_DICT_FIELDS, (
            f"{fname} missing from _INT_KEY_DICT_FIELDS — JSON round-trip "
            "will drop or string-key this field."
        )


def test_from_dict_serialize_round_trip_preserves_int_keys():
    """Full round-trip: dataclass → to_dict → JSON-shaped str keys → from_dict → int keys."""
    import json

    state = PipelineState()
    state["per_lf_distance_km"] = {1: 5.0, 2: 10.0}
    state["lf_thumbnail_drive_urls"] = {1: "https://drive/lf1.jpg"}
    state["per_lf_regions"] = {1: ["Kerry", "Portfable"]}

    # Round-trip through JSON to simulate disk write/read
    serialized = state.to_dict()
    rehydrated = PipelineState.from_dict(json.loads(json.dumps(serialized)))

    assert rehydrated.per_lf_distance_km == {1: 5.0, 2: 10.0}
    assert rehydrated.lf_thumbnail_drive_urls == {1: "https://drive/lf1.jpg"}
    assert rehydrated.per_lf_regions == {1: ["Kerry", "Portfable"]}
