"""Tests for the operator-side route_stats sidecar."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gomoveshift.pipeline.route_stats_sidecar import (
    ROUTE_STATS_FILENAME,
    build_route_stats_payload,
    write_route_stats_sidecar,
)

from _state_helpers import make_state


def _state_full():
    return make_state(
        per_lf_titles={
            0: "Killarney to Dingle via N86, Kerry",
            1: "Dingle to Tralee via N86, Kerry",
        },
        per_lf_recording_date={0: "2026-05-17", 1: "2026-05-17"},
        per_lf_distance_km={0: 47.234, 1: 38.0},
        per_lf_elevation_gain_m={0: 823.4, 1: 412.0},
        per_lf_regions={0: ["County Kerry", "Munster"], 1: ["County Kerry"]},
        per_lf_trim_ratio={0: 0.617, 1: 0.92},
    )


def test_payload_aggregates_all_lf_indices() -> None:
    payload = build_route_stats_payload(_state_full())

    assert "generated_at" in payload
    assert len(payload["lfs"]) == 2
    assert [lf["lf_index"] for lf in payload["lfs"]] == [0, 1]


def test_payload_rounds_numerics_to_one_decimal() -> None:
    payload = build_route_stats_payload(_state_full())

    lf0 = payload["lfs"][0]
    assert lf0["distance_km"] == 47.2
    assert lf0["elevation_gain_m"] == 823.4
    assert lf0["trim_ratio"] == 0.6


def test_payload_missing_fields_become_null_not_failures() -> None:
    state = make_state(
        per_lf_titles={0: "A", 1: "B"},
        # No distance, no elevation, no regions, no trim_ratio
    )
    payload = build_route_stats_payload(state)

    assert len(payload["lfs"]) == 2
    for lf in payload["lfs"]:
        assert lf["distance_km"] is None
        assert lf["elevation_gain_m"] is None
        assert lf["regions"] is None
        assert lf["trim_ratio"] is None


def test_payload_tolerates_string_keys_from_json_roundtrip() -> None:
    state = make_state(
        per_lf_titles={"0": "A"},
        per_lf_distance_km={"0": 12.0},
    )
    payload = build_route_stats_payload(state)

    assert payload["lfs"][0]["lf_index"] == 0
    assert payload["lfs"][0]["title"] == "A"
    assert payload["lfs"][0]["distance_km"] == 12.0


def test_payload_skips_unparseable_keys() -> None:
    state = make_state(
        per_lf_titles={0: "ok", "not-an-int": "garbage"},
    )
    payload = build_route_stats_payload(state)

    assert [lf["lf_index"] for lf in payload["lfs"]] == [0]


def test_payload_regions_coerced_to_list_of_strings() -> None:
    state = make_state(per_lf_regions={0: ["A", "", None, "B"]})
    payload = build_route_stats_payload(state)

    assert payload["lfs"][0]["regions"] == ["A", "B"]


def test_payload_empty_state_yields_no_lfs() -> None:
    payload = build_route_stats_payload(make_state())
    assert payload["lfs"] == []


def test_write_creates_file_at_expected_path(tmp_path: Path) -> None:
    written = write_route_stats_sidecar(tmp_path, _state_full())

    assert written is not None
    assert written.name == ROUTE_STATS_FILENAME
    assert written.parent == tmp_path
    assert written.exists()


def test_write_emits_valid_json_with_sorted_lf_order(tmp_path: Path) -> None:
    state = _state_full()
    # Insert LFs in non-sorted order to test sort-on-emit.
    state.per_lf_titles = {1: "B", 0: "A"}

    path = write_route_stats_sidecar(tmp_path, state)
    assert path is not None

    data = json.loads(path.read_text())
    assert [lf["lf_index"] for lf in data["lfs"]] == [0, 1]


def test_write_skips_when_no_lf_data(tmp_path: Path) -> None:
    written = write_route_stats_sidecar(tmp_path, make_state())
    assert written is None
    assert not (tmp_path / ROUTE_STATS_FILENAME).exists()


def test_write_creates_missing_output_dir(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "subdir"
    assert not target.exists()

    written = write_route_stats_sidecar(target, _state_full())
    assert written is not None
    assert target.is_dir()


def test_write_overwrites_existing_file(tmp_path: Path) -> None:
    path = tmp_path / ROUTE_STATS_FILENAME
    path.write_text('{"stale": true}')

    write_route_stats_sidecar(tmp_path, _state_full())

    data = json.loads(path.read_text())
    assert "stale" not in data
    assert "lfs" in data


def test_write_handles_oserror_gracefully(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """OSError on write returns None instead of raising — sidecar is best-effort."""

    def boom(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", boom)

    result = write_route_stats_sidecar(tmp_path, _state_full())
    assert result is None
