"""Tests for the generic Phase BC mapping-completeness invariant.

``_assert_mapping_complete`` is the deduplicated primitive that the four
per-field invariants (trim_ratio, distance, elevation, regions) delegate
to. Tests cover the three behaviors the per-field wrappers depend on:

- no-op on empty ``expected_keys`` (graceful for GPS-less drives where
  the upstream key set is empty);
- no-op when every expected key is present (the happy path);
- raise ``RuntimeError`` with the sorted missing-index list, the
  field-name framing, and the optional example-fix tail.

Also pins string-key tolerance on ``actual`` — state files that
round-trip through ``pipeline_state.json`` come back with str keys; the
invariant must coerce them before comparison so post-resume runs don't
spuriously fire.
"""

import pytest

from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    _assert_mapping_complete,
)


class TestAssertMappingComplete:
    def test_empty_expected_keys_noops(self) -> None:
        # GPS-less drive path: per_lf_gpx_path empty → expected_keys empty
        # → invariant must NOT raise (legacy WNC-style cohorts).
        _assert_mapping_complete(set(), {}, field_name="per_lf_distance_km")
        _assert_mapping_complete(set(), {1: 2.0}, field_name="per_lf_distance_km")  # also no-op

    def test_complete_mapping_noops(self) -> None:
        _assert_mapping_complete(
            {1, 2, 3},
            {1: "a", 2: "b", 3: "c"},
            field_name="per_lf_regions",
        )

    def test_missing_indices_raises_with_field_name(self) -> None:
        with pytest.raises(RuntimeError) as excinfo:
            _assert_mapping_complete(
                {1, 2, 3},
                {1: "a"},
                field_name="per_lf_distance_km",
            )
        msg = str(excinfo.value)
        assert "per_lf_distance_km" in msg
        assert "[2, 3]" in msg
        assert "expected 3" in msg
        assert "got 1" in msg

    def test_missing_list_sorted_in_message(self) -> None:
        # Set iteration is not order-stable; the message must sort missing.
        with pytest.raises(RuntimeError) as excinfo:
            _assert_mapping_complete(
                {5, 1, 3, 2, 4},
                {},
                field_name="per_lf_elevation_gain_m",
            )
        assert "[1, 2, 3, 4, 5]" in str(excinfo.value)

    def test_string_keys_in_actual_are_coerced(self) -> None:
        # JSON round-trip: ints become strings. The invariant must coerce
        # and treat "1", "2", "3" as the int keys they represent.
        _assert_mapping_complete(
            {1, 2, 3},
            {"1": "a", "2": "b", "3": "c"},
            field_name="per_lf_regions",
        )

    def test_non_coercible_keys_skipped(self) -> None:
        # Garbage keys (e.g. accidental "foo") are silently skipped — they
        # contribute nothing to ``have`` so the missing-set comparison
        # still surfaces the real gap.
        with pytest.raises(RuntimeError) as excinfo:
            _assert_mapping_complete(
                {1, 2},
                {"1": "a", "foo": "b"},
                field_name="per_lf_regions",
            )
        assert "[2]" in str(excinfo.value)
        assert "got 1" in str(excinfo.value)

    def test_example_fix_appended_when_provided(self) -> None:
        with pytest.raises(RuntimeError) as excinfo:
            _assert_mapping_complete(
                {1},
                {},
                field_name="per_lf_distance_km",
                example_fix="Re-run phase_a to recover.",
            )
        assert "Re-run phase_a to recover." in str(excinfo.value)

    def test_expected_source_desc_threaded(self) -> None:
        with pytest.raises(RuntimeError) as excinfo:
            _assert_mapping_complete(
                {1},
                {},
                field_name="per_lf_distance_km",
                expected_source_desc="per_lf_gpx_path",
            )
        assert "from per_lf_gpx_path" in str(excinfo.value)
