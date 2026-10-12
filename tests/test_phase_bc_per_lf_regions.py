"""Per-LF regions computation in phase_bc_day_first.

The ``🗺️ Through:`` description line is sourced from each
LF's canonical ``ResolvedLocation.regions`` — the SINGLE source of truth the
title/OST and the location-claim gates already consume — NOT an independent
fourth geocode pass over the raw GPX. That fourth pass
(``RouteAnalyzer.get_regions_passed_through``) was ungated and could name a
county no other surface named; reading ``resolved.regions`` makes that
divergence structurally impossible.

- Compute path: ``_compute_per_lf_regions`` reads ``resolved.regions`` for
  every LF that has a GPX path (so the Phase BC completeness invariant still
  holds); no geocoder / GPX parse involved.
- Invariant: ``_assert_per_lf_regions_complete`` raises listing missing
  indices when ``per_lf_gpx_path`` has entries the regions map lacks AND the
  map is non-empty.
"""

import inspect
from types import SimpleNamespace

import pytest

from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    _assert_per_lf_regions_complete,
    _compute_per_lf_regions,
)
from gomoveshift.pipeline.longform_metadata import _format_regions_line


# ────────────────────────────────────────────────────────────────────
# _compute_per_lf_regions — sourced from ResolvedLocation.regions
# ────────────────────────────────────────────────────────────────────


class TestComputePerLfRegionsFromResolved:
    def test_regions_come_from_resolved_model(self):
        resolved = SimpleNamespace(regions=["Nelson County, VA", "Amherst County, VA"])
        result = _compute_per_lf_regions({0: "/lf0.gpx"}, per_lf_resolved={0: resolved})
        assert result[0] == [{"name": "Nelson County, VA"}, {"name": "Amherst County, VA"}]

    def test_through_line_matches_resolved_not_divergent_geocode(self):
        # A raw-GPX 4th pass could clip a corner and name "Buckingham County";
        # the resolved model (title/OST/gates) never voted it. The Through line
        # must reflect resolved.regions, so "Buckingham" can never surface.
        resolved = SimpleNamespace(regions=["Nelson County, VA"])
        result = _compute_per_lf_regions({0: "/lf0.gpx"}, per_lf_resolved={0: resolved})
        line = _format_regions_line(result[0], "Blue Ridge Parkway, VA, USA")
        assert line == "🗺️ Through: Nelson County, VA\n"
        assert "Buckingham" not in line

    def test_no_geocode_pass_signature(self):
        # The independent 4th geocode pass is deleted: the producer no longer
        # accepts a geocoder / parse_fn and derives regions without gps_journey.
        params = inspect.signature(_compute_per_lf_regions).parameters
        assert "geocoder" not in params
        assert "parse_fn" not in params

    def test_empty_per_lf_gpx_path_returns_empty(self):
        resolved = SimpleNamespace(regions=["County Kerry"])
        assert _compute_per_lf_regions({}, per_lf_resolved={0: resolved}) == {}

    def test_gps_lf_without_resolved_gets_empty_list(self):
        # Honest measured-empty: KEY present (invariant satisfied), value [].
        assert _compute_per_lf_regions({0: "/lf0.gpx"}, per_lf_resolved={}) == {0: []}

    def test_string_key_normalized_to_int(self):
        resolved = SimpleNamespace(regions=["County Kerry"])
        result = _compute_per_lf_regions({"0": "/lf0.gpx"}, per_lf_resolved={"0": resolved})
        assert result[0] == [{"name": "County Kerry"}]

    def test_blank_region_labels_filtered(self):
        resolved = SimpleNamespace(regions=["County Kerry", "", None])
        result = _compute_per_lf_regions({0: "/lf0.gpx"}, per_lf_resolved={0: resolved})
        assert result[0] == [{"name": "County Kerry"}]

    def test_resolved_as_plain_dict_supported(self):
        # State can rehydrate a legacy entry as a plain dict, not a dataclass.
        result = _compute_per_lf_regions(
            {0: "/lf0.gpx"}, per_lf_resolved={0: {"regions": ["County Cork"]}}
        )
        assert result[0] == [{"name": "County Cork"}]


# ────────────────────────────────────────────────────────────────────
# _assert_per_lf_regions_complete
# ────────────────────────────────────────────────────────────────────


class TestAssertPerLfRegionsComplete:
    def test_full_coverage_does_not_raise(self):
        per_lf_gpx_path = {0: "/a.gpx", 1: "/b.gpx"}
        mapping = {0: [{"name": "County Kerry"}], 1: []}  # empty list still = coverage
        _assert_per_lf_regions_complete(per_lf_gpx_path, mapping)  # no raise

    def test_missing_entry_raises_with_indices(self):
        per_lf_gpx_path = {0: "/a.gpx", 1: "/b.gpx"}
        mapping = {0: [{"name": "County Kerry"}]}
        with pytest.raises(RuntimeError) as exc:
            _assert_per_lf_regions_complete(per_lf_gpx_path, mapping)
        assert "1" in str(exc.value)
        assert "per_lf_regions" in str(exc.value)

    def test_empty_gpx_path_tolerated(self):
        _assert_per_lf_regions_complete({}, {})  # no raise

    def test_empty_mapping_tolerated_no_raise(self):
        """Empty map = GPS-less drive (no resolved regions) — graceful."""
        _assert_per_lf_regions_complete({0: "/a.gpx"}, {})  # no raise

    def test_extra_entries_in_map_do_not_raise(self):
        per_lf_gpx_path = {0: "/a.gpx"}
        mapping = {0: [{"name": "County Kerry"}], 99: [{"name": "County Stale"}]}
        _assert_per_lf_regions_complete(per_lf_gpx_path, mapping)  # no raise

    def test_string_keyed_per_lf_gpx_path_supported(self):
        per_lf_gpx_path = {"0": "/a.gpx"}
        mapping = {0: [{"name": "County Kerry"}]}
        _assert_per_lf_regions_complete(per_lf_gpx_path, mapping)  # no raise
