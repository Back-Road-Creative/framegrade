"""A bundle's cache address carries its SHAPE, not just its sources.

``bundle_dir_for`` keyed a bundle by footprint + source versions + terrain
exaggeration. It did not carry the zoom ranges the bundle was built for, nor
any version for the imagery-band ladder that decides which source covers which
zooms. Two consequences, both live:

* Re-running ``prepare`` after the ladder changed shape returned the OLD
  bundle. ``_find_cached_bundle`` short-circuits the whole fetch on any
  existing directory whose manifest verifies clean, and the directory was the
  same one, so a ladder fix could not reach a trip that had already been
  prepared -- the operator's only remedy was to know the hash and delete it by
  hand.
* A bundle built for a narrow zoom range satisfied a request for a wide one.
  ``_report_from_existing`` reports the REQUESTED zooms, never the ones on
  disk, so the receipt claimed z2-15 for a bundle holding only z13-15 and the
  render found nothing at the wide end.

Putting the shape in the address fixes both by construction: a differently
shaped bundle is a different directory, so it can neither be silently reused
nor silently reported as something it is not.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from gomoveshift.pipeline.route_story import bands, bundle, prepare

FOOTPRINT = bundle.Footprint(west=-84.0, south=34.5, east=-83.0, north=35.5)
WIDE_IMAGERY = tuple(range(2, 16))
NARROW_IMAGERY = tuple(range(13, 16))
TERRAIN = (9, 10, 11, 12, 13)
RECORDED_ON = date(2026, 9, 16)


@pytest.fixture(autouse=True)
def _cache_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GMS_ROUTE_STORY_CACHE_DIR", str(tmp_path / "cache"))


class TestTheAddressCarriesTheShape:
    def test_a_different_imagery_zoom_range_is_a_different_bundle(self) -> None:
        wide = bundle.bundle_dir_for(
            FOOTPRINT, {"osm": "v1"}, imagery_zooms=WIDE_IMAGERY, terrain_zooms=TERRAIN
        )
        narrow = bundle.bundle_dir_for(
            FOOTPRINT, {"osm": "v1"}, imagery_zooms=NARROW_IMAGERY, terrain_zooms=TERRAIN
        )
        assert wide != narrow

    def test_a_different_terrain_zoom_range_is_a_different_bundle(self) -> None:
        deep = bundle.bundle_dir_for(
            FOOTPRINT, {"osm": "v1"}, imagery_zooms=WIDE_IMAGERY, terrain_zooms=TERRAIN
        )
        shallow = bundle.bundle_dir_for(
            FOOTPRINT, {"osm": "v1"}, imagery_zooms=WIDE_IMAGERY, terrain_zooms=(9, 10)
        )
        assert deep != shallow

    def test_the_zooms_are_canonicalised_like_the_source_versions(self) -> None:
        """Order and duplicates are not shape. A caller that passes the same
        zooms in a different order must land on its own bundle, not fetch a
        second copy of it."""
        one = bundle.bundle_dir_for(
            FOOTPRINT, {"osm": "v1"}, imagery_zooms=(13, 14, 15), terrain_zooms=(9, 10)
        )
        other = bundle.bundle_dir_for(
            FOOTPRINT, {"osm": "v1"}, imagery_zooms=(15, 13, 14, 13), terrain_zooms=(10, 9, 9)
        )
        assert one == other

    def test_omitting_the_zooms_still_addresses_the_pre_shape_bundles(self) -> None:
        """Back-compat, deliberately: bundles already on disk were addressed
        without a shape and must keep their address, or every cached trip is
        orphaned rather than superseded. Only a caller that states its shape
        gets a shape-keyed address."""
        assert bundle.bundle_dir_for(FOOTPRINT, {"osm": "v1"}) == bundle.bundle_dir_for(
            FOOTPRINT, {"osm": "v1"}, imagery_zooms=None, terrain_zooms=None
        )


def _versions(**overrides: object) -> dict[str, str]:
    kwargs: dict[str, object] = {
        "gibs_chosen": "blue_marble",
        "recorded_on": RECORDED_ON,
        "terrain_focal_name": prepare.DEM_3DEP_PROVENANCE_NAME,
        "terrain_regional_name": prepare.DEM_3DEP_PROVENANCE_NAME,
        "regional_margin_km": bands.DEFAULT_REGIONAL_MARGIN_KM,
    }
    kwargs.update(overrides)
    return prepare._band_source_versions(**kwargs)  # type: ignore[arg-type]


class TestTheLadderVersionIsPartOfTheKey:
    def test_prepare_puts_the_band_plan_version_in_its_source_versions(self) -> None:
        """The zoom RANGE is not enough: the five-band ladder changed which
        source covers which zoom without changing the range. Only an explicit
        version for the plan's shape can invalidate that."""
        assert _versions()["bands"] == bands.BAND_PLAN_VERSION

    def test_bumping_the_band_plan_version_moves_the_bundle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        before = _versions()
        monkeypatch.setattr(prepare, "BAND_PLAN_VERSION", "bands-test-next")
        after = _versions()
        assert bundle.bundle_dir_for(FOOTPRINT, before) != bundle.bundle_dir_for(FOOTPRINT, after)


def _write_verifiable_bundle(root: Path) -> None:
    """The three files ``_report_from_existing`` reads, and nothing else.

    ``verify_bundle`` is stubbed clean by the test that uses this: what is
    under test is which directory gets consulted, not what makes one valid.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps({"files": ["imagery/13/0/0.jpg", "terrain/9/0/0.png"], "labels": {}}),
        encoding="utf-8",
    )
    (root / "labels.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8"
    )
    (root / "route.geojson").write_text(
        json.dumps({"type": "Feature", "properties": {}}), encoding="utf-8"
    )


class TestPrepareWillNotReuseADifferentlyShapedBundle:
    def test_a_narrow_bundle_is_not_handed_back_for_a_wide_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point. Before this, the narrow bundle WAS returned, and
        its report claimed the wide zooms it did not have."""
        monkeypatch.setattr(prepare, "verify_bundle", lambda _root: [])
        _write_verifiable_bundle(
            bundle.bundle_dir_for(
                FOOTPRINT,
                _versions(),
                terrain_exaggeration=1.0,
                imagery_zooms=NARROW_IMAGERY,
                terrain_zooms=TERRAIN,
            )
        )
        assert (
            prepare._find_cached_bundle(
                FOOTPRINT,
                WIDE_IMAGERY,
                TERRAIN,
                RECORDED_ON,
                1.0,
                bands.DEFAULT_REGIONAL_MARGIN_KM,
                False,
            )
            is None
        )

    def test_the_same_shape_is_still_a_cache_hit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The fix must not turn every repeat run into a re-fetch."""
        monkeypatch.setattr(prepare, "verify_bundle", lambda _root: [])
        root = bundle.bundle_dir_for(
            FOOTPRINT,
            _versions(),
            terrain_exaggeration=1.0,
            imagery_zooms=WIDE_IMAGERY,
            terrain_zooms=TERRAIN,
        )
        _write_verifiable_bundle(root)
        report = prepare._find_cached_bundle(
            FOOTPRINT,
            WIDE_IMAGERY,
            TERRAIN,
            RECORDED_ON,
            1.0,
            bands.DEFAULT_REGIONAL_MARGIN_KM,
            False,
        )
        assert report is not None
        assert report.bundle_dir == root
        assert report.imagery_zooms == WIDE_IMAGERY
