"""Opt-in ``--map-inset roads``: the route inset plus nearby roads under the route line.

``route`` stays byte-identical; ``off`` is unchanged. Fixture bundles keep schema /1 and
add ``roads.geojson`` plus the manifest ``roads`` key, which is all the readers check.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import ImageColor
from test_map_inset import CREDIT, _make_bundle, _route

from gomoveshift.pipeline import map_inset
from gomoveshift.pipeline.edit_timeline import EditTimeline
from gomoveshift.pipeline.map_inset import MapInsetRefused, plan_route_inset
from gomoveshift.pipeline.route_claims import RouteClaimRefusedError, map_inset_inputs
from gomoveshift.pipeline.route_story import bundle as bundle_mod

FRAME = (1920, 1080)
GPS_DAY = [(float(t), 35.2 + 0.0002 * t, -83.6 + 0.0004 * t) for t in range(0, 101, 2)]


def _road(coords, highway="primary", osm_id=1):
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [list(c) for c in coords]},
        "properties": {"highway": highway, "osm_id": osm_id},
    }


def _parallel_road(highway="primary"):
    """A road beside the route (about 60 px away in the inset), never crossing it."""
    coords = [(-83.592 + 0.0004 * t, 35.196 + 0.0002 * t) for t in range(0, 81, 4)]
    return _road(coords, highway)


def _add_roads(root: Path, features, **manifest_roads) -> Path:
    """Write roads.geojson and the manifest ``roads`` key onto a schema /1 bundle."""
    (root / "roads.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest.pop("files", None)
    manifest["roads"] = {
        "bbox": [-83.7, 35.1, -83.4, 35.3],
        "count": len(features),
        "degraded": False,
        "problems": [],
        "provenance": "test",
        **manifest_roads,
    }
    bundle_mod.write_manifest(root, manifest)
    assert bundle_mod.verify_bundle(root) == []
    return root


@pytest.fixture()
def route_bundle(tmp_path):
    return _make_bundle(tmp_path / "plain")


@pytest.fixture()
def roads_bundle(tmp_path):
    return _add_roads(_make_bundle(tmp_path / "roads"), [_parallel_road()])


def _plan(bundle, tmp_path, **kw):
    return plan_route_inset(
        bundle,
        [_route()],
        EditTimeline.from_segments([(0.0, 20.0)]),
        FRAME,
        workdir=tmp_path / "work",
        **kw,
    )


def _drive(bundle, mode="roads", **kw):
    opts = dict(
        map_inset=mode,
        map_inset_bundle=str(bundle),
        map_inset_trim_start_m=150.0,
        map_inset_trim_end_m=150.0,
        map_inset_privacy_zones=[],
    )
    opts.update(kw)
    return SimpleNamespace(**opts)


# -- config / CLI -----------------------------------------------------------------------


def test_config_accepts_roads_and_rejects_junk():
    from gomoveshift.config import DriveConfig

    assert DriveConfig(map_inset="roads").map_inset == "roads"
    assert DriveConfig(map_inset="route").map_inset == "route"
    with pytest.raises(ValueError, match="map_inset"):
        DriveConfig(map_inset="streets")


def test_cli_map_inset_choices_include_roads():
    from unittest.mock import patch

    with patch("importlib.metadata.version", return_value="0"):
        from gomoveshift.cli.parser import _build_parser

        parser = _build_parser()
    sub = next(
        a for a in parser._actions if hasattr(a, "choices") and a.choices and "drive" in a.choices
    )
    action = next(a for a in sub.choices["drive"]._actions if a.dest == "map_inset")
    assert action.choices == ["off", "route", "roads"]
    assert "roads" in action.help


# -- refusals before any encode ---------------------------------------------------------


def test_roads_mode_accepts_a_good_bundle(roads_bundle):
    inputs = map_inset_inputs(_drive(roads_bundle), GPS_DAY, date="d")
    assert inputs.roads is True


def test_missing_roads_file_and_key_are_listed_together(route_bundle):
    with pytest.raises(RouteClaimRefusedError) as caught:
        map_inset_inputs(_drive(route_bundle), GPS_DAY, date="d")
    message = str(caught.value)
    assert "roads.geojson" in message
    assert "no roads" in message and "manifest" in message
    assert "nothing was encoded" in message


def test_degraded_roads_name_their_problems(tmp_path):
    root = _add_roads(
        _make_bundle(tmp_path / "b"),
        [_parallel_road()],
        degraded=True,
        problems=["overpass timed out"],
    )
    with pytest.raises(RouteClaimRefusedError, match="degraded.*overpass timed out"):
        map_inset_inputs(_drive(root), GPS_DAY, date="d")


def test_zero_road_features_refuse(tmp_path):
    root = _add_roads(_make_bundle(tmp_path / "b"), [])
    with pytest.raises(RouteClaimRefusedError, match="zero road"):
        map_inset_inputs(_drive(root), GPS_DAY, date="d")


def test_every_roads_reason_sits_beside_the_existing_ones(route_bundle):
    drive = _drive(route_bundle, map_inset_trim_start_m=0.0)
    with pytest.raises(RouteClaimRefusedError) as caught:
        map_inset_inputs(drive, [], date="d")
    message = str(caught.value)
    for fragment in ("roads.geojson", "trim", "no usable GPS"):
        assert fragment in message


def test_route_mode_ignores_roads_entirely(route_bundle, roads_bundle):
    # A bundle with no road layer is fine for route mode.
    assert map_inset_inputs(_drive(route_bundle, "route"), GPS_DAY, date="d").roads is False
    assert map_inset_inputs(_drive(roads_bundle, "route"), GPS_DAY, date="d").roads is False


# -- signature --------------------------------------------------------------------------


def _legacy_route_signature(bundle: Path) -> str:
    from gomoveshift.pipeline.route_claims import day_route_segments

    segments = day_route_segments(GPS_DAY)
    return hashlib.sha256(
        json.dumps(
            {
                "mode": "route",
                "bundle_manifest_sha256": hashlib.sha256(
                    (bundle / "manifest.json").read_bytes()
                ).hexdigest(),
                "trim": [150.0, 150.0],
                "zones": [],
                "layout": ["bottom-left", 0.20],
                "route": [[(p.time_s, p.latitude, p.longitude) for p in s] for s in segments],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def test_route_signature_is_unchanged_even_with_roads_present(route_bundle, roads_bundle):
    for bundle in (route_bundle, roads_bundle):
        got = map_inset_inputs(_drive(bundle, "route"), GPS_DAY, date="d").signature
        assert got == _legacy_route_signature(bundle)


def test_roads_signature_differs_and_tracks_the_roads_file(tmp_path):
    root = _add_roads(_make_bundle(tmp_path / "b"), [_parallel_road()])
    route_sig = map_inset_inputs(_drive(root, "route"), GPS_DAY, date="d").signature
    first = map_inset_inputs(_drive(root), GPS_DAY, date="d").signature
    assert first != route_sig
    assert map_inset_inputs(_drive(root), GPS_DAY, date="d").signature == first
    _add_roads(root, [_parallel_road(), _parallel_road("residential")])
    assert map_inset_inputs(_drive(root), GPS_DAY, date="d").signature != first


# -- rendering --------------------------------------------------------------------------


def _pixel(plan, lat, lon):
    x, y = plan.project(lat, lon)
    return plan.base.getpixel((round(x), round(y)))[:3]


def _near(plan, lat, lon):
    """Pixels within one pixel of a position (a thin line may straddle the rounding)."""
    x, y = (round(v) for v in plan.project(lat, lon))
    return {plan.base.getpixel((x + dx, y + dy))[:3] for dx in (-1, 0, 1) for dy in (-1, 0, 1)}


def test_roads_are_drawn_away_from_the_route(route_bundle, roads_bundle, tmp_path):
    plain = _plan(route_bundle, tmp_path / "a")
    with_roads = _plan(roads_bundle, tmp_path / "b", roads=True)
    lat, lon = 35.196 + 0.0002 * 40, -83.592 + 0.0004 * 40  # mid road
    road = ImageColor.getrgb(map_inset._ROAD_MAJOR_RGB)
    assert road in _near(with_roads, lat, lon)
    assert road not in _near(plain, lat, lon)


def test_road_width_follows_class(roads_bundle):
    assert map_inset.road_width_px("motorway_link", 384) > map_inset.road_width_px(
        "residential", 384
    )
    assert map_inset.road_width_px("primary", 384) == map_inset.road_width_px("trunk", 384)
    assert map_inset.road_width_px("residential", 1920) >= map_inset.road_width_px(
        "residential", 384
    )


def test_the_route_is_drawn_over_a_coincident_road(tmp_path):
    along = _road([(-83.60 + 0.0004 * t, 35.20 + 0.0002 * t) for t in range(0, 101, 5)])
    root = _add_roads(_make_bundle(tmp_path / "b"), [along])
    plan = _plan(root, tmp_path, roads=True)
    got = _near(plan, 35.2 + 0.0002 * 50, -83.6 + 0.0004 * 50)
    assert ImageColor.getrgb(map_inset._LINE_RGB) in got
    assert ImageColor.getrgb(map_inset._ROAD_MAJOR_RGB) not in got


def test_route_mode_bytes_ignore_a_roads_file(route_bundle, roads_bundle, tmp_path):
    plain = _plan(route_bundle, tmp_path / "a")
    beside = _plan(roads_bundle, tmp_path / "b")
    assert plain.base.tobytes() == beside.base.tobytes()
    assert plain.credits == beside.credits == (CREDIT,)


def test_attribution_text_only_in_roads_mode():
    assert map_inset.credit_text(CREDIT, roads=False) == f"Imagery: {CREDIT}"
    assert (
        map_inset.credit_text(CREDIT, roads=True)
        == f"Imagery: {CREDIT} · Roads: © OpenStreetMap contributors"
    )


def test_roads_credit_strip_is_drawn(route_bundle, roads_bundle, tmp_path):
    plain = _plan(route_bundle, tmp_path / "a")
    with_roads = _plan(roads_bundle, tmp_path / "b", roads=True)
    w, h = plain.size

    def white(plan):
        return int((np.asarray(plan.base.crop((0, h - 40, w, h)))[..., :3] == 255).all(-1).sum())

    assert white(with_roads) > white(plain)
    assert "Roads" in with_roads.credits[-1]


def test_zero_roads_in_view_refuses_naming_the_reason(tmp_path):
    far = _road([(-80.0, 40.0), (-80.01, 40.01)])
    root = _add_roads(_make_bundle(tmp_path / "b"), [far])
    with pytest.raises(MapInsetRefused, match="no roads in view"):
        _plan(root, tmp_path, roads=True)


def test_roads_mode_refuses_when_the_layer_is_gone(route_bundle, tmp_path):
    with pytest.raises(MapInsetRefused, match="roads"):
        _plan(route_bundle, tmp_path, roads=True)


def test_receipt_records_the_mode_only_for_roads(tmp_path):
    from dataclasses import replace

    from test_regular_lf_map_inset import _inset_receipt

    base = _inset_receipt()
    assert "mode" not in base.to_dict()
    roads = replace(base, mode="roads")
    assert roads.to_dict()["mode"] == "roads"
    path = tmp_path / "r.json"
    path.write_text(json.dumps(roads.to_dict()), encoding="utf-8")
    assert map_inset.InsetReceipt.load(path).mode == "roads"
