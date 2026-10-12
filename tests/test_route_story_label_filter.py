"""The scene sends back the labels it can actually draw, not all of them.

``labels.geojson`` for a typical scene bundle holds tens of thousands of
features. ``projectLabels`` ran ``map.project`` over every one of them for
every frame and marshalled the whole array back across the browser boundary,
and then every Python consumer discarded all but the handful that are on canvas.

Dropping the off-canvas entries in the browser is safe because EVERY consumer
already requires ``visible``: ``proof.place_labels`` skips a label without it
and both marker lookups test ``entry.get("visible")``. The marker entry is
kept regardless of its own flag: it is not a label, and a future consumer
that wants to know where an off-screen marker went must still be able to.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from gomoveshift.pipeline.route_story import proof
from gomoveshift.pipeline.route_story.frame_qa import marker_xy_from_labels
from gomoveshift.pipeline.route_story.scene import capture as capture_mod

_SCENE_JS = (Path(capture_mod.__file__).parent / "scene.js").read_text(encoding="utf-8")


def _projected(name, x, y, visible, kind="peak"):
    return {"name": name, "kind": kind, "x": x, "y": y, "visible": visible}


class TestTheBrowserSendsOnlyWhatCanBeDrawn:
    def test_the_projection_is_filtered_before_it_is_returned(self):
        assert "filter(" in _SCENE_JS
        assert "entry.visible" in _SCENE_JS

    def test_the_marker_is_pushed_after_the_filter_so_it_is_never_dropped(self):
        marker_at = _SCENE_JS.index('kind: "marker"')
        filter_at = _SCENE_JS.index("entry.visible")
        assert filter_at < marker_at, "the marker must survive the label filter"


_NODE = shutil.which("node")


def _function_source(name: str) -> str:
    """One scene.js function's own shipped source, lifted out of the IIFE."""
    start = _SCENE_JS.index(f"function {name}(")
    depth = 0
    for index in range(start, len(_SCENE_JS)):
        if _SCENE_JS[index] == "{":
            depth += 1
        elif _SCENE_JS[index] == "}":
            depth -= 1
            if depth == 0:
                return _SCENE_JS[start : index + 1]
    raise AssertionError(f"{name} has no closing brace")


def _feature(name: str, lon: float, lat: float) -> dict:
    return {
        "geometry": {"coordinates": [lon, lat]},
        "properties": {"name": name, "kind": "peak"},
    }


_FILM_VIEW = {
    "zoom": 15.32,
    "pitch": 60.0,
    "fov": 36.87,
    "lng": -83.5604,
    "lat": 35.2357,
    "elevation": 1400.0,
}


def _run_project_labels(features: list[dict], view: dict, marker: list[float] | None) -> dict:
    """Run the shipped ``projectLabels`` under node against a counting fake map.

    The fake puts every projected point on canvas: what is being pinned is
    which labels reach ``map.project`` at all, since that call is the cost.
    """
    script = (
        "const labels = " + json.dumps(features) + ";\n"
        "const V = " + json.dumps(view) + ";\n"
        "const LABEL_CULL_FLOOR_M = 500;\n"
        "let calls = 0;\n"
        "const map = {\n"
        "  getCanvas: () => ({width: 1920, height: 923}),\n"
        "  getCenter: () => ({lng: V.lng, lat: V.lat}),\n"
        "  getZoom: () => V.zoom, getPitch: () => V.pitch,\n"
        "  getVerticalFieldOfView: () => V.fov,\n"
        "  getCameraTargetElevation: () => V.elevation,\n"
        "  project: () => { calls++; return {x: 10, y: 10}; },\n"
        "};\n"
        + "\n".join(
            _function_source(name)
            for name in ("labelCullRadiusM", "cullLabelsToRadius", "projectLabels")
        )
        + "\nconst out = projectLabels("
        + json.dumps(marker)
        + ");\nconsole.log(JSON.stringify({calls: calls, out: out}));\n"
    )
    result = subprocess.run(
        [_NODE or "node", "--input-type=module"],
        input=script,  # stdin, not -e: 5,000 features overflow an argv
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip())


def _cull_radius_m(**overrides: float) -> float | None:
    """The shipped ``labelCullRadiusM`` for the film's own view, in metres
    (``None`` for Infinity, which JSON cannot carry)."""
    view = {
        "zoom": 15.32,
        "pitchDeg": 60.0,
        "latDeg": 35.2357,
        "fovDeg": 36.87,
        "width": 1920,
        "height": 923,
        "centreElevationM": 1400.0,
        "floorM": 500.0,
        **overrides,
    }
    script = (
        _function_source("labelCullRadiusM")
        + "\nconst r = labelCullRadiusM("
        + json.dumps(view)
        + ");\nconsole.log(JSON.stringify(isFinite(r) ? r : null));\n"
    )
    result = subprocess.run(
        [_NODE or "node", "--input-type=module"],
        input=script,  # stdin, not -e: 5,000 features overflow an argv
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip())


@pytest.mark.skipif(_NODE is None, reason="node is needed to run scene.js's own source")
class TestOnlyLabelsThatCanBeOnScreenAreProjected:
    """``map.project`` is terrain-aware, so projecting every label of a large
    scene bundle dominates the cost of each frame even though only a handful
    are drawn. With the cull, a pose settles far faster and returns the
    identical visible labels at identical pixels."""

    def test_a_region_of_far_labels_costs_no_projections(self):
        far = [_feature(f"far-{i}", -86.0 + i * 0.0001, 33.5) for i in range(5000)]
        near = [_feature("Rocky Bald", -83.55, 35.22), _feature("Tellico Gap", -83.57, 35.23)]

        got = _run_project_labels(far + near, _FILM_VIEW, marker=[-83.56, 35.235])

        assert got["calls"] == len(near) + 1, "far labels must be culled before map.project"
        assert [e["name"] for e in got["out"]] == ["Rocky Bald", "Tellico Gap", ""]

    def test_the_marker_is_projected_wherever_it_is(self):
        got = _run_project_labels([], _FILM_VIEW, marker=[-80.0, 30.0])

        assert [e["kind"] for e in got["out"]] == ["marker"]
        assert got["calls"] == 1

    def test_the_radius_reaches_past_where_the_top_corner_ray_lands(self):
        """The bound is the camera's own geometry, NOT ``map.getBounds()``:
        with terrain on, the bounds shrink to where the corner rays hit a near
        ridge, and a cull built on them would drop a label that is on screen
        (here "Tellico Gap"). At the film's pose the camera sits ~1.1 km
        over the centre and its top corner ray lands ~5.9x its drop away."""
        radius = _cull_radius_m()

        assert radius is not None
        assert 15_000 < radius < 30_000

    def test_lower_ground_only_ever_widens_the_radius(self):
        """The ray travels farthest over the lowest ground, so a higher centre
        (more room to fall to the floor) must never shrink the bound."""
        assert _cull_radius_m(centreElevationM=3000.0) > _cull_radius_m(centreElevationM=0.0)

    def test_a_view_with_the_horizon_in_frame_is_not_culled(self):
        assert _cull_radius_m(pitchDeg=80.0) is None

    def test_a_continental_view_is_not_culled(self):
        """One latitude's metres-per-pixel does not describe a wide view; the
        cull stands down rather than drop a label on bad arithmetic."""
        assert _cull_radius_m(zoom=5.0, pitchDeg=0.0) is None

    def test_an_unculled_view_projects_every_label(self):
        features = [_feature("a", -120.0, 45.0), _feature("b", -70.0, 30.0)]

        got = _run_project_labels(features, {**_FILM_VIEW, "pitch": 80.0}, marker=None)

        assert got["calls"] == len(features)


class TestDroppingInvisibleLabelsChangesNothingDownstream:
    """The filter is only safe while every consumer ignores invisible entries.
    These pin that contract: if one ever starts reading them, this fails."""

    def test_place_labels_gives_the_same_pills_either_way(self):
        everything = [
            _projected("Rocky Bald", 600.0, 230.0, True),
            _projected("Offscreen Knob", -400.0, 230.0, False),
            _projected("Black Bald", 900.0, 350.0, True),
            _projected("Way Below", 640.0, 9000.0, False),
        ]
        onscreen = [entry for entry in everything if entry["visible"]]

        assert [p.text for p in proof.place_labels(everything, 1280, 720)] == [
            p.text for p in proof.place_labels(onscreen, 1280, 720)
        ]

    def test_an_invisible_marker_is_no_marker_position_either_way(self):
        offscreen = [_projected("", -5.0, -5.0, False, kind="marker")]

        assert marker_xy_from_labels(offscreen) is None
        assert marker_xy_from_labels([]) is None

    def test_a_visible_marker_still_reports_its_position(self):
        assert marker_xy_from_labels([_projected("", 12.0, 34.0, True, kind="marker")]) == (
            12.0,
            34.0,
        )
