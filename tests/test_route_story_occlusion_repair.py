"""A single hidden vertex between two visible ones is an artifact, not a ridge.

scene.js decides whether terrain hides a route vertex by projecting it to its
own screen point and asking MapLibre to unproject that pixel back: a visible
vertex round-trips to float precision, one behind a ridge recovers the ridge's
own (very different) position instead. Near a terrain tile edge -- or wherever
the DEM sample the ray happens to hit belongs to the next post over -- that
round trip can come back a few metres off for ONE vertex while both of its
neighbours recover exactly.

That single false verdict used to break the drawn line. ``strokeRouteParts``
plotted vertices in a streaming pass, so a hidden one ended the current subpath
and the next visible one started a fresh ``moveTo``. When the artifact landed
two vertices short of the marker, the remaining stub was a fraction of a pixel
long and drew nothing, leaving the travelled line short of a marker that is
always drawn.

One vertex a metre or so over the 12.5 m tolerance, between visible
neighbours and with the marker itself visible, is enough to break the line
short of the marker: the "red dot lags behind the bright line" defect that
frame_qa's pixel ceiling catches.

Repairing a run of ONE cannot draw a line through a mountain: a real ridge
hides a contiguous run (genuinely hidden vertices recover 37-990 m, and every
span is subdivided to OCCLUSION_SUBDIVIDE_PX so a real crossing always
produces several in a row). Any run of two or more is left hidden, which the
cases below pin down.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from gomoveshift.pipeline.route_story import layers

SCENE_JS = Path(layers.__file__).parent / "scene" / "scene.js"
NODE = shutil.which("node")


def _function_source(name: str) -> str:
    """A single top-level function's own source, lifted out of the IIFE.

    scene.js runs inside a browser page and keeps everything in one closure, so
    there is nothing to import. Extracting the function by brace matching lets
    the real shipped source -- not a copy that can drift -- be exercised
    directly.
    """
    source = SCENE_JS.read_text(encoding="utf-8")
    start = source.index("function " + name + "(")
    depth = 0
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(name + " has no closing brace")


def _repair_function_source() -> str:
    """``repairIsolatedOcclusions``'s own source (see ``_function_source``)."""
    return _function_source("repairIsolatedOcclusions")


def _repair(flags: list[bool]) -> list[bool]:
    """Run the shipped repair over ``flags`` (True = occluded) under node."""
    script = (
        _repair_function_source()
        + "\nconst samples = "
        + json.dumps([{"occluded": flag} for flag in flags])
        + ";\nrepairIsolatedOcclusions(samples);"
        + "\nconsole.log(JSON.stringify(samples.map((s) => s.occluded)));\n"
    )
    result = subprocess.run(
        [NODE or "node", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip())


@pytest.mark.skipif(NODE is None, reason="node is needed to run scene.js's own source")
class TestAnIsolatedOcclusionIsRepaired:
    def test_one_hidden_vertex_among_visible_ones(self) -> None:
        """The last eight vertices of a refused pose: only index 5 is hidden."""
        assert _repair([False] * 5 + [True] + [False] * 2) == [False] * 8

    def test_a_single_hidden_vertex_between_two_visible_ones(self) -> None:
        assert _repair([False, True, False]) == [False, False, False]

    def test_several_separate_artifacts_are_each_repaired(self) -> None:
        assert _repair([False, True, False, True, False]) == [False] * 5


@pytest.mark.skipif(NODE is None, reason="node is needed to run scene.js's own source")
class TestARealRidgeStaysHidden:
    def test_two_hidden_vertices_in_a_row_are_left_alone(self) -> None:
        """The narrowest real occlusion still spans more than one vertex."""
        assert _repair([False, True, True, False]) == [False, True, True, False]

    def test_a_long_hidden_run_is_left_alone(self) -> None:
        assert _repair([False] + [True] * 6 + [False]) == [False] + [True] * 6 + [False]

    def test_repairing_one_gap_never_cascades_into_its_neighbour(self) -> None:
        """The pass walks in place, so a just-repaired vertex becomes a visible
        neighbour for the next one it looks at. That must not let a real
        occlusion through: in ``[vis, hidden, vis, hidden, hidden, vis]`` the
        lone artifact at index 1 is repaired and the genuine pair at 3-4
        survives untouched, because each of them still has an occluded
        neighbour on one side."""
        assert _repair([False, True, False, True, True, False]) == [
            False,
            False,
            False,
            True,
            True,
            False,
        ]

    def test_a_hidden_vertex_at_either_end_has_no_neighbours_to_vouch_for_it(
        self,
    ) -> None:
        assert _repair([True, False, False]) == [True, False, False]
        assert _repair([False, False, True]) == [False, False, True]


class TestTheRepairIsWiredIntoTheDrawing:
    """An unreferenced repair would pass every case above and fix nothing."""

    def test_stroke_route_parts_repairs_before_it_draws(self) -> None:
        source = SCENE_JS.read_text(encoding="utf-8")
        start = source.index("function strokeRouteParts(")
        body = source[start : start + 2500]
        repair_at = body.index("repairIsolatedOcclusions(samples)")
        draw_at = body.index("ctx.beginPath()")
        assert repair_at < draw_at, (
            "strokeRouteParts must repair isolated occlusions before it opens "
            "the path, or the drawing pass sees the raw verdicts"
        )

    def test_the_streaming_plotter_that_could_not_see_neighbours_is_gone(self) -> None:
        assert "plotVertex" not in SCENE_JS.read_text(encoding="utf-8")


def _force_trailing(flags: list[bool], marker_visible: bool) -> list[bool]:
    """Run the shipped ``forceTrailingSpanToMarker`` over ``flags`` under node."""
    script = (
        _function_source("forceTrailingSpanToMarker")
        + "\nconst samples = "
        + json.dumps([{"occluded": flag} for flag in flags])
        + "\nforceTrailingSpanToMarker(samples, "
        + json.dumps(marker_visible)
        + ");"
        + "\nconsole.log(JSON.stringify(samples.map((s) => s.occluded)));\n"
    )
    result = subprocess.run(
        [NODE or "node", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip())


@pytest.mark.skipif(NODE is None, reason="node is needed to run scene.js's own source")
class TestTheTravelledLineAlwaysReachesAVisibleMarker:
    """A false-occlusion RUN of two or more consecutive vertices (not a lone
    artifact ``repairIsolatedOcclusions`` already fixes) right before a
    visible marker leaves the last drawn subpath a few pixels short of a dot
    that is always drawn. Since the ground under a visible marker is
    by construction not occluded, the trailing run adjacent to it is forced
    visible too -- but only that trailing run, never a hidden span further
    back or the marker's own vertex when the marker itself is hidden.
    """

    def test_a_false_occlusion_run_before_a_visible_marker_is_forced_visible(self) -> None:
        """Two falsely-occluded vertices sit right before a visible marker."""
        assert _force_trailing([False, False, True, True, False], True) == [False] * 5

    def test_the_mountain_case_a_hidden_run_away_from_the_marker_stays_hidden(
        self,
    ) -> None:
        """The marker is visible and the tail is already clear, but a real
        ridge earlier along the path hides a contiguous run: that run is not
        adjacent to the marker, so forcing must leave it alone."""
        assert _force_trailing([False, True, True, False, False], True) == [
            False,
            True,
            True,
            False,
            False,
        ]

    def test_an_invisible_marker_gets_no_forced_span(self) -> None:
        """When the marker itself is hidden behind a ridge, nothing is forced --
        a hidden marker must not fabricate a line to a point that will not be
        drawn."""
        assert _force_trailing([False, False, True, True, False], False) == [
            False,
            False,
            True,
            True,
            False,
        ]


def _repair_shallow(samples: list[dict[str, object]]) -> list[bool]:
    """Run the shipped ``repairShallowOcclusionRuns`` over ``samples`` under node."""
    script = (
        _function_source("repairShallowOcclusionRuns")
        + "\nvar SHALLOW_OCCLUSION_MAX_M = "
        + _constant("SHALLOW_OCCLUSION_MAX_M")
        + ";\nconst samples = "
        + json.dumps(samples)
        + ";\nrepairShallowOcclusionRuns(samples);"
        + "\nconsole.log(JSON.stringify(samples.map((s) => s.occluded)));\n"
    )
    result = subprocess.run(
        [NODE or "node", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip())


def _constant(name: str) -> str:
    """A ``var NAME = value;`` constant's value text from scene.js."""
    source = SCENE_JS.read_text(encoding="utf-8")
    start = source.index("var " + name + " = ") + len("var " + name + " = ")
    return source[start : source.index(";", start)]


def _misses(*miss_m: float) -> list[dict[str, object]]:
    """Samples from their raycast misses: 0 = recovered exactly (visible)."""
    return [{"occluded": miss > 12.5, "missM": miss} for miss in miss_m]


@pytest.mark.skipif(NODE is None, reason="node is needed to run scene.js's own source")
class TestAShallowFalseOcclusionRunIsRepaired:
    """A shallow false-occlusion run: the marker recovers 0.0 m, but the two
    samples 8-13 px before it recover 17.0 m and 21.3 m against a 12.5 m
    tolerance, with a visible sample between them and the marker --
    so neither the lone-vertex repair nor the trailing force touched the run,
    and the bright line broke right before the dot. Genuinely hidden vertices
    recover 37-990 m; a run that stays under SHALLOW_OCCLUSION_MAX_M between
    visible samples is DEM noise, not a ridge.
    """

    def test_a_shallow_run_between_visible_samples_is_repaired(self) -> None:
        assert _repair_shallow(_misses(0.0, 10.2, 17.0, 21.3, 0.0, 0.0)) == [False] * 6

    def test_a_real_ridge_run_stays_hidden(self) -> None:
        assert _repair_shallow(_misses(0.0, 37.0, 990.0, 0.0)) == [False, True, True, False]

    def test_one_deep_sample_keeps_the_whole_run_hidden(self) -> None:
        """A run is noise only if EVERY sample in it is shallow."""
        assert _repair_shallow(_misses(0.0, 17.0, 40.0, 17.0, 0.0)) == [
            False,
            True,
            True,
            True,
            False,
        ]

    def test_a_run_touching_either_end_has_nothing_to_vouch_for_it(self) -> None:
        assert _repair_shallow(_misses(17.0, 17.0, 0.0)) == [True, True, False]
        assert _repair_shallow(_misses(0.0, 17.0, 17.0)) == [False, True, True]

    def test_a_sample_with_no_measured_miss_is_never_called_shallow(self) -> None:
        """Flags-only samples (no ``missM``) are treated as a real ridge."""
        samples: list[dict[str, object]] = [
            {"occluded": False},
            {"occluded": True},
            {"occluded": True},
            {"occluded": False},
        ]
        assert _repair_shallow(samples) == [False, True, True, False]

    def test_the_cap_stays_below_the_shallowest_real_ridge(self) -> None:
        cap = float(_constant("SHALLOW_OCCLUSION_MAX_M"))
        assert 21.3 < cap < 37.0


class TestTheShallowRepairIsWiredIntoTheDrawing:
    def test_samples_carry_their_miss_and_the_repair_runs_before_drawing(self) -> None:
        source = SCENE_JS.read_text(encoding="utf-8")
        assert "missM:" in _function_source("vertexSample")
        start = source.index("function strokeRouteParts(")
        body = source[start : start + 2500]
        shallow_at = body.index("repairShallowOcclusionRuns(samples)")
        assert shallow_at < body.index("forceTrailingSpanToMarker(samples")
        assert shallow_at < body.index("ctx.beginPath()")


class TestTheForceIsWiredIntoTheDrawing:
    """An unreferenced force would pass every case above and fix nothing."""

    def test_force_trailing_runs_after_the_isolated_repair(self) -> None:
        source = SCENE_JS.read_text(encoding="utf-8")
        start = source.index("function strokeRouteParts(")
        body = source[start : start + 2500]
        repair_at = body.index("repairIsolatedOcclusions(samples)")
        force_at = body.index("forceTrailingSpanToMarker(samples")
        draw_at = body.index("ctx.beginPath()")
        assert repair_at < force_at < draw_at, (
            "forceTrailingSpanToMarker must run after the isolated-occlusion "
            "repair and before the drawing pass opens the path"
        )

    def test_only_the_travelled_line_is_forced_not_the_dim_route(self) -> None:
        source = SCENE_JS.read_text(encoding="utf-8")
        start = source.index("function drawRouteOverlay(")
        end = source.index("\n  }\n\n  var map = null;")
        body = source[start:end]
        assert body.count("strokeRouteParts(ctx, pose.route") == 2
        assert body.count("strokeRouteParts(ctx, pose.travelled") == 1
        route_calls_start = body.index("strokeRouteParts(ctx, pose.route")
        travelled_call_start = body.index("strokeRouteParts(ctx, pose.travelled")
        route_calls = body[route_calls_start:travelled_call_start]
        assert "markerVisible" not in route_calls
        travelled_call = body[travelled_call_start : travelled_call_start + 200]
        assert "markerVisible" in travelled_call
