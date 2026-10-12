"""A refused frame must survive the run that refused it.

``SceneMapRenderer`` raises ``FrameQaError`` with the offending pixels attached,
but every frame a film renders lives in a ``TemporaryDirectory`` that is deleted
on the way out -- so the refusal used to leave nothing behind but a log line and
a pose dict. Diagnosing one meant re-rendering that pose by hand against the
same bundle, roughly forty minutes of work to look at a picture the run had
already drawn and thrown away.

The evidence lands beside ``output``, which is an operator-chosen directory that
outlives the staging root. Writing it must never raise: the refusal is the real
failure and an unwritable evidence directory must not replace it with an
unrelated ``OSError``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.route_story.film import render_route_story_film
from gomoveshift.pipeline.route_story.frame_qa import (
    FrameQaError,
    FrameQaResult,
    save_refusal_evidence,
)
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.satellite import SatelliteRaster

METERS_PER_DEGREE_LAT = math.pi / 180 * 6_371_008.8

# A pose of the shape frame QA refuses, trimmed to the keys a reader
# needs to re-render it: the whole point of the JSON is that this survives.
REFUSED_POSE = {
    "center": [-83.5, 35.2],
    "zoom": 15.32,
    "pitch": 60.0,
    "bearing": 214.7,
}


def _refusal(rgb: np.ndarray | None = None) -> FrameQaError:
    result = FrameQaResult(
        frame=5586,
        void_fraction=0.0,
        unloaded_fraction=0.0,
        black_fraction=0.0,
        problems=("travelled line ends 7.2px from the marker (max 4.0px)",),
        travelled_line_width_px=3.39,
        line_end_to_marker_px=7.2,
    )
    return FrameQaError(result, REFUSED_POSE, rgb=rgb)


def _frame() -> np.ndarray:
    rgb = np.zeros((8, 12, 3), dtype=np.uint8)
    rgb[2:5, 3:9] = (255, 207, 63)  # the bright route colour, so a reader sees it
    return rgb


class TestTheRefusedFrameIsWrittenWhereItSurvives:
    def test_the_refused_pixels_land_beside_the_output(self, tmp_path: Path) -> None:
        rgb = _frame()
        written = save_refusal_evidence(_refusal(rgb), tmp_path / "film.mp4")

        pngs = [path for path in written if path.suffix == ".png"]
        assert len(pngs) == 1, written
        assert pngs[0].parent == tmp_path
        assert np.array_equal(np.asarray(Image.open(pngs[0]).convert("RGB")), rgb)

    def test_the_pose_and_the_complaint_land_as_json(self, tmp_path: Path) -> None:
        written = save_refusal_evidence(_refusal(_frame()), tmp_path / "film.mp4")

        jsons = [path for path in written if path.suffix == ".json"]
        assert len(jsons) == 1, written
        record = json.loads(jsons[0].read_text(encoding="utf-8"))
        assert record["frame"] == 5586
        assert record["pose"] == REFUSED_POSE
        assert record["problems"] == ["travelled line ends 7.2px from the marker (max 4.0px)"]
        # The measured numbers, not just the sentence: a threshold argument
        # needs the value that lost, and re-deriving it costs another render.
        assert record["line_end_to_marker_px"] == pytest.approx(7.2)
        assert record["travelled_line_width_px"] == pytest.approx(3.39)

    def test_the_evidence_is_named_after_the_film_it_refused(self, tmp_path: Path) -> None:
        """Two films rendering into one directory must not overwrite each other."""
        first = save_refusal_evidence(_refusal(_frame()), tmp_path / "alpha.mp4")
        second = save_refusal_evidence(_refusal(_frame()), tmp_path / "beta.mp4")

        assert {path.name for path in first}.isdisjoint({path.name for path in second})
        assert all("alpha" in path.name for path in first), first
        assert all("beta" in path.name for path in second), second

    def test_a_refusal_carrying_no_pixels_still_writes_its_pose(self, tmp_path: Path) -> None:
        """``assert_frame_ok`` raises without an rgb; the pose is still the
        expensive half to recover, so it is written on its own."""
        written = save_refusal_evidence(_refusal(None), tmp_path / "film.mp4")

        assert [path.suffix for path in written] == [".json"]
        assert json.loads(written[0].read_text(encoding="utf-8"))["pose"] == REFUSED_POSE

    def test_an_unwritable_directory_never_replaces_the_real_failure(self, tmp_path: Path) -> None:
        """The refusal is the failure worth reporting. An evidence write that
        raises would surface as an OSError about a directory, burying it."""
        blocked = tmp_path / "not-a-directory"
        blocked.write_text("this is a file, so film.mp4 cannot live inside it\n")

        assert save_refusal_evidence(_refusal(_frame()), blocked / "film.mp4") == []


def _point(north_m: float, time_s: float) -> RoutePoint:
    return RoutePoint(
        time_s=time_s,
        latitude=40.0 + north_m / METERS_PER_DEGREE_LAT,
        longitude=-74.0,
        elevation_m=100.0,
    )


class _RefusingRenderer:
    """Draws one frame, then refuses the way SceneMapRenderer refuses."""

    name = "refusing"

    def measuring(self, segments):
        return self

    def render(self, segments, at_time, width, height) -> Image.Image:
        return Image.new("RGB", (width, height), "#223344")

    def render_leg(self, segments, times, width, height) -> Iterator[Image.Image]:
        yield Image.new("RGB", (width, height), "#223344")
        raise _refusal(_frame())


class TestTheFilmLeavesItsEvidenceBehind:
    def test_a_refused_frame_outlives_the_staging_directory(self, tmp_path: Path) -> None:
        overview = tmp_path / "overview.jpg"
        local = tmp_path / "local.jpg"
        Image.new("RGB", (64, 32), "#3c6f83").save(overview)
        Image.new("RGB", (64, 32), "#4e7540").save(local)
        output = tmp_path / "out" / "film.mp4"

        with pytest.raises(FrameQaError):
            render_route_story_film(
                segments=[[_point(i * 5.0, i * 2.0) for i in range(11)]],
                national=SatelliteRaster(overview, -180, -90, 180, 90, "Test overview"),
                local=SatelliteRaster(local, -74.1, 39.9, -73.9, 40.1, "Test local"),
                photos=[],
                output=output,
                fps=2,
                width=320,
                height=180,
                intro_seconds=1,
                map_seconds=1,
                photo_seconds=1,
                recap_seconds=1,
                map_renderer=_RefusingRenderer(),
            )

        evidence = sorted(path.name for path in output.parent.glob("*qa-refused*"))
        assert len(evidence) == 2, evidence
        assert any(name.endswith(".png") for name in evidence), evidence
        assert any(name.endswith(".json") for name in evidence), evidence
