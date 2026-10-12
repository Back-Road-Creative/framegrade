"""A failed frame must cost one frame, not the whole render.

A long render once captured for hours, reached film frame 4353, and then died
with no mp4, no receipt and no evidence:

    FrameQaError: frame 11503 failed frame QA:
    frame 11503: travelled line ends 4.0px from the marker (max 4.0px)

Four separate defects met in that one sentence, and each has a test here:

1. ``11503`` is not a frame. ``scene_renderer`` passed ``int(at_time)`` --
   trip SECONDS -- and the message called it a frame, which cost two wrong
   diagnoses before anyone read the call site.
2. The comparison is strictly ``>`` but both numbers printed at ``.1f``, so a
   4.03px overshoot rendered as "4.0px ... (max 4.0px)" and reads as a PASS.
3. A four-pixel cosmetic overshoot aborted the entire run. The gate is right --
   it is the detector for the marker lagging the travelled line -- but the cost
   of it firing must be one recorded fault, not the artifact.
4. ``save_refusal_evidence`` swallowed its own ``OSError``/``ValueError`` and
   returned an empty list, so the run that most needed the refused pixels was
   the run guaranteed not to explain why it had none.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pytest

from gomoveshift.pipeline.route_story.frame_qa import (
    FrameQaError,
    FrameQaLedger,
    FrameQaResult,
    format_px,
    save_refusal_evidence,
)
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.scene_renderer import SceneMapRenderer

WIDTH, HEIGHT = 320, 180
MAP_HEIGHT = int(HEIGHT * 0.855)


class _BlackCapture:
    """A capture whose every frame is flat black -- a guaranteed QA fault."""

    def __init__(self, width: int = WIDTH, height: int = MAP_HEIGHT) -> None:
        self.width = width
        self.height = height

    def __enter__(self) -> "_BlackCapture":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def render(self, pose, light=None):
        return np.zeros((self.height, self.width, 3), dtype=np.uint8), []


def _segments(count: int = 90):
    return [
        [
            RoutePoint(float(i), 35.0 + i * 0.0004, -80.0 + i * 0.0006, 100.0 + i)
            for i in range(count)
        ]
    ]


def _renderer(*, workers: int = 1, ledger: FrameQaLedger | None = None) -> SceneMapRenderer:
    return SceneMapRenderer(
        capture=_BlackCapture(),
        trip_epoch=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        max_zoom=16.0,
        follow_zoom=14.0,
        fps=30.0,
        workers=workers,
        capture_factory=(lambda: _BlackCapture()) if workers > 1 else None,
        qa_ledger=ledger,
    )


def _render_all(renderer: SceneMapRenderer, frame_count: int = 20):
    times = [float(i + 1) for i in range(frame_count)]
    return list(renderer.render_leg(_segments(), times, WIDTH, HEIGHT)), times


class TestAFaultIsRecordedNotFatal:
    def test_a_faulted_frame_does_not_abort_the_leg(self):
        """The lost render in one assertion: every frame still arrives."""
        ledger = FrameQaLedger()
        frames, times = _render_all(_renderer(ledger=ledger))

        assert len(frames) == len(times)

    def test_every_faulted_frame_is_recorded(self):
        ledger = FrameQaLedger()
        _, times = _render_all(_renderer(ledger=ledger))

        assert ledger.fault_count == len(times)
        assert ledger.faults, "a recorded fault keeps its result for the report"

    def test_the_ledger_keeps_the_pixels_of_the_first_fault_only(self):
        """Thousands of held frames is an out-of-memory bug, not a report."""
        ledger = FrameQaLedger()
        _render_all(_renderer(ledger=ledger))

        with_pixels = [fault for fault in ledger.faults if fault.rgb is not None]
        assert len(with_pixels) == 1
        assert with_pixels[0] is ledger.faults[0]

    def test_faults_from_every_capture_worker_land_in_one_ledger(self):
        """The collector is written from worker THREADS -- nothing may be lost."""
        ledger = FrameQaLedger()
        _, times = _render_all(_renderer(workers=3, ledger=ledger))

        assert ledger.fault_count == len(times)

    def test_without_a_ledger_a_faulted_frame_still_raises(self):
        """preflight.py judges a handful of frames and wants the refusal."""
        with pytest.raises(FrameQaError):
            _render_all(_renderer(ledger=None), frame_count=4)


class TestAFailingValueNeverPrintsAsTheLimit:
    def test_an_overshoot_is_distinguishable_from_the_ceiling_it_broke(self):
        assert format_px(4.03) != format_px(4.0)

    def test_the_problem_sentence_shows_the_value_that_actually_lost(self):
        ledger = FrameQaLedger()
        _render_all(_renderer(ledger=ledger), frame_count=2)

        sentence = ledger.faults[0].result.problems[0]
        assert "frame" in sentence


class TestTheFrameNumberIsAFrameNumber:
    def test_a_recorded_fault_counts_frames_not_trip_seconds(self):
        """``int(at_time)`` is trip seconds; a leg of 20 frames must report
        indices inside that leg, never a five-digit clock reading."""
        ledger = FrameQaLedger()
        _, times = _render_all(_renderer(ledger=ledger))

        recorded = sorted(fault.result.frame for fault in ledger.faults)
        assert recorded == list(range(len(times)))


class TestAFaultedFilmIsStillWrittenBeforeItFails:
    """The loss this file is named for, end to end: four hours of capture, then nothing.

    A faulted render must still FAIL -- but only after the mp4 and the fault
    report exist, so the operator can watch what the gate objected to instead
    of re-rendering to find out.
    """

    def _render(self, tmp_path):
        from PIL import Image

        from gomoveshift.pipeline.route_story.film import render_route_story_film
        from gomoveshift.pipeline.route_story.satellite import SatelliteRaster

        overview = tmp_path / "overview.jpg"
        local = tmp_path / "local.jpg"
        Image.new("RGB", (720, 360), "#3c6f83").save(overview)
        Image.new("RGB", (640, 360), "#4e7540").save(local)
        output = tmp_path / "film.mp4"

        renderer = SceneMapRenderer(
            capture=_BlackCapture(WIDTH, int(HEIGHT * 0.855)),
            trip_epoch=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            max_zoom=16.0,
            follow_zoom=14.0,
            fps=2.0,
            workers=1,
        )
        with pytest.raises(FrameQaError) as raised:
            render_route_story_film(
                segments=[
                    [RoutePoint(0, 35.0, -80.0, 100), RoutePoint(10, 35.0001, -79.9999, 110)]
                ],
                national=SatelliteRaster(overview, -180, -90, 180, 90, "Test overview"),
                local=SatelliteRaster(local, -80.1, 34.9, -79.9, 35.1, "Test local"),
                photos=[],
                output=output,
                map_renderer=renderer,
                fps=2,
                width=WIDTH,
                height=HEIGHT,
                intro_seconds=1,
                map_seconds=1,
                recap_seconds=1,
            )
        return output, raised.value

    def test_the_film_exists_even_though_the_run_failed(self, tmp_path):
        output, _ = self._render(tmp_path)

        assert output.is_file()
        assert output.stat().st_size > 0

    def test_the_fault_report_lands_beside_the_film(self, tmp_path):
        import json

        output, _ = self._render(tmp_path)
        report = output.parent / "film.qa-faults.json"

        assert report.is_file()
        summary = json.loads(report.read_text(encoding="utf-8"))
        assert summary["frames_faulted"] > 0
        assert summary["problems"], "the report names WHICH defect, with a count"

    def test_the_failure_says_the_film_was_written(self, tmp_path):
        output, error = self._render(tmp_path)

        assert str(output) in str(error)

    def test_the_refused_pixels_are_saved(self, tmp_path):
        output, _ = self._render(tmp_path)

        assert (output.parent / "film.qa-refused.png").is_file()


class TestEvidenceExplainsItsOwnFailure:
    def test_an_unwritable_destination_is_logged_not_swallowed(self, tmp_path, caplog):
        result = FrameQaResult(
            frame=7,
            void_fraction=1.0,
            unloaded_fraction=0.0,
            black_fraction=0.0,
            problems=("frame 7: all void",),
        )
        error = FrameQaError(result, {"zoom": 4.0}, rgb=np.zeros((4, 4, 3), dtype=np.uint8))
        blocked = tmp_path / "not-a-dir"
        blocked.write_text("this is a file, so mkdir of a child must fail", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            written = save_refusal_evidence(error, blocked / "child" / "film.mp4")

        assert written == []
        assert caplog.records, "losing the evidence must never be silent"

    def test_a_writable_destination_still_returns_what_landed(self, tmp_path):
        result = FrameQaResult(
            frame=7,
            void_fraction=1.0,
            unloaded_fraction=0.0,
            black_fraction=0.0,
            problems=("frame 7: all void",),
        )
        error = FrameQaError(result, {"zoom": 4.0}, rgb=np.zeros((4, 4, 3), dtype=np.uint8))

        written = save_refusal_evidence(error, tmp_path / "film.mp4")

        assert [path.name for path in written] == [
            "film.qa-refused.png",
            "film.qa-refused.json",
        ]
