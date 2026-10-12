"""No wait in the parallel scene capture may be unbounded.

A 1080p30 film render froze for good at film frame 2258: one of six capture
workers never got MapLibre's "load" event, its ``ready`` promise never
resolved, ``page.evaluate`` has no timeout, and the ordered consumer waited on
that worker's queue while the other five filled theirs. Zero CPU, no error, no
exit code.
"""

from __future__ import annotations

import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from gomoveshift.pipeline.route_story import proof
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.scene import capture as capture_mod
from gomoveshift.pipeline.route_story.scene_renderer import SceneMapRenderer

_SCENE_JS = (Path(capture_mod.__file__).parent / "scene.js").read_text(encoding="utf-8")


class _ReadyPage:
    """A page that reports ready, or never does, and counts forced frames.

    ``unframed_polls`` is the number of times the ready flag was read without a
    frame having been forced since the previous read. It must stay zero: a
    passive wait is the bug this harness exists to catch, because headless
    Chromium paints only when something asks for pixels and MapLibre's "load"
    fires on the first painted frame.
    """

    def __init__(self, log: list[str], *, never_ready: bool) -> None:
        self.log = log
        self.never_ready = never_ready
        self.frames = 0
        self.unframed_polls = 0
        self._framed_since_last_poll = False

    def screenshot(self, *, clip: dict[str, int] | None = None) -> bytes:
        self.frames += 1
        self._framed_since_last_poll = True
        return b""

    def evaluate(self, script: str, *args: object) -> object:
        if "diagnose" in script:
            return {"initialised": True, "loadFired": False, "labelsLoaded": True}
        assert "isReady" in script
        if not self._framed_since_last_poll:
            self.unframed_polls += 1
        self._framed_since_last_poll = False
        return not self.never_ready


def _capture_with_fake_browser(tmp_path, monkeypatch, *, timeouts: int, **kwargs):
    kwargs.setdefault("ready_timeout_s", 0.05)
    capture = capture_mod.SceneCapture(tmp_path, 2, 2, **kwargs)
    log: list[str] = []
    remaining = [timeouts]
    pages: list[_ReadyPage] = []

    def _open_page() -> None:
        log.append("open")
        never_ready = remaining[0] > 0
        remaining[0] -= 1
        capture._page = _ReadyPage(log, never_ready=never_ready)
        pages.append(capture._page)

    monkeypatch.setattr(capture, "_open_page", _open_page)
    monkeypatch.setattr(capture, "_close", lambda: log.append("close"))
    capture._pages = pages  # type: ignore[attr-defined]
    return capture, log


class TestSceneReadinessIsBounded:
    def test_ready_wait_carries_a_finite_driver_side_timeout(self, tmp_path, monkeypatch):
        capture, _log = _capture_with_fake_browser(
            tmp_path, monkeypatch, timeouts=99, ready_attempts=1, ready_timeout_s=0.2
        )
        started = time.monotonic()

        with pytest.raises(capture_mod.SceneRenderError, match="did not become ready"):
            capture.__enter__()

        assert time.monotonic() - started < 5

    def test_every_readiness_poll_forces_a_frame_first(self, tmp_path, monkeypatch):
        # The failure this prevents: style, tiles and labels all loaded, zero
        # tile failures, and "load" still never fired after 240 s -- because
        # headless Chromium paints only when something asks it to. One 8x8
        # screenshot brought the same scene ready in 1.3 s.
        capture, _log = _capture_with_fake_browser(
            tmp_path, monkeypatch, timeouts=99, ready_attempts=1, ready_timeout_s=0.2
        )

        with pytest.raises(capture_mod.SceneRenderError):
            capture.__enter__()

        page = capture._pages[0]
        assert page.frames > 1
        assert page.unframed_polls == 0

    def test_a_ready_scene_is_not_made_to_wait(self, tmp_path, monkeypatch):
        capture, log = _capture_with_fake_browser(tmp_path, monkeypatch, timeouts=0)
        started = time.monotonic()

        assert capture.__enter__() is capture

        assert log == ["open"]
        assert capture._pages[0].frames == 1
        assert time.monotonic() - started < 1

    def test_a_scene_that_never_loads_gets_a_fresh_browser(self, tmp_path, monkeypatch, caplog):
        capture, log = _capture_with_fake_browser(tmp_path, monkeypatch, timeouts=1)

        with caplog.at_level("WARNING"):
            capture.__enter__()

        assert log == ["open", "close", "open"]
        assert "fresh browser" in caplog.text

    def test_the_default_budget_survives_four_missed_loads(self, tmp_path, monkeypatch):
        # A real 1080p render had one worker miss "load" twice and start only on
        # its third and last browser: a three-try default was one miss from
        # costing a multi-hour render.
        capture, log = _capture_with_fake_browser(tmp_path, monkeypatch, timeouts=4)

        assert capture.__enter__() is capture

        assert log.count("open") == 5
        assert log.count("close") == 4

    def test_every_attempt_timing_out_raises_instead_of_waiting(self, tmp_path, monkeypatch):
        capture, log = _capture_with_fake_browser(
            tmp_path, monkeypatch, timeouts=99, ready_attempts=2
        )

        with pytest.raises(capture_mod.SceneRenderError, match="did not become ready"):
            capture.__enter__()

        assert log.count("open") == 2
        assert log.count("close") == 2

    def test_an_unrelated_startup_error_is_not_retried(self, tmp_path, monkeypatch):
        capture = capture_mod.SceneCapture(tmp_path, 2, 2)
        opened: list[int] = []

        def _open_page() -> None:
            opened.append(1)
            raise RuntimeError("Executable doesn't exist")

        monkeypatch.setattr(capture, "_open_page", _open_page)
        monkeypatch.setattr(capture, "_close", lambda: None)

        with pytest.raises(RuntimeError, match="Executable doesn't exist"):
            capture.__enter__()
        assert opened == [1]

    def test_ready_attempts_must_be_positive(self, tmp_path):
        with pytest.raises(ValueError, match="ready_attempts"):
            capture_mod.SceneCapture(tmp_path, 2, 2, ready_attempts=0)

    def test_the_scene_page_exposes_a_pollable_ready_flag(self):
        assert "isReady:" in _SCENE_JS

    def test_a_readiness_failure_carries_the_scenes_own_diagnosis(
        self, tmp_path, monkeypatch, caplog
    ):
        capture, _log = _capture_with_fake_browser(
            tmp_path, monkeypatch, timeouts=99, ready_attempts=1, ready_timeout_s=0.05
        )

        with pytest.raises(capture_mod.SceneRenderError) as excinfo:
            capture.__enter__()

        # Which HALF of readiness is outstanding is the whole question: "load"
        # and the label fetch are independent waits, and "not ready" alone
        # cannot tell them apart from outside the page's closure.
        assert "scene reports" in str(excinfo.value)
        assert '"loadFired": false' in str(excinfo.value)

    def test_the_ready_poll_asks_for_a_render_pass_when_the_map_is_loaded_but_load_never_fired(
        self,
    ):
        # MapLibre fires "load" only from inside a render pass, and schedules a
        # pass only when something is dirty. When the last source finishes
        # between two passes, map.loaded() flips true with no pass to follow and
        # "load" never comes -- and a forced screenshot cannot help, because no
        # frame is requested. Measured on one browser at a time: 9 of
        # 12 fresh browsers sat the whole ready budget this way. The poll the
        # driver already makes is the one place that can ask for that pass.
        start = _SCENE_JS.index("isReady: function () {")
        body = _SCENE_JS[start : _SCENE_JS.index("},", start)]
        assert "map.loaded()" in body
        assert "map.redraw()" in body
        assert "!loadFired" in body

    def test_the_scene_page_reports_which_half_of_readiness_is_outstanding(self):
        assert "diagnose:" in _SCENE_JS
        assert "loadFired" in _SCENE_JS
        assert "labelsLoaded" in _SCENE_JS


class TestConsumerNeverWaitsForever:
    def test_a_silent_queue_raises_once_the_stall_budget_passes(self):
        started = time.monotonic()

        with pytest.raises(capture_mod.SceneRenderError, match="no frame"):
            proof._get_or_raise(queue.Queue(), threading.Event(), [], stall_timeout_s=0.2)

        assert time.monotonic() - started < 5

    def test_a_recorded_failure_still_wins_over_the_stall(self):
        cancel = threading.Event()
        cancel.set()

        with pytest.raises(RuntimeError, match="capture crashed"):
            proof._get_or_raise(
                queue.Queue(), cancel, [RuntimeError("capture crashed")], stall_timeout_s=60
            )

    def test_the_default_stall_budget_is_finite(self):
        assert 0 < proof.WORKER_STALL_TIMEOUT_S < 60 * 60

    def test_join_gives_up_on_a_worker_that_never_returns(self, caplog):
        release = threading.Event()
        stuck = threading.Thread(target=release.wait, daemon=True)
        stuck.start()
        started = time.monotonic()
        try:
            with caplog.at_level("ERROR"):
                proof._drain_and_join([queue.Queue()], [stuck], deadline_s=0.3)
        finally:
            release.set()

        assert time.monotonic() - started < 5
        assert "still running" in caplog.text


class _LegCapture:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def __enter__(self) -> "_LegCapture":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def render(self, pose, light=None):
        return np.full((self.height, self.width, 3), 90, dtype=np.uint8), []


class TestALegWithAStuckWorkerAborts:
    def test_a_worker_stuck_in_startup_fails_the_leg_instead_of_hanging_it(self):
        width, height = 320, 180
        map_height = int(height * 0.855)
        release = threading.Event()
        opened = {"n": 0}
        lock = threading.Lock()

        class _StuckOnce(_LegCapture):
            def __enter__(self):
                with lock:
                    opened["n"] += 1
                    mine = opened["n"]
                if mine == 2:
                    release.wait()
                return self

        segments = [
            [RoutePoint(float(i), 35.0 + i * 0.0004, -80.0 + i * 0.0006, 100.0) for i in range(90)]
        ]
        renderer = SceneMapRenderer(
            capture=_LegCapture(width, map_height),
            trip_epoch=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            max_zoom=16.0,
            follow_zoom=14.0,
            fps=30.0,
            workers=3,
            capture_factory=lambda: _StuckOnce(width, map_height),
            stall_timeout_s=0.3,
            join_deadline_s=0.3,
        )
        times = [float(i + 1) for i in range(60)]
        started = time.monotonic()
        try:
            with pytest.raises(capture_mod.SceneRenderError, match="no frame"):
                list(renderer.render_leg(segments, times, width, height))
        finally:
            release.set()

        assert time.monotonic() - started < 10


# --- the browser's own account of a readiness failure -------------------------


class _ProblemRecorder(capture_mod.SceneCapture):
    """Just the problem-recording surface, with no browser behind it.

    `SceneCapture.__init__` needs no browser, so this exercises the real
    methods rather than a reimplementation of them.
    """

    def __init__(self) -> None:
        super().__init__(Path("/nonexistent"), 1920, 1080)
        self._page_errors: list[str] = []


def test_a_readiness_failure_carries_the_browsers_own_error():
    """Two renders died reporting only THAT the scene never became ready.

    The browser knew why the whole time -- a 404 on a bundle tile -- and the
    message threw it away, so the only way to learn the cause was to re-run a
    four-hour render. A later render named it on the first try once these listeners
    existed.
    """
    cap = _ProblemRecorder()
    cap._record_page_problem("HTTP 404: http://x/imagery/9/140/203.jpg")
    cap._record_page_problem("console.error: Failed to load resource")

    report = cap._page_problem_report()

    assert "HTTP 404: http://x/imagery/9/140/203.jpg" in report
    assert "console.error: Failed to load resource" in report


def test_repeated_problems_are_recorded_once_and_capped():
    """One bad tile URL repeats per tile request; a message full of the same
    line teaches nothing and hides the second, different failure behind it."""
    cap = _ProblemRecorder()
    for _ in range(5):
        cap._record_page_problem("HTTP 404: same-tile.jpg")
    for i in range(40):
        cap._record_page_problem(f"HTTP 404: tile-{i}.jpg")

    assert cap._page_errors.count("HTTP 404: same-tile.jpg") == 1
    assert len(cap._page_errors) == capture_mod.SceneCapture._MAX_PAGE_PROBLEMS


def test_a_silent_browser_says_so_rather_than_reading_as_clean():
    """An empty report must not look like "no problems found" -- the scene
    still failed, and the message has to say the browser offered no reason."""
    cap = _ProblemRecorder()

    report = cap._page_problem_report()

    assert "no error of its own" in report
