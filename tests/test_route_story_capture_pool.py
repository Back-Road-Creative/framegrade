"""Capture workers live for the whole film, not one map leg.

A render once booted six fresh Chromiums at the start of EVERY map
leg, and on later legs two of the six missed the 60 s ready timeout
and were relaunched. Frames drain in strict order, so the four healthy
workers then idled behind the slow two: roughly 1.5-2 minutes lost per leg,
for a film of a dozen or more legs. A worker's browser is opened once and
reused across legs; ``close`` is what finally shuts it.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

import numpy as np
import pytest

from gomoveshift.pipeline.route_story.gpx import RoutePoint
from gomoveshift.pipeline.route_story.scene import capture as capture_mod
from gomoveshift.pipeline.route_story.scene_renderer import SceneMapRenderer

WIDTH, HEIGHT = 320, 180
MAP_HEIGHT = int(HEIGHT * 0.855)


class _CountingCapture:
    """A fake worker capture that records its own open/close on a shared ledger."""

    def __init__(self, ledger: dict, lock: threading.Lock) -> None:
        self.ledger = ledger
        self.lock = lock
        self.width = WIDTH
        self.height = MAP_HEIGHT

    def __enter__(self) -> "_CountingCapture":
        with self.lock:
            self.ledger["opened"] += 1
        return self

    def __exit__(self, *exc: object) -> None:
        with self.lock:
            self.ledger["closed"] += 1

    def render(self, pose, light=None):
        return np.full((self.height, self.width, 3), 90, dtype=np.uint8), []


def _segments() -> list[list[RoutePoint]]:
    return [[RoutePoint(float(i), 35.0 + i * 0.0004, -80.0 + i * 0.0006, 100.0) for i in range(90)]]


def _renderer(factory, workers: int = 3, **overrides) -> SceneMapRenderer:
    return SceneMapRenderer(
        capture=factory(),
        trip_epoch=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        max_zoom=16.0,
        follow_zoom=14.0,
        fps=30.0,
        workers=workers,
        capture_factory=factory,
        stall_timeout_s=overrides.pop("stall_timeout_s", 5.0),
        join_deadline_s=overrides.pop("join_deadline_s", 2.0),
        **overrides,
    )


def _leg(renderer: SceneMapRenderer, start: int, count: int) -> list:
    times = [float(start + i) for i in range(count)]
    return list(renderer.render_leg(_segments(), times, WIDTH, HEIGHT))


class TestWorkersOutliveALeg:
    def test_three_legs_open_one_capture_per_worker_not_per_leg(self):
        ledger = {"opened": 0, "closed": 0}
        lock = threading.Lock()
        factory = lambda: _CountingCapture(ledger, lock)  # noqa: E731
        renderer = _renderer(factory, workers=3)
        ledger["opened"] = ledger["closed"] = 0  # discard the renderer's own capture

        for start in (1, 31, 61):
            assert len(_leg(renderer, start, 24)) == 24

        assert ledger["opened"] == 3, ledger
        assert ledger["closed"] == 0, "a worker's browser is not shut between legs"

    def test_close_shuts_every_worker_capture_once(self):
        ledger = {"opened": 0, "closed": 0}
        lock = threading.Lock()
        factory = lambda: _CountingCapture(ledger, lock)  # noqa: E731
        renderer = _renderer(factory, workers=3)
        ledger["opened"] = ledger["closed"] = 0
        _leg(renderer, 1, 12)

        renderer.close()
        renderer.close()

        assert ledger == {"opened": 3, "closed": 3}

    def test_close_before_any_leg_is_a_no_op(self):
        ledger = {"opened": 0, "closed": 0}
        lock = threading.Lock()
        renderer = _renderer(lambda: _CountingCapture(ledger, lock), workers=3)
        ledger["opened"] = ledger["closed"] = 0

        renderer.close()

        assert ledger == {"opened": 0, "closed": 0}

    def test_a_leg_after_close_gets_fresh_workers(self):
        ledger = {"opened": 0, "closed": 0}
        lock = threading.Lock()
        renderer = _renderer(lambda: _CountingCapture(ledger, lock), workers=2)
        ledger["opened"] = ledger["closed"] = 0
        _leg(renderer, 1, 8)
        renderer.close()

        assert len(_leg(renderer, 9, 8)) == 8
        renderer.close()

        assert ledger == {"opened": 4, "closed": 4}


class TestAFailedWorkerStillAbortsTheLeg:
    def test_the_original_exception_surfaces_and_the_next_leg_starts_clean(self):
        ledger = {"opened": 0, "closed": 0}
        lock = threading.Lock()
        calls = {"n": 0}

        class _CrashOnce(_CountingCapture):
            def render(self, pose, light=None):
                with lock:
                    calls["n"] += 1
                    mine = calls["n"]
                if mine == 5:
                    raise RuntimeError("capture crashed")
                return super().render(pose, light)

        renderer = _renderer(lambda: _CrashOnce(ledger, lock), workers=3)
        ledger["opened"] = ledger["closed"] = 0

        with pytest.raises(RuntimeError, match="capture crashed"):
            _leg(renderer, 1, 30)
        first_opened = ledger["opened"]
        assert first_opened == 3

        # The leg after a failure never inherits a worker whose browser may
        # be mid-crash: the pool is discarded and rebuilt.
        assert len(_leg(renderer, 31, 12)) == 12
        assert ledger["opened"] == 6, ledger
        renderer.close()
        assert ledger["closed"] == 6, ledger

    def test_a_worker_stuck_in_startup_still_fails_the_leg_quickly(self):
        release = threading.Event()
        opened = {"n": 0}
        lock = threading.Lock()
        ledger = {"opened": 0, "closed": 0}

        class _StuckOnce(_CountingCapture):
            def __enter__(self):
                with lock:
                    opened["n"] += 1
                    mine = opened["n"]
                if mine == 3:  # the renderer's own capture is never entered
                    release.wait()
                return super().__enter__()

        renderer = _renderer(
            lambda: _StuckOnce(ledger, lock), workers=3, stall_timeout_s=0.3, join_deadline_s=0.3
        )
        started = time.monotonic()
        try:
            with pytest.raises(capture_mod.SceneRenderError, match="no frame"):
                _leg(renderer, 1, 60)
        finally:
            release.set()

        assert time.monotonic() - started < 10


def test_the_film_closes_its_scene_renderer_when_the_export_ends():
    """film.py owns the renderer's lifetime: the workers' browsers are daemon
    threads, so nothing but an explicit close ends them at export end."""
    from pathlib import Path

    from gomoveshift.pipeline.route_story import film

    source = Path(film.__file__).read_text(encoding="utf-8")
    assert "map_renderer.close()" in source
