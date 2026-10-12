"""The console schedule ticker -- a background thread that drains
due schedules on an interval so ``dispatch-queue`` runs without crontab or
an operator hitting ``GET /jobs``."""

import contextlib
import sys
import threading
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from gomoveshift.console import jobs, pubqueue
from gomoveshift.console.app import create_app
from gomoveshift.console.jobs import JobStore


@pytest.fixture(autouse=True)
def _isolate_tenant_storage(tmp_path, monkeypatch):
    """``_lifespan`` now claims tenant storage (``console/tenant.py``)
    off ``Config.load().seo.storage_path`` before anything else -- isolate
    it from the real ``config.yaml``'s absolute host path so a `with
    TestClient(...)` in this file (which DOES run lifespan) never writes a
    stamp into production SEO storage."""
    monkeypatch.setattr(
        pubqueue.Config,
        "load",
        staticmethod(
            lambda *a, **k: SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
        ),
    )


class StepWaiter:
    """Lock-step replacement for ``Ticker``'s tick-boundary wait. Each call
    (made by the ticker thread, after ``run_due`` has already run for that
    tick) sets ``_ready`` and blocks until the test calls ``release_tick()``.
    ``wait_for_tick()`` blocks the test until that tick has genuinely
    arrived, so a test can assert an exact call count and stop the ticker
    with no race -- no sleeping, no wall-clock deadline (a
    sleep-loop version flaked under a loaded CI runner)."""

    def __init__(self) -> None:
        self._proceed = threading.Event()
        self._ready = threading.Event()
        self.calls = 0
        #: Set by ``lockstep_ticker`` so teardown can stop the thread it
        #: parked; ``None`` when a test drives a ``Ticker`` it built itself.
        self.ticker = None

    def wait_for_tick(self, timeout: float = 60.0) -> None:
        # The timeout is a HANG GUARD, not a measurement: nothing here passes or
        # fails on how long a tick took. Without it a ticker thread that died
        # blocks the test for ever, which on a runner is worse than a flake.
        if not self._ready.wait(timeout):
            raise AssertionError(f"no tick arrived within {timeout}s (calls={self.calls})")
        self._ready.clear()

    def release_tick(self) -> None:
        self._proceed.set()

    def __call__(self, timeout: float) -> bool:
        self.calls += 1
        self._ready.set()
        # Park until released -- but notice a stop() while parked. Without
        # this, a ticker whose owner is shutting down (``_lifespan`` calls
        # ``stop()``) makes that join wait out its full 5s against a thread
        # nobody is going to release.
        while not self._proceed.wait(0.05):
            ticker = self.ticker
            if ticker is not None and ticker._stop_event.is_set():
                return True  # what Event.wait returns when the stop is set
        self._proceed.clear()
        return False


@contextlib.contextmanager
def lockstep_ticker(monkeypatch):
    """Make the ticker ``create_app``'s lifespan starts lock-step.

    ``create_app`` builds its own ``Ticker`` with the real stop-aware sleep,
    so a test that reaches into ``app.state.console_ticker`` is racing that
    thread's FIRST tick -- which finishes whenever it finishes and overwrites
    ``last_tick_ok_at`` with ``now``. Raising ``GMS_CONSOLE_TICK_SECONDS``
    only postpones the SECOND tick; it does nothing about the first. Losing
    that race makes the test flaky.

    Yields the ``StepWaiter``: ``wait_for_tick()`` returns only once a full
    tick body has completed (the waiter is called at the tick boundary,
    after ``last_tick_ok_at`` is set), and no further tick can start until
    ``release_tick()``. So a test can backdate a timestamp knowing nothing
    is about to overwrite it.
    """
    waiter = StepWaiter()

    def _start(store, log_dir, tick_seconds):
        ticker = jobs.Ticker(store, log_dir, tick_seconds, wait=waiter)
        waiter.ticker = ticker
        ticker.start()
        return ticker

    monkeypatch.setattr(jobs, "start_ticker", _start)
    try:
        yield waiter
    finally:
        # Ask the loop to stop BEFORE unparking it, so it re-checks the stop
        # event instead of running another tick -- otherwise the lifespan's
        # own ``stop()`` waits out its 5s join against a parked thread.
        if getattr(waiter, "ticker", None) is not None:
            waiter.ticker._stop_event.set()
            waiter.release_tick()


def lock_step(store, log_dir, waiter):
    """A ticker whose tick boundary is ``waiter``, already started."""
    ticker = jobs.Ticker(store, log_dir, tick_seconds=0, wait=waiter)
    ticker.start()
    return ticker


def let_ticks_land(waiter, n):
    """Block until exactly ``n`` ticks have run, leaving the ticker thread
    parked in the waiter — so the caller can assert an exact count and stop it
    with no race."""
    for i in range(n):
        waiter.wait_for_tick()
        if i < n - 1:
            waiter.release_tick()


def stop_parked(waiter, ticker):
    """Stop a ticker parked in the waiter, without letting one more tick run."""
    ticker._stop_event.set()
    waiter.release_tick()
    ticker.stop()


def test_start_ticker_calls_run_due_on_interval_until_stopped(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(jobs, "run_due", lambda store, now, log_dir: calls.append(now) or [])
    store = JobStore(tmp_path / "jobs.sqlite3")

    waiter = StepWaiter()
    ticker = lock_step(store, tmp_path / "logs", waiter)
    let_ticks_land(waiter, 3)
    stop_parked(waiter, ticker)

    assert len(calls) == 3
    assert not ticker.is_alive()


def test_failing_queue_observer_never_kills_the_ticker(tmp_path, monkeypatch):
    """A queue-DB error (permissions, missing dir) in the notify observer is
    logged and skipped; the schedule drain keeps ticking."""
    calls = []
    monkeypatch.setattr(jobs, "run_due", lambda store, now, log_dir: calls.append(now) or [])

    def boom():
        raise PermissionError("schema lock")

    monkeypatch.setattr(jobs.notify, "check_queue_regressions", boom)
    store = JobStore(tmp_path / "jobs.sqlite3")
    waiter = StepWaiter()
    ticker = lock_step(store, tmp_path / "logs", waiter)
    let_ticks_land(waiter, 3)
    still_alive = ticker.is_alive()
    stop_parked(waiter, ticker)
    assert len(calls) == 3
    assert still_alive


def test_stop_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "run_due", lambda store, now, log_dir: [])
    store = JobStore(tmp_path / "jobs.sqlite3")
    ticker = jobs.start_ticker(store, tmp_path / "logs", tick_seconds=0.01)
    ticker.stop()
    ticker.stop()  # must not raise or hang
    assert not ticker.is_alive()


def test_ticker_disabled_by_default_no_thread_started(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("GMS_CONSOLE_TICKER", raising=False)
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert app.state.console_ticker is None


def test_ticker_enabled_env_starts_thread_and_shutdown_stops_it(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GMS_CONSOLE_TICKER", "1")
    monkeypatch.setenv("GMS_CONSOLE_TICK_SECONDS", "0.05")
    app = create_app()
    with TestClient(app) as client:
        client.get("/healthz")
        ticker = app.state.console_ticker
        assert ticker is not None
        assert ticker.is_alive()
    assert not ticker.is_alive()


def test_run_due_exception_does_not_kill_the_ticker_thread(tmp_path, monkeypatch):
    """One bad tick must not silently end the background thread."""
    calls = []

    def _boom(store, now, log_dir):
        calls.append(now)
        raise RuntimeError("boom")

    monkeypatch.setattr(jobs, "run_due", _boom)
    store = JobStore(tmp_path / "jobs.sqlite3")

    waiter = StepWaiter()
    ticker = lock_step(store, tmp_path / "logs", waiter)
    let_ticks_land(waiter, 3)
    still_alive = ticker.is_alive()
    stop_parked(waiter, ticker)

    assert len(calls) == 3
    assert still_alive


def test_healthz_reports_ticker_alive_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GMS_CONSOLE_TICKER", "1")
    monkeypatch.setenv("GMS_CONSOLE_TICK_SECONDS", "60")
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["ticker_alive"] is True


def test_healthz_reports_ticker_alive_false_when_thread_died(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GMS_CONSOLE_TICKER", "1")
    monkeypatch.setenv("GMS_CONSOLE_TICK_SECONDS", "60")
    app = create_app()
    with TestClient(app) as client:
        client.get("/healthz")
        app.state.console_ticker._thread.join(timeout=0)  # no-op, thread still alive
        app.state.console_ticker._stop_event.set()
        app.state.console_ticker._thread.join(timeout=5.0)
        resp = client.get("/healthz")
        assert resp.json()["ticker_alive"] is False


def test_healthz_reports_ticker_alive_none_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("GMS_CONSOLE_TICKER", raising=False)
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/healthz")
        assert resp.json()["ticker_alive"] is None


def test_ticker_reaps_a_failed_job_and_notifies_without_a_jobs_get(tmp_path, monkeypatch):
    """A 2am job failure must push within one tick, not wait for someone to
    open /jobs (which is the only thing that reaped before this fix)."""
    monkeypatch.setattr(jobs, "_validate_argv", lambda argv: None)
    monkeypatch.setattr(jobs, "run_due", lambda store, now, log_dir: [])
    calls = []
    monkeypatch.setattr(jobs.notify, "notify_job_failed", lambda job: calls.append(job))
    store = JobStore(tmp_path / "jobs.sqlite3")
    argv = [sys.executable, "-c", "import sys; sys.exit(1)"]
    job_id = jobs.launch(store, "status", argv, tmp_path / "logs")

    # The only wait in this file that is NOT lock-stepped, because what it waits
    # for is not a tick: a real child process has to exit before any tick can
    # reap it. So the deadline is a hang guard sized for a loaded machine, not a
    # measurement — the loop leaves as soon as the reap lands.
    ticker = jobs.start_ticker(store, tmp_path / "logs", tick_seconds=0.02)
    deadline = time.time() + 60.0
    while not calls and time.time() < deadline:
        time.sleep(0.02)
    ticker.stop()

    assert len(calls) == 1
    assert calls[0]["id"] == job_id
    assert store.get(job_id)["status"] == "failed"


def test_ticker_gate_observer_uses_job_store_output_dirs_and_survives_failure(
    tmp_path, monkeypatch
):
    # Wired to the job store's own argvs; a raising check_pending_gates must
    # not kill the drain -- same guard shape as check_queue_regressions.
    monkeypatch.setattr(jobs, "run_due", lambda store, now, log_dir: [])
    monkeypatch.setattr(jobs, "_validate_argv", lambda argv: None)
    calls = []

    def _boom(output_dirs, **kwargs):
        calls.append(output_dirs)
        raise RuntimeError("boom")

    monkeypatch.setattr(jobs.notify, "check_pending_gates", _boom)
    store = JobStore(tmp_path / "jobs.sqlite3")
    argv = [sys.executable, "-m", "gomoveshift", "drive", "d", "--output-dir", "/out/a"]
    jobs.launch(store, "drive", argv, tmp_path / "logs")

    waiter = StepWaiter()
    ticker = lock_step(store, tmp_path / "logs", waiter)
    let_ticks_land(waiter, 3)
    still_alive = ticker.is_alive()
    stop_parked(waiter, ticker)

    assert calls[0] == ["/out/a"]
    assert len(calls) == 3
    assert still_alive


def test_healthz_returns_503_when_ticker_thread_died(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GMS_CONSOLE_TICKER", "1")
    monkeypatch.setenv("GMS_CONSOLE_TICK_SECONDS", "60")
    app = create_app()
    with TestClient(app) as client:
        client.get("/healthz")
        app.state.console_ticker._stop_event.set()
        app.state.console_ticker._thread.join(timeout=5.0)
        resp = client.get("/healthz")
        assert resp.status_code == 503
        assert resp.json()["status"] == "degraded"
        assert resp.json()["ticker_alive"] is False


def test_healthz_returns_503_before_first_tick_completes_past_the_grace_window(
    monkeypatch, tmp_path
):
    # A wedged first tick (last_tick_ok_at stays None forever) used to read
    # healthy forever; staleness now falls back to process start.
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GMS_CONSOLE_TICKER", "1")
    # Long enough that the thread's first pass is its only pass: with a 10s
    # interval a real tick could land between the backdate below and the
    # assertion and refresh the very timestamp under test.
    monkeypatch.setenv("GMS_CONSOLE_TICK_SECONDS", "3600")
    monkeypatch.setattr(
        jobs, "run_due", lambda store, now, log_dir: (_ for _ in ()).throw(RuntimeError("wedged"))
    )
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200  # inside the grace window
        ticker = app.state.console_ticker
        from datetime import datetime, timezone

        ticker.started_at = datetime.now(timezone.utc) - timedelta(seconds=3 * 3600 + 60)
        resp = client.get("/healthz")
        assert resp.status_code == 503
        assert resp.json()["status"] == "degraded"
        assert resp.json()["ticker_alive"] is True


def test_healthz_returns_503_when_last_tick_is_stale(monkeypatch, tmp_path):
    """A ticker thread can stay ``is_alive()`` while its own tick body is
    wedged (e.g. a hung DB call) -- staleness of the last successful tick is
    the only way to see that.

    Lock-step, not lucky: the previous version backdated ``last_tick_ok_at``
    while the thread's first tick was still running, and a first tick that
    finished after the backdate wrote ``now`` back over it. See
    ``lockstep_ticker``.
    """
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GMS_CONSOLE_TICKER", "1")
    monkeypatch.setenv("GMS_CONSOLE_TICK_SECONDS", "3600")
    with lockstep_ticker(monkeypatch) as waiter:
        app = create_app()
        with TestClient(app) as client:
            resp = client.get("/healthz")
            assert resp.status_code == 200
            ticker = app.state.console_ticker
            from datetime import datetime, timezone

            # The one tick that could overwrite the field under test has now
            # finished, and the next cannot start until release_tick().
            waiter.wait_for_tick()
            assert ticker.last_tick_ok_at is not None

            # /healthz's threshold is 3 x tick_seconds (`healthz` in the console app), so the
            # backdate is expressed against the interval rather than
            # hard-coded to the 31s that only worked while it was 10.
            ticker.last_tick_ok_at = datetime.now(timezone.utc) - timedelta(seconds=3 * 3600 + 60)
            resp = client.get("/healthz")
            assert resp.status_code == 503
            assert resp.json()["status"] == "degraded"
            # Pin WHY it is degraded. Stopping the thread would also return
            # 503, so without this the test can pass for the wrong reason.
            assert resp.json()["ticker_alive"] is True


def test_a_tick_finishing_late_cannot_overwrite_the_backdated_timestamp(monkeypatch, tmp_path):
    """The race itself, made deterministic: a first tick whose body takes
    longer than the test does. Under the old shape the backdate landed while
    that tick was mid-flight, and the tick's own ``last_tick_ok_at = now``
    then erased it -- /healthz answered 200 and the assertion read ``200 ==
    503``. Waiting for the tick boundary is what makes that impossible."""
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GMS_CONSOLE_TICKER", "1")
    monkeypatch.setenv("GMS_CONSOLE_TICK_SECONDS", "3600")

    tick_entered = threading.Event()
    let_tick_finish = threading.Event()
    real_run_due = jobs.run_due

    def _slow_run_due(store, now, log_dir):
        tick_entered.set()
        # A real wedged/slow tick body: a hung DB call, a loaded runner.
        let_tick_finish.wait(30)
        return real_run_due(store, now, log_dir)

    monkeypatch.setattr(jobs, "run_due", _slow_run_due)

    with lockstep_ticker(monkeypatch) as waiter:
        app = create_app()
        with TestClient(app) as client:
            from datetime import datetime, timezone

            assert tick_entered.wait(30), "the ticker thread never started its first tick"
            ticker = app.state.console_ticker
            # Release the slow tick and wait for the BOUNDARY, not for luck.
            let_tick_finish.set()
            waiter.wait_for_tick()

            ticker.last_tick_ok_at = datetime.now(timezone.utc) - timedelta(seconds=3 * 3600 + 60)
            assert client.get("/healthz").status_code == 503


def test_ticker_checks_for_missed_slots_and_survives_a_failure(tmp_path, monkeypatch):
    """The missed-slot observer gets the same independent guard as its two
    siblings. It is the only one that fires on something NOT happening, so a
    queue-DB error here must not be the thing that stops the drain."""
    monkeypatch.setattr(jobs, "run_due", lambda store, now, log_dir: [])
    monkeypatch.setattr(jobs.notify, "check_queue_regressions", lambda: None)
    monkeypatch.setattr(jobs.notify, "check_pending_gates", lambda output_dirs: None)
    calls = []

    def boom():
        calls.append(1)
        raise PermissionError("schema lock")

    monkeypatch.setattr(jobs.notify, "check_missed_slots", boom)
    store = JobStore(tmp_path / "jobs.sqlite3")

    waiter = StepWaiter()
    ticker = lock_step(store, tmp_path / "logs", waiter)
    let_ticks_land(waiter, 3)
    still_alive = ticker.is_alive()
    stop_parked(waiter, ticker)

    assert len(calls) == 3
    assert still_alive
