## Route Story capture workers outlive a map leg (`gomoveshift/pipeline/route_story/scene_renderer.py`)

`_capture_leg_parallel` used to open each worker's browser with `with factory() as worker_capture`
*inside* the per-leg thread body, so a film of N map legs paid `workers × N` Chromium launches.
A measured render showed the cost: on two consecutive legs, two of six workers missed the
60 s readiness timeout and logged `scene not ready after 60000 ms (attempt 1 of 5); launching a
fresh browser`. Because frames drain in strict frame order, the four healthy workers stall behind
the slow two for the whole relaunch — ~1.5–2 min per leg, repeated per leg.

- **The thread is what persists, not just the browser handle.** Playwright's sync API pins a
  browser to the thread that opened it, so a pooled capture cannot be handed to a fresh thread each
  leg. `_ensure_pool` starts one long-lived daemon thread per worker; `_worker_loop` opens its
  capture on that thread and then blocks on a `jobs` queue, rendering one `_CaptureJob` (a striped
  slice of frame indices) per leg.
- **The capture opens on the first job, not at thread start**, so a launch failure surfaces through
  that leg's own `failure` list — the consumer's error path — rather than dying in a thread nobody
  is waiting on yet.
- **A failed leg tears the pool down** (`close()` in the `except`): a browser that just crashed, or
  is stuck inside a call with no timeout, must not be inherited by the next leg. The next leg opens
  fresh browsers.
- **Per-leg waiting is on the job, never the thread.** `_drain_until_done` waits for each worker's
  `done` event (and unsticks any worker blocked writing to an abandoned queue), because in a pool
  the thread only exits at `close`. `proof._drain_and_join`, which joins threads, still serves
  `proof._render_frames_parallel`, whose workers really are per-call.
- **`film.py` owns the lifetime**: its `finally` calls `map_renderer.close()` before
  `scene_capture.__exit__`. The workers are daemon threads, so without that an export would leave
  browsers alive until the process exits. `close()` is idempotent and a no-op before any leg ran.
