## Route Story bundle coverage truth (`gomoveshift/pipeline/route_story/bundle.py`, `scene/`)

`write_tile_pyramid` skips any tile the source raster only partly covers, so a requested zoom can
end up with **no** tiles — at a small footprint every tile of a wide zoom extends past the fetched
raster. A manifest that still advertises that zoom makes the scene request tiles that answer 404,
so `verify_bundle` now reports every zoom in a kind's declared `minzoom..maxzoom` range that holds
no tile on disk. Bundle writers derive the declared range from the tiles actually written; the
verifier is the gate that catches any writer that forgets.

The failure it prevents is silent and expensive: MapLibre marks an errored tile permanently
unloaded, `areTilesLoaded()` never turns true again, and the scene's old unbounded
wait-for-settled then hung **every remaining frame of the film**, not just the offending one (one
900-frame proof render died twice with no output before this was found). `renderPose(pose,
settleTimeoutMs)` now takes a budget, re-checks on a timer as well as on `idle` (that event may
never fire again after a tile errors), collects MapLibre `error` events, and rejects naming the
failing tile URL. `SceneCapture(..., settle_timeout_s=…)` sets the budget and raises
`SceneRenderError` carrying the tile and the pose.

A pose that is only **slow** is a different failure and gets a different answer. The scene words a
bare deadline as `tiles still loading` (no tile errored); `SceneCapture` renders that pose again,
up to `settle_attempts` times (default 3), doubling the budget each time (20 s, 40 s, 80 s), and
logs a warning per retry. A frame is only ever captured from a fully settled scene, so a retry
never changes a pixel — it only waits longer. A **named** tile failure still raises on the first
attempt, because waiting cannot load a missing tile, and a pose that never settles raises after
the last attempt naming the attempt count and the pose. Failure mode: a film render once died 22 minutes in, all frames lost, when one parallel worker's pose missed its 20 s budget on a
heavily loaded host (a `nice`d render beside other work). Do not `nice` a film render on a busy host.

**No wait in the scene capture is unbounded.** `page.evaluate` has no timeout, so nothing the
capture awaits through it may be open-ended. Startup polls the scene's `isReady()` flag with
`wait_for_function` under `ready_timeout_s` (default 60 s) — a deadline the Playwright driver
enforces, so it fires even when the page is wedged — and a browser that misses it is closed and a
**fresh one launched**, up to `ready_attempts` (default 5 — a loaded box has been seen to need all of three), before `SceneRenderError`. Behind that,
the ordered consumer of the parallel capture carries a stall budget (`WORKER_STALL_TIMEOUT_S`,
15 min; a healthy worker's worst case is under six): no frame from the worker it is waiting on for
that long raises instead of waiting, and the join that follows gives up on a worker that never
returns after `WORKER_JOIN_DEADLINE_S` and logs it, since a daemon thread cannot hold the process
open. Failure mode: a film render once froze for good at film frame 2258 — one of six
workers' MapLibre never fired `load` (its GPU process used 5 s of CPU against the others' 2 min),
the consumer waited on that worker's queue while the other five filled theirs, and the process sat
at zero CPU with no error and no exit code.

`tiles_covering` applies its `1e-9` epsilon symmetrically on both edges of a footprint (it already
subtracted it on the east/south max side): a west or north edge landing a float-rounding hair past
a tile boundary now rounds back onto that boundary instead of pulling in a whole extra tile column
or row the raster then can't fully cover.

<a id="route-story-frame-cache"></a>
