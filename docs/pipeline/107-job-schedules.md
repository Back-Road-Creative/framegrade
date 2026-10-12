### Job schedules (`console/jobs.py`'s `schedules` table, `POST /jobs/schedules*`)

A `schedules` table (id/name/subcommand/argv/interval_minutes/enabled/last_run_at), created
idempotently alongside `jobs` so an existing DB gains it on next open. `add_schedule` builds
`argv` via the SAME `build_argv()` the tabs use -- never free-text off the form -- raising
`ValueError` (400 at the route) for a non-allowlisted subcommand. `due_schedules(store, now)`
returns enabled schedules that never ran or whose `last_run_at + interval_minutes <= now`;
`run_due(store, now, log_dir)` launches each due one through the existing allowlisted `launch()`,
passing `env=settings.brand_launch_env()` so a scheduled launch sees the same per-brand
overrides a manual launch does, and stamps `last_run_at`. `GET /jobs` never drains (a read
must never start work); when the ticker is disabled the page instead shows a "not draining"
notice and a `POST /jobs/drain` button that calls `run_due` explicitly, still serialized against
the ticker by `_RUN_DUE_LOCK`. `POST /jobs/schedules` creates a row, `.../toggle` flips `enabled`,
`.../delete` removes it.

**A refusal is an inline correction here too.** `interval_minutes` used to be typed
`int = Form(...)`, so "every hour" never reached the route -- FastAPI answered with a 422 whose body
is a list of pydantic error dicts. It is a string now, parsed by the same `_number` helper the
drives and photos forms use, and the route re-renders `jobs.html` (whole tab, via `_jobs_context()`)
with the submitted name, subcommand and interval plus the message. `add_schedule` also range-checks
the interval itself: the form marks it `min="1"`, which a browser enforces and a direct POST does
not, and below 1 it meant a schedule due on every tick forever. The input is `type="text"
inputmode="numeric"` for the same reason as the photos dials -- a browser blanks a `type="number"`
input it cannot parse, which is exactly the value the operator needs to see.

#### Ticker (`jobs.Ticker`/`start_ticker`, `create_app`'s startup/shutdown hooks)

A real background drain: `Ticker` is a daemon `threading.Thread` that calls `run_due(store, now,
log_dir)` every `tick_seconds`, sleeping via `threading.Event.wait()` so `stop()` (idempotent,
joins the thread) wakes it immediately instead of waiting out the interval. `create_app()` wires
it opt-in -- `GMS_CONSOLE_TICKER=1` starts it on the FastAPI `startup` event (`GMS_CONSOLE_TICK_SECONDS`,
default `60`, sets the interval); the `shutdown` event stops it. Off by default so every test/instance
that doesn't ask for it never spawns a thread; turning it on by default in the compose file is a
follow-up, not part of this change. Each tick also calls `jobs.reap_finished()` -- the same reap
`GET /jobs`/`GET /jobs/{id}` do on read -- so a job failure notifies within one tick instead of
waiting for someone to open the tab. A tick that raises is logged and the loop continues rather
than dying silently; the `Ticker` stamps `last_tick_ok_at`/`last_observer_ok_at` after each
successful pass, plus `started_at` at construction so a wedged first pass is still
catchable. Two further independently-guarded passes each tick call
`notify.check_pending_gates()` with output dirs from the job store's own argvs and
`notify.check_missed_slots()` (both below), each also stamping `last_observer_ok_at`. The
missed-slot pass runs last, after `run_due`, so a slot this very tick drained is already out of
the window it would be judged against.
`GET /healthz` reports `ticker_alive`, `observer_ok` (same tri-state) and `notify_configured`
(`GMS_CONSOLE_NOTIFY_URL` set); while enabled it returns HTTP 503 `status: "degraded"` when the
thread has died or `ticker_alive`/`observer_ok` staleness exceeds 3x the tick interval -- measured
from `last_tick_ok_at`/`last_observer_ok_at` when set, else `started_at` (previously `None` read
as healthy forever). The compose healthcheck (`curl -f`) then flips the container unhealthy.
`run_due` itself is lock-guarded so this ticker and the explicit `POST /jobs/drain` above can't
double-launch the same due schedule.
