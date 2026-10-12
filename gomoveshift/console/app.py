"""FastAPI app factory: fronts the CLIs, never re-implements pipeline logic."""

import logging
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from ..cli.handlers.enqueue_asset import build_enqueue_asset_argv
from ..cli.handlers.route_story_film import build_route_story_film_argv
from ..pipeline import comments as comments_pipeline
from ..pipeline.comments import flag_reasons
from ..pipeline.route_story import profile as route_story_profile
from ..web.app import _FrameEncodeError, _read_frame_sync
from . import (
    drives,
    insights,
    jobs,
    journeys,
    library,
    notify,
    photos,
    pubqueue,
    pubqueue_calendar,
    route_story,
    schedule,
    seo,
    shorts,
    tenant,
)
from .auth import ConsoleAuthMiddleware, is_authenticated
from .route_story_progress import parse_progress
from .settings import (
    brand_disabled_tabs,
    brand_launch_env,
    console_data_dir,
    credential_status,
    current_brand,
    known_brands,
    load_brand_settings,
    load_registry,
    registry_status,
    save_brand_settings,
    youtube_identity_status,
)
from .tabs import TabGateMiddleware

logger = logging.getLogger(__name__)

#: Left-nav entries (slug, title), in display order.
NAV_ENTRIES: list[tuple[str, str]] = [
    ("library", "Media library"),
    ("drives", "Drives"),
    ("shorts", "Shorts"),
    ("photos", "Photos"),
    ("seo", "SEO & Performance"),
    ("comments", "Comments & corrections"),
    ("journeys", "Journeys & site"),
    ("publishing", "Publishing"),
    ("schedule", "Schedule"),
    ("settings", "Settings"),
    ("jobs", "Jobs"),
]

_NAV_SLUGS = {slug for slug, _title in NAV_ENTRIES}

#: POST /jobs subcommand -> owning nav tab, for the same ``_require_tab``
#: gate the tab's own routes apply. ``status`` has no owning tab.
#: ``schedule`` and ``reconcile-playlists`` are deliberately absent -- no
#: nav tab owns either yet, so they launch ungated from the generic form.
_SUBCOMMAND_TAB: dict[str, str] = {
    "drive": "drives",
    "photos": "photos",
    "sync-site": "journeys",
    "publish-journal": "journeys",
    "enqueue-plan": "publishing",
    "enqueue-asset": "publishing",
    "dispatch-queue": "publishing",
    "publishing-status": "publishing",
}


#: Label per develop-form number, so a refusal names the field the way the
#: page does. `float | None = Form(None)` never got this far -- a typo was
#: answered by FastAPI's 422 handler with a list of pydantic error dicts.
_PHOTO_NUMBERS: dict[str, str] = {
    "vignette": "Vignette",
    "clarity": "Clarity",
    "grain": "Grain",
    "grain_size": "Grain size",
    "halation": "Halation",
    "white_balance_g": "White balance, green-magenta",
    "white_balance_r": "White balance, red",
    "white_balance_b": "White balance, blue",
    "white_balance_r_high": "White balance, red highlights",
    "white_balance_b_high": "White balance, blue highlights",
    "white_balance_g_high": "White balance, green highlights",
}


def _route_story_receipt_for_job(argv: list[str]) -> str | None:
    """The job's launch profile's ``receipt`` key, if it named one -- read
    back from ``argv``'s ``--profile`` path since the stored argv is the
    CLI invocation, not the film argv built in-process."""
    try:
        profile_path = argv[argv.index("--profile") + 1]
    except ValueError:
        return None
    try:
        profile = route_story_profile.load_profile(Path(profile_path))
    except route_story_profile.ProfileError:
        return None
    receipt = profile.get("receipt")
    return str(receipt) if receipt else None


def create_app() -> FastAPI:
    """Build the console FastAPI app. ``GMS_CONSOLE_BRAND`` (default
    "gomoveshift") selects which nav tabs are enabled -- e.g. an instance
    without a drive pipeline hides the drive pipeline tabs."""

    def _job_store() -> jobs.JobStore:
        return jobs.JobStore(console_data_dir() / "jobs.sqlite3")

    def _log_dir() -> Path:
        return console_data_dir() / "job_logs"

    ticker_enabled = os.environ.get("GMS_CONSOLE_TICKER") == "1"
    tick_seconds = float(os.environ.get("GMS_CONSOLE_TICK_SECONDS", "60"))

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        """Opt-in (``GMS_CONSOLE_TICKER=1``) background drain of due
        schedules -- see docs/pipeline/107-job-schedules.md.
        Off by default so tests/instances that don't ask for it never spawn
        a thread."""
        # Claim/verify the tenant stamp before anything else -- console/tenant.py.
        # BOTH of this instance's storage directories, for the same reason: the
        # image bakes identical container-side paths for every brand, so only
        # the host side of each bind mount distinguishes them and no path
        # comparison in here can see a collision. The data dir holds
        # jobs.sqlite3, the job logs and each brand's saved settings -- sharing
        # it is the same leak as sharing the queue, and the queue's stamp
        # cannot see it.
        brand = current_brand()
        tenant.claim_storage(pubqueue.queue_db_path().parent, brand)
        tenant.claim_storage(console_data_dir(), brand)
        # A restart loses every in-process Popen handle, so any DB row
        # still "running" from before the restart is stranded -- sweep it
        # to "interrupted" before anything else reads the jobs table.
        jobs.sweep_orphaned_jobs(_job_store())
        app.state.console_ticker = (
            jobs.start_ticker(_job_store(), _log_dir(), tick_seconds) if ticker_enabled else None
        )
        yield
        if app.state.console_ticker is not None:
            app.state.console_ticker.stop()
            app.state.console_ticker = None

    app = FastAPI(title="GoMoveShift Console", lifespan=_lifespan)
    from .night import router as night_router
    from .route_story import router as route_story_router

    app.include_router(night_router)
    app.include_router(route_story_router)
    app.add_middleware(ConsoleAuthMiddleware)
    app.add_middleware(TabGateMiddleware)
    templates_dir = Path(__file__).parent / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))
    disabled_tabs = brand_disabled_tabs(current_brand())
    nav_entries = [entry for entry in NAV_ENTRIES if entry[0] not in disabled_tabs]
    # _base.html renders the nav on EVERY page, including the error page, so the
    # list has to be reachable without each route remembering to pass it. A
    # global is right here because it is brand-level configuration fixed for the
    # life of the app, not per-request state.
    templates.env.globals["nav_entries"] = nav_entries

    def _require_tab(slug: str) -> None:
        """404 a direct GET/POST to a tab this brand doesn't enable."""
        if slug in disabled_tabs:
            raise HTTPException(status_code=404, detail=f"Unknown console tab: {slug!r}")

    def _wants_json(request: Request) -> bool:
        """API-ish callers ask for JSON explicitly; browsers get the HTML
        error page (with the normal nav) even on an unhandled exception."""
        accept = request.headers.get("accept", "")
        return "application/json" in accept and "text/html" not in accept

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException):
        if _wants_json(request):
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        return templates.TemplateResponse(
            request,
            "error.html",
            {
                "nav_entries": nav_entries,
                "status_code": exc.status_code,
                "message": exc.detail,
            },
            status_code=exc.status_code,
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("unhandled console exception on %s", request.url.path)
        notify.notify(
            "console_error",
            {"path": request.url.path, "exc_type": type(exc).__name__},
            str(exc),
        )
        if _wants_json(request):
            return JSONResponse({"detail": "Internal Server Error"}, status_code=500)
        return templates.TemplateResponse(
            request,
            "error.html",
            {
                "nav_entries": nav_entries,
                "status_code": 500,
                "message": "Internal Server Error",
            },
            status_code=500,
        )

    def _stale(last: "datetime | None", started_at: datetime) -> bool:
        """Staleness of the last completed tick/observer pass -- ``last is
        None`` (never completed) falls back to ``started_at`` so a wedged
        first pass is caught too, instead of reading healthy forever."""
        reference = last if last is not None else started_at
        return (datetime.now(timezone.utc) - reference) > timedelta(seconds=3 * tick_seconds)

    @app.get("/healthz")
    def healthz(request: Request):
        brands = known_brands() or ["gomoveshift"]
        ticker = getattr(app.state, "console_ticker", None)
        ticker_alive = ticker.is_alive() if ticker is not None else None
        notify_configured = bool(os.environ.get("GMS_CONSOLE_NOTIFY_URL", "").strip())
        observer_ok = (
            None if ticker is None else not _stale(ticker.last_observer_ok_at, ticker.started_at)
        )
        body = {
            "status": "ok",
            "brands": brands,
            "ticker_alive": ticker_alive,
            "observer_ok": observer_ok,
            "notify_configured": notify_configured,
        }
        degraded = ticker_enabled and (
            ticker_alive is False
            or _stale(ticker.last_tick_ok_at, ticker.started_at)
            or not observer_ok
        )
        status_code = 503 if degraded else 200
        if degraded:
            body["status"] = "degraded"
        # /healthz is exempt from ConsoleAuthMiddleware (the compose
        # healthcheck can't carry a token), but an unauthenticated caller
        # still shouldn't see brand slugs / ticker / observer internals --
        # only the status code and a minimal body. The status code itself
        # (200 vs 503) is identical either way so the container
        # healthcheck never depends on auth.
        if not is_authenticated(request):
            return JSONResponse({"status": body["status"]}, status_code=status_code)
        return JSONResponse(body, status_code=status_code)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        # Local import: this route is the only caller in this file.
        from . import overview  # noqa: PLC0415

        context = {"nav_entries": nav_entries, **overview.build_overview(_job_store())}
        return templates.TemplateResponse(request, "index.html", context)

    def _jobs_context() -> dict:
        store = _job_store()
        return {
            "job_rows": jobs.list_jobs(store),
            "allowed": sorted(jobs.GENERIC_SUBCOMMANDS),
            "schedule_rows": jobs.list_schedules(store),
            "ticker_enabled": ticker_enabled,
        }

    @app.get("/jobs", response_class=HTMLResponse)
    def jobs_list(request: Request):
        # A GET must never start work -- draining (ticker tick or the
        # explicit POST /jobs/drain below) is the only thing that launches
        # a due schedule.
        return templates.TemplateResponse(request, "jobs.html", _jobs_context())

    @app.post("/jobs/drain")
    def jobs_drain():
        jobs.run_due(_job_store(), datetime.now(timezone.utc), _log_dir())
        return RedirectResponse("/jobs", status_code=303)

    @app.post("/jobs/schedules")
    def schedules_create(
        request: Request,
        name: str = Form(""),
        subcommand: str = Form(""),
        # Was `int = Form(...)`, so "every hour" never reached this route
        # -- FastAPI answered it with a 422 whose body is a list of pydantic
        # error dicts. A string parsed by `_number` says which field and why.
        interval_minutes: str = Form(""),
    ):
        form = {"name": name, "subcommand": subcommand, "interval_minutes": interval_minutes}

        def _refuse(message: str):
            return templates.TemplateResponse(
                request,
                "jobs.html",
                {**_jobs_context(), "form": form, "error": message},
                status_code=400,
            )

        try:
            minutes = _number(interval_minutes, "Interval (minutes)", int)
            if minutes is None:
                raise ValueError("Interval (minutes): required")
            jobs.add_schedule(_job_store(), name, subcommand, minutes)
        except ValueError as exc:
            return _refuse(str(exc))
        return RedirectResponse("/jobs", status_code=303)

    @app.post("/jobs/schedules/{schedule_id}/toggle")
    def schedules_toggle(schedule_id: int):
        jobs.toggle_schedule(_job_store(), schedule_id)
        return RedirectResponse("/jobs", status_code=303)

    @app.post("/jobs/schedules/{schedule_id}/delete")
    def schedules_delete(schedule_id: int):
        jobs.delete_schedule(_job_store(), schedule_id)
        return RedirectResponse("/jobs", status_code=303)

    @app.post("/jobs")
    def jobs_create(subcommand: str = Form(...)):
        tab = _SUBCOMMAND_TAB.get(subcommand)
        if tab is not None:
            _require_tab(tab)
        try:
            argv = jobs.build_argv(subcommand)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        job_id = jobs.launch(_job_store(), subcommand, argv, _log_dir(), env=brand_launch_env())
        return RedirectResponse(f"/jobs/{job_id}", status_code=303)

    @app.post("/route-stories/film")
    def route_story_film_create(profile: str = Form(...), dry_run: bool = Form(False)):
        try:
            resolved = route_story.resolve_profile(profile)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        argv = build_route_story_film_argv(resolved, dry_run=dry_run)
        job_id = jobs.launch(
            _job_store(), "route-story-film", argv, _log_dir(), env=brand_launch_env()
        )
        return RedirectResponse(f"/jobs/{job_id}", status_code=303)

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_detail(request: Request, job_id: int):
        row = jobs.get_job(_job_store(), job_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
        tail = jobs.tail_log(row["log_path"], 100) if row["log_path"] else []
        route_story_ctx = None
        if row["name"] == "route-story-film":
            route_story_ctx = parse_progress(tail)
            enqueue_cmd = None
            if route_story_ctx["finished"] and route_story_ctx["output"]:
                receipt = _route_story_receipt_for_job(row["argv"])
                if receipt:
                    enqueue_cmd = (
                        f"gomoveshift enqueue-asset {route_story_ctx['output']} "
                        f"--receipt {receipt} --platform youtube --at <ISO>"
                    )
            route_story_ctx["enqueue_cmd"] = enqueue_cmd
        return templates.TemplateResponse(
            request,
            "job_detail.html",
            {
                "job": row,
                "log_tail": tail,
                "route_story": route_story_ctx,
                # Same decision the POST route enforces, so the page never
                # shows a button that is going to be refused.
                "rerun_blocked": jobs.rerun_blocked_reason(row),
                # What this console observed, for a row it could never get an
                # exit code for. None for every job it actually reaped.
                "outcome_evidence": jobs.interrupted_evidence_sentence(row),
            },
        )

    @app.post("/jobs/{job_id}/rerun")
    def job_rerun(job_id: int):
        store = _job_store()
        # Reap first: a job whose child has since exited is `done`/`failed`
        # by the time we decide, so a finished job is never refused merely
        # because nothing had looked at it yet.
        row = jobs.get_job(store, job_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
        blocked = jobs.rerun_blocked_reason(row)
        if blocked is not None:
            raise HTTPException(status_code=409, detail=blocked)
        launch_brand = row.get("brand")
        # The argv is the old job's, the environment is resolved from the
        # settings as they stand right now. That hybrid is what an operator
        # usually wants, but it is neither "retry that job" nor "start again
        # with current settings", and until `rerun_of` was recorded the new job
        # carried no trace of which job it came from -- so the page could not
        # say which of the two it had done.
        env = brand_launch_env(launch_brand) if launch_brand else brand_launch_env()
        new_id = jobs.launch(
            store,
            row["name"],
            row["argv"],
            _log_dir(),
            env=env,
            brand=launch_brand,
            rerun_of=job_id,
        )
        return RedirectResponse(f"/jobs/{new_id}", status_code=303)

    @app.get("/library", response_class=HTMLResponse)
    def library_tab(
        request: Request,
        media_type: "str | None" = None,
        journey: "str | None" = None,
        date: "str | None" = None,
        q: "str | None" = None,
        page: "str | None" = None,
    ):
        _require_tab("library")
        from ..catalog.night import read_night_projects

        context = library.build_context(
            media_type=media_type or None,
            journey=journey or None,
            date=date or None,
            q=q or None,
            page=page or 1,
        )
        # The active filters, so the pager links can carry them forward; a
        # "next" that dropped them lands on a different collection entirely.
        context.update({"media_type": media_type, "journey": journey, "date": date, "q": q})
        if "photos" not in brand_disabled_tabs(current_brand()):
            context["night_library"] = read_night_projects(
                console_data_dir() / "night-projects", current_brand()
            )
        context["filter_query"] = urlencode(
            {
                k: v
                for k, v in (
                    ("media_type", media_type),
                    ("journey", journey),
                    ("date", date),
                    ("q", q),
                )
                if v
            }
        )
        return templates.TemplateResponse(request, "library.html", context)

    @app.post("/library/{catalog_id:path}/tags")
    def library_tags_update(catalog_id: str, tag: str = Form(...), action: str = Form("add")):
        _require_tab("library")
        if action == "remove":
            library.remove_library_tag(catalog_id, tag)
        else:
            library.add_library_tag(catalog_id, tag)
        return RedirectResponse("/library", status_code=303)

    @app.get("/library/media/{catalog_id:path}")
    def library_media(catalog_id: str):
        _require_tab("library")
        found = library.resolve_photo_media(catalog_id)
        if found is None:
            raise HTTPException(status_code=404, detail="Photo unavailable")
        path, content_type = found
        return FileResponse(path, media_type=content_type)

    @app.get("/library/{catalog_id:path}", response_class=HTMLResponse)
    def library_asset(request: Request, catalog_id: str):
        _require_tab("library")
        context = library.build_asset_context(catalog_id)
        if context is None:
            raise HTTPException(status_code=404, detail=f"Unknown catalog_id: {catalog_id!r}")
        return templates.TemplateResponse(request, "library_asset.html", context)

    def _drives_context() -> dict:
        return {
            "brand": current_brand(),
            "brands": known_brands() or ["gomoveshift"],
            "weekdays": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
            "lf_modes": drives.LF_MODE_CHOICES,
        }

    @app.get("/drives", response_class=HTMLResponse)
    def drives_tab(request: Request):
        _require_tab("drives")
        return templates.TemplateResponse(request, "drives.html", _drives_context())

    def _number(raw: str, field: str, cast):
        """``int("seven")`` says "invalid literal for int() with base 10",
        which names neither the field nor anything an operator can act on.
        This form has thirteen inputs; the message has to say which one."""
        if not raw.strip():
            return None
        try:
            return cast(raw)
        except ValueError as exc:
            raise ValueError(f"{field}: {raw.strip()!r} is not a number") from exc

    def _require_schedule(publish: str, lf_day: str) -> None:
        """Every argv the console builds is a real ``drive`` launch (neither
        tab exposes ``--list-lfs``/``--label-dry-run``), and the ``drive``
        handler now hard-refuses (``SystemExit``) an unscheduled launch
        before any Phase A work. Catch that here, at submit time, so an
        operator gets a form refusal instead of a job that crashes after
        launch."""
        if not publish.strip() and not lf_day.strip():
            raise ValueError(
                "Pick a Publish day (--publish) or a Long-form day (--lf-day) "
                "before launching -- drive refuses an unscheduled run."
            )

    @app.post("/drives/run")
    def drives_run(
        request: Request,
        directory: str = Form(...),
        output_dir: str = Form(...),
        region: str = Form(""),
        dry_run: bool = Form(False),
        no_review: bool = Form(False),
        publish: str = Form(""),
        lf_day: str = Form(""),
        lf_mode: str = Form(""),
        lf_target_duration: str = Form(""),
        superlong: bool = Form(False),
        shorts_target: str = Form(""),
        shorts_gap: str = Form(""),
        brand: str = Form(""),
    ):
        _require_tab("drives")
        # Every value the operator typed, kept together so a refusal can
        # hand the whole form back. A 400 page discarded twelve good fields to
        # report one bad one, and a browser will not reliably restore a POSTed
        # form -- so the mistyped digit cost the other twelve.
        form = {
            "directory": directory,
            "output_dir": output_dir,
            "region": region,
            "dry_run": dry_run,
            "no_review": no_review,
            "publish": publish,
            "lf_day": lf_day,
            "lf_mode": lf_mode,
            "lf_target_duration": lf_target_duration,
            "superlong": superlong,
            "shorts_target": shorts_target,
            "shorts_gap": shorts_gap,
            "brand": brand,
        }

        def _refuse(message: str):
            return templates.TemplateResponse(
                request,
                "drives.html",
                {**_drives_context(), "form": form, "error": message},
                status_code=400,
            )

        slug = brand or current_brand()
        if slug not in (known_brands() or ["gomoveshift"]):
            # The brand is a field on this form like any other, so a bad one
            # is a correction to make here, not a separate page.
            return _refuse(f"Brand: {slug!r} is not a brand this console knows.")
        try:
            _require_schedule(publish, lf_day)
            argv = drives.build_drive_argv(
                directory,
                output_dir=output_dir,
                region=region or None,
                dry_run=dry_run,
                no_review=no_review,
                publish=publish or None,
                lf_day=lf_day or None,
                lf_mode=lf_mode or None,
                lf_target_duration=_number(lf_target_duration, "Long-form target duration", float),
                superlong=superlong,
                shorts_target=_number(shorts_target, "Shorts per long-form", int),
                shorts_gap=_number(shorts_gap, "Minimum seconds between shorts", float),
            )
        except ValueError as exc:
            return _refuse(str(exc))
        job_id = jobs.launch(
            _job_store(), "drive", argv, _log_dir(), env=brand_launch_env(slug), brand=slug
        )
        return RedirectResponse(f"/jobs/{job_id}", status_code=303)

    def _console_output_dirs() -> "list[str]":
        rows = jobs.list_jobs(_job_store())
        return drives.output_dirs_from_job_argvs([row["argv"] for row in rows])

    @app.get("/drives/preview", response_class=HTMLResponse)
    def drives_preview(request: Request, output_dir: str):
        _require_tab("drives")
        resolved = drives.resolve_allowed_output_dir(output_dir, _console_output_dirs())
        if resolved is None:
            raise HTTPException(status_code=404, detail="Unknown output_dir")
        previews = drives.collect_metadata_previews(resolved)
        if not previews:
            raise HTTPException(
                status_code=404,
                detail="No generated metadata found in output_dir",
            )
        return templates.TemplateResponse(
            request,
            "drives_preview.html",
            {"output_dir": str(resolved), "previews": previews},
        )

    @app.get("/drives/gates/pending", response_class=HTMLResponse)
    def drives_gates_pending(request: Request):
        _require_tab("drives")
        pending = drives.list_pending_gates(_console_output_dirs())
        return templates.TemplateResponse(
            request, "drives_gates_pending.html", {"pending": pending}
        )

    @app.get("/drives/gates", response_class=HTMLResponse)
    def drives_gates(request: Request, output_dir: str, date_key: str):
        _require_tab("drives")
        resolved = drives.resolve_allowed_output_dir(output_dir, _console_output_dirs())
        if resolved is None or not drives.is_valid_gate_date_key(date_key):
            raise HTTPException(status_code=404, detail="Unknown output_dir")
        question = drives.read_gate_question(resolved, date_key)
        media_items = drives.gate_media_items(resolved, question)
        return templates.TemplateResponse(
            request,
            "drives_gates.html",
            {
                "output_dir": str(resolved),
                "date_key": date_key,
                "question": question,
                "media_items": media_items,
            },
        )

    @app.get("/drives/gates/frame")
    def drives_gates_frame(output_dir: str, date_key: str, index: int):
        """A still frame for one video this gate covers -- the ONLY input
        taken from the request is ``index`` into the gate's own artifact
        list; the filesystem path itself is never accepted as a parameter
        (see ``drives.gate_frame_path``)."""
        _require_tab("drives")
        resolved = drives.resolve_allowed_output_dir(output_dir, _console_output_dirs())
        if resolved is None or not drives.is_valid_gate_date_key(date_key):
            raise HTTPException(status_code=404, detail="Unknown output_dir")
        question = drives.read_gate_question(resolved, date_key)
        path = drives.gate_frame_path(resolved, question, index)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="Video not found")
        try:
            frame_bytes = _read_frame_sync(path, 0.0)
        except _FrameEncodeError:
            raise HTTPException(status_code=500, detail="Frame encoding failed") from None
        if frame_bytes is None:
            raise HTTPException(status_code=404, detail="Frame not found")
        return Response(content=frame_bytes, media_type="image/jpeg")

    @app.get("/drives/gates/video")
    def drives_gates_video(output_dir: str, date_key: str, index: int):
        """The video itself, so the reviewer can actually play what the gate
        asks them to confirm they played. Same containment as the frame route
        above -- ``index`` into the gate's own artifact list is the only input
        taken from the request. ``FileResponse`` answers Range requests, so
        the player can seek instead of downloading the whole file to move."""
        _require_tab("drives")
        resolved = drives.resolve_allowed_output_dir(output_dir, _console_output_dirs())
        if resolved is None or not drives.is_valid_gate_date_key(date_key):
            raise HTTPException(status_code=404, detail="Unknown output_dir")
        question = drives.read_gate_question(resolved, date_key)
        media = drives.resolve_gate_video_media(resolved, question, index)
        if media is None:
            raise HTTPException(status_code=404, detail="Video not found")
        path, content_type = media
        return FileResponse(path, media_type=content_type)

    @app.post("/drives/gates")
    def drives_gates_answer(
        # The hidden output_dir/date_key form fields (drives_gates.html) only
        # NAME which run this POST is answering -- they are never trusted as
        # authority; resolve_allowed_output_dir/is_valid_gate_date_key below
        # are what actually gate the read/write/delete.
        output_dir: str = Form(...),
        date_key: str = Form(...),
        decision: str = Form(...),
        note: str = Form(""),
    ):
        _require_tab("drives")
        resolved = drives.resolve_allowed_output_dir(output_dir, _console_output_dirs())
        if resolved is None or not drives.is_valid_gate_date_key(date_key):
            raise HTTPException(status_code=404, detail="Unknown output_dir")
        try:
            drives.write_gate_answer(resolved, date_key, decision, note)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RedirectResponse(
            f"/drives/gates?output_dir={resolved}&date_key={date_key}", status_code=303
        )

    @app.get("/shorts", response_class=HTMLResponse)
    def shorts_tab(request: Request):
        _require_tab("shorts")
        return templates.TemplateResponse(
            request,
            "shorts.html",
            {
                "brands": known_brands() or ["gomoveshift"],
                "brand": current_brand(),
                "weekdays": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
            },
        )

    @app.post("/shorts/run")
    def shorts_run(
        directory: str = Form(...),
        output_dir: str = Form(...),
        region: str = Form(""),
        dry_run: bool = Form(False),
        publish: str = Form(""),
        lf_day: str = Form(""),
        shorts_target: str = Form(""),
        shorts_gap: str = Form(""),
        brand: str = Form(""),
    ):
        _require_tab("shorts")
        slug = brand or current_brand()
        if slug not in (known_brands() or ["gomoveshift"]):
            raise HTTPException(status_code=400, detail=f"unknown brand: {slug!r}")
        try:
            _require_schedule(publish, lf_day)
            argv = shorts.build_shorts_argv(
                directory,
                output_dir=output_dir,
                region=region or None,
                dry_run=dry_run,
                publish=publish or None,
                lf_day=lf_day or None,
                shorts_target=int(shorts_target) if shorts_target.strip() else None,
                shorts_gap=float(shorts_gap) if shorts_gap.strip() else None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        job_id = jobs.launch(
            _job_store(), "shorts", argv, _log_dir(), env=brand_launch_env(slug), brand=slug
        )
        return RedirectResponse(f"/jobs/{job_id}", status_code=303)

    @app.get("/shorts/csv", response_class=HTMLResponse)
    def shorts_csv(request: Request, output_dir: str):
        _require_tab("shorts")
        resolved = drives.resolve_allowed_output_dir(output_dir, _console_output_dirs())
        if resolved is None:
            raise HTTPException(status_code=404, detail="Unknown output_dir")
        csv_path = shorts.find_shorts_csv(resolved)
        if csv_path is None:
            raise HTTPException(status_code=404, detail="No shorts CSV found in output_dir")
        header, rows = shorts.read_shorts_csv_rows(csv_path)
        return templates.TemplateResponse(
            request,
            "shorts_csv.html",
            {"output_dir": str(resolved), "header": header, "rows": rows},
        )

    @app.get("/shorts/csv/download")
    def shorts_csv_download(output_dir: str):
        _require_tab("shorts")
        resolved = drives.resolve_allowed_output_dir(output_dir, _console_output_dirs())
        if resolved is None:
            raise HTTPException(status_code=404, detail="Unknown output_dir")
        csv_path = shorts.find_shorts_csv(resolved)
        if csv_path is None:
            raise HTTPException(status_code=404, detail="No shorts CSV found in output_dir")
        return FileResponse(csv_path, media_type="text/csv", filename=csv_path.name)

    def _photos_context(journey: str = "") -> dict:
        return {
            "looks": sorted(photos.resolved_looks()),
            "styles": photos.resolved_styles(),
            "capabilities": photos.capability_status(),
            "stages": photos.journey_stages(journey),
            "journey": journey,
            "photo_options": photos.journey_photo_options(journey),
        }

    @app.get("/photos", response_class=HTMLResponse)
    def photos_tab(request: Request, journey: str = ""):
        # The selected journey rides in the query string so the stage
        # links can carry it on to review/schedule and back again.
        return templates.TemplateResponse(request, "photos.html", _photos_context(journey))

    @app.post("/photos/run")
    def photos_run(
        request: Request,
        action: str = Form(...),
        card_dir: str = Form(""),
        journey: str = Form(""),
        day: str = Form(""),
        location: str = Form(""),
        dry_run: bool = Form(False),
        paths: str = Form(""),
        picked_paths: list[str] = Form([]),
        out: str = Form(""),
        styles: str = Form(""),
        look: str = Form(""),
        export_format: str = Form(""),
        vignette: str = Form(""),
        clarity: str = Form(""),
        grain: str = Form(""),
        grain_size: str = Form(""),
        halation: str = Form(""),
        white_balance_g: str = Form(""),
        white_balance_r: str = Form(""),
        white_balance_b: str = Form(""),
        white_balance_r_high: str = Form(""),
        white_balance_b_high: str = Form(""),
        white_balance_g_high: str = Form(""),
    ):
        # Every field as typed, so a refusal can hand the whole thing back.
        # `action` rides along: the page carries two forms and only the one
        # that was submitted is refilled.
        form = {
            "action": action,
            "card_dir": card_dir,
            "journey": journey,
            "day": day,
            "location": location,
            "dry_run": dry_run,
            "paths": paths,
            "out": out,
            "styles": styles,
            "look": look,
            "export_format": export_format,
            "vignette": vignette,
            "clarity": clarity,
            "grain": grain,
            "grain_size": grain_size,
            "halation": halation,
            "white_balance_g": white_balance_g,
            "white_balance_r": white_balance_r,
            "white_balance_b": white_balance_b,
            "white_balance_r_high": white_balance_r_high,
            "white_balance_b_high": white_balance_b_high,
            "white_balance_g_high": white_balance_g_high,
        }

        def _refuse(message: str):
            return templates.TemplateResponse(
                request,
                "photos.html",
                {**_photos_context(), "form": form, "error": message},
                status_code=400,
            )

        try:
            numbers = {
                name: _number(form[name], label, float) for name, label in _PHOTO_NUMBERS.items()
            }
        except ValueError as exc:
            return _refuse(str(exc))
        try:
            if action == "ingest":
                argv = photos.build_photos_argv(
                    "ingest",
                    card_dir=card_dir,
                    journey=journey,
                    day=day,
                    location=location or None,
                    dry_run=dry_run,
                )
            elif action == "develop":
                # Picked photos (from the picker) are additive to typed
                # paths, deduplicated in order -- both routes reach the SAME
                # validation and the same job launcher below, never a second
                # path into it.
                develop_paths = list(
                    dict.fromkeys(
                        [p for p in paths.splitlines() if p.strip()]
                        + [p for p in picked_paths if p.strip()]
                    )
                )
                # The typed textarea lost its HTML `required` now that
                # the picker is an additive second way to supply photos, so
                # "gave neither" is checked here instead of by the browser --
                # still upstream of the job launch, never a bare argparse
                # failure buried in the log.
                if not develop_paths:
                    return _refuse("paths: required (type a path or pick a photo)")
                # Refuse before the job launches, naming the
                # specific filesystem problem -- never a generic "invalid
                # path" the operator has to re-derive by hand.
                path_errors = photos.validate_develop_paths(develop_paths, out)
                if path_errors:
                    return _refuse("; ".join(path_errors))
                argv = photos.build_photos_argv(
                    "develop",
                    paths=develop_paths,
                    out=out,
                    styles=styles or None,
                    look=look or None,
                    export_format=export_format or None,
                    **numbers,
                )
            else:
                raise ValueError(f"unsupported photos action: {action!r}")
        except ValueError as exc:
            return _refuse(str(exc))
        job_id = jobs.launch(_job_store(), "photos", argv, _log_dir(), env=brand_launch_env())
        return RedirectResponse(f"/jobs/{job_id}", status_code=303)

    def _seo_context(journey: str = "") -> dict:
        # Damage is shown with the editor withheld: offering an "add correction"
        # form over bytes we failed to read is how curation used to be lost.
        # The journey selection rides here too, for the visible indicator
        # and the nav links -- but corrections/denylist are a global
        # place-name list and performance rows carry no journey identity
        # (`insights.performance_view` docstring), so nothing below is
        # actually narrowed by it. The template says so rather than
        # pretending to filter data that cannot honestly be filtered.
        context: dict = {
            "corrections": [],
            "denylist": [],
            "revision": "",
            "corrupt": "",
            "journey": journey,
        }
        try:
            ledger = seo.read()
        except seo.LedgerCorruptError as exc:
            context["corrupt"] = str(exc)
        else:
            context.update(
                corrections=ledger.corrections,
                denylist=ledger.denylist,
                revision=ledger.revision,
            )
        # The performance half of the tab its own name has always
        # promised. It is a separate read with its own failure, kept out of
        # the `corrupt` branch above on purpose -- an unreadable ledger must
        # not hide the numbers, and unavailable numbers must not withhold the
        # editor. `performance_view` never raises; it returns its reason.
        context.update(performance=insights.performance_view())
        # Shadow-only hypotheses; `anomaly_view` never raises either.
        context.update(anomalies=insights.anomaly_view())
        return context

    @app.get("/seo", response_class=HTMLResponse)
    def seo_tab(request: Request, journey: str = ""):
        return templates.TemplateResponse(request, "seo.html", _seo_context(journey))

    @app.post("/seo/corrections")
    def seo_corrections_add(
        request: Request,
        wrong: str = Form(""),
        correct: str = Form(""),
        denylist_name: str = Form(""),
        revision: str = Form(""),
    ):
        # The form always renders a revision; an absent one is not our own page.
        expected = revision if revision else None
        # `kind` is which of this page's two forms was submitted. Only that
        # one is refilled -- putting a denylist name into the correction boxes
        # would be text in a form nobody touched.
        kind = "correction" if (wrong.strip() or correct.strip()) else ""
        if not kind and denylist_name.strip():
            kind = "denylist"
        form = {
            "kind": kind,
            "wrong": wrong,
            "correct": correct,
            "denylist_name": denylist_name,
        }

        def _refuse(message: str, status: int):
            # The context is re-READ, not the one this submission was built
            # against: on a 409 someone else has written, and the whole point of
            # handing the page back is that it now shows what they wrote. The
            # fresh revision rides along, so the operator's next submit is an
            # informed one rather than a retry that fails the same way forever.
            return templates.TemplateResponse(
                request,
                "seo.html",
                {**_seo_context(), "form": form, "error": message},
                status_code=status,
            )

        try:
            # A half-filled correction goes to the writer, not the generic
            # message below: the writer says which side is missing, and the
            # operator plainly meant to add a correction, not a denylist name.
            if kind == "correction":
                seo.add_correction(wrong.strip(), correct.strip(), expected_revision=expected)
            elif kind == "denylist":
                seo.add_denylist_entry(denylist_name.strip(), expected_revision=expected)
            else:
                return _refuse("Provide wrong + correct, or a denylist name.", 400)
        except seo.LedgerInvalidError as exc:
            # The edit itself is the problem, not the ledger's state -- 400,
            # and the reason is the whole point of refusing it.
            return _refuse(str(exc), 400)
        except (seo.LedgerCorruptError, seo.LedgerConflictError) as exc:
            return _refuse(str(exc), 409)
        return RedirectResponse("/seo", status_code=303)

    def _comments_path() -> Path:
        raw = os.environ.get("GMS_COMMENTS_PATH")
        return Path(raw) if raw and raw.strip() else comments_pipeline.default_comments_path()

    @app.get("/comments", response_class=HTMLResponse)
    def comments_tab(request: Request):
        rows = comments_pipeline.load_comments(_comments_path())
        for row in rows:
            reasons = flag_reasons(row.get("text", ""))
            row["triage"] = ", ".join(reasons) if reasons else "clear"
        return templates.TemplateResponse(request, "comments.html", {"rows": rows})

    @app.get("/journeys", response_class=HTMLResponse)
    def journeys_tab(request: Request):
        prs, error = journeys.list_site_prs()
        return templates.TemplateResponse(
            request,
            "journeys.html",
            {"prs": prs, "error": error, "site_repo": journeys.site_repo()},
        )

    @app.post("/journeys/sync-site")
    def journeys_sync_site(request: Request):
        # Same rule as the Media library: the site path is this brand's own
        # config (`GMS_SITE_PATH`), never an operator-supplied string -- a
        # form field here would let one brand's console write into another
        # brand's checkout.
        site_path = journeys.resolve_site_path()
        if site_path is None:
            prs, error = journeys.list_site_prs()
            return templates.TemplateResponse(
                request,
                "journeys.html",
                {
                    "prs": prs,
                    "error": error,
                    "site_repo": journeys.site_repo(),
                    "sync_error": (
                        "No site path configured for this brand. Set GMS_SITE_PATH in Settings."
                    ),
                },
                status_code=400,
            )
        argv = journeys.build_sync_site_argv(site_path)
        job_id = jobs.launch(_job_store(), "sync-site", argv, _log_dir(), env=brand_launch_env())
        return RedirectResponse(f"/jobs/{job_id}", status_code=303)

    @app.post("/journeys/publish-journal-replay")
    def journeys_publish_journal_replay():
        argv = journeys.build_publish_journal_replay_argv()
        job_id = jobs.launch(
            _job_store(), "publish-journal", argv, _log_dir(), env=brand_launch_env()
        )
        return RedirectResponse(f"/jobs/{job_id}", status_code=303)

    def _publishing_context(page: "str | int | None" = 1) -> dict:
        # One bounded page, with the LIMIT and the lifecycle order both in
        # the SQL (``pubqueue.queue_page``). The pager keys go into every
        # context this tab renders, refusals included, or the template raises
        # on a missing name instead of showing the refusal.
        read = pubqueue.queue_page(page=page or 1)
        rows = read.pop("rows")
        return {
            "rows_by_state": pubqueue.rows_by_state(rows),
            # The per-state counts in the table are per-state within this page;
            # the pager line needs how many rows the page itself carries.
            "rows_on_page": len(rows),
            "platforms": pubqueue.known_platforms(),
            "paused": pubqueue.paused_platforms(),
            "daily_caps": pubqueue.platform_daily_caps(),
            **read,
        }

    @app.get("/publishing", response_class=HTMLResponse)
    def publishing_tab(request: Request, page: "str | None" = None):
        _require_tab("publishing")
        return templates.TemplateResponse(request, "pubqueue.html", _publishing_context(page))

    @app.get("/publishing/calendar", response_class=HTMLResponse)
    def publishing_calendar(request: Request, week: "str | None" = None):
        _require_tab("publishing")
        anchor = None
        if week:
            try:
                anchor = date.fromisoformat(week)
            except ValueError:
                anchor = None
        result = pubqueue_calendar.week_calendar(anchor=anchor)
        return templates.TemplateResponse(request, "pubqueue_calendar.html", result)

    @app.post("/publishing/pause")
    def publishing_platform_pause(
        request: Request,
        platform: str = Form(""),
        action: str = Form("pause"),
    ):
        """Pause or resume one platform. The rule and the storage are the
        queue's -- the drain's own SELECT excludes a paused platform, so this
        governs the CLI and a future worker too, not just the button above.
        A refusal comes back on this page rather than a separate error page."""
        _require_tab("publishing")
        try:
            if action not in ("pause", "resume"):
                raise ValueError(f"unsupported publishing action: {action!r}")
            pubqueue.set_platform_pause(
                pubqueue.queue_db_path(), platform, paused=action == "pause"
            )
        except ValueError as exc:
            # Re-READ, not the context this submission was built against: a
            # refusal is exactly when the operator needs the current pause list.
            return templates.TemplateResponse(
                request,
                "pubqueue.html",
                {**_publishing_context(), "error": str(exc)},
                status_code=400,
            )
        return RedirectResponse("/publishing", status_code=303)

    @app.post("/publishing/override-state")
    def publishing_override_state(
        request: Request,
        platform: str = Form(""),
        state: str = Form(""),
        cap: str = Form(""),
    ):
        """An operator manually pins one platform's bounded-response
        state/cap (``pubqueue.set_state_override``), overriding what the
        nightly ``metrics collect`` run would otherwise decide -- the
        override sticks until cleared (``state="clear"``) because
        ``dispatch.apply_suppression_response`` skips any ``manual:``-
        reasoned row. Same refusal shape as ``publishing_platform_pause``:
        a bad platform/state/cap comes back on this page, never a separate
        error page."""
        _require_tab("publishing")
        try:
            cap_value: "int | None" = None
            if cap.strip():
                try:
                    cap_value = int(cap)
                except ValueError as exc:
                    raise ValueError(f"cap must be a whole number, got {cap!r}") from exc
            pubqueue.set_state_override(pubqueue.queue_db_path(), platform, state, cap=cap_value)
        except ValueError as exc:
            return templates.TemplateResponse(
                request,
                "pubqueue.html",
                {**_publishing_context(), "error": str(exc)},
                status_code=400,
            )
        return RedirectResponse("/publishing", status_code=303)

    @app.get("/publishing/{delivery_id}", response_class=HTMLResponse)
    def publishing_detail(request: Request, delivery_id: int):
        _require_tab("publishing")
        try:
            row = pubqueue.delivery_by_id(delivery_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown delivery: {delivery_id}") from exc
        return templates.TemplateResponse(request, "pubqueue_detail.html", {"row": row})

    @app.post("/publishing/enqueue")
    def publishing_enqueue(
        state_path: str = Form(...),
        output_dir: str = Form(""),
        shadow: bool = Form(False),
    ):
        _require_tab("publishing")
        # Both paths arrive as free text in a form field. Contain them the way
        # every other path-taking route here does, against the runs this
        # console's own job argvs record -- a plan file belonging to another
        # deployment must not be enqueueable by typing where it lives.
        known = _console_output_dirs()
        resolved_state = drives.resolve_allowed_run_file(state_path, known)
        if resolved_state is None:
            raise HTTPException(status_code=404, detail="Unknown state_path")
        resolved_output = ""
        if output_dir:
            allowed = drives.resolve_allowed_output_dir(output_dir, known)
            if allowed is None:
                raise HTTPException(status_code=404, detail="Unknown output_dir")
            resolved_output = str(allowed)
        argv = pubqueue.build_enqueue_argv(
            str(resolved_state), output_dir=resolved_output, shadow=shadow
        )
        job_id = jobs.launch(_job_store(), "enqueue-plan", argv, _log_dir(), env=brand_launch_env())
        return RedirectResponse(f"/jobs/{job_id}", status_code=303)

    @app.post("/publishing/enqueue-asset")
    def publishing_enqueue_asset(
        video_path: str = Form(...),
        platform: "list[str]" = Form(...),
        at: str = Form(...),
        title: str = Form(...),
        description: str = Form(""),
        confirm: str = Form(...),
        receipt_path: str = Form(""),
        shadow: bool = Form(False),
    ):
        _require_tab("publishing")
        known = _console_output_dirs()
        resolved_video = drives.resolve_allowed_run_file(video_path, known)
        if resolved_video is None:
            raise HTTPException(status_code=404, detail="Unknown video_path")
        resolved_receipt = ""
        if receipt_path:
            allowed_receipt = drives.resolve_allowed_run_file(receipt_path, known)
            if allowed_receipt is None:
                raise HTTPException(status_code=404, detail="Unknown receipt_path")
            resolved_receipt = str(allowed_receipt)
        argv = build_enqueue_asset_argv(
            str(resolved_video),
            platforms=platform,
            at=at,
            title=title,
            description=description,
            confirm=confirm,
            receipt=resolved_receipt,
            shadow=shadow,
        )
        job_id = jobs.launch(
            _job_store(), "enqueue-asset", argv, _log_dir(), env=brand_launch_env()
        )
        return RedirectResponse(f"/jobs/{job_id}", status_code=303)

    @app.post("/publishing/drain")
    def publishing_drain(sidecar_dir: str = Form("")):
        _require_tab("publishing")
        resolved_sidecar = ""
        if sidecar_dir:
            allowed = drives.resolve_allowed_output_dir(sidecar_dir, _console_output_dirs())
            if allowed is None:
                raise HTTPException(status_code=404, detail="Unknown sidecar_dir")
            resolved_sidecar = str(allowed)
        argv = pubqueue.build_drain_argv(sidecar_dir=resolved_sidecar)
        job_id = jobs.launch(
            _job_store(), "dispatch-queue", argv, _log_dir(), env=brand_launch_env()
        )
        return RedirectResponse(f"/jobs/{job_id}", status_code=303)

    @app.post("/publishing/mark-published/{delivery_id}")
    def publishing_mark_published(
        delivery_id: int,
        permalink: str = Form(""),
        platform_post_id: str = Form(""),
    ):
        _require_tab("publishing")
        try:
            pubqueue.mark_published(
                pubqueue.queue_db_path(),
                delivery_id,
                permalink=permalink or None,
                platform_post_id=platform_post_id or None,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown delivery: {delivery_id}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return RedirectResponse("/publishing", status_code=303)

    def _detail_context(delivery_id: int, **extra) -> dict:
        return {"row": pubqueue.delivery_by_id(delivery_id), **extra}

    # Registered BEFORE the {action} route below, which would otherwise match
    # this path and 404 it as an unknown action.
    @app.post("/publishing/{delivery_id}/reschedule", response_class=HTMLResponse)
    def publishing_reschedule(
        request: Request,
        delivery_id: int,
        scheduled_at: str = Form(""),
    ):
        """Move one delivery's slot. Every rule is the queue's -- a
        terminal row, a live foreign lease, and a timestamp the drain could not
        compare -- so this only decides where the refusal is shown. It comes
        back on THIS page with the value still in the field: an operator who
        worked out a timestamp should not have to work it out twice."""
        _require_tab("publishing")
        try:
            pubqueue.reschedule(pubqueue.queue_db_path(), delivery_id, scheduled_at)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown delivery: {delivery_id}") from exc
        except ValueError as exc:
            # 409 when the row's own state refused it, 400 when the value did:
            # one is "not now", the other "not like that".
            refused_by_state = "is already" in str(exc) or "is held by" in str(exc)
            return templates.TemplateResponse(
                request,
                "pubqueue_detail.html",
                _detail_context(delivery_id, error=str(exc), scheduled_at=scheduled_at),
                status_code=409 if refused_by_state else 400,
            )
        return RedirectResponse(f"/publishing/{delivery_id}", status_code=303)

    @app.post("/publishing/{delivery_id}/{action}")
    def publishing_operator_action(delivery_id: int, action: str):
        """Cancel or retry one delivery. Both rules live in the queue layer --
        terminal rows are final, a live foreign lease may already be
        mid-publish -- so this route only maps its refusal onto HTTP rather
        than re-deciding anything. Redirects back to the DETAIL page, not the
        list: the operator is watching this one row change."""
        _require_tab("publishing")
        if action not in ("cancel", "retry"):
            raise HTTPException(status_code=404, detail=f"Unknown action: {action}")
        try:
            pubqueue.operator_action(pubqueue.queue_db_path(), delivery_id, action)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown delivery: {delivery_id}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return RedirectResponse(f"/publishing/{delivery_id}", status_code=303)

    @app.get("/settings", response_class=HTMLResponse)
    def settings_tab(request: Request):
        overrides = load_brand_settings(current_brand())
        form_rows = [
            {**row, "value": overrides.get(row["name"]) or ""} for row in registry_status()
        ]
        return templates.TemplateResponse(
            request,
            "settings.html",
            {
                "registry_rows": registry_status(),
                "credential_rows": credential_status(),
                "youtube_rows": youtube_identity_status(),
                "form_rows": form_rows,
            },
        )

    @app.post("/settings")
    async def settings_save(request: Request):
        submitted = dict(await request.form())
        registry_names = {row.name for row in load_registry()}
        unknown = sorted(set(submitted) - registry_names)
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown setting(s): {', '.join(unknown)}")
        slug = current_brand()
        values = load_brand_settings(slug)
        for name, raw_value in submitted.items():
            stripped = str(raw_value).strip()
            values[name] = stripped or None
        save_brand_settings(slug, values)
        return RedirectResponse("/settings", status_code=303)

    @app.get("/schedule", response_class=HTMLResponse)
    def schedule_tab(request: Request, journey: str = ""):
        _require_tab("schedule")
        # The journey selection follows an operator here from
        # /library or /photos via ?journey=, the same query parameter --
        # never a second convention.
        return templates.TemplateResponse(
            request, "schedule.html", {"report": schedule.schedule_view(journey=journey or None)}
        )

    @app.get("/{slug}", response_class=HTMLResponse)
    def tab(request: Request, slug: str):
        if slug not in _NAV_SLUGS or slug in disabled_tabs:
            raise HTTPException(status_code=404, detail=f"Unknown console tab: {slug!r}")
        title = dict(NAV_ENTRIES)[slug]
        return templates.TemplateResponse(request, "placeholder.html", {"title": title})

    return app
