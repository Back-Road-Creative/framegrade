"""Structural 404 for a route under a tab this brand disabled.

``create_app()`` (app.py) defines ``_require_tab`` and calls it from every
drives/shorts/library/publishing/schedule/jobs route, but ten routes across
photos, seo, comments, journeys and settings never call it -- found by
reading every ``@app.get``/``@app.post`` decorator in app.py. For those
routes a brand that disables the tab still hides the nav link but serves the
page (and runs the POST) on a direct request -- a tenant-visibility hole.

Rather than add ten more ``_require_tab()`` call sites -- leaving the
eleventh route, written next month, free to forget the same way -- this
middleware derives the tab slug from the request path's first segment and
gates every route, present and future, in one place. The first segment IS
the tab slug for every console route: ``/library/media/x`` -> ``library``,
``/jobs/{id}`` -> ``jobs``, ``/publishing/{id}/{action}`` -> ``publishing``.
A path whose first segment doesn't name a tab at all (``/``, ``/healthz``)
never matches a disabled slug and passes straight through unaffected -- this
middleware only ever narrows what already 404s via the ``/{slug}`` catch-all
and the per-route ``_require_tab`` calls, never widens it.

The ``_require_tab()`` calls inside handlers are deliberately kept in place
rather than deleted: they are redundant behind this middleware, but they stay
as defence in depth, so a guard cannot silently disappear if the middleware
is ever bypassed or reordered.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from .settings import brand_disabled_tabs, current_brand

_templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


class TabGateMiddleware(BaseHTTPMiddleware):
    """404s a request whose first path segment names a disabled tab.

    Raising ``HTTPException`` here wouldn't reach the app's own
    ``@app.exception_handler(HTTPException)`` -- Starlette builds
    ``ServerErrorMiddleware`` -> user middleware (this one, added via
    ``app.add_middleware``) -> ``ExceptionMiddleware`` -> router
    (``starlette/applications.py:build_middleware_stack``), so this
    middleware sits OUTSIDE ``ExceptionMiddleware`` and a raised
    ``HTTPException`` would propagate to ``ServerErrorMiddleware`` as a bare
    500 instead. This builds the same 404 shape the handler produces
    directly instead: JSON ``{"detail": ...}`` for an API caller, the usual
    templated error page (with nav filtered the same way) for a browser.

    ``disabled_tabs`` is brand-level config fixed for the app's lifetime
    (see the comment on the ``nav_entries`` global in ``create_app``) --
    resolved once here, not per request.
    """

    def __init__(self, app):
        super().__init__(app)
        self._disabled = brand_disabled_tabs(current_brand())

    async def dispatch(self, request: Request, call_next):
        if not self._disabled:
            return await call_next(request)

        slug = request.url.path.strip("/").split("/", 1)[0]
        if slug not in self._disabled:
            return await call_next(request)

        detail = f"Unknown console tab: {slug!r}"
        accept = request.headers.get("accept", "")
        if "application/json" in accept and "text/html" not in accept:
            return JSONResponse({"detail": detail}, status_code=404)

        # Deferred import: app.py imports TabGateMiddleware from this module,
        # so importing NAV_ENTRIES back at module load time would cycle.
        from .app import NAV_ENTRIES

        nav_entries = [entry for entry in NAV_ENTRIES if entry[0] not in self._disabled]
        return _templates.TemplateResponse(
            request,
            "error.html",
            {"nav_entries": nav_entries, "status_code": 404, "message": detail},
            status_code=404,
        )
