"""Opt-in bearer-token auth for the console.

``GMS_CONSOLE_TOKEN`` unset/empty means the middleware is a no-op --
byte-for-byte the console's prior (unauthenticated) behavior. Set it and
every route but ``/healthz`` (the compose healthcheck) requires the token,
supplied via ``Authorization: Bearer <token>``, a ``?token=`` query param
(once -- it then sets a session cookie so a phone bookmark keeps working),
or the cookie itself.

The token compare is constant-time (``hmac.compare_digest``) so a wrong
guess can't be timed. A GET/HEAD authenticated via ``?token=`` is redirected
to the same URL with ``token`` stripped -- otherwise the secret sits in the
address bar, browser history, and the ``Referer`` header of every outbound
link forever. A POST/PUT/etc. carrying ``?token=`` can't be redirected
without either dropping the body or silently turning it into a GET, so it
authenticates and sets the cookie on the original response instead; the
token still lingers in that one request/log line, but never past it -- the
session cookie covers every later request. The session cookie is marked
``Secure`` only when the request itself arrived over HTTPS
(``request.url.scheme``) -- never from a forwarded-proto header, which is
attacker-controlled, and never unconditionally, which would silently break
legitimate local HTTP-on-loopback operation.
"""

from __future__ import annotations

import hmac
import os
from urllib.parse import urlencode

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse

#: Session cookie set after a successful ``?token=`` request.
COOKIE_NAME = "gms_console_token"

#: Paths reachable with no token even when auth is enabled.
_EXEMPT_PATHS = {"/healthz"}

#: Methods safe to redirect (no body, idempotent) when stripping ?token=.
_REDIRECTABLE_METHODS = {"GET", "HEAD"}


def console_token() -> str:
    """``GMS_CONSOLE_TOKEN`` or ``""`` (auth disabled)."""
    return os.environ.get("GMS_CONSOLE_TOKEN", "").strip()


def _supplied_token(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[len("bearer ") :].strip()
    query_token = request.query_params.get("token")
    if query_token:
        return query_token
    return request.cookies.get(COOKIE_NAME)


def _secret_matches(supplied: str, token: str) -> bool:
    """The one place that compares a caller-supplied value against
    ``GMS_CONSOLE_TOKEN``. Constant-time over UTF-8 bytes, not ``str`` --
    ``hmac.compare_digest(str, str)`` raises ``TypeError`` when either side
    has a non-ASCII character, and ``supplied`` is attacker-controlled
    (header/query/cookie), so that would turn a wrong non-ASCII guess into
    an unauthenticated 500 instead of a 401. Encoding first (UTF-8, matching
    how Starlette decodes the query string and cookie values) removes the
    ASCII restriction entirely and keeps the compare itself constant-time
    over the encoded bytes."""
    return hmac.compare_digest(supplied.encode("utf-8"), token.encode("utf-8"))


def is_authenticated(request: Request) -> bool:
    """True when auth is disabled, or the request carries the token via
    header/query/cookie. Used both by the middleware and by ``/healthz``
    (exempt from the middleware, but not from telling callers apart)."""
    token = console_token()
    if not token:
        return True
    supplied = _supplied_token(request)
    if supplied is None:
        return False
    return _secret_matches(supplied, token)


def _redirect_location_without_token(request: Request) -> str:
    """Path + remaining query only -- never an absolute URL built from the
    client-supplied ``Host`` header (which is what
    ``str(request.url.remove_query_params(...))`` would produce)."""
    remaining = [(k, v) for k, v in request.query_params.multi_items() if k != "token"]
    query = urlencode(remaining)
    return f"{request.url.path}?{query}" if query else request.url.path


def _set_session_cookie(response, token: str, request: Request) -> None:
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )


class ConsoleAuthMiddleware(BaseHTTPMiddleware):
    """Guards every route except ``/healthz`` behind ``GMS_CONSOLE_TOKEN``
    when it is set; a no-op when it is not (see module docstring)."""

    async def dispatch(self, request: Request, call_next):
        token = console_token()
        if not token or request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        if not is_authenticated(request):
            return JSONResponse({"detail": "unauthorized"}, status_code=401)

        query_token = request.query_params.get("token")
        authenticated_via_query = query_token is not None and _secret_matches(query_token, token)

        if authenticated_via_query and request.method in _REDIRECTABLE_METHODS:
            redirect = RedirectResponse(
                url=_redirect_location_without_token(request), status_code=303
            )
            _set_session_cookie(redirect, token, request)
            return redirect

        response = await call_next(request)
        if authenticated_via_query:
            _set_session_cookie(response, token, request)
        return response
