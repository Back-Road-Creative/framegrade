"""The one Meta Graph request helper the Facebook, Instagram and photo
clients share.

Each client keeps its own ``raise_for_error`` hook: the Graph error body is
the same for every platform, but the exception it maps to is not
(``FacebookPermanentError`` vs ``InstagramPermanentError`` vs
``PhotoPermanentError``, each with its own recovery wording). This module
owns only what was identical across the three: the base URL, the JSON
decode, and the "status >= 400 OR an ``error`` object in a 200 body" gate.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from gomoveshift.publishing.meta_auth import DEFAULT_GRAPH_VERSION

#: ``(status_code, body)`` -> raises the platform's own error. Called only
#: when the response is a Graph error; a return from it is never expected.
GraphErrorHook = Callable[[int, dict], None]


def graph_base_url(graph_version: Optional[str] = None) -> str:
    """``https://graph.facebook.com/<version>``; the default version applies
    when a credential record omits one."""
    return f"https://graph.facebook.com/{graph_version or DEFAULT_GRAPH_VERSION}"


def _graph_json(response: Any, raise_for_error: GraphErrorHook) -> Dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        body = {}
    # Graph can answer 200 with an `error` object in the body (the exact
    # shape that hid the TikTok creator_info defect). Gating only on the
    # status code lets such a body through as success: `body["id"]` then
    # raises a bare KeyError, which the dispatcher classifies as RETRYABLE
    # and retries a permanently-rejected post forever.
    if response.status_code >= 400 or (body or {}).get("error"):
        raise_for_error(response.status_code, body)
    return body


def graph_post(session: Any, url: str, raise_for_error: GraphErrorHook, **kwargs) -> Dict[str, Any]:
    """POST through ``session`` and return the decoded body, or let
    ``raise_for_error`` raise the platform error."""
    return _graph_json(session.post(url, **kwargs), raise_for_error)


def graph_get(session: Any, url: str, raise_for_error: GraphErrorHook, **kwargs) -> Dict[str, Any]:
    """GET through ``session``; same contract as ``graph_post``."""
    return _graph_json(session.get(url, **kwargs), raise_for_error)
