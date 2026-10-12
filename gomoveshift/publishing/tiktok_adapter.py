"""TikTok Direct Post adapter: creator_info, the public-routing gate, chunk
planning, upload-host validation, Direct Post init, the chunked upload
transfer, and the publish-status poll loop (there is no post-publish privacy
readback).

See ``docs/social-publishing-tiktok.md`` for the full contract and what is
deferred (delivery reconciliation against the publishing queue). The audit
gate below is required together with the upload path: uploading is never
enabled without it.

Never logs or raises with token material: a request-level failure (timeout,
DNS, undecodable body) raises a fixed, generic message naming only which
call failed. A TikTok API-level failure (``error.code`` other than
``"ok"``, whether carried on an HTTP error or a 200) raises a message
including TikTok's own ``error.code``, ``error.message``, and ``log_id`` —
support tickets need the ``log_id`` — with any accidental occurrence of the
access token scrubbed from that echoed text before it's raised. Raw
response headers and the access token itself are never included.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import requests

CREATOR_INFO_URL = "https://open.tiktokapis.com/v2/post/publish/creator_info/query/"
INIT_URL = "https://open.tiktokapis.com/v2/post/publish/video/init/"
# Inbox/draft upload: video lands in the account's TikTok inbox for a human
# to publish in-app — never auto-published, so no privacy_level applies.
# Candidate unaudited-client path (docs/social-publishing-tiktok.md).
INBOX_INIT_URL = "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/"
STATUS_URL = "https://open.tiktokapis.com/v2/post/publish/status/fetch/"
#: Read-only listing of the account's own posts (scope ``video.list``).
VIDEO_LIST_URL = (
    "https://open.tiktokapis.com/v2/video/list/"
    "?fields=id,create_time,video_description,title,duration"
)

# Direct Post upload hosts — the ONLY hosts video bytes may be sent to.
# Validated (scheme + host) before any upload, never trusted from a response
# body alone.
# EXACT hostnames only — never a suffix match. This list is what stops us
# sending video bytes to whatever host an (untrusted) API response names, and
# "endswith tiktokapis.com" would happily accept tiktokapis.com.evil.com.
# open-upload.tiktokapis.us is TikTok's US data-residency upload host (it
# serves a certificate for *.tiktokapis.us).
ALLOWED_UPLOAD_HOSTS = frozenset(
    {
        "upload.us.tiktokapis.com",
        "open-upload.tiktokapis.com",
        "open-upload.tiktokapis.us",
    }
)

MIN_CHUNK_BYTES = 5 * 1024 * 1024
MAX_CHUNK_BYTES = 64 * 1024 * 1024
MAX_FINAL_CHUNK_BYTES = 128 * 1024 * 1024
MAX_CHUNK_COUNT = 1000
MAX_VIDEO_BYTES = 4 * 1024 * 1024 * 1024


class TikTokAdapterError(RuntimeError):
    """Adapter-level failure. Message is always generic — never includes
    token/credential material."""


#: TikTok's code for "this account already has too many uploaded drafts
#: waiting to be published." TikTok refuses
#: further inbox inits while too many uploaded drafts are pending.
#: Nothing the pipeline does resolves it -- a human publishes or dismisses
#: the drafts in the app.
THROTTLE_ERROR_CODES = frozenset({"spam_risk_too_many_pending_share"})


class TikTokThrottledError(TikTokAdapterError):
    """TikTok is refusing further work until a human clears the backlog.

    Separate from the base class so ``drain_queue`` can exempt it from
    ``max_attempts``: the delivery's media and credentials are both fine,
    and retrying cannot change the answer. See ``THROTTLE_ERROR_CODES``."""


def _throttle_type(detail: str, default: type) -> type:
    """`TikTokThrottledError` when `detail` names a throttle code, else
    `default`. Substring match on the already-formatted detail string,
    which carries `code=...` for both the HTTP and the error-body paths --
    one place to key off rather than two."""
    return TikTokThrottledError if any(c in detail for c in THROTTLE_ERROR_CODES) else default


def _tiktok_error_detail(payload: Dict[str, Any]) -> Optional[str]:
    """Return a diagnostic string for `payload`'s `error` object if it
    signals a failure, else ``None``. TikTok answers many failures with
    HTTP 200 and ``error.code != "ok"`` in the body, so this is checked
    regardless of ``resp.raise_for_status()`` already passing."""
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code", "")
    if not code or code == "ok":
        return None
    return f"code={code!r} message={error.get('message', '')!r} log_id={error.get('log_id', '')!r}"


def _http_failure_detail(resp: "requests.Response") -> str:
    """``HTTP <status>`` plus TikTok's own error object when the body carries
    one, else a short slice of the raw body. Never raises: this runs on a
    failure path, and a second exception here would replace a diagnosable
    error with an undiagnosable one."""
    status = getattr(resp, "status_code", "?")
    try:
        error = (resp.json() or {}).get("error") or {}
        if error:
            return (
                f"HTTP {status} code={error.get('code', '')!r} "
                f"message={error.get('message', '')!r} log_id={error.get('log_id', '')!r}"
            )
        body = (resp.text or "").strip()
    except Exception:  # noqa: BLE001 — any decode failure falls back to the text
        body = ""
        try:
            body = (resp.text or "").strip()
        except Exception:  # noqa: BLE001
            body = ""
    return f"HTTP {status} {body[:200]}" if body else f"HTTP {status}"


def _tiktok_call(
    perform: Callable[[], "requests.Response"],
    call_desc: str,
    access_token: Optional[str],
    *,
    exc_type: type = TikTokAdapterError,
) -> Dict[str, Any]:
    """Run `perform()` (issue the request), raise `exc_type` on any
    request-level failure (timeout, DNS, HTTP >= 400, undecodable body) with
    a fixed generic message naming `call_desc`, or on a decoded body whose
    `error.code` isn't `"ok"`/absent — folding TikTok's `code`/`message`/
    `log_id` into the raised message so the failure is diagnosable without a
    separate raw probe. `access_token`, if given, is scrubbed from that
    echoed text before raising, so an API that echoed it back never leaks
    it. Returns the decoded JSON body (or `{}` for a bodyless success, e.g.
    a chunk-upload PUT) on success."""
    try:
        resp = perform()
    except requests.RequestException:
        # No response exists to quote: a timeout or a DNS failure has no
        # status and no body. The fixed message is all there is.
        raise exc_type(f"{call_desc} request failed") from None
    try:
        resp.raise_for_status()
    except requests.RequestException:
        # An HTTP failure DOES carry TikTok's diagnosis, and dropping it hides
        # the cause: the message must include the status, code and log_id,
        # not only "<call> request failed". Same scrub as the error-body path
        # below.
        detail = _http_failure_detail(resp)
        if access_token:
            detail = detail.replace(access_token, "<redacted>")
        raise _throttle_type(detail, exc_type)(f"{call_desc} failed: {detail}") from None
    try:
        # An empty body (e.g. a bodyless 2xx on a chunk-upload PUT) is a
        # normal success, not a decode failure — resp.json() would raise.
        payload = {} if not resp.content else resp.json()
    except ValueError:
        raise exc_type(f"{call_desc} request failed") from None
    if not isinstance(payload, dict):
        payload = {}
    detail = _tiktok_error_detail(payload)
    if detail is not None:
        if access_token:
            detail = detail.replace(access_token, "<redacted>")
        # TikTok answers some failures with HTTP 200 and error.code set, so
        # the throttle can arrive down this path too.
        raise _throttle_type(detail, exc_type)(f"{call_desc} failed: {detail}") from None
    return payload


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    video_size: int
    chunk_size: int
    total_chunk_count: int


def plan_chunks(video_size: int, chunk_size: Optional[int] = None) -> ChunkPlan:
    """Direct Post chunk arithmetic (TikTok's documented contract):
    ``total_chunk_count = floor(video_size / chunk_size)``, a trailing
    remainder folds into the final chunk when it still fits the 128 MB
    final-chunk ceiling, otherwise it becomes its own extra chunk. A whole
    file smaller than ``MAX_CHUNK_BYTES`` uploads as a single chunk (no 5 MB
    floor applies — there is nothing to split).

    A file LARGER than ``MAX_CHUNK_BYTES`` must upload in at least two chunks
    — TikTok rejects a single-chunk plan for one at init. The sizes are
    binary: the ceiling is 64 MiB, not decimal 64,000,000."""
    if video_size <= 0:
        raise TikTokAdapterError("video_size must be positive")
    if chunk_size is None:
        chunk_size = min(video_size, MAX_CHUNK_BYTES)
        # "Videos with a total size greater than 64 MB must be uploaded in
        # multiple chunks" -- and total_chunk_count is floor(video_size /
        # chunk_size), which is 1 for EVERYTHING between 64 MiB and 128 MiB.
        # So the line above would propose a single chunk for a file the API
        # requires to be split, and init rejects that with 400 invalid_params
        # "The chunk size is invalid". Halving keeps both chunks inside the
        # ceiling: the file is under 128 MiB here, so each half is under
        # 64 MiB.
        if video_size > MAX_CHUNK_BYTES and video_size // chunk_size < 2:
            chunk_size = video_size // 2
    if chunk_size > MAX_CHUNK_BYTES:
        raise TikTokAdapterError("chunk_size exceeds the 64 MB maximum")
    if chunk_size < MIN_CHUNK_BYTES and chunk_size != video_size:
        raise TikTokAdapterError("chunk_size is below the 5 MB minimum")

    total_chunk_count = max(video_size // chunk_size, 1)
    remainder = video_size - chunk_size * total_chunk_count
    if remainder:
        if (chunk_size + remainder) > MAX_FINAL_CHUNK_BYTES:
            total_chunk_count += 1
        # else: remainder folds into the final chunk's actual transfer size

    if video_size > MAX_CHUNK_BYTES and total_chunk_count < 2:
        # Reachable only with a caller-supplied chunk_size; the default above
        # cannot produce it. Refused here rather than at init, where the only
        # evidence is a 400 with a message that does not name the real rule.
        raise TikTokAdapterError(
            "a video larger than 64 MB must upload in multiple chunks, but "
            f"chunk_size={chunk_size} yields {total_chunk_count}"
        )
    if total_chunk_count > MAX_CHUNK_COUNT:
        raise TikTokAdapterError(f"chunk plan exceeds the {MAX_CHUNK_COUNT}-chunk ceiling")
    if video_size > MAX_VIDEO_BYTES:
        raise TikTokAdapterError("video exceeds the 4 GB TikTok upload limit")

    return ChunkPlan(
        video_size=video_size, chunk_size=chunk_size, total_chunk_count=total_chunk_count
    )


def validate_upload_host(url: str) -> None:
    """Refuse to send video bytes anywhere but an allowlisted TikTok upload
    host over https — the response that names the host is untrusted input."""
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise TikTokAdapterError("upload URL must use https")
    if parts.hostname not in ALLOWED_UPLOAD_HOSTS:
        raise TikTokAdapterError("upload URL host is not allowlisted")


def query_creator_info(access_token: str) -> Dict[str, Any]:
    """``POST .../creator_info/query/``, returning the response's ``data``
    object — NOT the ``{"data": ..., "error": ...}`` envelope around it.

    Returning the envelope would be a defect: callers index
    ``creator_info["privacy_level_options"]``, which lives inside ``data``,
    so they would always see ``None`` and every privacy level — ``SELF_ONLY``
    included — would be rejected by ``assert_public_routing_allowed`` as "not
    offered".

    The offered ``privacy_level_options`` are necessary but NOT sufficient
    for public routing; see ``assert_public_routing_allowed``."""
    payload = _tiktok_call(
        lambda: requests.post(
            CREATOR_INFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30,
        ),
        "creator_info",
        access_token,
    )
    return payload.get("data") or {}


def list_videos(
    access_token: str, *, cursor: Optional[int] = None, max_count: int = 20
) -> Tuple[List[Dict[str, Any]], Optional[int], bool]:
    """``POST .../video/list/`` (read-only): one page of the account's posts,
    newest first, as ``(videos, cursor, has_more)``. Each video carries
    ``id``, ``create_time`` (epoch seconds), ``video_description``, ``title``
    and ``duration``. Pass the returned ``cursor`` back for the next page; a
    first page sends no ``cursor`` key."""
    body: Dict[str, Any] = {"max_count": max_count}
    if cursor is not None:
        body["cursor"] = cursor
    payload = _tiktok_call(
        lambda: requests.post(
            VIDEO_LIST_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=30,
        ),
        "video list",
        access_token,
    )
    data = payload.get("data") or {}
    return list(data.get("videos") or []), data.get("cursor"), bool(data.get("has_more"))


def assert_public_routing_allowed(
    privacy_level: str, audit_approved: bool, creator_info: Dict[str, Any]
) -> None:
    """The ENFORCEMENT POINT for the public-routing rule — the deferred submit path
    MUST call this before ``POST /v2/post/publish/video/init/``, never just
    consult a boolean.

    An unaudited client is 403'd by ``.../video/init/`` with
    ``unaudited_client_can_only_post_to_private_accounts`` regardless of the
    requested ``privacy_level``, including a SELF_ONLY request. The
    constraint TikTok enforces is on the ACCOUNT's public/private state, not
    the post's audience, so an unaudited client is refused here for EVERY
    privacy_level, not just PUBLIC_TO_EVERYONE. Do not try to infer the
    account's public/private state from ``privacy_level_options`` — that list
    tracks it as an observed signal only, never a documented contract."""
    options = creator_info.get("privacy_level_options", [])
    if privacy_level not in options:
        raise TikTokAdapterError("requested privacy_level is not offered by creator_info")
    if not audit_approved:
        raise TikTokAdapterError(
            "unaudited_client_can_only_post_to_private_accounts: "
            "DriveConfig.publishing_tiktok_audit_approved must be set before any "
            "TikTok Direct Post, not just PUBLIC_TO_EVERYONE ones"
        )


def init_direct_post(
    access_token: str,
    *,
    title: str = "",
    privacy_level: str = "",
    audit_approved: bool = False,
    creator_info: Optional[Dict[str, Any]] = None,
    video_size: int,
    chunk_size: Optional[int] = None,
    brand_content_toggle: bool = False,
    brand_organic_toggle: bool = False,
    # False: GMS shorts are real captured driving footage, not AI-generated —
    # True renders a "Creator labeled as AI-generated" badge on the post.
    # Never flip this default without a change to the production process.
    is_aigc: bool = False,
    # "direct" (default, unchanged): Direct Post, gated by
    # assert_public_routing_allowed. "inbox": the unaudited-client
    # candidate — see INBOX_INIT_URL — skips the gate entirely.
    mode: str = "direct",
) -> Dict[str, Any]:
    """Initialize a Direct Post or inbox/draft FILE_UPLOAD session
    (``mode``). For ``mode="direct"`` calls ``assert_public_routing_allowed``
    FIRST — the enforcement point, not an advisory check — so an unaudited
    PUBLIC request never reaches ``POST .../video/init/``. Validates the
    returned ``upload_url`` against the allowlist before handing it back."""
    if mode not in ("direct", "inbox"):
        raise TikTokAdapterError(f"unknown init mode: {mode!r}")
    if mode == "direct":
        assert_public_routing_allowed(privacy_level, audit_approved, creator_info or {})
    chunk_plan = plan_chunks(video_size, chunk_size)
    source_info = {
        "source": "FILE_UPLOAD",
        "video_size": chunk_plan.video_size,
        "chunk_size": chunk_plan.chunk_size,
        "total_chunk_count": chunk_plan.total_chunk_count,
    }
    if mode == "inbox":
        url = INBOX_INIT_URL
        body: Dict[str, Any] = {"source_info": source_info}
        call_desc = "Inbox upload init"
    else:
        url = INIT_URL
        body = {
            "post_info": {
                "title": title,
                "privacy_level": privacy_level,
                "brand_content_toggle": brand_content_toggle,
                "brand_organic_toggle": brand_organic_toggle,
                "is_aigc": is_aigc,
            },
            "source_info": source_info,
        }
        call_desc = "Direct Post init"
    payload = _tiktok_call(
        lambda: requests.post(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            json=body,
            timeout=30,
        ),
        call_desc,
        access_token,
    )
    try:
        data = payload["data"]
        publish_id = data["publish_id"]
        upload_url = data["upload_url"]
    except KeyError:
        raise TikTokAdapterError(f"{call_desc} request failed") from None

    validate_upload_host(upload_url)
    return {"publish_id": publish_id, "upload_url": upload_url, "chunk_plan": chunk_plan}


def upload_video_chunks(
    upload_url: str, chunk_plan: ChunkPlan, chunk_reader: Callable[[int], bytes]
) -> None:
    """Send every chunk in ``chunk_plan`` to ``upload_url`` in order, never
    parallelized. ``chunk_reader(index)`` must return exactly that chunk's
    bytes (the last chunk may exceed ``chunk_size``, up to the 128 MB
    ceiling — the folded-in remainder from ``plan_chunks``). Re-validates
    the host regardless of what the caller already checked."""
    validate_upload_host(upload_url)
    video_size = chunk_plan.video_size
    chunk_size = chunk_plan.chunk_size
    total = chunk_plan.total_chunk_count
    for index in range(total):
        start = index * chunk_size
        end = video_size - 1 if index == total - 1 else start + chunk_size - 1
        length = end - start + 1
        data = chunk_reader(index)
        if len(data) != length:
            raise TikTokAdapterError("chunk reader returned the wrong byte count")
        _tiktok_call(
            lambda: requests.put(
                upload_url,
                headers={
                    "Content-Range": f"bytes {start}-{end}/{video_size}",
                    "Content-Length": str(length),
                    "Content-Type": "video/mp4",
                },
                data=data,
                timeout=120,
            ),
            "chunk upload",
            None,
        )


# --- Publish-status poll loop ------------------------------------------------

# Terminal states for POST .../status/fetch/. Every other observed value
# (e.g. PROCESSING_DOWNLOAD, PROCESSING_UPLOAD) is non-terminal and keeps
# polling.
PUBLISH_COMPLETE_STATUS = "PUBLISH_COMPLETE"
PUBLISH_FAILED_STATUS = "FAILED"
TERMINAL_PUBLISH_STATUSES = frozenset({PUBLISH_COMPLETE_STATUS, PUBLISH_FAILED_STATUS})

DEFAULT_POLL_MAX_ATTEMPTS = 10
_POLL_BACKOFF_CAP_SECONDS = 30.0


def _poll_backoff_seconds(attempt: int) -> float:
    """Exponential backoff between poll attempts, capped so a long-running
    poll never sleeps more than ``_POLL_BACKOFF_CAP_SECONDS`` between tries."""
    return min(2.0**attempt, _POLL_BACKOFF_CAP_SECONDS)


class TikTokPollTimeoutError(TikTokAdapterError):
    """A poll timeout is UNKNOWN, never FAILED — TikTok may finish publishing
    after our last poll. Preserves ``publish_id`` for reconciliation; NEVER
    resubmit a fresh Direct Post on this error, since resubmitting a
    possibly-already-published video creates a duplicate post."""

    def __init__(self, publish_id: str):
        self.publish_id = publish_id
        super().__init__(
            "publish-status poll exhausted its attempt budget without reaching a "
            f"terminal state for publish_id={publish_id}; the post may exist on "
            "TikTok — reconcile this publish_id before ever resubmitting"
        )


def poll_publish_status(
    access_token: str,
    publish_id: str,
    *,
    max_attempts: int = DEFAULT_POLL_MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> Dict[str, Any]:
    """Poll ``POST .../status/fetch/`` for ``publish_id`` until it reaches a
    terminal status (see ``TERMINAL_PUBLISH_STATUSES``), sleeping with capped
    exponential backoff between attempts. Returns the response's ``data``
    dict once terminal, else raises ``TikTokPollTimeoutError`` (never the
    token) — see that class's docstring for why a timeout must never trigger
    a resubmit."""
    data: Dict[str, Any] = {}
    for attempt in range(1, max_attempts + 1):
        payload = _tiktok_call(
            lambda: requests.post(
                STATUS_URL,
                headers={"Authorization": f"Bearer {access_token}"},
                json={"publish_id": publish_id},
                timeout=30,
            ),
            "publish-status poll",
            access_token,
        )
        data = payload.get("data", {})
        if data.get("status") in TERMINAL_PUBLISH_STATUSES:
            return data
        if attempt < max_attempts:
            sleep(_poll_backoff_seconds(attempt))
    raise TikTokPollTimeoutError(publish_id)


@dataclass(frozen=True, slots=True)
class PublishOutcome:
    publish_id: str
    delivery_status: str  # "published" or "failed_permanent"
    message: Optional[str] = None
    #: The audience we ASKED for. TikTok gives us no way to read back what
    #: was actually applied (see ``confirm_publish``), so this is an audit
    #: record of the request, never a confirmation of the result.
    requested_privacy_level: Optional[str] = None


def confirm_publish(
    access_token: str,
    publish_id: str,
    requested_privacy_level: str,
    *,
    max_attempts: int = DEFAULT_POLL_MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> PublishOutcome:
    """Poll to a terminal publish status and report the delivery outcome.

    THERE IS NO POST-PUBLISH PRIVACY READBACK, and there cannot be one.
    TikTok's API gives no readback: publishing a SELF_ONLY video
    returns ``{"status": "PUBLISH_COMPLETE"}`` — one key. TikTok does not
    report the applied ``privacy_level`` on ``status/fetch``, and without the
    ``video.list`` scope ``video/list``/``video/query`` both answer HTTP 401
    ``scope_not_authorized`` (an app holding only ``user.info.basic``,
    ``video.publish``). With ``video.list``/``video.upload`` granted,
    ``video/query`` returns HTTP 200 with real stats
    (``tracking.collector.TikTokCollector``). The readback gap itself still
    holds regardless of scopes.

    Comparing ``privacy_level`` anyway would be wrong: because the key is
    never present, every successful publish would be marked
    ``failed_permanent`` — the video live on TikTok, the queue claiming
    permanent failure, and a human "fixing" it posting a duplicate.

    A readback is a LATE GATE regardless: it runs after the post is already
    public, so it cannot prevent the harm it checks for, only report it.
    The audience guarantee therefore lives entirely UPSTREAM, in
    ``assert_public_routing_allowed`` before ``init_direct_post`` — the
    point where a wrong audience can still be refused. ``requested_privacy_level``
    is carried on the outcome as an audit record of what we asked for.
    """
    status_data = poll_publish_status(
        access_token, publish_id, max_attempts=max_attempts, sleep=sleep
    )
    if status_data.get("status") == PUBLISH_FAILED_STATUS:
        return PublishOutcome(
            publish_id=publish_id,
            delivery_status="failed_permanent",
            message="TikTok reported the publish attempt failed",
            requested_privacy_level=requested_privacy_level,
        )
    return PublishOutcome(
        publish_id=publish_id,
        delivery_status="published",
        message=None,
        requested_privacy_level=requested_privacy_level,
    )
