"""Registry wiring the four platform adapters into ``dispatch.drain_queue``'s
injected ``handlers`` mapping (see that module's docstring — it deliberately
has no import edge onto any adapter). ``build_handlers`` is the single seam
a caller (the future CLI dispatch subcommand) constructs once and hands to
``drain_queue``. Full contract: ``docs/social-publishing-dispatch.md``.

Every adapter is INJECTED, never constructed here — a test never needs live
credentials to exercise a handler's error-translation behaviour.

Translation is deliberately narrow: only an adapter's OWN named terminal
exception (or, for TikTok, its own terminal *return value*) becomes
``PermanentDeliveryError``. Everything else — including every adapter's own
retryable subclasses and any exception this module doesn't recognise —
propagates unchanged, so ``drain_queue``'s default retryable path handles
it. Never widen a ``except`` clause here to a base class "to be safe": that
is exactly the mistake that would silently discard a publishable short."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .dispatch import PermanentDeliveryError, ThrottledDeliveryError, _release_own_lease
from .facebook_adapter import (
    FacebookAuthBlockedError,
    FacebookPermanentError,
    FacebookShortsAdapter,
)
from .instagram_adapter import (
    InstagramAuthBlockedError,
    InstagramPermanentError,
    InstagramShortsAdapter,
)
from .models import FORMAT_CAROUSEL, FORMAT_PHOTO, MediaFingerprint, ShortPublishRecord
from .photo_adapter import PhotoAuthBlockedError, PhotoCarouselAdapter, PhotoPermanentError
from .queue import (
    AWAITING_MANUAL_PUBLISH_NEXT_ATTEMPT,
    STATUS_AWAITING_MANUAL_PUBLISH,
    acquire_lease,
    find_delivery,
    transition,
)
from .queue import NEXT_ATTEMPT_AWAITING_MANUAL as AWAITING_MANUAL
from .queue import PUBLISHED_AT_PLATFORM_CONFIRMED as PLATFORM
from .tiktok_adapter import (
    TikTokThrottledError,
    confirm_publish,
    init_direct_post,
    query_creator_info,
    upload_video_chunks,
)
from .tiktok_pack import pack_and_park
from .youtube_adapter import AmbiguousUploadMatchError, YouTubeShortsAdapter

RecordLoader = Callable[[str, int], ShortPublishRecord]
Handler = Callable[[Dict[str, Any]], None]

# Ceiling on TikTok's init-through-upload attempt's worker lease — mirrors
# the other three adapters' LEASE_SEC (a worker that dies mid-attempt frees
# this window for another worker, or this one on retry, to reclaim).
_TIKTOK_LEASE_SECONDS = 900

#: The app is unaudited (docs/social-publishing-tiktok.md), so Direct Post is
#: unavailable, and inbox uploads report SEND_TO_USER_INBOX yet never reach
#: the owner's inbox. The default is "pack" -- write the H.264 file and
#: its caption to the TikTok Studio pack folder and park the row for the owner
#: to schedule. "inbox" and "direct" stay available by ``GMS_TIKTOK_MODE``.
_DEFAULT_TIKTOK_MODE = "pack"
_TIKTOK_MODES = ("pack", "inbox", "direct")

#: Stop uploading inbox drafts once this many already sit unpublished.
#: TikTok enforces its own undocumented ceiling with
#: `spam_risk_too_many_pending_share`, which blocks the account entirely
#: once enough drafts are pending. Three leaves room for a normal day's
#: slots (12:00Z and 16:00Z) while staying well clear of that ceiling.
#: Inbox mode only: Direct Post creates no draft.
_MAX_PENDING_INBOX_DRAFTS = 3

# ``STATUS_AWAITING_MANUAL_PUBLISH`` and the 9999 park time live in
# ``queue.py`` (shared with ``tiktok_pack``); the names stay importable here.
_AWAITING_MANUAL_PUBLISH_NEXT_ATTEMPT = AWAITING_MANUAL_PUBLISH_NEXT_ATTEMPT


def make_youtube_handler(
    adapter: YouTubeShortsAdapter,
    conn,
    record_loader: RecordLoader,
    video_path_loader: Callable[[ShortPublishRecord], Path],
) -> Handler:
    """``AmbiguousUploadMatchError`` is terminal — two or more channel
    uploads match a retrying short by title/recency, and adopting either
    risks stealing the wrong video; a human must resolve it, retrying never
    will (see the adapter's own docstring). Everything else — including
    ``YouTubeQuotaError``, which the adapter already treats as retryable by
    transitioning its own row and re-raising — propagates unrecognised,
    handled by ``drain_queue``'s default retryable path."""

    def handler(row: Dict[str, Any]) -> None:
        record = record_loader(row["project_name"], row["short_index"])
        video_path = video_path_loader(record)
        try:
            adapter.publish(record, conn=conn, delivery_id=row["id"], video_path=video_path)
        except AmbiguousUploadMatchError as exc:
            raise PermanentDeliveryError(str(exc)) from exc

    return handler


def make_instagram_handler(
    adapter: InstagramShortsAdapter,
    conn,
    record_loader: RecordLoader,
    video_bytes_loader: Callable[[ShortPublishRecord], bytes],
) -> Handler:
    """``InstagramPermanentError`` (incl. ``UnsafeUploadHostError``, one of
    its subclasses) is terminal: the platform rejected the request outright
    or the response named an unsafe upload host.

    ``InstagramAuthBlockedError`` is ALSO terminal here, deliberately: an
    expired/revoked access token is recoverable by a human re-authorizing,
    but NOT by this queue's retry loop — the token is still dead on attempt
    8, so retrying only burns the attempt budget and delays a human
    noticing. Marking it permanent surfaces the problem immediately instead.

    ``InstagramRetryableError`` / ``InstagramPublishingLimitError`` (the
    account-level quota refusal — the window self-heals) are NOT
    translated, nor is any exception this module doesn't recognise: both
    propagate as themselves."""

    def handler(row: Dict[str, Any]) -> None:
        record = record_loader(row["project_name"], row["short_index"])
        video_bytes = video_bytes_loader(record)
        try:
            adapter.publish(record, conn=conn, delivery_id=row["id"], video_bytes=video_bytes)
        except (InstagramPermanentError, InstagramAuthBlockedError) as exc:
            raise PermanentDeliveryError(str(exc)) from exc

    return handler


def make_facebook_handler(
    adapter: FacebookShortsAdapter,
    conn,
    record_loader: RecordLoader,
    video_bytes_loader: Callable[[ShortPublishRecord], bytes],
    *,
    video_state: str = "PUBLISHED",
) -> Handler:
    """Mirrors ``make_instagram_handler``'s classification and reasoning:
    ``FacebookPermanentError`` (incl. ``UnsafeUploadHostError`` and
    ``ScheduleWindowError``, both subclasses) and ``FacebookAuthBlockedError``
    are terminal for the same reasons as their Instagram counterparts.
    ``FacebookRetryableError`` and any unrecognised exception propagate."""

    def handler(row: Dict[str, Any]) -> None:
        record = record_loader(row["project_name"], row["short_index"])
        video_bytes = video_bytes_loader(record)
        try:
            adapter.publish(
                record,
                conn=conn,
                delivery_id=row["id"],
                video_bytes=video_bytes,
                video_state=video_state,
            )
        except (FacebookPermanentError, FacebookAuthBlockedError) as exc:
            raise PermanentDeliveryError(str(exc)) from exc

    return handler


def make_photo_handler(
    adapter: PhotoCarouselAdapter,
    conn,
    record_loader: RecordLoader,
    media_image_url_loader: Callable[[MediaFingerprint], str],
) -> Handler:
    """A ``format=photo``/``format=carousel`` record's handler for one
    platform. Same ``handler(row) -> None`` contract as the video handlers.

    ``media_image_url_loader`` resolves a ``MediaFingerprint`` to the
    publicly-reachable ``image_url`` it publishes under -- NOT raw bytes
    (see ``photo_graph_client.py``'s module docstring).

    ``PhotoPermanentError`` (incl. ``UnsupportedPhotoFormatError`` and
    ``PartialCarouselError``, both subclasses -- a partial carousel is
    ALWAYS terminal, never auto-retried, see ``photo_adapter``'s module
    docstring) and ``PhotoAuthBlockedError`` are terminal, mirroring the
    video handlers' AuthBlocked reasoning. ``PhotoRetryableError`` and any
    unrecognised exception propagate, handled by ``drain_queue``'s default
    retryable path."""

    def handler(row: Dict[str, Any]) -> None:
        record = record_loader(row["project_name"], row["short_index"])
        try:
            adapter.publish(
                record,
                conn=conn,
                delivery_id=row["id"],
                media_image_url_loader=media_image_url_loader,
            )
        except (PhotoPermanentError, PhotoAuthBlockedError) as exc:
            raise PermanentDeliveryError(str(exc)) from exc

    return handler


def _dispatch_by_format(
    record_loader: RecordLoader,
    video_handler: Optional[Handler],
    photo_handler: Optional[Handler],
) -> Handler:
    """Wraps a platform's video and photo handlers behind ONE handler keyed
    by the delivery row's ``platform`` column, routing on the loaded
    record's ``format`` -- ``queue.py``'s ``platform`` column is
    "instagram"/"facebook" regardless of format (``models.RouteSelection``
    only ever names one of ``KNOWN_PLATFORMS``, never a format-qualified
    variant), so ``drain_queue``'s ``handlers`` mapping has room for exactly
    one handler per platform. Neither ``make_instagram_handler`` nor
    ``make_facebook_handler`` is touched by this -- when only one of the two
    handlers is wired, ``build_handlers`` registers it directly (see below),
    so existing (video-only) callers see no behaviour change at all."""

    def handler(row: Dict[str, Any]) -> None:
        record = record_loader(row["project_name"], row["short_index"])
        if record.format in (FORMAT_PHOTO, FORMAT_CAROUSEL):
            if photo_handler is None:
                raise PermanentDeliveryError(
                    f"short_index={row['short_index']} is format={record.format!r} but "
                    f"platform={row['platform']!r} has no photo handler wired"
                )
            photo_handler(row)
        else:
            if video_handler is None:
                raise PermanentDeliveryError(
                    f"short_index={row['short_index']} is format={record.format!r} but "
                    f"platform={row['platform']!r} has no video handler wired"
                )
            video_handler(row)

    return handler


@dataclass(frozen=True, slots=True)
class TikTokCredentials:
    access_token: str
    audit_approved: bool = False


def _lease_expiry(seconds: int = _TIKTOK_LEASE_SECONDS) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def _chunk_reader(video_bytes: bytes, chunk_plan) -> Callable[[int], bytes]:
    def reader(index: int) -> bytes:
        start = index * chunk_plan.chunk_size
        end = (
            chunk_plan.video_size
            if index == chunk_plan.total_chunk_count - 1
            else start + chunk_plan.chunk_size
        )
        return video_bytes[start:end]

    return reader


def _init_translating_throttle(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """``init_direct_post``, with TikTok's own throttle refusal translated to
    :class:`ThrottledDeliveryError`.

    Narrow by design, like every other translation here: only the adapter's
    named throttle subclass is caught. A plain ``TikTokAdapterError`` stays
    retryable, because it is usually something retrying CAN fix."""
    try:
        return init_direct_post(*args, **kwargs)
    except TikTokThrottledError as exc:
        raise ThrottledDeliveryError(str(exc)) from exc


def resolve_tiktok_mode(mode: Optional[str] = None) -> str:
    """``mode``, else ``$GMS_TIKTOK_MODE``, else ``_DEFAULT_TIKTOK_MODE``.
    Raises ``ValueError`` for anything but pack/inbox/direct."""
    resolved = mode if mode is not None else os.environ.get("GMS_TIKTOK_MODE", "").strip()
    resolved = resolved or _DEFAULT_TIKTOK_MODE
    if resolved not in _TIKTOK_MODES:
        raise ValueError(f"unknown GMS_TIKTOK_MODE: {resolved!r}")
    return resolved


def make_tiktok_handler(
    credentials: TikTokCredentials,
    conn,
    record_loader: RecordLoader,
    video_bytes_loader: Callable[[ShortPublishRecord], bytes],
    *,
    worker_id: str = "adapter",
    chunk_size: Optional[int] = None,
    mode: Optional[str] = None,
    rendition_path_loader: Optional[Callable[[ShortPublishRecord], Path]] = None,
) -> Handler:
    """``tiktok_adapter.py`` is a free-function toolkit, not a
    ``*ShortsAdapter`` class (see its module docstring) — this orchestrates
    creator_info -> init -> chunked upload -> publish-status confirm, wired
    to ``queue.py`` the same "persist before the next irreversible call"
    pattern the other three adapters use.

    ``mode`` is ``"pack"``, ``"direct"`` or ``"inbox"``; it defaults
    (``None``) to ``$GMS_TIKTOK_MODE`` or ``_DEFAULT_TIKTOK_MODE`` ("pack";
    see ``docs/social-publishing-tiktok.md``). ``"pack"`` makes NO
    TikTok API call: ``rendition_path_loader`` (the verified H.264 file's
    path) feeds ``tiktok_pack.pack_and_park``, which copies the file and its
    caption into the TikTok Studio pack folder and parks the row
    ``awaiting_manual_publish`` -- no inbox cap applies. ``direct``/``inbox``
    select ``init_direct_post``'s ``mode``. In ``"inbox"`` mode there is no
    public-routing gate to satisfy and no publish-status to confirm: TikTok
    never auto-publishes an inbox upload, so the delivery lands
    ``STATUS_AWAITING_MANUAL_PUBLISH`` once the chunked upload completes and
    a human publishes it from the TikTok app -- ``confirm_publish`` is never
    called in this mode. ``"direct"`` mode is unchanged.

    ``TikTokPollTimeoutError`` is NEVER translated: the adapter's own
    docstring says a poll timeout is an unknown outcome, never a failure,
    and resubmitting risks a duplicate post. It propagates unrecognised, so
    ``drain_queue`` retries — and because ``publish_id`` is persisted
    BEFORE the poll ever starts, a retry re-enters at the confirm step,
    never re-running ``init_direct_post``.

    ``confirm_publish``'s ``delivery_status == "failed_permanent"`` (a
    TikTok-reported failure, or a post-publish privacy-readback mismatch —
    see that function's docstring) IS translated to
    ``PermanentDeliveryError``: that's TikTok's own terminal signal, just
    returned rather than raised. Any other exception (from ``creator_info``,
    ``init``, or the chunked upload) is not recognised here and propagates,
    retryable — TikTok exposes only ``TikTokAdapterError`` for both
    transient network failures and permanent misconfiguration, with no
    dedicated permanent subclass to key off; per the conservative default,
    it stays retryable rather than being guessed at."""
    resolved_mode = resolve_tiktok_mode(mode)

    def handler(row: Dict[str, Any]) -> None:
        record = record_loader(row["project_name"], row["short_index"])
        payload = record.payloads.tiktok
        if payload is None:
            raise ValueError(f"short_index={record.short_index} carries no TikTok payload")

        delivery_id = row["id"]
        existing = find_delivery(conn, record.project_name, record.short_index, "tiktok")
        current_status = existing["status"] if existing else "queued"
        publish_id = existing.get("platform_upload_id") if existing else None

        if current_status == STATUS_AWAITING_MANUAL_PUBLISH:
            # Nothing to do until a human publishes the post (console's
            # mark-published action) -- never reprocessed automatically. The
            # row was parked after the drain selected it, so give back the
            # lease the drain took (only if this worker still holds it).
            _release_own_lease(conn, delivery_id, worker_id)
            return

        if resolved_mode == "pack":
            if rendition_path_loader is None:
                raise ValueError(
                    "TikTok pack mode needs a rendition_path_loader (the verified H.264 "
                    "file's path); none was wired"
                )
            if not acquire_lease(conn, delivery_id, worker_id, _lease_expiry()):
                raise RuntimeError(
                    f"delivery {delivery_id} is leased by another worker — refusing to "
                    "pack it twice"
                )
            pack_and_park(conn, delivery_id, record, rendition_path_loader(record))
            return

        if not publish_id:
            if resolved_mode == "inbox":
                # Upstream gate: refuse BEFORE the upload, the irreversible
                # act. A burst of uploads in a short window,
                # on top of drafts already parked, can trip TikTok's own spam
                # guard and block every further upload on the account. Counting
                # our own parked rows sees that coming; the adapter's
                # TikTokThrottledError below is the backstop for when this
                # count is wrong (a draft dismissed in the app, say).
                pending = conn.execute(
                    "SELECT COUNT(*) FROM deliveries WHERE platform='tiktok' AND status=?",
                    (STATUS_AWAITING_MANUAL_PUBLISH,),
                ).fetchone()[0]
                if pending >= _MAX_PENDING_INBOX_DRAFTS:
                    raise ThrottledDeliveryError(
                        f"{pending} TikTok drafts already await a manual publish "
                        f"(cap {_MAX_PENDING_INBOX_DRAFTS}) — publish or dismiss them "
                        "in the app inbox; uploading more risks TikTok's spam guard"
                    )
            if not acquire_lease(conn, delivery_id, worker_id, _lease_expiry()):
                raise RuntimeError(
                    f"delivery {delivery_id} is leased by another worker — refusing to "
                    "risk a duplicate Direct Post init"
                )
            transition(conn, delivery_id, current_status, "submitting")
            creator_info = (
                query_creator_info(credentials.access_token) if resolved_mode == "direct" else None
            )
            video_bytes = video_bytes_loader(record)
            init = _init_translating_throttle(
                credentials.access_token,
                # TikTok has ONE text field: post_info.title IS the
                # caption (2200 UTF-16 runes, hashtags included). `title`
                # is the short console label and is only the fallback, for
                # sidecars written before the caption field existed.
                title=payload.caption or payload.title,
                privacy_level=payload.privacy_level,
                audit_approved=credentials.audit_approved,
                creator_info=creator_info,
                video_size=len(video_bytes),
                chunk_size=chunk_size,
                brand_content_toggle=payload.disclose_branded_content,
                brand_organic_toggle=payload.disclose_your_brand,
                mode=resolved_mode,
            )
            publish_id = init["publish_id"]
            # Persist BEFORE the chunked upload — mirrors every other
            # adapter's "persist before the next irreversible call" pattern
            # so a lost response after this point is recoverable, not
            # orphaned/duplicated. The status stays "submitting": the id is
            # durable, the bytes are not up yet, and saying "uploaded" here
            # would make those two indistinguishable to a resuming worker.
            transition(conn, delivery_id, "submitting", "submitting", platform_upload_id=publish_id)
            chunk_plan = init["chunk_plan"]
            upload_video_chunks(
                init["upload_url"], chunk_plan, _chunk_reader(video_bytes, chunk_plan)
            )
            # Only now is the video actually on TikTok.
            transition(conn, delivery_id, "submitting", "uploaded")
            current_status = "uploaded"
        elif existing["status"] == "submitting":
            # A publish id with the upload still unfinished: the worker died
            # mid-chunk. Neither way forward is safe to take silently --
            # re-running init would create a SECOND TikTok post, and skipping
            # to the next step would treat an incomplete upload as a finished
            # one. In inbox mode, parking the row as a draft "awaiting manual
            # publish" would leave a draft that is not in anyone's app inbox:
            # nothing reprocesses that status automatically, and it would hold
            # one of the pending slots for good. In direct mode, polling would
            # wait on a publish that cannot complete until the attempt budget
            # runs out. So the row is failed where a person can see it, to be
            # re-enqueued.
            raise PermanentDeliveryError(
                f"TikTok upload for publish_id {publish_id} never finished — the worker "
                "stopped mid-upload. Re-enqueue this short; resuming would either "
                "duplicate the post or confirm a video that was never fully uploaded."
            )
        else:
            current_status = existing["status"]

        if resolved_mode == "inbox":
            transition(
                conn,
                delivery_id,
                current_status,
                STATUS_AWAITING_MANUAL_PUBLISH,
                platform_upload_id=publish_id,
                next_attempt_at=_AWAITING_MANUAL_PUBLISH_NEXT_ATTEMPT,
                next_attempt_reason=AWAITING_MANUAL,
            )
            return

        outcome = confirm_publish(credentials.access_token, publish_id, payload.privacy_level)
        if outcome.delivery_status == "failed_permanent":
            raise PermanentDeliveryError(
                outcome.message or "TikTok reported the publish attempt failed"
            )
        transition(
            conn,
            delivery_id,
            current_status,
            "published",
            platform_post_id=publish_id,
            published_at_source=PLATFORM,
        )

    return handler


def build_handlers(
    conn,
    record_loader: RecordLoader,
    *,
    worker_id: str = "adapter",
    youtube: Optional[Tuple[YouTubeShortsAdapter, Callable[[ShortPublishRecord], Path]]] = None,
    instagram: Optional[
        Tuple[InstagramShortsAdapter, Callable[[ShortPublishRecord], bytes]]
    ] = None,
    facebook: Optional[Tuple[FacebookShortsAdapter, Callable[[ShortPublishRecord], bytes]]] = None,
    tiktok: Optional[Tuple[TikTokCredentials, Callable[[ShortPublishRecord], bytes]]] = None,
    tiktok_rendition_path_loader: Optional[Callable[[ShortPublishRecord], Path]] = None,
    instagram_photo: Optional[
        Tuple[PhotoCarouselAdapter, Callable[[MediaFingerprint], str]]
    ] = None,
    facebook_photo: Optional[Tuple[PhotoCarouselAdapter, Callable[[MediaFingerprint], str]]] = None,
) -> Dict[str, Handler]:
    """Build the ``platform -> handler`` mapping ``drain_queue`` needs, from
    whichever adapters/credentials the caller has already constructed.
    Passing ``None`` (the default) for a platform leaves it unrouted —
    ``drain_queue`` skips it (``DrainResult.skipped_no_handler``), never
    acquiring its lease, until that platform is wired here.

    ``instagram_photo``/``facebook_photo`` wire a ``PhotoCarouselAdapter``
    for that platform's photo/carousel-format records. When BOTH a video and
    a photo tuple are given for a platform, the registered handler routes on
    the loaded record's ``format`` (see ``_dispatch_by_format``); when only
    one is given, that handler is registered directly, unchanged from
    before this parameter existed."""
    handlers: Dict[str, Handler] = {}
    if youtube is not None:
        adapter, video_path_loader = youtube
        handlers["youtube"] = make_youtube_handler(adapter, conn, record_loader, video_path_loader)
    if instagram is not None or instagram_photo is not None:
        video_handler = None
        if instagram is not None:
            adapter, video_bytes_loader = instagram
            video_handler = make_instagram_handler(adapter, conn, record_loader, video_bytes_loader)
        photo_handler = None
        if instagram_photo is not None:
            photo_adapter, photo_image_url_loader = instagram_photo
            photo_handler = make_photo_handler(
                photo_adapter, conn, record_loader, photo_image_url_loader
            )
        handlers["instagram"] = (
            _dispatch_by_format(record_loader, video_handler, photo_handler)
            if video_handler is not None and photo_handler is not None
            else (photo_handler or video_handler)
        )
    if facebook is not None or facebook_photo is not None:
        video_handler = None
        if facebook is not None:
            adapter, video_bytes_loader = facebook
            video_handler = make_facebook_handler(adapter, conn, record_loader, video_bytes_loader)
        photo_handler = None
        if facebook_photo is not None:
            photo_adapter, photo_image_url_loader = facebook_photo
            photo_handler = make_photo_handler(
                photo_adapter, conn, record_loader, photo_image_url_loader
            )
        handlers["facebook"] = (
            _dispatch_by_format(record_loader, video_handler, photo_handler)
            if video_handler is not None and photo_handler is not None
            else (photo_handler or video_handler)
        )
    if tiktok is not None:
        credentials, video_bytes_loader = tiktok
        handlers["tiktok"] = make_tiktok_handler(
            credentials,
            conn,
            record_loader,
            video_bytes_loader,
            worker_id=worker_id,
            rendition_path_loader=tiktok_rendition_path_loader,
        )
    return handlers
