"""Tests for ``gomoveshift.publishing.handlers``: the registry that wires the
four platform adapters into ``dispatch.drain_queue``'s injected ``handlers``
mapping, and translates each adapter's error hierarchy into
``dispatch.PermanentDeliveryError`` for genuinely terminal cases only.

The "unrecognised exception" test on each platform is the one that matters
most: an unrecognised exception must propagate AS ITSELF so
``drain_queue``'s default retryable path handles it — never silently
converted to permanent. See ``docs/social-publishing-dispatch.md``."""

from __future__ import annotations

from pathlib import Path

import dataclasses

import pytest
from test_publishing_models import _record

from gomoveshift.publishing.dispatch import PermanentDeliveryError, ThrottledDeliveryError
from gomoveshift.publishing.facebook_adapter import (
    FacebookAuthBlockedError,
    FacebookPermanentError,
    FacebookRetryableError,
    UnsafeUploadHostError as FacebookUnsafeUploadHostError,
)
from gomoveshift.publishing.handlers import (
    TikTokCredentials,
    build_handlers,
    make_facebook_handler,
    make_instagram_handler,
    make_tiktok_handler,
    make_youtube_handler,
)
from gomoveshift.publishing.instagram_adapter import (
    InstagramAuthBlockedError,
    InstagramPermanentError,
    InstagramPublishingLimitError,
    InstagramRetryableError,
    UnsafeUploadHostError as InstagramUnsafeUploadHostError,
)
from gomoveshift.pipeline.exceptions import YouTubeQuotaError
from gomoveshift.publishing.queue import (
    acquire_lease,
    enqueue,
    find_delivery,
    open_queue_db,
    transition,
)
from gomoveshift.publishing.tiktok_adapter import (
    ChunkPlan,
    PublishOutcome,
    TikTokPollTimeoutError,
    TikTokThrottledError,
)
from gomoveshift.publishing.youtube_adapter import AmbiguousUploadMatchError

FAKE_TOKEN = "act.faketoken12345"  # noqa: S105 — test fixture, not a real credential


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def _row(conn, platform: str, project_name: str = "proj", short_index: int = 0) -> dict:
    delivery_id = enqueue(conn, project_name, short_index, platform, "hash-a")
    return dict(conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone())


class _FakeAdapter:
    """Stand-in for a `*ShortsAdapter`: `publish(record, *, conn, delivery_id,
    **kwargs)` raises whatever the test configures, so translation is
    exercised without any real network/credential plumbing."""

    def __init__(self, to_raise: Exception | None = None):
        self._to_raise = to_raise
        self.calls = []

    def publish(self, record, *, conn, delivery_id, **kwargs):
        self.calls.append((record, delivery_id, kwargs))
        if self._to_raise is not None:
            raise self._to_raise


def _loader(*, path: bool = False):
    if path:
        return lambda record: Path("/tmp/short.mp4")
    return lambda record: b"video-bytes"


# --- YouTube -----------------------------------------------------------


def test_youtube_ambiguous_upload_match_is_permanent(conn):
    adapter = _FakeAdapter(to_raise=AmbiguousUploadMatchError("2 candidates match"))
    handler = make_youtube_handler(adapter, conn, lambda p, i: _record(), _loader(path=True))
    with pytest.raises(PermanentDeliveryError):
        handler(_row(conn, "youtube"))


def test_youtube_quota_error_stays_retryable_not_permanent(conn):
    adapter = _FakeAdapter(to_raise=YouTubeQuotaError("daily quota exhausted"))
    handler = make_youtube_handler(adapter, conn, lambda p, i: _record(), _loader(path=True))
    with pytest.raises(YouTubeQuotaError):
        handler(_row(conn, "youtube"))


def test_youtube_unrecognised_exception_stays_retryable(conn):
    adapter = _FakeAdapter(to_raise=RuntimeError("transient network blip"))
    handler = make_youtube_handler(adapter, conn, lambda p, i: _record(), _loader(path=True))
    with pytest.raises(RuntimeError):
        handler(_row(conn, "youtube"))


def test_youtube_success_calls_adapter_with_video_path(conn):
    adapter = _FakeAdapter()
    handler = make_youtube_handler(adapter, conn, lambda p, i: _record(), _loader(path=True))
    handler(_row(conn, "youtube"))
    assert len(adapter.calls) == 1
    assert adapter.calls[0][2]["video_path"] == Path("/tmp/short.mp4")


# --- Instagram -----------------------------------------------------------


def test_instagram_permanent_error_is_permanent(conn):
    adapter = _FakeAdapter(to_raise=InstagramPermanentError("content policy rejection"))
    handler = make_instagram_handler(adapter, conn, lambda p, i: _record(), _loader())
    with pytest.raises(PermanentDeliveryError):
        handler(_row(conn, "instagram"))


def test_instagram_unsafe_upload_host_is_permanent(conn):
    adapter = _FakeAdapter(to_raise=InstagramUnsafeUploadHostError("not allowlisted"))
    handler = make_instagram_handler(adapter, conn, lambda p, i: _record(), _loader())
    with pytest.raises(PermanentDeliveryError):
        handler(_row(conn, "instagram"))


def test_instagram_auth_blocked_is_permanent_not_retried_forever(conn):
    # Deliberate: an expired/revoked token can't be fixed by retrying — only
    # a human re-authorizing fixes it, so retrying just burns the attempt
    # budget before the human notices.
    adapter = _FakeAdapter(to_raise=InstagramAuthBlockedError("token expired"))
    handler = make_instagram_handler(adapter, conn, lambda p, i: _record(), _loader())
    with pytest.raises(PermanentDeliveryError):
        handler(_row(conn, "instagram"))


def test_instagram_retryable_error_stays_retryable(conn):
    adapter = _FakeAdapter(to_raise=InstagramRetryableError("transient Graph API error"))
    handler = make_instagram_handler(adapter, conn, lambda p, i: _record(), _loader())
    with pytest.raises(InstagramRetryableError):
        handler(_row(conn, "instagram"))


def test_instagram_publishing_limit_error_stays_retryable(conn):
    adapter = _FakeAdapter(to_raise=InstagramPublishingLimitError("100/100 used"))
    handler = make_instagram_handler(adapter, conn, lambda p, i: _record(), _loader())
    with pytest.raises(InstagramPublishingLimitError):
        handler(_row(conn, "instagram"))


def test_instagram_unrecognised_exception_stays_retryable(conn):
    adapter = _FakeAdapter(to_raise=ConnectionError("reset by peer"))
    handler = make_instagram_handler(adapter, conn, lambda p, i: _record(), _loader())
    with pytest.raises(ConnectionError):
        handler(_row(conn, "instagram"))


# --- Facebook (mirrors instagram's classification) -----------------------


def test_facebook_permanent_error_is_permanent(conn):
    adapter = _FakeAdapter(to_raise=FacebookPermanentError("content policy rejection"))
    handler = make_facebook_handler(adapter, conn, lambda p, i: _record(), _loader())
    with pytest.raises(PermanentDeliveryError):
        handler(_row(conn, "facebook"))


def test_facebook_unsafe_upload_host_is_permanent(conn):
    adapter = _FakeAdapter(to_raise=FacebookUnsafeUploadHostError("not allowlisted"))
    handler = make_facebook_handler(adapter, conn, lambda p, i: _record(), _loader())
    with pytest.raises(PermanentDeliveryError):
        handler(_row(conn, "facebook"))


def test_facebook_auth_blocked_is_permanent(conn):
    adapter = _FakeAdapter(to_raise=FacebookAuthBlockedError("token expired"))
    handler = make_facebook_handler(adapter, conn, lambda p, i: _record(), _loader())
    with pytest.raises(PermanentDeliveryError):
        handler(_row(conn, "facebook"))


def test_facebook_retryable_error_stays_retryable(conn):
    adapter = _FakeAdapter(to_raise=FacebookRetryableError("transient Graph API error"))
    handler = make_facebook_handler(adapter, conn, lambda p, i: _record(), _loader())
    with pytest.raises(FacebookRetryableError):
        handler(_row(conn, "facebook"))


def test_facebook_unrecognised_exception_stays_retryable(conn):
    adapter = _FakeAdapter(to_raise=TimeoutError("upstream timeout"))
    handler = make_facebook_handler(adapter, conn, lambda p, i: _record(), _loader())
    with pytest.raises(TimeoutError):
        handler(_row(conn, "facebook"))


# --- TikTok (no `*ShortsAdapter` class — orchestrated in handlers.py) ----


@pytest.fixture
def tiktok_stubs(monkeypatch):
    """Stub the four free functions `make_tiktok_handler` orchestrates,
    patched onto the `handlers` module (where they're imported by name)."""
    import gomoveshift.publishing.handlers as handlers_mod

    creator_info = {"privacy_level_options": ["PUBLIC_TO_EVERYONE"]}
    chunk_plan = ChunkPlan(video_size=11, chunk_size=11, total_chunk_count=1)
    monkeypatch.setattr(handlers_mod, "query_creator_info", lambda token: creator_info)
    monkeypatch.setattr(
        handlers_mod,
        "init_direct_post",
        lambda *a, **k: {
            "publish_id": "pub-1",
            "upload_url": "https://upload.us.tiktokapis.com/x",
            "chunk_plan": chunk_plan,
        },
    )
    monkeypatch.setattr(handlers_mod, "upload_video_chunks", lambda *a, **k: None)
    return handlers_mod


def test_tiktok_confirm_publish_failed_permanent_is_permanent(conn, tiktok_stubs, monkeypatch):
    monkeypatch.setattr(
        tiktok_stubs,
        "confirm_publish",
        lambda *a, **k: PublishOutcome(
            publish_id="pub-1", delivery_status="failed_permanent", message="rejected"
        ),
    )

    creds = TikTokCredentials(access_token=FAKE_TOKEN, audit_approved=True)
    handler = make_tiktok_handler(creds, conn, lambda p, i: _record(), _loader(), mode="direct")
    with pytest.raises(PermanentDeliveryError):
        handler(_row(conn, "tiktok"))


def test_tiktok_poll_timeout_stays_retryable(conn, tiktok_stubs, monkeypatch):
    def raise_timeout(*a, **k):
        raise TikTokPollTimeoutError("pub-1")

    monkeypatch.setattr(tiktok_stubs, "confirm_publish", raise_timeout)

    creds = TikTokCredentials(access_token=FAKE_TOKEN, audit_approved=True)
    handler = make_tiktok_handler(creds, conn, lambda p, i: _record(), _loader(), mode="direct")
    with pytest.raises(TikTokPollTimeoutError):
        handler(_row(conn, "tiktok"))


def test_tiktok_unrecognised_exception_stays_retryable(conn, tiktok_stubs, monkeypatch):
    def raise_runtime(*a, **k):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(tiktok_stubs, "confirm_publish", raise_runtime)

    creds = TikTokCredentials(access_token=FAKE_TOKEN, audit_approved=True)
    handler = make_tiktok_handler(creds, conn, lambda p, i: _record(), _loader(), mode="direct")
    with pytest.raises(RuntimeError):
        handler(_row(conn, "tiktok"))


def test_tiktok_success_transitions_delivery_to_published(conn, tiktok_stubs, monkeypatch):
    monkeypatch.setattr(
        tiktok_stubs,
        "confirm_publish",
        lambda *a, **k: PublishOutcome(
            publish_id="pub-1", delivery_status="published", message=None
        ),
    )

    creds = TikTokCredentials(access_token=FAKE_TOKEN, audit_approved=True)
    handler = make_tiktok_handler(creds, conn, lambda p, i: _record(), _loader(), mode="direct")
    row = _row(conn, "tiktok")
    handler(row)
    status = conn.execute(
        "SELECT status, platform_post_id FROM deliveries WHERE id=?", (row["id"],)
    ).fetchone()
    assert status["status"] == "published"
    assert status["platform_post_id"] == "pub-1"


# --- TikTok inbox mode -----------------------------------------------


def test_tiktok_env_inbox_keeps_the_inbox_behaviour(conn, tiktok_stubs, monkeypatch):
    # The default moved to "pack"; inbox stays available when asked for.
    monkeypatch.setenv("GMS_TIKTOK_MODE", "inbox")

    def fail_if_called(*a, **k):
        raise AssertionError("confirm_publish must never run in inbox mode")

    monkeypatch.setattr(tiktok_stubs, "confirm_publish", fail_if_called)

    creds = TikTokCredentials(access_token=FAKE_TOKEN, audit_approved=False)
    handler = make_tiktok_handler(creds, conn, lambda p, i: _record(), _loader())
    row = _row(conn, "tiktok", project_name="example-2026-08", short_index=3)
    handler(row)  # must not raise -- no confirm_publish, no routing gate

    status = conn.execute(
        "SELECT status, platform_upload_id, next_attempt_at, next_attempt_reason "
        "FROM deliveries WHERE id=?",
        (row["id"],),
    ).fetchone()
    assert status["status"] == "awaiting_manual_publish"
    assert status["platform_upload_id"] == "pub-1"
    # Parked far in the future so dispatch._select_due never reselects it.
    assert status["next_attempt_at"] > "2100-01-01"
    assert status["next_attempt_reason"] == "awaiting_manual"


def test_tiktok_inbox_mode_skips_creator_info(conn, tiktok_stubs, monkeypatch):
    calls = []
    monkeypatch.setattr(tiktok_stubs, "query_creator_info", lambda token: calls.append(token))

    creds = TikTokCredentials(access_token=FAKE_TOKEN, audit_approved=False)
    handler = make_tiktok_handler(creds, conn, lambda p, i: _record(), _loader(), mode="inbox")
    handler(_row(conn, "tiktok", project_name="example-2026-08", short_index=3))

    assert calls == []


def test_tiktok_awaiting_manual_publish_is_never_reprocessed(conn, tiktok_stubs, monkeypatch):
    init_calls = []
    real_init = tiktok_stubs.init_direct_post
    monkeypatch.setattr(
        tiktok_stubs,
        "init_direct_post",
        lambda *a, **k: init_calls.append(1) or real_init(*a, **k),
    )

    creds = TikTokCredentials(access_token=FAKE_TOKEN, audit_approved=False)
    handler = make_tiktok_handler(creds, conn, lambda p, i: _record(), _loader(), mode="inbox")
    row = _row(conn, "tiktok", project_name="example-2026-08", short_index=3)
    handler(row)
    handler(row)  # second drain pass, same stale row snapshot

    assert init_calls == [1]  # never re-initialized once awaiting manual publish


def test_tiktok_invalid_mode_env_raises(conn, monkeypatch):
    monkeypatch.setenv("GMS_TIKTOK_MODE", "bogus")
    creds = TikTokCredentials(access_token=FAKE_TOKEN, audit_approved=False)
    with pytest.raises(ValueError):
        make_tiktok_handler(creds, conn, lambda p, i: _record(), _loader())


# --- build_handlers -------------------------------------------------------


def test_build_handlers_only_wires_platforms_it_was_given(conn):
    youtube_adapter = _FakeAdapter()
    instagram_adapter = _FakeAdapter()
    handlers = build_handlers(
        conn,
        lambda p, i: _record(),
        youtube=(youtube_adapter, _loader(path=True)),
        instagram=(instagram_adapter, _loader()),
    )
    assert set(handlers) == {"youtube", "instagram"}


def test_build_handlers_empty_call_wires_nothing(conn):
    assert build_handlers(conn, lambda p, i: _record()) == {}


# --- TikTok throttling: the cap that keeps the inbox publishable ---------


def _park(conn, count: int) -> None:
    """`count` TikTok rows already parked awaiting a human publish."""
    for i in range(count):
        delivery_id = enqueue(conn, "proj", 100 + i, "tiktok", "hash-parked")
        conn.execute(
            "UPDATE deliveries SET status='awaiting_manual_publish', "
            "platform_upload_id=? WHERE id=?",
            (f"v_inbox_file~v2.{i}", delivery_id),
        )
    conn.commit()


def test_tiktok_inbox_refuses_to_upload_past_the_pending_cap(conn, tiktok_stubs, monkeypatch):
    """The upstream gate. A burst of uploads on top of one already parked
    trips TikTok's own spam guard and blocks the account. Refuse BEFORE the upload -- the irreversible act -- rather
    than discovering it from a 400 afterwards."""
    called = []
    monkeypatch.setattr(tiktok_stubs, "init_direct_post", lambda *a, **k: called.append(1))
    _park(conn, 3)

    creds = TikTokCredentials(access_token=FAKE_TOKEN, audit_approved=False)
    handler = make_tiktok_handler(creds, conn, lambda p, i: _record(), _loader(), mode="inbox")
    with pytest.raises(ThrottledDeliveryError) as excinfo:
        handler(_row(conn, "tiktok"))

    assert called == [], "must not reach init_direct_post"
    assert "3" in str(excinfo.value)


def test_tiktok_inbox_uploads_normally_below_the_cap(conn, tiktok_stubs):
    """The gate must not become a permanent stop: two parked rows is the
    ordinary state after a couple of slots and has to keep working."""
    _park(conn, 2)

    creds = TikTokCredentials(access_token=FAKE_TOKEN, audit_approved=False)
    handler = make_tiktok_handler(creds, conn, lambda p, i: _record(), _loader(), mode="inbox")
    handler(_row(conn, "tiktok"))

    row = dict(
        conn.execute(
            "SELECT status FROM deliveries WHERE platform='tiktok' AND short_index=0"
        ).fetchone()
    )
    assert row["status"] == "awaiting_manual_publish"


def test_tiktok_direct_mode_ignores_the_pending_cap(conn, tiktok_stubs, monkeypatch):
    """Direct Post never creates an inbox draft, so pending drafts are
    irrelevant to it -- gating it would block a path the throttle can't
    affect."""
    monkeypatch.setattr(
        tiktok_stubs,
        "confirm_publish",
        lambda *a, **k: PublishOutcome(publish_id="pub-1", delivery_status="published", message=""),
    )
    _park(conn, 5)

    creds = TikTokCredentials(access_token=FAKE_TOKEN, audit_approved=True)
    handler = make_tiktok_handler(creds, conn, lambda p, i: _record(), _loader(), mode="direct")
    handler(_row(conn, "tiktok"))


def test_tiktok_adapter_throttle_becomes_the_dispatch_throttle(conn, tiktok_stubs, monkeypatch):
    """Defense behind the local count: if our tally is wrong, TikTok's own
    refusal must still reach the drain as a throttle, not as a generic
    retryable that burns the attempt budget."""

    def raise_throttled(*a, **k):
        raise TikTokThrottledError("Inbox upload init failed: spam_risk_too_many_pending_share")

    monkeypatch.setattr(tiktok_stubs, "init_direct_post", raise_throttled)

    creds = TikTokCredentials(access_token=FAKE_TOKEN, audit_approved=False)
    handler = make_tiktok_handler(creds, conn, lambda p, i: _record(), _loader(), mode="inbox")
    with pytest.raises(ThrottledDeliveryError):
        handler(_row(conn, "tiktok"))


def _with_tiktok_caption(record, caption):
    """`record` with only the TikTok caption changed. The payload dataclasses
    are frozen, so this rebuilds rather than mutates."""
    tt = dataclasses.replace(record.payloads.tiktok, caption=caption)
    return dataclasses.replace(record, payloads=dataclasses.replace(record.payloads, tiktok=tt))


def test_tiktok_direct_post_sends_the_caption_as_the_title(conn, tiktok_stubs, monkeypatch):
    """TikTok has one text field. `post_info.title` IS the caption -- 2200
    UTF-16 runes, hashtags included -- so sending the short label there would
    publish "Road trip West Cork" where Facebook gets a full paragraph."""
    seen = {}

    def capture(*a, **k):
        seen.update(k)
        return {
            "publish_id": "pub-1",
            "upload_url": "https://upload.us.tiktokapis.com/x",
            "chunk_plan": ChunkPlan(video_size=11, chunk_size=11, total_chunk_count=1),
        }

    monkeypatch.setattr(tiktok_stubs, "init_direct_post", capture)
    monkeypatch.setattr(
        tiktok_stubs,
        "confirm_publish",
        lambda *a, **k: PublishOutcome(publish_id="pub-1", delivery_status="published", message=""),
    )

    record = _with_tiktok_caption(_record(), "Full caption #POVDriving Filmed September 2025.")
    creds = TikTokCredentials(access_token=FAKE_TOKEN, audit_approved=True)
    handler = make_tiktok_handler(creds, conn, lambda p, i: record, _loader(), mode="direct")
    handler(_row(conn, "tiktok"))

    assert seen["title"] == "Full caption #POVDriving Filmed September 2025."


def test_tiktok_falls_back_to_the_title_when_no_caption_is_stored(conn, tiktok_stubs, monkeypatch):
    """Older sidecars carry no caption. A queued delivery
    against one must still publish, with the old short title, rather than
    posting an empty caption."""
    seen = {}

    def capture(*a, **k):
        seen.update(k)
        return {
            "publish_id": "pub-1",
            "upload_url": "https://upload.us.tiktokapis.com/x",
            "chunk_plan": ChunkPlan(video_size=11, chunk_size=11, total_chunk_count=1),
        }

    monkeypatch.setattr(tiktok_stubs, "init_direct_post", capture)
    monkeypatch.setattr(
        tiktok_stubs,
        "confirm_publish",
        lambda *a, **k: PublishOutcome(publish_id="pub-1", delivery_status="published", message=""),
    )

    record = _with_tiktok_caption(_record(), "")
    creds = TikTokCredentials(access_token=FAKE_TOKEN, audit_approved=True)
    handler = make_tiktok_handler(creds, conn, lambda p, i: record, _loader(), mode="direct")
    handler(_row(conn, "tiktok"))

    assert seen["title"] == record.payloads.tiktok.title


def _interrupted_upload_row(conn, *, status: str) -> dict:
    """A row whose ``init_direct_post`` succeeded and whose chunked upload
    then did or did not finish -- the two states a killed worker leaves
    behind, told apart by ``status`` alone."""
    row = _row(
        conn, "tiktok", project_name=_record().project_name, short_index=_record().short_index
    )
    transition(conn, row["id"], "queued", "submitting")
    transition(conn, row["id"], "submitting", status, platform_upload_id="pub-1")
    return dict(conn.execute("SELECT * FROM deliveries WHERE id=?", (row["id"],)).fetchone())


def test_tiktok_is_not_uploaded_until_the_chunks_have_actually_gone_up(
    conn, tiktok_stubs, monkeypatch
):
    """``uploaded`` must mean the bytes are on TikTok, not that ``init``
    returned a publish id. The id is still persisted before the upload --
    that is what makes an interrupted upload recoverable rather than a
    duplicate post -- but it is persisted WITHOUT claiming the upload
    finished."""
    seen = {}

    def _watching_upload(*a, **k):
        seen["mid_upload"] = dict(
            conn.execute("SELECT status, platform_upload_id FROM deliveries").fetchone()
        )

    monkeypatch.setattr(tiktok_stubs, "upload_video_chunks", _watching_upload)

    creds = TikTokCredentials(access_token=FAKE_TOKEN, audit_approved=False)
    handler = make_tiktok_handler(creds, conn, lambda p, i: _record(), _loader(), mode="inbox")
    handler(_row(conn, "tiktok", project_name="example-2026-08", short_index=3))

    # The publish id is already durable while the bytes are still going up...
    assert seen["mid_upload"]["platform_upload_id"] == "pub-1"
    # ...but the row does not yet claim the upload completed.
    assert seen["mid_upload"]["status"] == "submitting"
    assert (
        find_delivery(conn, "example-2026-08", 3, "tiktok")["status"] == "awaiting_manual_publish"
    )


def test_tiktok_inbox_resume_after_an_interrupted_upload_parks_no_phantom_draft(
    conn, tiktok_stubs, monkeypatch
):
    """A worker killed DURING the chunked upload leaves a publish id and an
    incomplete upload. Resuming must not announce a draft waiting in the
    TikTok app inbox: there is nothing there for a human to publish, the row
    is never reprocessed automatically, and it holds one of the pending-draft
    slots forever."""
    init_calls = []
    monkeypatch.setattr(
        tiktok_stubs, "init_direct_post", lambda *a, **k: init_calls.append(1) or {}
    )
    row = _interrupted_upload_row(conn, status="submitting")

    creds = TikTokCredentials(access_token=FAKE_TOKEN, audit_approved=False)
    handler = make_tiktok_handler(creds, conn, lambda p, i: _record(), _loader(), mode="inbox")
    with pytest.raises(PermanentDeliveryError):
        handler(row)

    assert init_calls == []  # a second init would be a second TikTok post
    assert (
        find_delivery(conn, "example-2026-08", 3, "tiktok")["status"] != "awaiting_manual_publish"
    )


def test_tiktok_direct_resume_after_an_interrupted_upload_does_not_confirm_a_publish(
    conn, tiktok_stubs, monkeypatch
):
    """Same interrupted state, direct mode: confirming a publish whose video
    never finished uploading can only poll until it times out, burning every
    retry on a delivery that cannot complete."""
    confirm_calls = []
    monkeypatch.setattr(
        tiktok_stubs, "confirm_publish", lambda *a, **k: confirm_calls.append(1) or None
    )
    row = _interrupted_upload_row(conn, status="submitting")

    creds = TikTokCredentials(access_token=FAKE_TOKEN, audit_approved=True)
    handler = make_tiktok_handler(creds, conn, lambda p, i: _record(), _loader(), mode="direct")
    with pytest.raises(PermanentDeliveryError):
        handler(row)

    assert confirm_calls == []


def test_tiktok_inbox_resume_after_a_completed_upload_still_parks_the_draft(
    conn, tiktok_stubs, monkeypatch
):
    """The other half of the same distinction: an upload that DID finish
    resumes exactly as before -- no re-init, and the draft is parked for a
    human. This is what stops the fix above from turning every resume into
    a failure."""
    init_calls = []
    monkeypatch.setattr(
        tiktok_stubs, "init_direct_post", lambda *a, **k: init_calls.append(1) or {}
    )
    row = _interrupted_upload_row(conn, status="uploaded")

    creds = TikTokCredentials(access_token=FAKE_TOKEN, audit_approved=False)
    handler = make_tiktok_handler(creds, conn, lambda p, i: _record(), _loader(), mode="inbox")
    handler(row)

    assert init_calls == []
    assert (
        find_delivery(conn, "example-2026-08", 3, "tiktok")["status"] == "awaiting_manual_publish"
    )


def test_tiktok_direct_resume_after_a_completed_upload_confirms_without_reinit(
    conn, tiktok_stubs, monkeypatch
):
    init_calls = []
    monkeypatch.setattr(
        tiktok_stubs, "init_direct_post", lambda *a, **k: init_calls.append(1) or {}
    )
    monkeypatch.setattr(
        tiktok_stubs,
        "confirm_publish",
        lambda *a, **k: PublishOutcome(publish_id="pub-1", delivery_status="published", message=""),
    )
    row = _interrupted_upload_row(conn, status="uploaded")

    creds = TikTokCredentials(access_token=FAKE_TOKEN, audit_approved=True)
    handler = make_tiktok_handler(creds, conn, lambda p, i: _record(), _loader(), mode="direct")
    handler(row)

    assert init_calls == []
    assert find_delivery(conn, "example-2026-08", 3, "tiktok")["status"] == "published"


# --- TikTok pack mode (the default) -----------------------------------


@pytest.fixture
def pack_env(tmp_path, monkeypatch):
    monkeypatch.delenv("GMS_TIKTOK_MODE", raising=False)
    pack_dir = tmp_path / "pack"
    monkeypatch.setenv("GMS_TIKTOK_PACK_DIR", str(pack_dir))
    return pack_dir


@pytest.fixture
def tiktok_fail_stubs(monkeypatch):
    """Every injected TikTok function fails the test if it is reached."""
    import gomoveshift.publishing.handlers as handlers_mod

    def boom(name):
        def fail(*a, **k):
            raise AssertionError(f"pack mode must make no TikTok API call: {name}")

        return fail

    for name in (
        "query_creator_info",
        "init_direct_post",
        "upload_video_chunks",
        "confirm_publish",
    ):
        monkeypatch.setattr(handlers_mod, name, boom(name))
    return handlers_mod


def _no_bytes(record):
    raise AssertionError("pack mode never reads the video into memory for an upload")


def _pack_handler(conn, tmp_path, **kwargs):
    src = tmp_path / "short.mp4"
    src.write_bytes(b"h264")
    creds = TikTokCredentials(access_token=FAKE_TOKEN, audit_approved=False)
    return make_tiktok_handler(
        creds,
        conn,
        lambda p, i: _record(),
        _no_bytes,
        rendition_path_loader=lambda record: src,
        **kwargs,
    )


def test_tiktok_default_mode_is_pack():
    from gomoveshift.publishing.handlers import _DEFAULT_TIKTOK_MODE

    assert _DEFAULT_TIKTOK_MODE == "pack"


def test_tiktok_pack_is_the_default_and_makes_no_tiktok_call(
    conn, tmp_path, pack_env, tiktok_fail_stubs
):
    handler = _pack_handler(conn, tmp_path)
    row = _row(conn, "tiktok", project_name="example-2026-08", short_index=3)
    handler(row)

    parked = find_delivery(conn, "example-2026-08", 3, "tiktok")
    assert parked["status"] == "awaiting_manual_publish"
    assert parked["platform_upload_id"] is None
    assert parked["lease_owner"] is None
    assert len(list(pack_env.glob("*.mp4"))) == 1
    assert (pack_env / "captions.txt").is_file()


def test_tiktok_explicit_pack_mode_matches_the_default(conn, tmp_path, pack_env, tiktok_fail_stubs):
    handler = _pack_handler(conn, tmp_path, mode="pack")
    handler(_row(conn, "tiktok", project_name="example-2026-08", short_index=3))
    assert (
        find_delivery(conn, "example-2026-08", 3, "tiktok")["status"] == "awaiting_manual_publish"
    )


def test_tiktok_pack_mode_has_no_inbox_cap(conn, tmp_path, pack_env, tiktok_fail_stubs):
    _park(conn, 5)  # past the inbox cap of three
    handler = _pack_handler(conn, tmp_path)
    handler(_row(conn, "tiktok", project_name="example-2026-08", short_index=3))
    assert (
        find_delivery(conn, "example-2026-08", 3, "tiktok")["status"] == "awaiting_manual_publish"
    )


def test_tiktok_pack_mode_leaves_an_awaiting_row_alone(conn, tmp_path, pack_env, tiktok_fail_stubs):
    handler = _pack_handler(conn, tmp_path)
    row = _row(conn, "tiktok", project_name="example-2026-08", short_index=3)
    transition(conn, row["id"], "queued", "awaiting_manual_publish")
    handler(row)
    assert not pack_env.exists()


def test_tiktok_pack_mode_refuses_to_run_without_a_rendition_loader(
    conn, pack_env, tiktok_fail_stubs
):
    creds = TikTokCredentials(access_token=FAKE_TOKEN, audit_approved=False)
    handler = make_tiktok_handler(creds, conn, lambda p, i: _record(), _no_bytes)
    with pytest.raises(ValueError, match="rendition"):
        handler(_row(conn, "tiktok", project_name="example-2026-08", short_index=3))
    assert find_delivery(conn, "example-2026-08", 3, "tiktok")["status"] == "queued"


def test_build_handlers_wires_the_rendition_loader_into_the_tiktok_handler(
    conn, tmp_path, pack_env, tiktok_fail_stubs
):
    src = tmp_path / "short.mp4"
    src.write_bytes(b"h264")
    creds = TikTokCredentials(access_token=FAKE_TOKEN, audit_approved=False)
    handlers = build_handlers(
        conn,
        lambda p, i: _record(),
        tiktok=(creds, _no_bytes),
        tiktok_rendition_path_loader=lambda record: src,
    )
    handlers["tiktok"](_row(conn, "tiktok", project_name="example-2026-08", short_index=3))
    assert (
        find_delivery(conn, "example-2026-08", 3, "tiktok")["status"] == "awaiting_manual_publish"
    )


def test_tiktok_env_direct_still_selects_direct_mode(conn, tiktok_stubs, monkeypatch):
    monkeypatch.setenv("GMS_TIKTOK_MODE", "direct")
    seen = {}

    def capture(*a, **k):
        seen["mode"] = k["mode"]
        return {
            "publish_id": "pub-1",
            "upload_url": "https://upload.us.tiktokapis.com/x",
            "chunk_plan": ChunkPlan(video_size=11, chunk_size=11, total_chunk_count=1),
        }

    monkeypatch.setattr(tiktok_stubs, "init_direct_post", capture)
    monkeypatch.setattr(
        tiktok_stubs,
        "confirm_publish",
        lambda *a, **k: PublishOutcome(publish_id="pub-1", delivery_status="published", message=""),
    )
    creds = TikTokCredentials(access_token=FAKE_TOKEN, audit_approved=True)
    handler = make_tiktok_handler(creds, conn, lambda p, i: _record(), _loader())
    handler(_row(conn, "tiktok"))
    assert seen["mode"] == "direct"


_FAR = "2999-01-01T00:00:00+00:00"


def test_tiktok_pack_mode_releases_its_lease_when_the_row_is_already_parked(
    conn, tmp_path, pack_env, tiktok_fail_stubs
):
    """Another process parked the row between the drain's select and this call:
    the early return must not leave the drain's lease behind."""
    handler = _pack_handler(conn, tmp_path, worker_id="w1")
    row = _row(conn, "tiktok", project_name="example-2026-08", short_index=3)
    transition(conn, row["id"], "queued", "awaiting_manual_publish")
    assert acquire_lease(conn, row["id"], "w1", _FAR)

    handler(row)

    parked = find_delivery(conn, "example-2026-08", 3, "tiktok")
    assert parked["status"] == "awaiting_manual_publish"
    assert parked["lease_owner"] is None and parked["lease_expires_at"] is None


def test_tiktok_pack_mode_never_releases_a_lease_another_worker_holds(
    conn, tmp_path, pack_env, tiktok_fail_stubs
):
    handler = _pack_handler(conn, tmp_path, worker_id="w1")
    row = _row(conn, "tiktok", project_name="example-2026-08", short_index=3)
    transition(conn, row["id"], "queued", "awaiting_manual_publish")
    assert acquire_lease(conn, row["id"], "other", _FAR)

    handler(row)

    assert find_delivery(conn, "example-2026-08", 3, "tiktok")["lease_owner"] == "other"
