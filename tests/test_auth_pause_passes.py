"""Every pass that talks to a platform pauses it on a refused access token. Before, these
caught it per row (or as an unknown outcome) and went on."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import Mock

import pytest
from google.auth.exceptions import RefreshError
from test_first_comment_meta_correction import FakeGraph, Resp, _drain, _relinked
from test_publishing_facebook_photo_client import Response
from test_publishing_facebook_photo_client import client as photo_client
from test_publishing_facebook_photo_client import upload as photo_upload
from test_publishing_facebook_photo_delivery import row as photo_row
from test_publishing_facebook_photo_delivery import setup  # noqa: F401 -- fixture
from test_publishing_facebook_reconcile import CREDS, UNKNOWN, _resp, seed
from test_publishing_models import _record
from test_youtube_first_comment import (
    _FakeService,
    _make_http_error,
    _make_published_youtube_row,
)

from gomoveshift.pipeline.drive_phases.youtube_uploader import youtube_auth_refused
from gomoveshift.publishing import facebook_adapter as facebook
from gomoveshift.publishing import youtube_comments
from gomoveshift.publishing.dispatch import drain_queue
from gomoveshift.publishing.facebook_photo_client import PhotoReadbackAuthBlockedError
from gomoveshift.publishing.facebook_photo_delivery import PhotoDeliveryAuthBlockedError
from gomoveshift.publishing.handlers import (
    build_handlers,
    make_photo_handler,
    make_youtube_handler,
)
from gomoveshift.publishing.models import FORMAT_CAROUSEL, MediaFingerprint
from gomoveshift.publishing.photo_adapter import PhotoAuthBlockedError, PhotoCarouselAdapter
from gomoveshift.publishing.queue import (
    AUTH_BLOCKED_PAUSE_REASON,
    AUTH_PAUSED_ERROR_CLASS,
    enqueue,
    open_queue_db,
    transition,
)
from gomoveshift.publishing.youtube_adapter import confirm_scheduled_live

NOW = "2026-11-26T12:00:00+00:00"
REVOKED = RefreshError("invalid_grant: Token has been expired or revoked.")


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def _paused_for_auth(conn, platform):
    row = conn.execute("SELECT reason FROM platform_pauses WHERE platform=?", (platform,))
    return ((row.fetchone() or [""])[0] or "").startswith(AUTH_BLOCKED_PAUSE_REASON + " ")


def _status(conn, delivery_id):
    return dict(conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone())


# --- the shared pause --------------------------------------------------------


def test_a_second_refusal_keeps_the_first_pause_and_its_time(conn):
    from gomoveshift.publishing.queue import pause_for_auth

    pause_for_auth(conn, "youtube", REVOKED)
    first = conn.execute("SELECT * FROM platform_pauses").fetchone()
    pause_for_auth(conn, "youtube", RuntimeError("later"))
    assert dict(conn.execute("SELECT * FROM platform_pauses").fetchone()) == dict(first)


def test_youtube_names_only_a_refused_token():
    assert youtube_auth_refused(REVOKED)
    assert youtube_auth_refused(_make_http_error(401, reason="authError"))
    assert not youtube_auth_refused(RefreshError("server_error", retryable=True))
    assert not youtube_auth_refused(_make_http_error(403, reason="insufficientPermissions"))
    assert not youtube_auth_refused(_make_http_error(403, reason="quotaExceeded"))


# --- first comments -----------------------------------------------------------


class _EditRefusedGraph(FakeGraph):
    """Graph refuses the comment edit for the Page token (code 190)."""

    def post(self, url, data=None, **kw):
        if not url.endswith("/comments"):
            self.log.append("edit-refused-token")
            return Resp({"error": {"code": 190, "message": "token expired"}}, 400)
        return super().post(url, data=data, **kw)


def test_a_refused_token_on_a_stale_comment_correction_pauses_facebook(conn, tmp_path):
    inc = tmp_path / "incidents"
    did = _relinked(conn, "facebook")
    graph = _EditRefusedGraph("facebook", [("c1", "Watch the full drive: https://youtu.be/old")])
    _drain(conn, "facebook", graph, inc)  # finds the stale comment, records the correction

    result = _drain(conn, "facebook", graph, inc)

    assert graph.log.count("edit-refused-token") == 1
    assert _paused_for_auth(conn, "facebook")
    assert any(e.startswith("facebook: paused, access token refused") for e in result.errors)
    assert _status(conn, did)["first_comment_correction_attempts"] == 0


def test_a_refused_youtube_token_pauses_youtube_on_the_first_comment_drain(conn):
    ids = [_make_published_youtube_row(conn, i) for i in (61, 62)]
    service = _FakeService(exc=_make_http_error(401, reason="authError", message="Invalid"))

    result = youtube_comments.drain_first_comments(
        conn, service_factory=lambda: service, now="2026-08-23T00:06:00+00:00"
    )

    assert len(service._threads.calls) == 1  # the second row waits for the resume
    assert _paused_for_auth(conn, "youtube")
    assert [e.split(":")[0] for e in result.errors] == ["youtube"]
    assert result.errors[0].startswith("youtube: paused, access token refused: ")
    assert all(_status(conn, i)["first_comment_attempts"] == 0 for i in ids)


def test_the_youtube_drain_has_its_corrections_pause_on_a_refused_token(conn, monkeypatch):
    seen = {}
    monkeypatch.setattr(youtube_comments, "run_corrections", lambda conn, **kw: seen.update(kw))
    youtube_comments.drain_first_comments(conn, service_factory=Mock(), now=NOW)
    assert seen["auth_refused"] is youtube_auth_refused


# --- YouTube publish and live check -------------------------------------------


class _RefusedYouTube:
    def __init__(self):
        self.calls = []

    def publish(self, record, *, conn, delivery_id, **_kw):
        self.calls.append(delivery_id)
        transition(conn, delivery_id, "queued", "submitting")
        raise REVOKED


def _loader(project, index):
    return dataclasses.replace(_record(), project_name=project, short_index=index)


def test_a_refused_youtube_token_pauses_youtube_and_keeps_the_row_for_its_reconcile(conn, tmp_path):
    ids = [enqueue(conn, "example-2026-08", i, "youtube", "hash-a") for i in range(3)]
    adapter = _RefusedYouTube()
    handler = make_youtube_handler(adapter, conn, _loader, lambda record: Path("/v/x.mp4"))

    result = drain_queue(conn, {"youtube": handler}, now=NOW, incidents_dir=tmp_path / "inc")

    assert len(adapter.calls) == 1 and result.failed_permanent == 0
    assert _paused_for_auth(conn, "youtube")
    first = _status(conn, ids[0])
    # Still submitting: the adapter's next attempt checks the channel first.
    assert first["status"] == "submitting" and first["error_class"] == AUTH_PAUSED_ERROR_CLASS
    assert [_status(conn, i)["attempt_count"] for i in ids[1:]] == [0, 0]


class _LiveCheckService:
    def __init__(self, exc):
        self.exc, self.calls = exc, 0

    def videos(self):
        return self

    def list(self, **_kw):
        return self

    def execute(self):
        self.calls += 1
        raise self.exc


def test_a_refused_youtube_token_pauses_youtube_on_the_live_check(conn):
    for i in range(3):
        did = enqueue(conn, "example-2026-08", i, "youtube", "hash-a")
        transition(
            conn, did, "queued", "published", platform_upload_id=f"v{i}",
            published_at_source="platform_scheduled", platform_published_at=NOW,
        )  # fmt: skip
    service = _LiveCheckService(_make_http_error(401, reason="authError"))

    assert confirm_scheduled_live(conn, service, now="2026-11-27T12:00:00+00:00") == 0

    assert service.calls == 1 and _paused_for_auth(conn, "youtube")


# --- Facebook reel read-back ---------------------------------------------------


def test_a_refused_token_on_the_reel_read_back_pauses_facebook(conn, tmp_path):
    record, delivery = seed(conn, status=UNKNOWN)
    refused = _resp(400, {"error": {"code": 190, "message": "Session has expired"}})
    session = Mock(post=Mock(), get=Mock(return_value=refused))
    adapter = facebook.FacebookShortsAdapter(facebook.FacebookReelsAdapter(CREDS, session=session))
    handlers = build_handlers(conn, lambda p, i: record, facebook=(adapter, lambda r: b"x"))

    result = drain_queue(conn, handlers, now=NOW, incidents_dir=tmp_path / "inc")

    assert result.failed_permanent == 0 and _paused_for_auth(conn, "facebook")
    state = _status(conn, delivery)
    assert state["status"] == UNKNOWN and state["platform_upload_id"] == "v-1"
    assert state["error_class"] == AUTH_PAUSED_ERROR_CLASS
    session.post.assert_not_called()


# --- Facebook Page photo read-back ---------------------------------------------


def test_the_page_photo_client_names_a_refused_token_on_its_read():
    instance, _session = photo_client(
        Response({"id": "456"}),
        Response({"error": {"code": 190, "message": "expired"}}, status=400),
    )
    receipt = photo_upload(instance)
    with pytest.raises(PhotoReadbackAuthBlockedError) as caught:
        instance.read_status(receipt)
    assert caught.value.photo_id == "456"
    assert "expired" not in str(caught.value)  # still nothing from the response


def test_a_refused_page_token_on_the_photo_reconcile_pauses_facebook(setup):  # noqa: F811
    db, identifier, _, _, client, delivery = setup
    client.read_status.side_effect = PhotoReadbackAuthBlockedError("123", "456", None)
    with pytest.raises(PhotoDeliveryAuthBlockedError):
        delivery.deliver(photo_row(db, identifier))

    result = drain_queue(
        db, {}, owner="worker", now="2026-11-11T12:00:00+00:00",
        facebook_photo_reconciler=delivery.reconcile,
    )  # fmt: skip

    assert result.failed_permanent == 0 and _paused_for_auth(db, "facebook")
    assert photo_row(db, identifier)["platform_upload_id"] == "456"
    client.upload_photo.assert_called_once()  # never a second POST


# --- Instagram carousel ----------------------------------------------------------


class _CarouselClient:
    account_id = "acct-1"

    def __init__(self, fail_at):
        self.fail_at, self.items = fail_at, 0

    def create_carousel_item_container(self, image_url):
        if self.items == self.fail_at:
            raise PhotoAuthBlockedError("[graph http=400 code=190] Meta access token expired")
        self.items += 1
        return f"item{self.items}"


def _carousel(project, index):
    extra = MediaFingerprint("/v/y.jpg", 999, 0.0, 1080, 1080, "jpeg", "b" * 64)
    return dataclasses.replace(
        _loader(project, index), format=FORMAT_CAROUSEL, additional_media=(extra,)
    )


@pytest.mark.parametrize("fail_at", [0, 1])
def test_a_refused_token_on_a_carousel_pauses_instagram(conn, tmp_path, fail_at):
    did = enqueue(conn, "example-2026-08", 0, "instagram", "hash-a")
    adapter = PhotoCarouselAdapter(_CarouselClient(fail_at), "instagram")
    handler = make_photo_handler(adapter, conn, _carousel, lambda media: "https://img.invalid/x")

    drain_queue(conn, {"instagram": handler}, now=NOW, incidents_dir=tmp_path / "inc")

    assert _paused_for_auth(conn, "instagram")
    # Nothing created: it waits for the resume. Part-built: it stays failed, as
    # any partial carousel does, for a person to look at its orphaned items.
    assert _status(conn, did)["status"] == ("queued" if fail_at == 0 else "failed_permanent")
