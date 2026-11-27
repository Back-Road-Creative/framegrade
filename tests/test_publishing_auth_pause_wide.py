"""A refused token in the Facebook Page photo client and delivery, the Reel
delivery and the first-comment read-back pauses Facebook through the shared
path (``queue.is_auth_blocked`` / ``queue.pause_for_auth``). Before, each turned
it into an unknown outcome, or logged it and went on."""

from __future__ import annotations

import pytest
from test_first_comment_meta_correction import FakeGraph, Resp, _relinked
from test_publishing_facebook_photo_client import Response
from test_publishing_facebook_photo_client import client as photo_client
from test_publishing_facebook_photo_client import upload as photo_upload
from test_publishing_facebook_photo_delivery import NOW
from test_publishing_facebook_photo_delivery import row as photo_row
from test_publishing_facebook_photo_delivery import setup  # noqa: F401 -- fixture
from test_publishing_facebook_reel_delivery import row as reel_row
from test_publishing_facebook_reel_delivery import setup as reel_setup  # noqa: F401 -- fixture

from gomoveshift.publishing import meta_comments
from gomoveshift.publishing.dispatch import drain_queue
from gomoveshift.publishing.facebook_adapter import FacebookAuthBlockedError, FacebookReelsAdapter
from gomoveshift.publishing.facebook_photo_client import (
    PhotoCommentAuthBlockedError,
    PhotoUploadAuthBlockedError,
    PhotoUploadUnknownError,
)
from gomoveshift.publishing.facebook_photo_delivery import PhotoDeliveryAuthBlockedError
from gomoveshift.publishing.facebook_reel_delivery import ReelDeliveryAuthBlockedError
from gomoveshift.publishing.handlers import make_facebook_page_photo_handler
from gomoveshift.publishing.queue import (
    AUTH_BLOCKED_PAUSE_REASON,
    is_auth_blocked,
    open_queue_db,
)

EXPIRED = {"error": {"code": 190, "message": "Session has expired"}}


def _paused(conn, platform="facebook"):
    row = conn.execute("SELECT reason FROM platform_pauses WHERE platform=?", (platform,))
    return ((row.fetchone() or [""])[0] or "").startswith(AUTH_BLOCKED_PAUSE_REASON + " ")


# --- Page photo client --------------------------------------------------------


@pytest.mark.parametrize("status", [400, 401])
def test_the_photo_client_names_a_refused_token_on_its_upload(status):
    instance, session = photo_client(Response(EXPIRED, status=status))
    with pytest.raises(PhotoUploadAuthBlockedError) as caught:
        photo_upload(instance)
    assert isinstance(caught.value, PhotoUploadUnknownError)  # still no second POST
    assert is_auth_blocked(caught.value) and "expired" not in str(caught.value)
    assert len(session.calls) == 1


def test_the_photo_client_names_a_refused_token_on_its_comment():
    instance, _ = photo_client(Response(EXPIRED, status=400))
    with pytest.raises(PhotoCommentAuthBlockedError) as caught:
        instance.post_comment("123_902", "Watch the full drive")
    assert is_auth_blocked(caught.value) and caught.value.story_id == "123_902"


# --- Page photo delivery --------------------------------------------------------


def test_a_refused_token_on_the_photo_upload_pauses_facebook_and_holds_the_attempt(
    setup,  # noqa: F811
):
    db, identifier, _, _, client, delivery = setup
    client.upload_photo.side_effect = PhotoUploadAuthBlockedError("123")

    with pytest.raises(PhotoDeliveryAuthBlockedError) as caught:
        delivery.deliver(photo_row(db, identifier))

    assert is_auth_blocked(caught.value)
    held = photo_row(db, identifier)
    assert held["upload_kind"] is not None and not held["platform_upload_id"]
    client.upload_photo.assert_called_once()  # held: never a second POST


def test_the_drain_pauses_facebook_on_a_refused_photo_upload(setup):  # noqa: F811
    db, identifier, _, _, client, delivery = setup
    client.upload_photo.side_effect = PhotoUploadAuthBlockedError("123")

    drain_queue(
        db, {"facebook": make_facebook_page_photo_handler(delivery)},
        owner="worker", now=NOW.isoformat(),
    )  # fmt: skip

    assert _paused(db)
    client.upload_photo.assert_called_once()


# --- Reel delivery ---------------------------------------------------------------


def test_a_refused_token_on_the_reel_upload_keeps_the_attempt_and_names_the_refusal(
    reel_setup,  # noqa: F811
):
    conn, _, _, _, client, delivery = reel_setup
    client.upload_video.side_effect = FacebookAuthBlockedError("[graph code=190] expired")

    with pytest.raises(ReelDeliveryAuthBlockedError) as caught:
        delivery.deliver(42)

    assert is_auth_blocked(caught.value)
    assert reel_row(conn)["state"] == "upload_started"  # retained, never replayed
    client.finish_upload.assert_not_called()


def test_a_refused_token_on_the_reel_reconcile_names_the_refusal(reel_setup):  # noqa: F811
    conn, _, _, _, client, delivery = reel_setup
    client.finish_upload.side_effect = ConnectionError("lost")
    with pytest.raises(Exception):
        delivery.deliver(42)
    client.check_video_status.side_effect = FacebookAuthBlockedError("[graph code=190] expired")

    with pytest.raises(ReelDeliveryAuthBlockedError) as caught:
        delivery.reconcile(42)

    assert is_auth_blocked(caught.value)
    assert reel_row(conn)["state"] == "finish_unknown"


# --- first-comment read-back ------------------------------------------------------


class _ReadBackRefused(FakeGraph):
    """Posting works; reading the fresh comment back is refused for the token."""

    def get(self, url, params=None, **kw):
        if not url.endswith("/comments") and self.n:
            self.log.append("read-refused")
            return Resp(EXPIRED, 400)
        return super().get(url, params=params, **kw)


def test_a_refused_token_on_the_comment_read_back_pauses_facebook_and_keeps_the_post(tmp_path):
    conn = open_queue_db(tmp_path / "q")
    try:
        did = _relinked(conn, "facebook")
        graph = _ReadBackRefused("facebook", [])
        adapter = FacebookReelsAdapter({"page_id": "p", "page_access_token": "t"}, session=graph)

        result = meta_comments.drain_meta_first_comments(
            conn, adapter_factories={"facebook": lambda: adapter},
            now="2026-08-23T00:06:00+00:00", incidents_dir=tmp_path / "inc",
        )  # fmt: skip

        assert "read-refused" in graph.log and _paused(conn)
        assert any(e.startswith("facebook: paused, access token refused") for e in result.errors)
        live = conn.execute("SELECT first_comment_live_id FROM deliveries WHERE id=?", (did,))
        assert live.fetchone()[0] == "new1"  # the posted comment is still recorded
    finally:
        conn.close()
