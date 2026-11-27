"""A refused token met after the upload went through (YouTube's post-upload
``videos.list`` check and playlist assign, the Facebook reel read-back after
finish) pauses the platform like every other pass. Before, each logged it and
went on: the YouTube row published unchecked, the reel waited a pass."""

from __future__ import annotations

from unittest.mock import patch

from google.auth.exceptions import RefreshError
from test_publishing_youtube_adapter import (  # noqa: F401 -- fixtures
    _handle,
    _setup,
    mock_service,
    queue_conn,
)

from gomoveshift.publishing.queue import (
    AUTH_BLOCKED_PAUSE_REASON,
    resume_platform,
)

REVOKED = RefreshError("invalid_grant: Token has been expired or revoked.")


def _pause_reason(conn):
    row = conn.execute("SELECT reason FROM platform_pauses WHERE platform='youtube'").fetchone()
    return row["reason"] if row else ""


def test_a_refused_token_on_the_post_upload_check_pauses_youtube_and_keeps_the_id(
    mock_service,  # noqa: F811
    queue_conn,  # noqa: F811
    tmp_path,
):
    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)
    mock_service.videos.return_value.list.return_value.execute.side_effect = REVOKED

    row = _handle(adapter, record, video_path, queue_conn, delivery_id)

    assert _pause_reason(queue_conn).startswith(AUTH_BLOCKED_PAUSE_REASON + " ")
    assert (row["status"], row["platform_upload_id"]) == ("uploaded", "SHORTVID123")
    assert mock_service.videos.return_value.insert.call_count == 1


def test_after_the_resume_the_upload_is_reconciled_never_inserted_again(
    mock_service,  # noqa: F811
    queue_conn,  # noqa: F811
    tmp_path,
):
    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)
    execute = mock_service.videos.return_value.list.return_value.execute
    execute.side_effect = REVOKED
    _handle(adapter, record, video_path, queue_conn, delivery_id)

    assert resume_platform(queue_conn, "youtube") is True
    execute.side_effect = None
    execute.return_value = {"items": [{"id": "SHORTVID123"}]}
    with patch("gomoveshift.publishing.youtube_adapter.MediaFileUpload", create=True):
        adapter.publish(record, conn=queue_conn, delivery_id=delivery_id, video_path=video_path)

    row = queue_conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    assert row["status"] == "published" and row["platform_upload_id"] == "SHORTVID123"
    assert mock_service.videos.return_value.insert.call_count == 1


def test_a_refused_token_on_the_playlist_assign_pauses_youtube(
    mock_service,  # noqa: F811
    queue_conn,  # noqa: F811
    tmp_path,
):
    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)
    mock_service.videos.return_value.list.return_value.execute.return_value = {
        "items": [{"id": "SHORTVID123"}]
    }
    with patch.object(adapter, "_assign_playlist", side_effect=REVOKED):
        row = _handle(adapter, record, video_path, queue_conn, delivery_id)

    assert _pause_reason(queue_conn).startswith(AUTH_BLOCKED_PAUSE_REASON + " ")
    assert (row["status"], row["platform_upload_id"]) == ("uploaded", "SHORTVID123")


def test_a_refused_token_on_the_reel_read_back_after_finish_pauses_facebook(tmp_path):
    from test_publishing_facebook_adapter import CREDS, _finish_ok_session, _resp
    from test_publishing_models import _record

    from gomoveshift.publishing.dispatch import drain_queue
    from gomoveshift.publishing.facebook_adapter import (
        FacebookReelsAdapter,
        FacebookShortsAdapter,
    )
    from gomoveshift.publishing.handlers import build_handlers
    from gomoveshift.publishing.queue import enqueue, open_queue_db

    conn = open_queue_db(tmp_path / "seo")
    try:
        record = _record()
        delivery_id = enqueue(conn, record.project_name, record.short_index, "facebook", "h1")
        refused = _resp(400, {"error": {"code": 190, "message": "Session has expired"}})
        adapter = FacebookShortsAdapter(
            FacebookReelsAdapter(CREDS, session=_finish_ok_session(refused))
        )
        handlers = build_handlers(conn, lambda p, i: record, facebook=(adapter, lambda r: b"x"))

        result = drain_queue(
            conn, handlers, now="2026-11-26T12:00:00+00:00", incidents_dir=tmp_path / "inc"
        )

        pause = conn.execute("SELECT reason FROM platform_pauses WHERE platform='facebook'")
        assert (pause.fetchone() or [""])[0].startswith(AUTH_BLOCKED_PAUSE_REASON + " ")
        assert result.failed_permanent == 0
        row = conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
        assert (row["status"], row["platform_upload_id"]) == ("awaiting_facebook_reconcile", "v-1")
    finally:
        conn.close()
