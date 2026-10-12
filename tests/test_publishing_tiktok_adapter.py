"""Tests for the TikTok Direct Post adapter: the public-routing gate, chunk
planning, upload-host allowlisting, the init + chunked FILE_UPLOAD transfer,
the publish-status poll loop, and the post-publish privacy readback.

Credential loading and token refresh are covered in
``tests/test_publishing_tiktok_auth.py``; see also
``docs/social-publishing-tiktok.md``."""

from __future__ import annotations

from unittest import mock

import pytest
import requests

from gomoveshift.publishing.tiktok_adapter import (
    INBOX_INIT_URL,
    INIT_URL,
    ChunkPlan,
    PublishOutcome,
    TikTokAdapterError,
    TikTokPollTimeoutError,
    TikTokThrottledError,
    assert_public_routing_allowed,
    confirm_publish,
    init_direct_post,
    list_videos,
    plan_chunks,
    poll_publish_status,
    query_creator_info,
    upload_video_chunks,
    validate_upload_host,
)

MB = 1024 * 1024
GB = 1024 * MB


# The regression that matters most: audit_approved=False refuses
# PUBLIC_TO_EVERYONE even when creator_info offers it — as a raise, not a
# boolean a future caller could forget to check.
_OFFERS_ALL = {
    "privacy_level_options": ["PUBLIC_TO_EVERYONE", "MUTUAL_FOLLOW_FRIENDS", "SELF_ONLY"]
}


def test_unaudited_public_raises_naming_the_config_flag():
    with pytest.raises(TikTokAdapterError, match="publishing.tiktok.audit_approved"):
        assert_public_routing_allowed("PUBLIC_TO_EVERYONE", False, _OFFERS_ALL)


def test_audited_public_passes():
    assert assert_public_routing_allowed("PUBLIC_TO_EVERYONE", True, _OFFERS_ALL) is None


@pytest.mark.parametrize("privacy_level", ["SELF_ONLY", "MUTUAL_FOLLOW_FRIENDS"])
def test_non_public_passes_when_audited(privacy_level):
    assert assert_public_routing_allowed(privacy_level, True, _OFFERS_ALL) is None


# TikTok 403s a Direct Post init from an unaudited client with
# unaudited_client_can_only_post_to_private_accounts even when privacy_level
# is SELF_ONLY — the constraint is on the ACCOUNT's public/private state, not
# the post's audience. The guard must refuse every
# privacy_level when unaudited, not just PUBLIC_TO_EVERYONE, so a doomed
# delivery never burns an init attempt against an error it exists to prevent.
@pytest.mark.parametrize("privacy_level", ["SELF_ONLY", "MUTUAL_FOLLOW_FRIENDS"])
def test_unaudited_non_public_raises_naming_the_live_error_code(privacy_level):
    with pytest.raises(
        TikTokAdapterError, match="unaudited_client_can_only_post_to_private_accounts"
    ):
        assert_public_routing_allowed(privacy_level, False, _OFFERS_ALL)


def test_privacy_level_not_offered_raises():
    creator_info = {"privacy_level_options": ["SELF_ONLY"]}
    with pytest.raises(TikTokAdapterError):
        assert_public_routing_allowed("MUTUAL_FOLLOW_FRIENDS", True, creator_info)


# --- Chunk arithmetic -------------------------------------------------------


def test_chunk_plan_just_under_five_mb_is_a_single_chunk():
    size = 5 * MB - 1
    plan = plan_chunks(size)
    assert plan == ChunkPlan(video_size=size, chunk_size=size, total_chunk_count=1)


def test_chunk_plan_exactly_sixty_four_mb_is_a_single_chunk():
    size = 64 * MB
    plan = plan_chunks(size)
    assert plan.chunk_size == 64 * MB
    assert plan.total_chunk_count == 1


def test_chunk_plan_trailing_chunk_folds_into_final():
    size = 130 * MB
    plan = plan_chunks(size)
    assert plan.chunk_size == 64 * MB
    assert plan.total_chunk_count == 2
    final_chunk_bytes = size - plan.chunk_size * (plan.total_chunk_count - 1)
    assert MB <= final_chunk_bytes <= 128 * MB


# Guards the chunk plan for files between 64 MiB and 128 MiB. For a
# 97,000,000-byte short, floor(97,000,000 / 64 MiB) == 1, so a naive plan
# proposes ONE chunk for a file the API requires to be split, and init is
# rejected with HTTP 400 invalid_params "The chunk size is invalid". The
# single-chunk ceiling is 64 MiB (not decimal 64,000,000): a whole file below
# it may go as one chunk. The affected window is 64 MiB < video_size < 128 MiB,
# where a typical short lands.
def test_chunk_plan_between_64_and_128_mb_is_never_a_single_chunk():
    size = 97_000_000  # a typical short size, between 64 and 128 MiB
    plan = plan_chunks(size)
    assert plan.total_chunk_count >= 2
    assert plan.chunk_size <= 64 * MB
    assert plan.chunk_size >= 5 * MB
    # The whole file is still covered: the remainder folds into the final chunk.
    final_chunk_bytes = size - plan.chunk_size * (plan.total_chunk_count - 1)
    assert final_chunk_bytes <= 128 * MB


def test_chunk_plan_refuses_a_caller_supplied_single_chunk_over_64mb():
    """An explicit chunk_size that yields one chunk for a >64 MiB file is the
    same rejected plan, reached by a different door -- refuse it here rather
    than at init, where the only evidence is a 400."""
    with pytest.raises(TikTokAdapterError, match="multiple chunks"):
        plan_chunks(97_000_000, chunk_size=64 * MB)


def test_chunk_plan_whole_file_just_over_decimal_64mb_stays_one_chunk():
    """A whole file between decimal 64,000,000 bytes and binary 64 MiB is one
    valid chunk: the single-chunk ceiling is 64 MiB, so this must not be split."""
    size = 66_000_000
    plan = plan_chunks(size)
    assert plan == ChunkPlan(video_size=size, chunk_size=size, total_chunk_count=1)


def test_chunk_plan_over_thousand_chunks_rejected():
    with pytest.raises(TikTokAdapterError, match="1000-chunk"):
        plan_chunks(1001 * (5 * MB), chunk_size=5 * MB)


def test_chunk_plan_over_four_gb_rejected():
    with pytest.raises(TikTokAdapterError, match="4 GB"):
        plan_chunks(4 * GB + 1)


@pytest.mark.parametrize("bad_chunk_size", [1 * MB, 65 * MB])
def test_chunk_plan_out_of_range_chunk_size_rejected(bad_chunk_size):
    with pytest.raises(TikTokAdapterError):
        plan_chunks(200 * MB, chunk_size=bad_chunk_size)


# --- Upload host allowlist ---------------------------------------------------


def test_allowlisted_https_upload_host_accepted():
    validate_upload_host("https://open-upload.tiktokapis.com/upload/v1/abc")


def test_non_allowlisted_host_rejected():
    with pytest.raises(TikTokAdapterError):
        validate_upload_host("https://evil.example.com/upload")


def test_http_scheme_rejected_even_on_allowlisted_host():
    with pytest.raises(TikTokAdapterError):
        validate_upload_host("http://open-upload.tiktokapis.com/upload/v1/abc")


# --- No token material in raised messages -----------------------------------


def test_creator_info_failure_never_leaks_token_in_message():
    with mock.patch(
        "gomoveshift.publishing.tiktok_adapter.requests.post",
        side_effect=requests.ConnectionError("boom"),
    ):
        with pytest.raises(TikTokAdapterError) as exc_info:
            query_creator_info("super-secret-access-token")
    assert "super-secret-access-token" not in str(exc_info.value)


# --- creator_info envelope unwrapping ----------------------------------------

# Sandbox response shape:
# {"data": {...8 keys incl. privacy_level_options...},
#  "error": {"code": "ok", "log_id": "...", "message": ""}}
# The adapter used to return this WHOLE envelope, so every caller reading
# `creator_info["privacy_level_options"]` got None -> [] -> every privacy level
# rejected as "not offered". SELF_ONLY included; TikTok publishing could not work.
_LIVE_CREATOR_INFO_ENVELOPE = {
    "data": {
        "comment_disabled": False,
        "creator_avatar_url": "https://example.invalid/a.jpg",
        "creator_nickname": "GoMoveShift",
        "creator_username": "example_account",
        "duet_disabled": False,
        "max_video_post_duration_sec": 3600,
        "privacy_level_options": [
            "PUBLIC_TO_EVERYONE",
            "MUTUAL_FOLLOW_FRIENDS",
            "SELF_ONLY",
        ],
        "stitch_disabled": False,
    },
    "error": {"code": "ok", "log_id": "20260101000000BBBBBBBBBBBBBBBBBB", "message": ""},
}


def _post_returning(payload):
    resp = mock.Mock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return mock.patch("gomoveshift.publishing.tiktok_adapter.requests.post", return_value=resp)


def _post_raising_http(status_code, body_text):
    """A response that fails ``raise_for_status`` the way requests does, with
    TikTok's error body intact — the shape whose detail must reach the log."""
    resp = mock.Mock()
    resp.status_code = status_code
    resp.text = body_text
    resp.content = body_text.encode()
    resp.json.return_value = __import__("json").loads(body_text)
    resp.raise_for_status.side_effect = requests.HTTPError(f"{status_code} Client Error")
    return mock.patch("gomoveshift.publishing.tiktok_adapter.requests.post", return_value=resp)


# An HTTP failure at init must carry the status, TikTok's error code, message
# and log_id, the same diagnostics an error-code body already carries, and must
# never echo the token. A bare "request failed" hides causes such as a
# one-chunk plan for a 97 MB file.
#: TikTok's own log correlation id, the value support asks for. Not a
#: credential -- it identifies a request in their logs and grants nothing --
#: but 32 hex characters trip detect-secrets' entropy heuristic.
_LOG_ID = "20260101000000AAAAAAAAAAAAAAAAAA"  # pragma: allowlist secret


def test_http_error_names_the_status_and_the_tiktok_error_body():
    body = (
        '{"error":{"code":"invalid_params","message":"The chunk size is invalid",'
        f'"log_id":"{_LOG_ID}"}}}}'
    )
    with _post_raising_http(400, body):
        with pytest.raises(TikTokAdapterError) as exc_info:
            init_direct_post("tok", video_size=5 * MB, mode="inbox")
    message = str(exc_info.value)
    assert "400" in message
    assert "invalid_params" in message
    assert "The chunk size is invalid" in message
    assert _LOG_ID in message


def test_http_error_body_echoing_the_token_does_not_leak_it():
    echoed = "super-secret-access-token"  # not a credential; the API echoes it back
    body = f'{{"error":{{"code":"unauthorized","message":"bad {echoed}","log_id":"z"}}}}'
    with _post_raising_http(401, body):
        with pytest.raises(TikTokAdapterError) as exc_info:
            init_direct_post(echoed, video_size=5 * MB, mode="inbox")
    assert echoed not in str(exc_info.value)


def test_a_transport_failure_still_raises_the_generic_message():
    """No response exists to quote — the message stays the fixed one."""
    with mock.patch(
        "gomoveshift.publishing.tiktok_adapter.requests.post",
        side_effect=requests.ConnectionError("dns"),
    ):
        with pytest.raises(TikTokAdapterError, match="request failed"):
            init_direct_post("tok", video_size=5 * MB, mode="inbox")


def test_query_creator_info_unwraps_the_data_envelope():
    """Returns `data`, not the whole envelope — the value callers index into."""
    with _post_returning(_LIVE_CREATOR_INFO_ENVELOPE):
        info = query_creator_info("tok")
    assert info["privacy_level_options"] == [
        "PUBLIC_TO_EVERYONE",
        "MUTUAL_FOLLOW_FRIENDS",
        "SELF_ONLY",
    ]
    assert info["creator_username"] == "example_account"
    assert "data" not in info and "error" not in info


def test_query_creator_info_result_satisfies_the_routing_guard():
    """The end-to-end seam the unit tests used to skip: the guard was always
    fed a hand-unwrapped dict, so nobody noticed the real call returned the
    envelope. SELF_ONLY passes once audited; an unaudited client is refused
    regardless of privacy_level."""
    with _post_returning(_LIVE_CREATOR_INFO_ENVELOPE):
        info = query_creator_info("tok")
    assert_public_routing_allowed("SELF_ONLY", True, info)
    with pytest.raises(TikTokAdapterError):
        assert_public_routing_allowed("SELF_ONLY", False, info)


def test_query_creator_info_raises_on_error_code():
    envelope = {
        "data": {},
        "error": {"code": "access_token_invalid", "log_id": "x", "message": "bad token"},
    }
    with _post_returning(envelope):
        with pytest.raises(TikTokAdapterError) as exc_info:
            query_creator_info("super-secret-access-token")
    message = str(exc_info.value)
    assert "access_token_invalid" in message
    assert "bad token" in message
    assert "x" in message
    assert "super-secret-access-token" not in message


def test_query_creator_info_error_body_echoing_token_does_not_leak_it():
    envelope = {
        "data": {},
        "error": {
            "code": "internal_error",
            "message": "token super-secret-access-token was rejected",
            "log_id": "abc123",
        },
    }
    with _post_returning(envelope):
        with pytest.raises(TikTokAdapterError) as exc_info:
            query_creator_info("super-secret-access-token")
    assert "super-secret-access-token" not in str(exc_info.value)


# --- Init + chunked FILE_UPLOAD transfer -------------------------------------


def _init_response(publish_id="pub123", upload_url="https://open-upload.tiktokapis.com/u/1"):
    resp = mock.Mock()
    resp.raise_for_status = mock.Mock()
    resp.json.return_value = {"data": {"publish_id": publish_id, "upload_url": upload_url}}
    return resp


_INIT_KWARGS = dict(
    title="A Drive",
    privacy_level="SELF_ONLY",
    audit_approved=True,
    creator_info={"privacy_level_options": ["SELF_ONLY"]},
    video_size=10 * MB,
)


def test_init_sends_exact_chunk_plan():
    video_size = 130 * MB
    expected_plan = plan_chunks(video_size)
    kwargs = {**_INIT_KWARGS, "video_size": video_size}
    with mock.patch(
        "gomoveshift.publishing.tiktok_adapter.requests.post",
        return_value=_init_response(),
    ) as mock_post:
        result = init_direct_post("tok", **kwargs)
    source_info = mock_post.call_args.kwargs["json"]["source_info"]
    assert source_info == {
        "source": "FILE_UPLOAD",
        "video_size": expected_plan.video_size,
        "chunk_size": expected_plan.chunk_size,
        "total_chunk_count": expected_plan.total_chunk_count,
    }
    assert result["publish_id"] == "pub123"
    assert result["chunk_plan"] == expected_plan


def test_init_default_is_aigc_false():
    # Real captured driving footage, not AI-generated — a True default would
    # put a false "Creator labeled as AI-generated" badge on every post.
    with mock.patch(
        "gomoveshift.publishing.tiktok_adapter.requests.post",
        return_value=_init_response(),
    ) as mock_post:
        init_direct_post("tok", **_INIT_KWARGS)
    assert mock_post.call_args.kwargs["json"]["post_info"]["is_aigc"] is False


def test_init_explicit_is_aigc_true_is_honoured():
    kwargs = {**_INIT_KWARGS, "is_aigc": True}
    with mock.patch(
        "gomoveshift.publishing.tiktok_adapter.requests.post",
        return_value=_init_response(),
    ) as mock_post:
        init_direct_post("tok", **kwargs)
    assert mock_post.call_args.kwargs["json"]["post_info"]["is_aigc"] is True


def test_unaudited_public_request_never_calls_init():
    kwargs = {
        **_INIT_KWARGS,
        "privacy_level": "PUBLIC_TO_EVERYONE",
        "audit_approved": False,
        "creator_info": _OFFERS_ALL,
    }
    with mock.patch(
        "gomoveshift.publishing.tiktok_adapter.requests.post",
        return_value=_init_response(),
    ) as mock_post:
        with pytest.raises(TikTokAdapterError):
            init_direct_post("tok", **kwargs)
    mock_post.assert_not_called()


def test_init_failure_surfaces_error_code_message_and_log_id():
    envelope = {
        "error": {
            "code": "spam_risk_too_many_posts",
            "message": "Too many posts in a short period",
            "log_id": "abc123",
        }
    }
    with _post_returning(envelope):
        with pytest.raises(TikTokAdapterError) as exc_info:
            init_direct_post("tok", **_INIT_KWARGS)
    message = str(exc_info.value)
    assert "spam_risk_too_many_posts" in message
    assert "Too many posts in a short period" in message
    assert "abc123" in message


def test_init_failure_never_leaks_token_in_message():
    with mock.patch(
        "gomoveshift.publishing.tiktok_adapter.requests.post",
        side_effect=requests.ConnectionError("boom"),
    ):
        with pytest.raises(TikTokAdapterError) as exc_info:
            init_direct_post("super-secret-access-token", **_INIT_KWARGS)
    assert "super-secret-access-token" not in str(exc_info.value)


def test_init_error_body_echoing_token_does_not_leak_it():
    envelope = {
        "error": {
            "code": "internal_error",
            "message": "token super-secret-access-token was rejected",
            "log_id": "abc123",
        }
    }
    with _post_returning(envelope):
        with pytest.raises(TikTokAdapterError) as exc_info:
            init_direct_post("super-secret-access-token", **_INIT_KWARGS)
    assert "super-secret-access-token" not in str(exc_info.value)


# --- Unaudited-path probe modes -----------------------------------------

_POST = "gomoveshift.publishing.tiktok_adapter.requests.post"


@mock.patch(_POST, return_value=_init_response())
def test_default_mode_is_direct_and_byte_unchanged(mock_post):
    """Omitting `mode` hits the same URL/body as before (SELF_ONLY is
    already in `_INIT_KWARGS`) — existing call sites pass no `mode`."""
    init_direct_post("tok", **_INIT_KWARGS)
    assert mock_post.call_args.args[0] == INIT_URL
    assert mock_post.call_args.kwargs["json"]["post_info"]["privacy_level"] == "SELF_ONLY"


@mock.patch(_POST, return_value=_init_response())
@mock.patch("gomoveshift.publishing.tiktok_adapter.assert_public_routing_allowed")
def test_inbox_mode_posts_to_inbox_url_with_no_post_info_or_gate(mock_gate, mock_post):
    """Inbox uploads are never auto-published: no privacy_level, and the
    audience gate must not even run."""
    result = init_direct_post("tok", video_size=10 * MB, mode="inbox")
    assert mock_post.call_args.args[0] == INBOX_INIT_URL
    body = mock_post.call_args.kwargs["json"]
    assert "post_info" not in body and "privacy_level" not in body
    assert body["source_info"]["video_size"] == 10 * MB
    assert result["publish_id"] == "pub123"
    mock_gate.assert_not_called()


@mock.patch(_POST, return_value=_init_response())
def test_unknown_mode_rejected_before_any_request(mock_post):
    with pytest.raises(TikTokAdapterError, match="unknown init mode"):
        init_direct_post("tok", video_size=10 * MB, mode="bogus")
    mock_post.assert_not_called()


def test_init_rejects_bad_upload_host():
    with mock.patch(
        "gomoveshift.publishing.tiktok_adapter.requests.post",
        return_value=_init_response(upload_url="https://evil.example.com/u/1"),
    ):
        with pytest.raises(TikTokAdapterError):
            init_direct_post("tok", **_INIT_KWARGS)


def _ok_put_response():
    resp = mock.Mock()
    resp.raise_for_status = mock.Mock()
    return resp


def test_upload_chunks_sequential_order_and_ranges_with_trailing_fold():
    video_size = 16 * MB
    chunk_size = 5 * MB
    plan = plan_chunks(video_size, chunk_size=chunk_size)
    assert plan.total_chunk_count == 3  # 5 + 5 + (5 + 1 remainder fold)

    def chunk_reader(index):
        start = index * chunk_size
        is_last = index == plan.total_chunk_count - 1
        end = video_size if is_last else start + chunk_size
        return b"x" * (end - start)

    with mock.patch(
        "gomoveshift.publishing.tiktok_adapter.requests.put",
        return_value=_ok_put_response(),
    ) as mock_put:
        upload_video_chunks("https://open-upload.tiktokapis.com/u/1", plan, chunk_reader)

    assert mock_put.call_count == plan.total_chunk_count
    ranges = [call.kwargs["headers"]["Content-Range"] for call in mock_put.call_args_list]
    assert ranges == [
        f"bytes 0-{chunk_size - 1}/{video_size}",
        f"bytes {chunk_size}-{2 * chunk_size - 1}/{video_size}",
        f"bytes {2 * chunk_size}-{video_size - 1}/{video_size}",
    ]
    # trailing chunk folds the 1 MB remainder in, so it's larger than chunk_size
    last_length = int(mock_put.call_args_list[-1].kwargs["headers"]["Content-Length"])
    assert last_length == video_size - 2 * chunk_size
    assert last_length > chunk_size


def test_upload_chunks_surfaces_error_code_from_a_200_response():
    plan = plan_chunks(5 * MB)

    def chunk_reader(index):
        return b"x" * plan.chunk_size

    resp = mock.Mock()
    resp.raise_for_status = mock.Mock()
    resp.json.return_value = {
        "error": {
            "code": "upload_failure",
            "message": "checksum mismatch",
            "log_id": "chunk-log-1",
        }
    }
    with mock.patch(
        "gomoveshift.publishing.tiktok_adapter.requests.put",
        return_value=resp,
    ):
        with pytest.raises(TikTokAdapterError) as exc_info:
            upload_video_chunks("https://open-upload.tiktokapis.com/u/1", plan, chunk_reader)
    message = str(exc_info.value)
    assert "upload_failure" in message
    assert "checksum mismatch" in message
    assert "chunk-log-1" in message


def test_upload_chunks_bad_host_refused_before_bytes():
    plan = plan_chunks(5 * MB)
    chunk_reader = mock.Mock(side_effect=AssertionError("must not be called"))
    with mock.patch(
        "gomoveshift.publishing.tiktok_adapter.requests.put",
        return_value=_ok_put_response(),
    ) as mock_put:
        with pytest.raises(TikTokAdapterError):
            upload_video_chunks("https://evil.example.com/u/1", plan, chunk_reader)
    mock_put.assert_not_called()
    chunk_reader.assert_not_called()


# --- Publish-status poll loop -------------------------------------------------


def _status_response(status, privacy_level=None):
    resp = mock.Mock()
    resp.raise_for_status = mock.Mock()
    data = {"status": status}
    if privacy_level is not None:
        data["privacy_level"] = privacy_level
    resp.json.return_value = {"data": data}
    return resp


@pytest.mark.parametrize("terminal_status", ["PUBLISH_COMPLETE", "FAILED"])
def test_poll_stops_immediately_on_each_terminal_state(terminal_status):
    sleep = mock.Mock()
    with mock.patch(
        "gomoveshift.publishing.tiktok_adapter.requests.post",
        return_value=_status_response(terminal_status),
    ) as mock_post:
        data = poll_publish_status("tok", "pub123", sleep=sleep)
    assert data["status"] == terminal_status
    assert mock_post.call_count == 1
    sleep.assert_not_called()


def test_poll_keeps_polling_a_non_terminal_state_up_to_the_cap():
    sleep = mock.Mock()
    responses = [_status_response("PROCESSING_UPLOAD") for _ in range(3)] + [
        _status_response("PUBLISH_COMPLETE")
    ]
    with mock.patch(
        "gomoveshift.publishing.tiktok_adapter.requests.post",
        side_effect=responses,
    ) as mock_post:
        data = poll_publish_status("tok", "pub123", max_attempts=10, sleep=sleep)
    assert data["status"] == "PUBLISH_COMPLETE"
    assert mock_post.call_count == 4
    assert sleep.call_count == 3


def test_poll_exhausting_attempts_has_bounded_increasing_backoff_and_raises():
    sleep = mock.Mock()
    with mock.patch(
        "gomoveshift.publishing.tiktok_adapter.requests.post",
        return_value=_status_response("PROCESSING_UPLOAD"),
    ):
        with pytest.raises(TikTokPollTimeoutError) as exc_info:
            poll_publish_status("super-secret-token", "pub-xyz", max_attempts=5, sleep=sleep)
    waits = [call.args[0] for call in sleep.call_args_list]
    assert waits == sorted(waits)
    assert len(set(waits)) > 1
    assert max(waits) <= 30
    message = str(exc_info.value)
    assert "pub-xyz" in message
    assert "super-secret-token" not in message
    assert exc_info.value.publish_id == "pub-xyz"


def test_poll_failure_surfaces_error_code_message_and_log_id():
    envelope = {
        "error": {
            "code": "publish_id_invalid",
            "message": "publish_id does not exist",
            "log_id": "poll-log-1",
        }
    }
    with _post_returning(envelope):
        with pytest.raises(TikTokAdapterError) as exc_info:
            poll_publish_status("tok", "pub123", sleep=mock.Mock())
    message = str(exc_info.value)
    assert "publish_id_invalid" in message
    assert "publish_id does not exist" in message
    assert "poll-log-1" in message


def test_poll_error_body_echoing_token_does_not_leak_it():
    envelope = {
        "error": {
            "code": "internal_error",
            "message": "token super-secret-token was rejected",
            "log_id": "poll-log-1",
        }
    }
    with _post_returning(envelope):
        with pytest.raises(TikTokAdapterError) as exc_info:
            poll_publish_status("super-secret-token", "pub123", sleep=mock.Mock())
    assert "super-secret-token" not in str(exc_info.value)


def test_poll_request_failure_never_leaks_token_in_message():
    with mock.patch(
        "gomoveshift.publishing.tiktok_adapter.requests.post",
        side_effect=requests.ConnectionError("boom"),
    ):
        with pytest.raises(TikTokAdapterError) as exc_info:
            poll_publish_status("super-secret-token", "pub123", sleep=mock.Mock())
    assert "super-secret-token" not in str(exc_info.value)


# --- Post-publish privacy readback --------------------------------------------


def test_confirm_publish_treats_publish_complete_as_published():
    """THE status/fetch payload. For a published SELF_ONLY video the API returns, in full,
    ``{"status": "PUBLISH_COMPLETE"}`` — one key, no ``privacy_level``.

    The old readback compared a ``privacy_level`` that is never present, so
    it marked every genuinely successful publish ``failed_permanent``. That
    is the worst available outcome: the video is live on TikTok, the queue
    says permanently failed, and a human "fixing" it publishes a duplicate."""
    with mock.patch(
        "gomoveshift.publishing.tiktok_adapter.requests.post",
        return_value=_status_response("PUBLISH_COMPLETE"),
    ):
        outcome = confirm_publish("tok", "pub123", "SELF_ONLY", sleep=mock.Mock())
    assert outcome.delivery_status == "published"
    assert outcome.publish_id == "pub123"
    # The REQUESTED audience is retained for audit even though it cannot be
    # verified after the fact.
    assert outcome.requested_privacy_level == "SELF_ONLY"


def test_confirm_publish_ignores_a_privacy_level_if_one_ever_appears():
    """If TikTok ever starts returning ``privacy_level``, it must not
    silently become load-bearing again: the audience guarantee lives in
    ``assert_public_routing_allowed`` BEFORE the post is submitted, which is
    the only point where the harm can still be prevented."""
    with mock.patch(
        "gomoveshift.publishing.tiktok_adapter.requests.post",
        return_value=_status_response("PUBLISH_COMPLETE", privacy_level="PUBLIC_TO_EVERYONE"),
    ):
        outcome = confirm_publish("tok", "pub123", "SELF_ONLY", sleep=mock.Mock())
    assert outcome.delivery_status == "published"


def test_confirm_publish_failed_status_marks_failed_permanent():
    with mock.patch(
        "gomoveshift.publishing.tiktok_adapter.requests.post",
        return_value=_status_response("FAILED"),
    ):
        outcome = confirm_publish("tok", "pub123", "SELF_ONLY", sleep=mock.Mock())
    assert outcome.delivery_status == "failed_permanent"


def test_us_data_residency_upload_host_is_allowlisted():
    """TikTok returned open-upload.tiktokapis.us (US data-residency namespace)
    as the upload_url host, so the allowlist must include the .us names as well
    as the .com ones; otherwise a legitimate Direct Post dies at
    validate_upload_host."""
    validate_upload_host("https://open-upload.tiktokapis.us/upload/v1/abc?sig=x")


def test_allowlist_is_exact_match_not_suffix():
    """The whole point of the allowlist: a lookalike that merely ENDS WITH a
    real TikTok domain must still be refused."""
    for hostile in (
        "https://open-upload.tiktokapis.us.evil.com/upload",
        "https://evil-open-upload.tiktokapis.us/upload",
        "http://open-upload.tiktokapis.us/upload",
    ):
        with pytest.raises(TikTokAdapterError):
            validate_upload_host(hostile)


# ─── throttling: too many drafts already pending in the account inbox ─────

#: TikTok refuses a new inbox upload once too many uploaded drafts sit
#: unpublished in the account; every further upload is blocked until they are
#: cleared.
_SPAM_RISK = "spam_risk_too_many_pending_share"


def test_spam_risk_raises_the_throttled_subclass_not_a_generic_error():
    """Only a human publishing or dismissing those drafts clears this, so it
    must be distinguishable from an ordinary failure -- the drain exempts it
    from max_attempts on that basis."""
    body = '{"error": {"code": "spam_risk_too_many_pending_share", "message": "spam_risk_too_many_pending_share", "log_id": "LOGID"}}'.replace(
        "LOGID", _LOG_ID
    )
    with _post_raising_http(400, body):
        with pytest.raises(TikTokThrottledError) as excinfo:
            init_direct_post("tok", video_size=MB, mode="inbox")
    assert _SPAM_RISK in str(excinfo.value)


def test_throttled_error_is_still_an_adapter_error():
    """Widening the hierarchy must not change what an existing `except
    TikTokAdapterError` already catches."""
    assert issubclass(TikTokThrottledError, TikTokAdapterError)


def test_an_unrelated_400_is_not_treated_as_throttling():
    """The chunk-size 400 stays an
    ordinary retryable failure; exempting it from max_attempts would let a
    genuinely broken delivery retry forever."""
    body = '{"error": {"code": "invalid_params", "message": "The chunk size is invalid", "log_id": "LOGID"}}'.replace(
        "LOGID", _LOG_ID
    )
    with _post_raising_http(400, body):
        with pytest.raises(TikTokAdapterError) as excinfo:
            init_direct_post("tok", video_size=MB, mode="inbox")
    assert not isinstance(excinfo.value, TikTokThrottledError)


# --- video.list (read-only; pack-mode reconcile) -----------------------------

_LIST_ENVELOPE = {
    "data": {
        "videos": [{"id": "7000000000001", "create_time": 1790000000}],
        "cursor": 1789000000000,
        "has_more": True,
    },
    "error": {"code": "ok", "message": "", "log_id": "x"},
}


def test_list_videos_returns_videos_cursor_and_has_more():
    with _post_returning(_LIST_ENVELOPE):
        videos, cursor, has_more = list_videos("tok")
    assert [v["id"] for v in videos] == ["7000000000001"]
    assert cursor == 1789000000000
    assert has_more is True


def test_list_videos_posts_fields_in_the_url_and_omits_a_missing_cursor():
    with _post_returning(_LIST_ENVELOPE) as post:
        list_videos("tok", max_count=5)
    args, kwargs = post.call_args
    assert args[0].startswith("https://open.tiktokapis.com/v2/video/list/?fields=")
    assert "id,create_time,video_description,title,duration" in args[0]
    assert kwargs["json"] == {"max_count": 5}
    assert kwargs["headers"]["Authorization"] == "Bearer tok"


def test_list_videos_sends_the_cursor_it_is_given():
    with _post_returning(_LIST_ENVELOPE) as post:
        list_videos("tok", cursor=42)
    assert post.call_args.kwargs["json"] == {"max_count": 20, "cursor": 42}


def test_list_videos_http_failure_raises_without_the_token():
    body = '{"error": {"code": "scope_not_authorized", "message": "tok", "log_id": "L"}}'
    with _post_raising_http(401, body):
        with pytest.raises(TikTokAdapterError) as exc:
            list_videos("tok")
    assert "scope_not_authorized" in str(exc.value)
    assert "tok" not in str(exc.value).replace("scope_not_authorized", "")


def test_list_videos_with_an_empty_data_object_returns_nothing():
    with _post_returning({"error": {"code": "ok"}}):
        assert list_videos("tok") == ([], None, False)
