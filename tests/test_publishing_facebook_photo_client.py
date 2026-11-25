"""Offline contracts for the unwired Page-photo byte transport."""

import hashlib
import traceback

import pytest
import requests

from gomoveshift.publishing.facebook_photo_client import (
    FacebookPagePhotoClient,
    PhotoCommentUnknownError,
    PhotoReadbackUnknownError,
    PhotoUploadUnknownError,
)

JPEG = b"\xff\xd8\xffsynthetic-test-bytes\xff\xd9"
SHA = hashlib.sha256(JPEG).hexdigest()
TOKEN = "test-private-token-never-in-errors"  # noqa: S105 -- synthetic sentinel


class Response:
    def __init__(self, body, status=200, version="v26.0"):
        self.body = body
        self.status_code = status
        self.headers = {"facebook-api-version": version}

    def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


class Session:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.adapters = {}

    def mount(self, prefix, adapter):
        self.adapters[prefix] = adapter

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def client(*responses, **credentials):
    session = Session(*responses)
    instance = FacebookPagePhotoClient(
        {"page_id": "123", "page_access_token": TOKEN, **credentials}, session=session
    )
    return instance, session


def upload(instance, **kwargs):
    return instance.upload_photo(
        JPEG,
        caption="weekend memories\n\nExact caption 🎵",
        expected_sha256=SHA,
        published=kwargs.pop("published", False),
        **kwargs,
    )


@pytest.mark.parametrize("published", [False, True])
def test_upload_exact_private_bytes_explicit_intent_and_versions(published):
    instance, session = client(Response({"id": "456"}))
    receipt = upload(instance, published=published)
    assert (receipt.photo_id, receipt.page_id, receipt.sha256) == ("456", "123", SHA)
    assert receipt.published_requested is published
    assert (receipt.requested_version, receipt.response_version) == ("v21.0", "v26.0")
    method, url, options = session.calls[0]
    assert (method, url) == ("POST", "https://graph.facebook.com/v21.0/123/photos")
    assert options["data"] == {
        "caption": "weekend memories\n\nExact caption 🎵",
        "published": "true" if published else "false",
    }
    assert options["files"] == {"source": ("photo.jpg", JPEG, "image/jpeg")}
    assert options["headers"] == {"Authorization": f"Bearer {TOKEN}"}
    assert options["allow_redirects"] is False
    assert options["timeout"] == (10, 60)
    assert session.adapters["https://"].max_retries.total == 0
    assert session.trust_env is False
    assert len(session.calls) == 1


@pytest.mark.parametrize("published", [None, "false", "true", 0, 1])
def test_intent_requires_actual_boolean_before_network(published):
    instance, session = client()
    with pytest.raises(ValueError):
        upload(instance, published=published)
    assert session.calls == []


def test_omitted_intent_refused():
    instance, session = client()
    with pytest.raises(TypeError):
        instance.upload_photo(JPEG, caption="", expected_sha256=SHA)
    assert session.calls == []


@pytest.mark.parametrize(
    "data,digest,caption",
    [
        (bytearray(JPEG), SHA, "ok"),
        (b"", SHA, "ok"),
        (b"not jpeg", SHA, "ok"),
        (JPEG, "0" * 64, "ok"),
        (JPEG, SHA, None),
        (JPEG, SHA.upper(), "ok"),
    ],
)
def test_invalid_snapshot_or_caption_never_sends(data, digest, caption):
    instance, session = client()
    with pytest.raises(ValueError):
        instance.upload_photo(data, caption=caption, expected_sha256=digest, published=False)
    assert session.calls == []


@pytest.mark.parametrize(
    "credentials",
    [
        {"page_id": "../me"},
        {"page_id": 123},
        {"page_id": "１２３"},
        {"page_access_token": ""},
        {"graph_version": "v21.0/../me"},
    ],
)
def test_invalid_destination_or_credentials_refused(credentials):
    with pytest.raises(ValueError):
        client(**credentials)


@pytest.mark.parametrize(
    "response",
    [
        requests.Timeout(TOKEN),
        Response(ValueError(TOKEN)),
        Response([]),
        Response(None),
        Response({}),
        Response({"id": "../me"}),
        Response({"id": 456}),
        Response({"error": {"message": TOKEN}}, 200),
        Response({"error": {"message": TOKEN}}, 400),
        Response({"id": "456"}, 302),
        Response({"id": "456"}, 500),
    ],
)
def test_ambiguous_or_error_upload_never_retries_or_leaks(response):
    instance, session = client(response)
    with pytest.raises(PhotoUploadUnknownError) as caught:
        upload(instance)
    assert caught.value.page_id == "123"
    assert caught.value.photo_id is None
    assert len(session.calls) == 1
    assert TOKEN not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert "retry" not in str(caught.value).lower()


@pytest.mark.parametrize("published", [False, True])
def test_readback_exact_photo_story_owner_and_strict_status(published):
    instance, session = client(
        Response({"id": "456"}),
        Response({"id": "456", "page_story_id": "123_789"}),
        Response({"id": "123_789", "is_published": published}),
        Response({"id": "123_789", "from": {"id": "123"}}),
    )
    receipt = upload(instance)
    status = instance.read_status(receipt)
    assert (status.photo_id, status.story_id, status.page_id) == ("456", "123_789", "123")
    assert status.is_published is published
    assert receipt.published_requested is False  # observation is independent of request
    assert [call[2]["params"]["fields"] for call in session.calls[1:]] == [
        "id,page_story_id",
        "id,is_published",
        "id,from",
    ]
    assert all(call[0] == "GET" for call in session.calls[1:])
    assert all(call[2]["allow_redirects"] is False for call in session.calls)


@pytest.mark.parametrize(
    "photo",
    [
        {},
        {"id": "wrong", "page_story_id": "123_789"},
        {"id": "456", "page_story_id": "999_789"},
        {"id": "456", "page_story_id": "123_789?token=bad"},
    ],
)
def test_unknown_photo_link_preserves_photo_id_and_stops(photo):
    instance, session = client(Response({"id": "456"}), Response(photo))
    receipt = upload(instance)
    with pytest.raises(PhotoReadbackUnknownError) as caught:
        instance.read_status(receipt)
    assert caught.value.photo_id == "456"
    assert caught.value.story_id is None
    assert len(session.calls) == 2


@pytest.mark.parametrize(
    "story",
    [
        {},
        {"id": "123_789"},
        {"id": "123_790", "is_published": True},
        {"id": "123_789", "is_published": "false"},
        {"id": "123_789", "is_published": 1},
    ],
)
def test_unknown_story_status_preserves_both_ids(story):
    instance, session = client(
        Response({"id": "456"}),
        Response({"id": "456", "page_story_id": "123_789"}),
        Response(story),
    )
    with pytest.raises(PhotoReadbackUnknownError) as caught:
        instance.read_status(upload(instance))
    assert (caught.value.photo_id, caught.value.story_id) == ("456", "123_789")
    assert len(session.calls) == 3


@pytest.mark.parametrize(
    "owner",
    [
        {},
        {"id": "123_789", "from": {"id": "999"}},
        {"id": "123_790", "from": {"id": "123"}},
        {"id": "123_789", "from": None},
        {"id": "123_789", "from": {"id": 123}},
        requests.Timeout(TOKEN),
        Response({"error": {"message": TOKEN}}, 400),
    ],
)
def test_owner_read_failure_preserves_ids_without_another_upload(owner):
    if isinstance(owner, dict):
        owner = Response(owner)
    instance, session = client(
        Response({"id": "456"}),
        Response({"id": "456", "page_story_id": "123_789"}),
        Response({"id": "123_789", "is_published": False}),
        owner,
    )
    receipt = upload(instance)
    with pytest.raises(PhotoReadbackUnknownError) as caught:
        instance.read_status(receipt)
    assert (caught.value.photo_id, caught.value.story_id, caught.value.page_id) == (
        "456",
        "123_789",
        "123",
    )
    assert TOKEN not in "".join(traceback.format_exception(caught.value))
    assert len(session.calls) == 4
    assert sum(call[0] == "POST" for call in session.calls) == 1


def test_account_rotation_refuses_status_network():
    first, _ = client(Response({"id": "456"}))
    receipt = upload(first)
    second, session = client(page_id="999")
    with pytest.raises(ValueError):
        second.read_status(receipt)
    assert session.calls == []


def test_configured_version_used_and_untrusted_response_version_discarded():
    instance, session = client(Response({"id": "456"}, version=TOKEN), graph_version="v26.0")
    receipt = upload(instance)
    assert receipt.requested_version == "v26.0"
    assert receipt.response_version is None
    assert session.calls[0][1].startswith("https://graph.facebook.com/v26.0/")


def test_default_session_disables_transport_retries_without_network():
    instance = FacebookPagePhotoClient({"page_id": "123", "page_access_token": TOKEN})
    assert instance.account_id == "123"
    assert instance._session.trust_env is False
    adapter = instance._session.get_adapter("https://graph.facebook.com/v21.0/123/photos")
    assert adapter.max_retries.total == 0
    instance._session.close()


def test_readback_can_be_repeated_without_repeating_upload():
    instance, session = client(
        Response({"id": "456"}),
        requests.Timeout(TOKEN),
        Response({"id": "456", "page_story_id": "123_789"}),
        Response({"id": "123_789", "is_published": False}),
        Response({"id": "123_789", "from": {"id": "123"}}),
    )
    receipt = upload(instance)
    with pytest.raises(PhotoReadbackUnknownError) as caught:
        instance.read_status(receipt)
    assert caught.value.photo_id == receipt.photo_id
    assert caught.value.__context__ is None
    assert instance.read_status(receipt).is_published is False
    assert sum(call[0] == "POST" for call in session.calls) == 1


COMMENT = "Day 9 on the journal: https://gomoveshift.com/journeys/ireland-2026-11/#day-9"


def comments(*items, after=None):
    body = {"data": list(items)}
    if after:
        body["paging"] = {"cursors": {"after": after}, "next": "https://elsewhere.example/x"}
    return Response(body)


def test_comment_reads_then_posts_once_with_bearer_and_no_redirects():
    instance, session = client(comments(), Response({"id": "902_77"}))
    assert instance.post_comment("123_902", COMMENT) == "902_77"
    (read, _, read_opts), (write, url, write_opts) = session.calls
    assert read == "GET" and read_opts["params"]["fields"] == "id,message,from"
    assert (write, url) == ("POST", "https://graph.facebook.com/v21.0/123_902/comments")
    assert write_opts["data"] == {"message": COMMENT}
    for _, call_url, options in session.calls:
        assert options["allow_redirects"] is False
        assert options["headers"] == {"Authorization": f"Bearer {TOKEN}"}
        assert TOKEN not in call_url and "access_token" not in str(options.get("params"))


def test_existing_own_comment_is_returned_without_posting_across_pages():
    stranger = {"id": "902_1", "message": COMMENT, "from": {"id": "999"}}
    ours = {"id": "902_2", "message": COMMENT, "from": {"id": "123"}}
    instance, session = client(comments(stranger, after="NEXT"), comments(ours))
    assert instance.post_comment("123_902", COMMENT) == "902_2"
    assert [c[0] for c in session.calls] == ["GET", "GET"]
    assert session.calls[1][2]["params"]["after"] == "NEXT"
    # The cursor is sent to our own base URL; the "next" link is never followed.
    assert all(c[1].startswith("https://graph.facebook.com/") for c in session.calls)


@pytest.mark.parametrize(
    "story,text",
    [
        ("999_902", COMMENT),
        ("123", COMMENT),
        ("123_902", ""),
        ("123_902", "  \n"),
        ("123_902", "x" * 1001),
        ("123_902", "tab\there"),
        ("123_902", None),
    ],
)
def test_invalid_comment_input_fails_before_network(story, text):
    instance, session = client()
    with pytest.raises(ValueError):
        instance.post_comment(story, text)
    assert session.calls == []


@pytest.mark.parametrize(
    "responses",
    [
        [Response({"error": {"message": TOKEN}}, status=403)],
        [Response({"nope": []})],
        [requests.ConnectionError(TOKEN)],
        [comments(), Response({"error": {"code": 200}}, status=400)],
        [comments(), Response({"id": "not an id"})],
        [comments(), requests.Timeout(TOKEN)],
    ],
)
def test_unconfirmed_comment_raises_without_sensitive_details(responses):
    instance, session = client(*responses)
    with pytest.raises(PhotoCommentUnknownError) as caught:
        instance.post_comment("123_902", COMMENT)
    assert caught.value.story_id == "123_902" and caught.value.__cause__ is None
    rendered = "".join(traceback.format_exception(caught.value))
    assert TOKEN not in rendered and COMMENT not in rendered
    if responses[0].__class__ is not Response or responses[0].status_code != 200:
        assert len(session.calls) == 1  # a failed read never posts
