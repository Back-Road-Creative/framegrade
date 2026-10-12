"""A photo post and a carousel get a real publish path -- an adapter,
a delivery row, a receipt -- never a CSV row and a hope.

``_FakeClient`` stands in for ``photo_adapter.PhotoGraphClient`` so these
tests exercise the adapter's receipt/atomicity behaviour without any real
network/credential plumbing, mirroring ``test_publishing_handlers.py``'s
``_FakeAdapter`` pattern.
"""

from __future__ import annotations

import dataclasses

import pytest
from test_publishing_models import _record

from gomoveshift.publishing.dispatch import PermanentDeliveryError
from gomoveshift.publishing.handlers import build_handlers, make_photo_handler
from gomoveshift.publishing.models import (
    FORMAT_CAROUSEL,
    FORMAT_PHOTO,
    FORMAT_SHORT,
    MediaFingerprint,
)
from gomoveshift.publishing.photo_adapter import (
    PartialCarouselError,
    PhotoCarouselAdapter,
    PhotoPermanentError,
    UnsupportedPhotoFormatError,
)
from gomoveshift.publishing.queue import enqueue, find_delivery, open_queue_db


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def _extra_media(path: str = "/v/y.jpg", sha: str = "b" * 64) -> MediaFingerprint:
    return MediaFingerprint(path, 999, 0.0, 1080, 1080, "jpeg", sha)


def _row(
    conn, platform: str = "instagram", project_name: str = "example-2026-08", short_index: int = 3
) -> dict:
    delivery_id = enqueue(conn, project_name, short_index, platform, "hash-a")
    return dict(conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone())


class _FakeClient:
    """Configurable stand-in for ``PhotoGraphClient``."""

    def __init__(self, *, post_id="post123", permalink="https://ig/p/x", account_id="acct-1"):
        self.account_id = account_id
        self._post_id = post_id
        self._permalink = permalink
        self.item_calls: list[str] = []
        self.published_container = None
        self._fail_item_at: int | None = None

    def fail_item_at(self, index: int) -> None:
        self._fail_item_at = index

    def create_single_container(self, image_url: str, caption: str) -> str:
        return "container-single"

    def create_carousel_item_container(self, image_url: str) -> str:
        index = len(self.item_calls)
        self.item_calls.append(image_url)
        if self._fail_item_at is not None and index == self._fail_item_at:
            raise RuntimeError(f"upload failed for item {index}")
        return f"item-{index}"

    def create_carousel_parent_container(self, item_ids, caption: str) -> str:
        self.published_container = ("carousel", item_ids)
        return "container-carousel"

    def publish_container(self, container_id: str) -> dict:
        self.published_container = self.published_container or ("single", container_id)
        if self._post_id is None:
            return {}
        return {"post_id": self._post_id, "permalink": self._permalink}


def _photo_record(**over):
    return dataclasses.replace(_record(), format=FORMAT_PHOTO, **over)


def _carousel_record(n_extra: int = 2, **over):
    extras = tuple(
        _extra_media(path=f"/v/y{i}.jpg", sha=chr(ord("b") + i) * 64) for i in range(n_extra)
    )
    return dataclasses.replace(_record(), format=FORMAT_CAROUSEL, additional_media=extras, **over)


def _loader(media: MediaFingerprint) -> str:
    return f"https://gomoveshift.com/images/journeys/x/{media.path}"


# --- single photo --------------------------------------------------------


def test_single_photo_publishes_and_records_post_id(conn):
    client = _FakeClient(post_id="ig_9001", permalink="https://instagram.com/p/9001")
    adapter = PhotoCarouselAdapter(client, "instagram")
    row = _row(conn)
    result = adapter.publish(
        _photo_record(), conn=conn, delivery_id=row["id"], media_image_url_loader=_loader
    )

    assert result.post_id == "ig_9001"
    assert result.permalink == "https://instagram.com/p/9001"
    assert result.item_count == 1

    stored = find_delivery(conn, "example-2026-08", 3, "instagram")
    assert stored["status"] == "published"
    assert stored["platform_post_id"] == "ig_9001"
    assert stored["destination_account"] == "acct-1"


def test_no_post_id_is_a_failure_not_a_blank_receipt(conn):
    client = _FakeClient(post_id=None)
    adapter = PhotoCarouselAdapter(client, "instagram")
    row = _row(conn)

    with pytest.raises(PhotoPermanentError, match="no post id"):
        adapter.publish(
            _photo_record(), conn=conn, delivery_id=row["id"], media_image_url_loader=_loader
        )

    stored = find_delivery(conn, "example-2026-08", 3, "instagram")
    assert stored["status"] != "published"
    assert stored["platform_post_id"] is None


# --- carousel --------------------------------------------------------------


def test_carousel_of_n_items_publishes(conn):
    client = _FakeClient(post_id="ig_carousel_1")
    adapter = PhotoCarouselAdapter(client, "instagram")
    row = _row(conn)
    record = _carousel_record(n_extra=3)  # 1 (media) + 3 additional = 4 items

    result = adapter.publish(
        record, conn=conn, delivery_id=row["id"], media_image_url_loader=_loader
    )

    assert result.item_count == 4
    assert len(client.item_calls) == 4
    assert result.post_id == "ig_carousel_1"


def test_mid_carousel_failure_reports_index_and_succeeded_count(conn):
    client = _FakeClient()
    client.fail_item_at(2)  # third item (0-based) fails
    adapter = PhotoCarouselAdapter(client, "instagram")
    row = _row(conn)
    record = _carousel_record(n_extra=3)  # 4 items total, indices 0..3

    with pytest.raises(PartialCarouselError) as excinfo:
        adapter.publish(record, conn=conn, delivery_id=row["id"], media_image_url_loader=_loader)

    err = excinfo.value
    assert err.failed_index == 2
    assert err.succeeded == 2
    assert err.total == 4
    assert "2" in str(err)

    # Never a partial success on the delivery row.
    stored = find_delivery(conn, "example-2026-08", 3, "instagram")
    assert stored["status"] != "published"


def test_carousel_never_auto_retries_a_partial_failure_via_handler(conn):
    """Pins the atomicity decision: a partial carousel failure is ALWAYS
    permanent through the handler translation, never left retryable --
    documented in photo_adapter.PartialCarouselError's docstring."""
    client = _FakeClient()
    client.fail_item_at(0)
    adapter = PhotoCarouselAdapter(client, "instagram")
    handler = make_photo_handler(adapter, conn, lambda p, i: _carousel_record(n_extra=1), _loader)
    row = _row(conn)

    with pytest.raises(PermanentDeliveryError):
        handler(row)


# --- unsupported format ----------------------------------------------------


def test_unsupported_format_names_the_format(conn):
    client = _FakeClient()
    adapter = PhotoCarouselAdapter(client, "instagram")
    row = _row(conn)
    record = dataclasses.replace(_record(), format=FORMAT_SHORT)

    with pytest.raises(UnsupportedPhotoFormatError, match=FORMAT_SHORT):
        adapter.publish(record, conn=conn, delivery_id=row["id"], media_image_url_loader=_loader)


# --- handlers.build_handlers wiring -----------------------------------------


def test_build_handlers_routes_photo_format_to_photo_adapter(conn):
    client = _FakeClient(post_id="ig_routed")
    photo_adapter = PhotoCarouselAdapter(client, "instagram")
    handlers = build_handlers(
        conn,
        lambda p, i: _photo_record(),
        instagram_photo=(photo_adapter, _loader),
    )
    row = _row(conn)
    handlers["instagram"](row)

    stored = find_delivery(conn, "example-2026-08", 3, "instagram")
    assert stored["platform_post_id"] == "ig_routed"


def test_build_handlers_still_routes_shorts_when_photo_not_wired(conn):
    """Existing (video-only) wiring is untouched: no ``instagram_photo``
    means a short-format record still goes through the ordinary video
    handler path, exactly as before this feature existed."""
    calls = []

    class _FakeVideoAdapter:
        def publish(self, record, *, conn, delivery_id, video_bytes):
            calls.append((record, delivery_id))

    handlers = build_handlers(
        conn,
        lambda p, i: _record(),  # format=None -> short-shaped, not photo/carousel
        instagram=(_FakeVideoAdapter(), lambda record: b"bytes"),
    )
    row = _row(conn)
    handlers["instagram"](row)
    assert len(calls) == 1
