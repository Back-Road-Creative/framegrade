"""Privacy report per post -- one status per publish-record media
item, reusing ``_privacy_status``'s strings rather than re-deriving them.
Video (the only format the drive pipeline emits today) never gets a
per-item catalog entry, so it must read as "not tracked here", never
blocked and never dressed up as a per-item assurance. Photo/carousel
records defer to the recorded catalog entry, matched by exported filename.
"""

from pathlib import Path

from gomoveshift.console import library
from gomoveshift.pipeline.photos.ingest import BLUR_REGIONS_FOUND
from gomoveshift.publishing.models import FORMAT_CAROUSEL, FORMAT_PHOTO, FORMAT_SHORT


def test_video_media_is_not_tracked_per_item_and_never_blocks():
    status = library.privacy_status_for_media("/x/short.mp4", FORMAT_SHORT)
    assert status == library._PRIVACY_STATUS_VIDEO_NOT_TRACKED
    assert not library.privacy_status_blocks_publish(status)


def test_none_format_reads_the_same_as_short():
    # ``ShortPublishRecord.format`` defaults to ``None`` for every record
    # the drive pipeline builds today (plan_builder.py always sets
    # FORMAT_SHORT explicitly, but plan_io treats an absent key as
    # FORMAT_SHORT too) -- ``None`` must not read as "photo".
    status = library.privacy_status_for_media("/x/short.mp4", None)
    assert status == library._PRIVACY_STATUS_VIDEO_NOT_TRACKED


def test_photo_media_with_no_resolvable_site_reads_unrecorded_and_blocks():
    status = library.privacy_status_for_media("/x/photo-001.jpg", FORMAT_PHOTO, site_path=None)
    assert status == library._PRIVACY_STATUS_UNRECORDED
    assert library.privacy_status_blocks_publish(status)


def test_photo_media_with_no_matching_catalog_entry_blocks(tmp_path):
    (tmp_path / "content" / "journeys").mkdir(parents=True)
    status = library.privacy_status_for_media(
        "/x/photo-001.jpg", FORMAT_PHOTO, site_path=str(tmp_path)
    )
    assert status == library._PRIVACY_STATUS_UNRECORDED
    assert library.privacy_status_blocks_publish(status)


def test_carousel_media_matching_a_recorded_entry_does_not_block(monkeypatch):
    entry = library.CatalogEntry(
        catalog_id="photo:journey-a:2025-01-01:photo-001.jpg",
        media_type="photo",
        journey="journey-a",
        date="2025-01-01",
        blur_state=BLUR_REGIONS_FOUND,
    )
    monkeypatch.setattr(
        library,
        "read_catalog",
        lambda site_path: library.CatalogResult(entries=[entry]),
    )
    status = library.privacy_status_for_media(
        "/x/photo-001.jpg", FORMAT_CAROUSEL, site_path="/anything"
    )
    assert status == library._privacy_status(entry)
    assert not library.privacy_status_blocks_publish(status)


def test_unrecognised_recorded_status_blocks(monkeypatch):
    entry = library.CatalogEntry(
        catalog_id="photo:journey-a:2025-01-01:photo-001.jpg",
        media_type="photo",
        journey="journey-a",
        date="2025-01-01",
        blur_state="some-future-value",
    )
    monkeypatch.setattr(
        library,
        "read_catalog",
        lambda site_path: library.CatalogResult(entries=[entry]),
    )
    status = library.privacy_status_for_media(
        "/x/photo-001.jpg", FORMAT_PHOTO, site_path="/anything"
    )
    assert library.privacy_status_blocks_publish(status)
