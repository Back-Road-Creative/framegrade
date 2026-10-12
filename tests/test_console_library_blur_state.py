"""The catalog reports a real per-photo blur outcome
(``CatalogEntry.blur_state``, ``catalog/reader.py``). The console must show
what was actually recorded -- four distinct states, never collapsed into
three, and never dressed up as an assurance this reader never made.
"""

from gomoveshift.console import library
from gomoveshift.pipeline.photos.ingest import (
    BLUR_NO_REGIONS,
    BLUR_REGIONS_FOUND,
    BLUR_SKIPPED,
)


def _entry(media_type="photo", blur_state=None):
    return library.CatalogEntry(
        catalog_id="photo:journey-a:2025-01-01:photo-001.jpg",
        media_type=media_type,
        journey="journey-a",
        date="2025-01-01",
        blur_state=blur_state,
    )


def test_unrecorded_blur_state_keeps_the_existing_wording():
    status = library._privacy_status(_entry(blur_state=None))
    assert status == library._PRIVACY_STATUS_UNRECORDED


def test_regions_found_is_reported_distinctly():
    status = library._privacy_status(_entry(blur_state=BLUR_REGIONS_FOUND))
    assert status != library._PRIVACY_STATUS_UNRECORDED
    lowered = status.lower()
    assert "verified" not in lowered
    assert "clean" not in lowered
    assert "safe" not in lowered


def test_no_regions_found_is_reported_distinctly_from_regions_found():
    found = library._privacy_status(_entry(blur_state=BLUR_REGIONS_FOUND))
    no_regions = library._privacy_status(_entry(blur_state=BLUR_NO_REGIONS))
    assert found != no_regions
    assert no_regions != library._PRIVACY_STATUS_UNRECORDED
    lowered = no_regions.lower()
    assert "verified" not in lowered
    assert "clean" not in lowered
    assert "safe" not in lowered


def test_skipped_reads_as_a_risk_not_a_normal_outcome():
    status = library._privacy_status(_entry(blur_state=BLUR_SKIPPED))
    lowered = status.lower()
    assert "verified" not in lowered
    assert "clean" not in lowered
    assert "safe" not in lowered
    # An operator reading this must understand the export shipped unblurred
    # by request -- a risk, not a status to skim past.
    assert "unblurred" in lowered or "not blurred" in lowered


def test_unrecognised_blur_state_does_not_crash_and_names_the_raw_value():
    status = library._privacy_status(_entry(blur_state="some-future-value"))
    assert "some-future-value" in status
    lowered = status.lower()
    assert "verified" not in lowered
    assert "clean" not in lowered
    assert "safe" not in lowered
    # Must not silently read as any of the three known states.
    assert status != library._privacy_status(_entry(blur_state=BLUR_REGIONS_FOUND))
    assert status != library._privacy_status(_entry(blur_state=BLUR_NO_REGIONS))
    assert status != library._privacy_status(_entry(blur_state=BLUR_SKIPPED))


def test_video_entry_never_reads_as_a_privacy_claim():
    # reader.py never populates blur_state for video entries -- it stays
    # None, and that must land on the same honest "not recorded" state as
    # a photo whose catalog predates this field, never a recorded outcome.
    status = library._privacy_status(_entry(media_type="video", blur_state=None))
    assert status == library._PRIVACY_STATUS_UNRECORDED
