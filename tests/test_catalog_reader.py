"""Read-only media-catalog reader: given a site checkout path, return
catalog entries for photos (journey markdown, ``photo_hashes``/``photo_provenance``)
and videos (``data/video_routing.yaml``) that already exist. Nothing writes,
nothing decodes, no network.
"""

import textwrap

import pytest

from gomoveshift.catalog import CatalogEntry, CatalogSourceUnavailableError, read_catalog

BAD_YAML_JOURNEY = textwrap.dedent("""\
    ---
    title: [unterminated
    days: []
    ---
    Body text
""")

NO_DELIMITER_JOURNEY = "not a journey file at all\n"

JOURNEY_A = textwrap.dedent("""\
    ---
    title: Journey A
    photo_count: 2
    days:
    - day_number: 1
      date: '2025-01-01'
      photos:
      - location: Place A
        photos:
        - photo-001.jpg
        - photo-legacy.jpg
        photo_hashes:
          photo-001.jpg: abc123
    - day_number: 2
      date: '2025-01-02'
      photos: []
    photo_provenance:
      photo-001.jpg: raw_decoder
    ---
    Body text
""")

JOURNEY_B = textwrap.dedent("""\
    ---
    title: Journey B
    days:
    - day_number: 1
      date: '2025-02-01'
      photos: []
    ---
    Body text
""")

ROUTING_YAML = textwrap.dedent("""\
    routes:
      vid-1:
        journey: journey-a
        date: '2025-01-01'
      vid-2:
        journey: journey-b
        date: '2025-02-01'
      vid-bad:
        journey: journey-a
    """)


def _site(tmp_path, *, journeys=True, routing=True):
    site = tmp_path / "site"
    if journeys:
        jdir = site / "content" / "journeys"
        jdir.mkdir(parents=True)
        (jdir / "journey-a.md").write_text(JOURNEY_A)
        (jdir / "journey-b.md").write_text(JOURNEY_B)
    if routing:
        (site / "data").mkdir(parents=True, exist_ok=True)
        (site / "data" / "video_routing.yaml").write_text(ROUTING_YAML)
    return site


def test_mixed_collection_reads_back(tmp_path):
    site = _site(tmp_path)
    entries = read_catalog(site).entries
    assert {e.media_type for e in entries} == {"photo", "video"}
    photo_ids = {e.catalog_id for e in entries if e.media_type == "photo"}
    assert len(photo_ids) == 2
    video_ids = {e.catalog_id for e in entries if e.media_type == "video"}
    # vid-bad has no date, so load_routing's own contract drops it.
    assert len(video_ids) == 2


def test_legacy_photo_no_hash_is_recorded_false_not_dropped(tmp_path):
    site = _site(tmp_path)
    entries = read_catalog(site, media_type="photo").entries
    legacy = next(e for e in entries if "photo-legacy.jpg" in e.catalog_id)
    assert legacy.hash_recorded is False
    assert legacy.content_hash is None

    hashed = next(e for e in entries if "photo-001.jpg" in e.catalog_id)
    assert hashed.hash_recorded is True
    assert hashed.content_hash == "abc123"
    assert hashed.decoder == "raw_decoder"


def test_missing_routing_file_yields_no_videos_no_error(tmp_path):
    site = _site(tmp_path, routing=False)
    entries = read_catalog(site).entries
    assert all(e.media_type != "video" for e in entries)
    assert any(e.media_type == "photo" for e in entries)


def test_missing_site_path_is_unavailable_not_empty(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(CatalogSourceUnavailableError):
        read_catalog(missing)


def test_missing_site_path_distinguishable_from_empty_catalog(tmp_path):
    # A real site with journeys dir but zero entries is a legitimate empty
    # catalog — not an error.
    site = tmp_path / "empty-site"
    (site / "content" / "journeys").mkdir(parents=True)
    result = read_catalog(site)
    assert result.entries == []
    assert result.damaged == []


def test_filter_by_type(tmp_path):
    site = _site(tmp_path)
    photos = read_catalog(site, media_type="photo").entries
    assert photos and all(e.media_type == "photo" for e in photos)
    videos = read_catalog(site, media_type="video").entries
    assert videos and all(e.media_type == "video" for e in videos)


def test_filter_by_date(tmp_path):
    site = _site(tmp_path)
    entries = read_catalog(site, date="2025-01-01").entries
    assert entries
    assert all(e.date == "2025-01-01" for e in entries)
    assert any(e.media_type == "photo" for e in entries)
    assert any(e.media_type == "video" for e in entries)


def test_filter_by_journey(tmp_path):
    site = _site(tmp_path)
    entries = read_catalog(site, journey="journey-b").entries
    assert entries
    assert all(e.journey == "journey-b" for e in entries)


def test_entry_is_frozen_dataclass_with_expected_fields(tmp_path):
    site = _site(tmp_path)
    entry = read_catalog(site, media_type="video").entries[0]
    assert isinstance(entry, CatalogEntry)
    assert entry.routing_id is not None
    with pytest.raises(Exception):
        entry.catalog_id = "mutated"


def test_empty_journey_is_not_reported_as_damaged(tmp_path):
    # journey-b has zero photos — a real, empty subset, never damage.
    site = _site(tmp_path)
    result = read_catalog(site, media_type="photo")
    assert result.damaged == []
    assert any(e.journey == "journey-b" for e in result.entries) is False


def test_damaged_journey_is_visible_and_others_still_read(tmp_path):
    site = _site(tmp_path)
    jdir = site / "content" / "journeys"
    (jdir / "journey-bad-yaml.md").write_text(BAD_YAML_JOURNEY)
    (jdir / "journey-no-delim.md").write_text(NO_DELIMITER_JOURNEY)

    result = read_catalog(site, media_type="photo")

    damaged_paths = {d.path for d in result.damaged}
    assert any("journey-bad-yaml.md" in p for p in damaged_paths)
    assert any("journey-no-delim.md" in p for p in damaged_paths)
    assert len(result.damaged) == 2
    for d in result.damaged:
        assert d.error

    # The two good journeys are still fully present.
    photo_ids = {e.catalog_id for e in result.entries}
    assert len(photo_ids) == 2


BAD_YAML_ROUTING = "routes:\n  vid-1:\n    journey: [unterminated\n"
NOT_A_MAPPING_ROUTING = "- vid-1\n- vid-2\n"


def _site_with_routing(tmp_path, text: str):
    site = _site(tmp_path, routing=False)
    (site / "data").mkdir(parents=True, exist_ok=True)
    (site / "data" / "video_routing.yaml").write_text(text)
    return site


def test_a_corrupt_routing_file_does_not_throw_away_the_photo_catalog(tmp_path):
    """The photo side already degrades one journey at a time. The video side
    had no error handling at all, so a single malformed video_routing.yaml
    aborted the whole read -- including every photo entry already gathered --
    and reached the console as a 500 rather than a visible state."""
    site = _site_with_routing(tmp_path, BAD_YAML_ROUTING)
    result = read_catalog(site)
    assert [e.catalog_id for e in result.entries if e.media_type == "photo"], (
        "a damaged routing file took the readable photo entries down with it"
    )
    assert [e for e in result.entries if e.media_type == "video"] == []
    assert any("video_routing.yaml" in d.path for d in result.damaged)


def test_the_damaged_routing_file_says_why(tmp_path):
    site = _site_with_routing(tmp_path, BAD_YAML_ROUTING)
    damaged = [d for d in read_catalog(site).damaged if "video_routing.yaml" in d.path]
    assert len(damaged) == 1
    assert damaged[0].error, "a damaged source with no reason is not actionable"


def test_a_routing_document_that_is_not_a_mapping_is_no_routes_not_a_crash(tmp_path):
    """``(yaml.safe_load(...) or {}).get("routes")`` raised AttributeError on a
    YAML document that parses to a list, which is neither of load_routing's two
    documented outcomes (missing file, malformed rows skipped) and crashed both
    of its callers."""
    site = _site_with_routing(tmp_path, NOT_A_MAPPING_ROUTING)
    result = read_catalog(site)
    assert [e for e in result.entries if e.media_type == "video"] == []
    assert [e for e in result.entries if e.media_type == "photo"]


def test_asking_only_for_videos_still_reports_the_damage(tmp_path):
    """A media_type filter must not hide the reason the answer is empty --
    that is precisely the "nothing here" vs "something broke" confusion the
    damaged list exists to prevent."""
    site = _site_with_routing(tmp_path, BAD_YAML_ROUTING)
    result = read_catalog(site, media_type="video")
    assert result.entries == []
    assert any("video_routing.yaml" in d.path for d in result.damaged)


def test_asking_only_for_photos_does_not_report_routing_damage(tmp_path):
    """The routing file is not read at all for a photo-only request, so
    claiming it is damaged would be a finding this read never made."""
    site = _site_with_routing(tmp_path, BAD_YAML_ROUTING)
    assert read_catalog(site, media_type="photo").damaged == []


def test_a_healthy_routing_file_is_never_reported_as_damaged(tmp_path):
    """The accept-case for all of the above: a guard with no accept-case is a
    guard that breaks the feature."""
    result = read_catalog(_site(tmp_path))
    assert result.damaged == []
    assert [e for e in result.entries if e.media_type == "video"]


def test_load_routing_returns_no_routes_for_a_non_mapping_document(tmp_path):
    """Pinned at the source, not only through the catalog: reconcile_journeys
    promises it "never raises on a routing/data inconsistency" and is the
    other caller of this function."""
    from gomoveshift.pipeline.site_sync.journey_reconcile import load_routing

    path = tmp_path / "video_routing.yaml"
    path.write_text(NOT_A_MAPPING_ROUTING)
    assert load_routing(path) == {}
    path.write_text("routes:\n  - vid-1\n")
    assert load_routing(path) == {}
