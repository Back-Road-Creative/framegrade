"""Tests for zero-match draft journey creation.

``resolve_journey_by_date`` returning ``None`` is the clean seam where a
minimal ``draft: true`` stub is created rather than leaving the run
unpublishable. No fabricated prose — only facts derivable from the
run (region, date) go into frontmatter.
"""

from __future__ import annotations

from pathlib import Path

from gomoveshift.pipeline.journal_publisher import load_journey
from gomoveshift.pipeline.site_sync.journey_by_date import (
    JourneyAlreadyExistsError,
    create_draft_journey,
    derive_journey_slug,
)


def test_derive_journey_slug_is_deterministic():
    assert derive_journey_slug("usa", "2026-05-01") == "usa-2026-05-01"
    assert derive_journey_slug("usa", "2026-05-01") == derive_journey_slug("usa", "2026-05-01")


def test_derive_journey_slug_normalizes_region():
    assert derive_journey_slug("American East", "2026-05-01") == "american-east-2026-05-01"


def test_create_draft_journey_writes_draft_true(tmp_path):
    site = tmp_path / "site"
    slug = create_draft_journey(site, "usa", "2026-05-01")
    journey_path = site / "content" / "journeys" / f"{slug}.md"
    assert journey_path.is_file()
    doc, _body = load_journey(journey_path)
    assert doc["draft"] is True
    assert doc["trip"]["start_date"] == "2026-05-01"
    assert doc["trip"]["end_date"] == "2026-05-01"
    assert len(doc["days"]) == 1
    assert doc["days"][0]["date"] == "2026-05-01"


def test_create_draft_journey_never_fabricates_prose(tmp_path):
    site = tmp_path / "site"
    slug = create_draft_journey(site, "usa", "2026-05-01")
    doc, _body = load_journey(site / "content" / "journeys" / f"{slug}.md")
    assert doc["title"] == ""
    assert doc["description"] == ""
    day = doc["days"][0]
    assert day["heading"] == ""
    assert day["narrative"] == ""
    assert day["locations"] == []


def test_create_draft_journey_refuses_to_overwrite_existing(tmp_path):
    site = tmp_path / "site"
    jdir = site / "content" / "journeys"
    jdir.mkdir(parents=True)
    slug = derive_journey_slug("usa", "2026-05-01")
    existing = jdir / f"{slug}.md"
    existing.write_text("---\ntitle: existing\n---\nbody\n")

    try:
        create_draft_journey(site, "usa", "2026-05-01")
        raise AssertionError("expected JourneyAlreadyExistsError")
    except JourneyAlreadyExistsError:
        pass

    assert existing.read_text() == "---\ntitle: existing\n---\nbody\n"


def test_create_draft_journey_no_dangling_featured_image(tmp_path):
    site = tmp_path / "site"
    slug = create_draft_journey(site, "usa", "2026-05-01")
    doc, _body = load_journey(site / "content" / "journeys" / f"{slug}.md")
    assert "featured_image" not in doc
