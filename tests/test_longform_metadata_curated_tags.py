"""TDD tests for curated whole-phrase tags from ``generate()``.

The SEO Manager's tag output word-splits the location string (e.g. it
returns ``["dingle", "peninsula", "toward"]`` for
``"Dingle Peninsula toward Gortmore, County Kerry"``). Shipping those
fragments as YouTube tags violates content integrity. ``generate()``
must build its ``tags`` list from ``GMS_BASE_TAGS`` + the curated
``_location_tags`` whole-phrase builder, exactly like the basic
fallback path already does — and must DISCARD the SEO manager's
single-word ``metadata.tags`` fragments.

Update: the wholesale discard is reversed for the optimizer's
MULTI-WORD analytics phrases only — those carry real search value and
now survive (`_phrase_tags`); single-word fragments stay dropped.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline.longform_metadata import (
    GMS_BASE_TAGS,
    LongFormMetadataGenerator,
    _cap_youtube_tags,
    _location_tags,
    _quoted_tag_width,
)


class _BasicStubGenerator(LongFormMetadataGenerator):
    """SEO Manager disabled — exercises ``_generate_basic``."""

    def _init_seo_manager(self):  # pragma: no cover - simple override
        self._manager = None
        self._optimizer = None
        self._seo_degraded = True

    def _init_ledger(self):  # pragma: no cover - skip ledger IO in tests
        self._ledger = None
        self._ledger_path = None


class _SeoManagerStubGenerator(LongFormMetadataGenerator):
    """SEO Manager path with controllable ``metadata.tags`` output.

    The SEO Manager (real or stub) is the source of the word-split
    bug. Mocking ``_manager`` + ``_optimizer`` lets us drive the
    ``_generate_with_seo_manager`` branch and assert the curated-tag
    contract.
    """

    seo_tags: list[str] = []

    def _init_seo_manager(self):  # pragma: no cover - simple override
        manager = MagicMock()
        manager.store.query_by_site.return_value = []

        def _make_metadata(*_args: Any, **_kwargs: Any) -> Any:
            meta = MagicMock()
            meta.title = "Drive Through The Place"
            meta.description = "A scenic drive."
            meta.tags = list(self.seo_tags)
            meta.keyword_density = 0.0
            meta.readability_score = 0.0
            meta.content_id = "stub-content-id"
            return meta

        manager.generate.side_effect = _make_metadata
        manager.save.return_value = "stub-event-id"
        self._manager = manager

        optimizer = MagicMock()
        optimizer.analyze_patterns.return_value = None
        optimizer.optimize_metadata.side_effect = _make_metadata
        self._optimizer = optimizer

        self._seo_degraded = False

    def _init_ledger(self):  # pragma: no cover - skip ledger IO in tests
        self._ledger = None
        self._ledger_path = None

    def _sync_youtube_analytics(self):  # pragma: no cover - skip network
        return None


@pytest.fixture
def basic_gen(tmp_path):
    return _BasicStubGenerator(seo_storage_path=tmp_path)


@pytest.fixture
def seo_gen(tmp_path):
    return _SeoManagerStubGenerator(seo_storage_path=tmp_path)


LOCATION = "Dingle Peninsula toward Gortmore, County Kerry"
RECORDING_DATE = "2025-09-14"


class TestCuratedTagsFromGenerate:
    def test_generate_does_not_emit_word_split_tags(self, seo_gen):
        """Single-word fragments from word-splitting must never appear."""
        type(seo_gen).seo_tags = ["dingle", "peninsula", "toward"]
        result = seo_gen.generate(
            location=LOCATION,
            video_duration=1800.0,
            project_name="t",
            recording_date=RECORDING_DATE,
        )
        tags_lower = [t.lower() for t in result["tags"]]
        assert "toward" not in tags_lower
        assert "peninsula" not in tags_lower
        assert "dingle" not in tags_lower
        # The whole-phrase component must be there.
        assert any("dingle peninsula toward gortmore" in t.lower() for t in result["tags"])
        assert any("county kerry" in t.lower() for t in result["tags"])

    def test_generate_includes_gms_base_tags(self, seo_gen):
        type(seo_gen).seo_tags = []
        result = seo_gen.generate(
            location=LOCATION,
            video_duration=1800.0,
            project_name="t",
            recording_date=RECORDING_DATE,
        )
        for base in GMS_BASE_TAGS:
            assert base in result["tags"], f"missing base tag: {base}"

    def test_generate_year_only_for_recording_date(self, seo_gen):
        type(seo_gen).seo_tags = []
        result = seo_gen.generate(
            location=LOCATION,
            video_duration=1800.0,
            project_name="t",
            recording_date=RECORDING_DATE,
        )
        assert "2025" in result["tags"]
        assert "2025-09-14" not in result["tags"]
        assert "09" not in result["tags"]
        assert "14" not in result["tags"]

    def test_generate_and_generate_for_lf_have_matching_tag_shape(self, basic_gen, seo_gen):
        """Both code paths (SEO-manager + basic fallback) must produce
        the same curated tag shape — neither emits fragments the other
        doesn't.
        """
        type(seo_gen).seo_tags = ["dingle", "peninsula", "toward", "irrelevant"]
        seo_result = seo_gen.generate(
            location=LOCATION,
            video_duration=1800.0,
            project_name="t",
            recording_date=RECORDING_DATE,
        )
        basic_result = basic_gen.generate(
            location=LOCATION,
            video_duration=1800.0,
            project_name="t",
            recording_date=RECORDING_DATE,
        )
        assert set(seo_result["tags"]) == set(basic_result["tags"])

    def test_generate_drops_word_split_fragments_keeps_multiword(self, seo_gen):
        """Single-word optimizer fragments stay discarded; the
        optimizer's MULTI-WORD analytics phrases now survive, appended
        after the curated core (deduped, capped at 30).
        """
        type(seo_gen).seo_tags = [
            "dingle",  # fragment → dropped
            "peninsula",  # fragment → dropped
            "real_tag",  # single token → dropped
            "mountain pass drive",  # multi-word phrase → kept
        ]
        result = seo_gen.generate(
            location=LOCATION,
            video_duration=1800.0,
            project_name="t",
            recording_date=RECORDING_DATE,
        )

        expected: list[str] = list(GMS_BASE_TAGS)
        for tag in _location_tags(LOCATION, RECORDING_DATE):
            if tag.lower() not in [t.lower() for t in expected]:
                expected.append(tag)
        expected.append("mountain pass drive")
        expected = expected[:30]

        assert result["tags"] == expected
        # Single-word fragments never leak in:
        assert "real_tag" not in result["tags"]
        assert "dingle" not in [t.lower() for t in result["tags"]]
        assert "peninsula" not in [t.lower() for t in result["tags"]]


class TestTagRecapAfterEnrichment:
    """Route-profile enrichment appends GPS towns/roads/features AFTER the
    :30 count cap with no length check; the FINAL tag list must still respect
    YouTube's 30-count + 500-char limits (mirrors the shorts _yt_tag_len guard).
    """

    def test_cap_youtube_tags_enforces_count_and_width(self):
        tags = [f"very long scenic town name number {i:02d}" for i in range(40)]
        capped = _cap_youtube_tags(tags)
        assert len(capped) <= 30
        assert _quoted_tag_width(capped) <= 480
        # Drops from the tail — result is a prefix of the input.
        assert capped == tags[: len(capped)]

    def test_enrichment_tags_stay_within_youtube_limits(self, basic_gen):
        """A GPS-rich drive whose route profile appends many long town/road/
        feature tags must not ship an over-limit tag set."""
        route_profile = {
            "towns_traversed": [f"Really Long Town Placename {i:02d}" for i in range(12)],
            "roads_traversed": [f"Very Long Named Country Road {i:02d}" for i in range(8)],
            "features": [f"Notable Scenic Overlook Feature {i:02d}" for i in range(8)],
            "states_traversed": [],
            "start_town": "",
            "end_town": "",
        }
        result = basic_gen.generate(
            location="Nelson County, VA",
            video_duration=1800.0,
            project_name="t",
            recording_date=RECORDING_DATE,
            route_profile=route_profile,
        )
        assert len(result["tags"]) <= 30
        assert _quoted_tag_width(result["tags"]) <= 480
