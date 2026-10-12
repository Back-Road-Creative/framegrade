"""Description hygiene.

- Long-form hashtags must be clean alphanumerics — no parens/punctuation
      leaking from a road-disambiguated label (#ColletonCounty(BlockerRunRoad)).
- The ``generate`` entry gate must reject coordinate / null-island /
      geocoder-error location strings, not only literal placeholders.
- The stale Wed/Sat + noon ``get_schedule_summary`` embed must be gone.
- Chapter end-location reverse-geocode must run through the budgeted
      ``NominatimChapterSource`` (MAX_SESSION_REQUESTS), not the raw geocoder.
- The single-video chapter path must pass ``included_segments`` so
      "describe only surviving footage" is no longer drive-path-only.
"""

import inspect
import re

import pytest

from gomoveshift.pipeline.longform_metadata import (
    LongFormMetadataGenerator,
    _clean_hashtag,
)


# ── Hashtag punctuation strip ──────────────────────────────────────────
class TestHashtagHygiene:
    def test_clean_hashtag_strips_parens_and_punctuation(self):
        assert _clean_hashtag("Colleton County (Blocker Run Road)") == (
            "ColletonCountyBlockerRunRoad"
        )
        assert _clean_hashtag("Dingle, Co. Kerry") == "DingleCoKerry"
        assert _clean_hashtag("Ireland.") == "Ireland"

    def test_hashtags_strip_parens_and_punctuation(self):
        gen = LongFormMetadataGenerator()
        desc = gen._build_gms_description(
            location="Colleton County (Blocker Run Road), SC",
            seo_description="POV drive.",
            duration_str="20m 0s",
            publish_day="Sunday",
        )
        # The location hashtag must be clean alphanumerics — no parens. The
        # broken form (parens leaked from a road-disambiguated label) must
        # never appear, and every hashtag body is bare ``[A-Za-z0-9]``.
        # (Whether the geo hashtag survives the [:12] slice is a separate concern.)
        assert "#ColletonCounty(BlockerRunRoad)" not in desc
        for ht in _hashtag_line(desc).split():
            assert re.fullmatch(r"#[A-Za-z0-9]+", ht), ht

    def test_country_hashtag_strips_punctuation(self):
        gen = LongFormMetadataGenerator()
        desc = gen._build_gms_description(
            location="Dingle, County Kerry, Ireland.",
            seo_description="POV drive.",
            duration_str="20m 0s",
            publish_day="Sunday",
        )
        line = _hashtag_line(desc)
        assert "." not in line


def _hashtag_line(desc: str) -> str:
    for ln in desc.splitlines():
        if ln.strip().startswith("#"):
            return ln
    raise AssertionError("no hashtag line found in description")


# ── Coordinate / null-island / error reject in entry gate ──────────────
class TestEntryGateRejectsGarbage:
    @pytest.mark.parametrize(
        "bad",
        [
            "34.12, -80.45",
            "0.0, 0.0",
            "0, 0",
            "None, None",
            "null",
            "12345",
        ],
    )
    def test_generate_rejects_coord_null_error_strings(self, bad):
        gen = LongFormMetadataGenerator()
        with pytest.raises(ValueError):
            gen.generate(location=bad)

    def test_generate_accepts_real_location(self):
        gen = LongFormMetadataGenerator()
        # Should not raise on a real label (may degrade SEO but must build).
        result = gen.generate(location="County Kerry, Ireland")
        assert result["title"]


# ── Stale schedule summary removed ────────────────────────────────────
class TestScheduleSummaryRemoved:
    def test_get_schedule_summary_method_gone(self):
        assert not hasattr(LongFormMetadataGenerator, "get_schedule_summary")


# ── Budgeted end-location geocode ──────────────────────────────────────
class TestChapterEndLocationBudget:
    def test_end_location_uses_budgeted_source(self):
        src = inspect.getsource(
            __import__(
                "gomoveshift.pipeline.chapters.generator",
                fromlist=["ChapterGenerator"],
            ).ChapterGenerator.generate
        )
        # The raw, unbudgeted call must be gone from generate().
        assert "self.geocoder.reverse_geocode(last_wp" not in src


# ── Single-video path passes included_segments ─────────────────────────
class TestSingleVideoIncludedSegments:
    def test_stage_passes_included_segments(self):
        from gomoveshift.pipeline.stages.chapters import ChapterGenerationStage

        src = inspect.getsource(ChapterGenerationStage.run)
        assert "included_segments=" in src
