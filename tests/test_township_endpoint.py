"""A US administrative TOWNSHIP is not a searchable settlement.

Root cause: a Gettysburg-area drive geocoded its
endpoint centroids to "Cumberland Township" (the rural admin subdivision, town/
village/hamlet all null) and stamped "Cumberland Township, PA" even
though **Gettysburg** — the highest-value search term — was in the trace's
place sequence. A township endpoint, like a county, is the rural fallback; it
must be nulled so the recognizable-place backfill surfaces the real
town. Ireland has no townships, so this is US-only by construction and cannot
demote a legitimate Irish coastal-hamlet endpoint.
"""

from types import SimpleNamespace

from gomoveshift.pipeline.title_builder import (
    _looks_like_township,
    _recognizable_settlements,
    build_lf_title,
)


class _TownshipGeo:
    """Endpoint centroids geocode to a township name (no town/village); the
    detailed geocode is empty so the in-place vote contributes nothing and the
    resolved sequence is the only settlement source.
    """

    def reverse_geocode(self, lat, lon):
        return SimpleNamespace(
            city=None,
            locality=None,
            suburb=None,
            name="Cumberland Township",
            town=None,
            place=None,
            hamlet=None,
            village=None,
        )

    def reverse_geocode_detailed(self, lat, lon):
        return {}


# Two fixes ~6 km apart (Gettysburg battlefield loop), under the via-road span.
_GPS = [{"lat": 39.832, "lon": -77.251}, {"lat": 39.808, "lon": -77.238}]


class TestLooksLikeTownship:
    def test_township_flagged(self):
        assert _looks_like_township("Cumberland Township") is True
        assert _looks_like_township("Freedom Township") is True

    def test_real_settlement_not_flagged(self):
        assert _looks_like_township("Gettysburg") is False
        assert _looks_like_township("Marion") is False
        # A real town that merely contains the substring elsewhere is safe —
        # the word-boundary regex only matches the standalone word.
        assert _looks_like_township("Townsend") is False


class TestRecognizableSettlementsDropTownships:
    def test_township_entries_dropped(self):
        seq = ["Cumberland Township", "Gettysburg", "Cumberland Township"]
        assert _recognizable_settlements(seq, "Adams County") == ["Gettysburg"]

    def test_all_township_sequence_empties(self):
        seq = ["Liberty Township", "Freedom Township", "Cumberland Township"]
        assert _recognizable_settlements(seq, "Adams County") == []


class TestBuildLfTitleTownship:
    def test_township_endpoints_surface_sequence_town(self):
        title = build_lf_title(
            [],
            _TownshipGeo(),
            "Adams County, Pennsylvania",
            gps_points=_GPS,
            resolved_places=["Cumberland Township", "Gettysburg", "Cumberland Township"],
        )
        assert title == "Gettysburg, PA"

    def test_all_township_sequence_falls_to_county(self):
        # No settlement anywhere in the trace — the title must NOT fabricate a
        # place; it degrades honestly to the county.
        title = build_lf_title(
            [],
            _TownshipGeo(),
            "Adams County, Pennsylvania",
            gps_points=_GPS,
            resolved_places=["Liberty Township", "Freedom Township", "Cumberland Township"],
        )
        assert title == "Adams County, PA"
