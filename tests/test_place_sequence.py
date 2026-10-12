"""Full place sequence + lift the 4-region description cap.

Covers: (1) ``dominant_location.ordered_places`` route-ordered town labels;
(2) ``build_resolved_location`` populates ``ordered_place_sequence``;
(3) ``_format_route_line`` ``🛤️ Route:`` formatter; (4) the lifted
``_REGIONS_DESCRIPTION_CAP`` (4 → 12) so long drives list every county;
(5) ``generate()`` splice of the route line from a resolved model.
"""

from gomoveshift.models import ResolvedLocation
from gomoveshift.pipeline.gps_enrichment import build_resolved_location
from gomoveshift.pipeline.longform_metadata import (
    _REGIONS_DESCRIPTION_CAP,
    LongFormMetadataGenerator,
    _format_regions_line,
    _format_route_line,
)
from gomoveshift.pipeline.title_builder import dominant_location

_A = (52.05, -9.50)  # Killarney-ish
_B = (51.88, -9.58)  # Kenmare-ish
_C = (51.83, -9.90)  # Sneem-ish


def _make_geocoder(by_point):
    class _Geo:
        def reverse_geocode_detailed(self, lat, lon):
            best, best_d = None, 0.15
            for (clat, clon), data in by_point.items():
                d = abs(lat - clat) + abs(lon - clon)
                if d < best_d:
                    best_d, best = d, data
            return dict(best) if best else {"county": None, "state": None}

    return _Geo()


_KERRY = {"county": "County Kerry", "state": "Munster"}
_GEO = lambda: _make_geocoder(  # noqa: E731
    {
        _A: {**_KERRY, "town": "Killarney"},
        _B: {**_KERRY, "town": "Kenmare"},
        _C: {**_KERRY, "village": "Sneem"},
    }
)


class TestDominantLocationOrderedPlaces:
    def test_ordered_places_route_order(self):
        # 3 distinct towns in travel order → ordered, deduped sequence
        pts = [
            {"lat": _A[0], "lon": _A[1]},
            {"lat": _B[0], "lon": _B[1]},
            {"lat": _C[0], "lon": _C[1]},
        ]
        result = dominant_location(pts, _GEO())
        assert result.ordered_places == ["Killarney", "Kenmare", "Sneem"]

    def test_consecutive_duplicates_collapsed(self):
        pts = [{"lat": _A[0], "lon": _A[1]}] * 3 + [{"lat": _C[0], "lon": _C[1]}]
        result = dominant_location(pts, _GEO())
        assert result.ordered_places == ["Killarney", "Sneem"]

    def test_default_empty_when_no_places(self):
        from gomoveshift.pipeline.title_builder import DominantLocation

        assert DominantLocation(county="X", state="Y").ordered_places == []


class TestBuilderPopulatesSequence:
    def test_resolved_location_ordered_place_sequence(self):
        pts = [
            {"lat": _A[0], "lon": _A[1]},
            {"lat": _B[0], "lon": _B[1]},
            {"lat": _C[0], "lon": _C[1]},
        ]
        rl = build_resolved_location(pts, geocoder=_GEO())
        assert rl.ordered_place_sequence == ["Killarney", "Kenmare", "Sneem"]


class TestFormatRouteLine:
    def test_basic(self):
        rl = ResolvedLocation(ordered_place_sequence=["Killarney", "Kenmare", "Sneem"])
        assert _format_route_line(rl) == "🛤️ Route: Killarney → Kenmare → Sneem\n"

    def test_single_or_empty_omitted(self):
        assert _format_route_line(ResolvedLocation(ordered_place_sequence=["Only"])) == ""
        assert _format_route_line(ResolvedLocation()) == ""
        assert _format_route_line(None) == ""

    def test_elides_head_tail_over_cap(self):
        seq = [f"T{i}" for i in range(12)]
        line = _format_route_line(ResolvedLocation(ordered_place_sequence=seq))
        assert "→ … →" in line
        assert "T0" in line and "T11" in line  # endpoints survive


class TestLiftedRegionsCap:
    def test_cap_lifted_to_12(self):
        assert _REGIONS_DESCRIPTION_CAP == 12

    def test_lists_all_regions_up_to_cap(self):
        regions = [{"name": f"County {i}"} for i in range(8)]
        line = _format_regions_line(regions, "Ireland")
        assert "… +" not in line  # no truncation — all 8 listed (old cap was 4)
        assert line.count(",") == 7

    def test_overflow_beyond_ceiling_still_elides(self):
        regions = [{"name": f"County {i}"} for i in range(15)]
        line = _format_regions_line(regions, "Ireland")
        assert "… +3 more" in line


class TestGenerateIntegration:
    def test_route_line_present_then_absent(self):
        gen = LongFormMetadataGenerator()
        rl = ResolvedLocation(is_us=False, ordered_place_sequence=["Killarney", "Kenmare", "Sneem"])
        meta = gen.generate(
            location="County Kerry, Ireland", video_duration=1800.0, resolved_location=rl
        )
        assert "🛤️ Route: Killarney → Kenmare → Sneem" in meta["description"]
        without = gen.generate(location="County Kerry, Ireland", video_duration=1800.0)
        assert "🛤️ Route:" not in without["description"]
