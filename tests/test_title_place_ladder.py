"""Recognizable-place preference ladder.

Titles collapsed to the bare county ("Jackson County via US
Highway 64", "Macon County via North 4th Street") even though the recognizable
settlements — Cashiers, Franklin, Lake Toxaway, Fort Fremont — were already in
the canonical ``ResolvedLocation.ordered_place_sequence``. The per-endpoint
centroid geocode missed them, so ``build_lf_title`` fell through to Level 4.
The ladder backfills a missing endpoint (and the single-location anchor) from the
travel-ordered place sequence so a real place leads the title.
"""

from types import SimpleNamespace

from gomoveshift.pipeline.title_builder import _recognizable_settlements, build_lf_title


class _MissGeo:
    """Endpoint geocode always misses (no town); detailed geocode empty so the
    in-place vote contributes nothing — the sequence is the only place source.
    """

    def reverse_geocode(self, lat, lon):
        return SimpleNamespace(
            city=None,
            locality=None,
            suburb=None,
            name=None,
            town=None,
            place=None,
            hamlet=None,
            village=None,
        )

    def reverse_geocode_detailed(self, lat, lon):
        return {}


# Two GPS fixes ~26 km apart (Cashiers → Franklin), under the 50 km via-road
# span guard so a real route would survive; here road stays None.
_GPS = [{"lat": 35.10, "lon": -83.10}, {"lat": 35.18, "lon": -83.38}]


class TestPlaceLadder:
    def test_two_place_crossing_backfills_from_sequence(self):
        title = build_lf_title(
            [],
            _MissGeo(),
            "Macon County, North Carolina",
            gps_points=_GPS,
            resolved_places=["Cashiers", "Franklin"],
        )
        assert title == "Cashiers to Franklin, NC"

    def test_single_settlement_anchors_level4(self):
        title = build_lf_title(
            [],
            _MissGeo(),
            "Beaufort County, South Carolina",
            gps_points=_GPS,
            resolved_places=["Fort Fremont"],
        )
        assert title == "Fort Fremont, SC"

    def test_county_only_sequence_stays_county(self):
        # The sequence carries only the county fallback (no settlement) — the
        # title must NOT fabricate a place; it stays on the county.
        title = build_lf_title(
            [],
            _MissGeo(),
            "Colleton County, South Carolina",
            gps_points=_GPS,
            resolved_places=["Colleton County"],
        )
        assert title == "Colleton County, SC"

    def test_business_name_in_sequence_rejected(self):
        # A possessive/business string in the sequence is dropped — never anchors.
        title = build_lf_title(
            [],
            _MissGeo(),
            "Colleton County, South Carolina",
            gps_points=_GPS,
            resolved_places=["Dickie's"],
        )
        assert title == "Colleton County, SC"

    def test_resolved_endpoints_not_overridden_by_sequence(self):
        # When the endpoints DO resolve (linear drive through one town), the
        # resolved town wins — the sequence never overrides it.
        class _HitSameTown:
            def reverse_geocode(self, lat, lon):
                return SimpleNamespace(
                    city="Cashiers",
                    locality=None,
                    suburb=None,
                    name=None,
                    town=None,
                    place=None,
                    hamlet=None,
                    village=None,
                )

            def reverse_geocode_detailed(self, lat, lon):
                return {}

        title = build_lf_title(
            [],
            _HitSameTown(),
            "Jackson County, North Carolina",
            gps_points=_GPS,
            resolved_places=["WrongTown"],
        )
        assert "Cashiers" in title
        assert "WrongTown" not in title

    def test_no_sequence_back_compat_county(self):
        # No sequence at all (GPS-less) — unchanged Level-4 county.
        title = build_lf_title(
            [],
            _MissGeo(),
            "Colleton County, South Carolina",
            gps_points=_GPS,
        )
        assert title == "Colleton County, SC"


class TestRecognizableSettlements:
    def test_filters_county_and_business(self):
        seq = ["Colleton County", "Green Pond", "Dickie's", "Green Pond", "Walterboro"]
        assert _recognizable_settlements(seq, "Colleton County") == ["Green Pond", "Walterboro"]

    def test_empty_on_all_county(self):
        assert _recognizable_settlements(["Macon County"], "Macon County") == []

    def test_dedups_consecutive(self):
        assert _recognizable_settlements(["Cashiers", "Cashiers", "Franklin"], "Macon County") == [
            "Cashiers",
            "Franklin",
        ]
