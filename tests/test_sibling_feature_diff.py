"""Deterministic sibling-LF feature differentiation.

When ≥2 LFs collapse to the same label (e.g. several rural backroad LFs
that all resolve only to the county/region), the pipeline
must NOT lean on a human to pick distinguishers or fall back to "(Part N of M)"
filler. It differentiates each LF by its modal named geographic feature via a
fully deterministic vote + stable assignment (same input → same output).
"""

from unittest.mock import MagicMock, patch

from gomoveshift.pipeline.title_builder import (
    _feature_stem,
    _insert_feature,
    differentiate_sibling_labels,
)


class TestFeatureStem:
    def test_strips_generic_road_suffix(self):
        assert _feature_stem("Blocker Run Road") == "Blocker Run"
        assert _feature_stem("Donnelley Drive") == "Donnelley"
        assert _feature_stem("Stocks Creek Road") == "Stocks Creek"
        assert _feature_stem("US Highway 64") == "US Highway 64"  # ref-bearing kept distinct enough
        assert _feature_stem("Main Street") == "Main"

    def test_keeps_name_without_generic_suffix(self):
        assert _feature_stem("Donnelley") == "Donnelley"
        assert _feature_stem("Ring of Kerry") == "Ring of Kerry"


class TestInsertFeature:
    def test_inserts_before_region(self):
        assert _insert_feature("ACE Basin, SC", "Donnelley") == "ACE Basin — Donnelley, SC"

    def test_appends_when_no_region(self):
        assert _insert_feature("ACE Basin Scenic Drive", "Donnelley") == (
            "ACE Basin Scenic Drive — Donnelley"
        )


class TestDifferentiateSiblingLabels:
    def test_sc_ace_basin_dominant_only(self):
        # Footage-weighted feature votes for four LFs that collapse to the same region label.
        labels = {3: "ACE Basin, SC", 4: "ACE Basin, SC", 5: "ACE Basin, SC", 6: "ACE Basin, SC"}
        features = {
            3: [("Donnelley", 10), ("Pineland", 6), ("G.W. Roberts", 3), ("Blocker Run", 3)],
            4: [("Blocker Run", 12), ("Savage", 2)],
            5: [("Blocker Run", 16), ("Stocks Creek", 6)],
            6: [("Stocks Creek", 5)],
        }
        out = differentiate_sibling_labels(labels, features)
        # DOMINANT-ONLY: each LF gets only its #1 feature.
        assert out[3] == "ACE Basin — Donnelley, SC"
        assert out[5] == "ACE Basin — Blocker Run, SC"  # LF5's count beats LF4's
        assert out[6] == "ACE Basin — Stocks Creek, SC"
        # LF4's dominant (Blocker Run) is the SAME road LF5 won → LF4 is NOT
        # forced onto a weak "Savage" 2nd-choice; it keeps the clean region.
        assert out[4] == "ACE Basin, SC"
        assert "Savage" not in out[4]
        # The three assigned features + LF4's region are all distinct.
        assert len(set(out.values())) == 4

    def test_shared_dominant_road_only_one_wins(self):
        # Two LFs whose dominant feature is the SAME road: only the stronger
        # claim is labelled; the other keeps the region (no fabricated distinction).
        labels = {1: "ACE Basin, SC", 2: "ACE Basin, SC"}
        features = {1: [("Blocker Run", 14), ("Savage", 3)], 2: [("Blocker Run", 18)]}
        out = differentiate_sibling_labels(labels, features)
        assert out[2] == "ACE Basin — Blocker Run, SC"
        assert out[1] == "ACE Basin, SC"  # NOT "Savage"

    def test_idempotent_same_input_same_output(self):
        labels = {1: "ACE Basin, SC", 2: "ACE Basin, SC"}
        features = {1: [("Donnelley", 5)], 2: [("Blocker Run", 4)]}
        a = differentiate_sibling_labels(labels, features)
        b = differentiate_sibling_labels(labels, features)
        assert a == b == {1: "ACE Basin — Donnelley, SC", 2: "ACE Basin — Blocker Run, SC"}

    def test_no_feature_left_unchanged_for_part_n_fallback(self):
        # LF with no unclaimed distinct feature is left as-is (ensure_unique_titles
        # then applies its (Part N) last resort downstream).
        labels = {1: "ACE Basin, SC", 2: "ACE Basin, SC"}
        features = {1: [("Donnelley", 5)], 2: []}  # LF2 has no feature
        out = differentiate_sibling_labels(labels, features)
        assert out[1] == "ACE Basin — Donnelley, SC"
        assert out[2] == "ACE Basin, SC"  # unchanged → Part-N handles it later

    def test_unique_labels_untouched(self):
        labels = {1: "Cashiers to Franklin, NC", 2: "Saluda to Lake Toxaway, NC"}
        out = differentiate_sibling_labels(labels, {1: [("X", 1)], 2: [("Y", 1)]})
        assert out == labels  # no collision → no change

    def test_empty(self):
        assert differentiate_sibling_labels({}, {}) == {}


class TestSiblingDiffRevalidated:
    """``differentiate_sibling_labels`` inserts a geographic feature into
    a colliding label, but that mutated string was written straight into the
    burn map — it bypassed ``validate_burned_ost_text`` (which ran only on the
    pre-diff label). A malformed feature (broken geocode → coordinate/None
    string) would burn ungated. The differentiated label must be re-validated,
    falling back to the already-validated original on failure.
    """

    def test_garbage_differentiated_label_falls_back_to_validated_original(self):
        from gomoveshift.pipeline import gps_enrichment as ge

        # Two LFs collide on the same clean, validated label.
        per_lf_ost_text_by_date = {
            "2026-02-20": {1: "ACE Basin, SC", 2: "ACE Basin, SC"},
        }
        # differentiate returns a broken coordinate string for LF 1 (the class
        # of garbage a malformed OSM feature produces) — validate_ost rejects
        # coordinate strings, so LF 1 must keep the clean original label.
        with patch.object(
            ge,
            "differentiate_sibling_labels",
            return_value={1: "34.5, -80.2", 2: "ACE Basin, SC"},
        ):
            ge._differentiate_and_revalidate_siblings(
                per_lf_ost_text_by_date,
                gps_points_by_lf={1: [{"lat": 34.5, "lon": -80.2}], 2: []},
                geocoder=MagicMock(),
            )
        day = per_lf_ost_text_by_date["2026-02-20"]
        assert day[1] == "ACE Basin, SC"  # garbage rejected, original restored
        assert "34.5" not in day[1]

    def test_valid_differentiated_label_is_applied(self):
        from gomoveshift.pipeline import gps_enrichment as ge

        per_lf_ost_text_by_date = {
            "2026-02-20": {1: "ACE Basin, SC", 2: "ACE Basin, SC"},
        }
        with patch.object(
            ge,
            "differentiate_sibling_labels",
            return_value={1: "ACE Basin — Donnelley, SC", 2: "ACE Basin, SC"},
        ):
            ge._differentiate_and_revalidate_siblings(
                per_lf_ost_text_by_date,
                gps_points_by_lf={1: [{"lat": 34.5, "lon": -80.2}], 2: []},
                geocoder=MagicMock(),
            )
        assert per_lf_ost_text_by_date["2026-02-20"][1] == "ACE Basin — Donnelley, SC"
