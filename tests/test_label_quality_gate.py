"""Label-quality backstop gate (``check_label_quality``).

WARN-records (greppable ``LABEL-QUALITY:``) when an LF's burned label / title
collapsed to the bare COUNTY even though the trace's place sequence carried a
recognizable settlement that should have anchored the title — the
defect class (``"Jackson County, NC Scenic Drive"`` while Cashiers was in the
sequence). Never aborts: a county-only label is "could be better", not a desync
(that is ``check_location_invariant``'s HARD job). Observability backstop to the
robust-by-construction fixes (surface the settlement, drop junk
via-roads).
"""

import pytest

from gomoveshift.pipeline.validation.content_gates import check_label_quality


def _meta(label, sequence):
    return {
        "location": label,
        "resolved_location": {
            "dominant_county": "Jackson County",
            "dominant_state": "North Carolina",
            "ordered_place_sequence": sequence,
        },
    }


class TestCheckLabelQuality:
    def test_county_only_with_available_settlement_warns(self, caplog):
        meta = _meta("Jackson County, NC Scenic Drive", ["Cashiers", "Sapphire"])
        with caplog.at_level("WARNING"):
            res = check_label_quality(meta)
        assert res.passed is True  # never aborts
        assert "LABEL-QUALITY" in caplog.text
        assert "Cashiers" in caplog.text

    def test_label_naming_settlement_does_not_warn(self, caplog):
        meta = _meta("Lake Toxaway to Cashiers, NC", ["Lake Toxaway", "Cashiers"])
        with caplog.at_level("WARNING"):
            res = check_label_quality(meta)
        assert res.passed is True
        assert "LABEL-QUALITY" not in caplog.text

    def test_no_settlement_in_sequence_does_not_warn(self, caplog):
        # The rural ACE Basin case — sequence is county-only, no settlement
        # was available, so a county label is the honest best and is NOT flagged.
        meta = _meta("Colleton County, SC Scenic Drive", ["Colleton County"])
        with caplog.at_level("WARNING"):
            res = check_label_quality(meta)
        assert res.passed is True
        assert "LABEL-QUALITY" not in caplog.text

    def test_no_resolved_location_skips(self):
        res = check_label_quality({"location": "Jackson County, NC Scenic Drive"})
        assert res.passed is True
        assert "skipped" in res.message

    def test_no_label_skips(self):
        res = check_label_quality({"resolved_location": {"ordered_place_sequence": ["Cashiers"]}})
        assert res.passed is True
        assert "skipped" in res.message

    def test_empty_sequence_does_not_warn(self, caplog):
        meta = _meta("Jackson County, NC Scenic Drive", [])
        with caplog.at_level("WARNING"):
            res = check_label_quality(meta)
        assert res.passed is True
        assert "LABEL-QUALITY" not in caplog.text

    def test_falls_back_to_title_when_no_location(self, caplog):
        meta = {
            "title": "Jackson County, NC Scenic Drive",
            "resolved_location": {"ordered_place_sequence": ["Cashiers"]},
        }
        with caplog.at_level("WARNING"):
            res = check_label_quality(meta)
        assert res.passed is True
        assert "LABEL-QUALITY" in caplog.text

    def test_title_prefixed_label_warns_title_not_county_only(self, caplog):
        # A title-shaped label: 'To Mizen Head, County Cork' is a
        # title (leads with the preposition "To"), not a place -- it must
        # not be reported as "county-only" even though it also contains the
        # literal word "County".
        meta = _meta("To Mizen Head, County Cork", ["West Cork"])
        with caplog.at_level("WARNING"):
            res = check_label_quality(meta)
        assert res.passed is True  # never escalates
        assert "LABEL-QUALITY" in caplog.text
        assert "title, not a place" in caplog.text
        assert "county-only" not in caplog.text

    def test_real_county_only_label_still_reported_as_county_only(self, caplog):
        meta = _meta("Jackson County, NC Scenic Drive", ["Cashiers"])
        with caplog.at_level("WARNING"):
            res = check_label_quality(meta)
        assert res.passed is True
        assert "LABEL-QUALITY" in caplog.text
        assert "county-only" in caplog.text
        assert "title, not a place" not in caplog.text

    def test_settlement_led_label_does_not_warn_county_only(self, caplog):
        # 'Goleen, County Cork': the lead component is a settlement, not a bare
        # county; it just is not in the trace's own ordered_place_sequence.
        meta = _meta("Goleen, County Cork", ["West Cork"])
        with caplog.at_level("WARNING"):
            res = check_label_quality(meta)
        assert res.passed is True
        assert "LABEL-QUALITY" not in caplog.text

    @pytest.mark.parametrize(
        "label",
        [
            "R591, County Cork",
            "US-64, Jackson County, NC",
            "Interstate 40, Buncombe County, NC",
            "US Highway 64, Jackson County, NC",
            "Cherohala Skyway, Graham County, NC",
            "I 77;I 81;US 52, Wythe County, VA",
            "NC Highway 28, Macon County, NC",
            "US 64 Business, Jackson County, NC",
            "US 64;NC 28, Macon County, NC",
            "Hwy. 64, Macon County, NC",
            "Capital Beltway, Montgomery County, MD",
            "M50 Motorway, County Dublin",
        ],
    )
    def test_road_plus_county_label_still_warns_county_only(self, caplog, label):
        # A road-shaped lead (ost_formatter's road+county labels) is the class
        # this gate exists to catch; the settlement carve-out must not hide it.
        meta = _meta(label, ["Some Town"])
        with caplog.at_level("WARNING"):
            res = check_label_quality(meta)
        assert res.passed is True
        assert "LABEL-QUALITY" in caplog.text
        assert "county-only" in caplog.text

    @pytest.mark.parametrize(
        "field, value",
        [
            ("road", "Wanderers Hollow"),
            ("road_refs", ["Wanderers Hollow"]),
            ("road_votes", {"Wanderers Hollow": 3}),
        ],
    )
    def test_resolver_recorded_road_warns_even_off_regex_shape(self, caplog, field, value):
        # The name fits no static road shape, but the resolver itself recorded
        # it as a road, which is a stronger signal than a string guess.
        meta = _meta("Wanderers Hollow, Watauga County, NC", ["Some Town"])
        meta["resolved_location"][field] = value
        with caplog.at_level("WARNING"):
            res = check_label_quality(meta)
        assert res.passed is True
        assert "county-only" in caplog.text

    def test_unrecorded_unrecognised_lead_is_read_as_settlement(self, caplog):
        meta = _meta("Wanderers Hollow, Watauga County, NC", ["Some Town"])
        with caplog.at_level("WARNING"):
            res = check_label_quality(meta)
        assert res.passed is True
        assert "LABEL-QUALITY" not in caplog.text
