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
