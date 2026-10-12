"""Tests for cross-episode OST uniqueness validation.

Duplicate OST titles baked into video are unfixable after
encode. The uniqueness check catches exact duplicates and near-duplicates
(differing only by whitespace) BEFORE rendering begins.
"""


class TestCheckOSTUniqueness:
    """validate_ost_uniqueness catches duplicate titles across episodes."""

    def test_all_unique_titles_pass(self):
        from gomoveshift.pipeline.ost_validator import validate_ost_uniqueness

        titles = [
            "Port Royal to Beaufort via Trask Parkway",
            "Seaside Road, South Carolina",
            "Coastal Highway through Hilton Head",
        ]
        dupes = validate_ost_uniqueness(titles)
        assert dupes == {}

    def test_exact_duplicate_detected(self):
        from gomoveshift.pipeline.ost_validator import validate_ost_uniqueness

        titles = [
            "Stocks Creek Road, SC",
            "Coastal Highway",
            "Stocks Creek Road, SC",
        ]
        dupes = validate_ost_uniqueness(titles)
        assert "Stocks Creek Road, SC" in dupes
        assert dupes["Stocks Creek Road, SC"] == [0, 2]

    def test_whitespace_near_duplicate_detected(self):
        """Titles differing only by whitespace are duplicates."""
        from gomoveshift.pipeline.ost_validator import validate_ost_uniqueness

        titles = [
            "Stocks Creek Road, SC",
            "Stocks Creek Road,  SC",  # extra space
            "Coastal Highway",
        ]
        dupes = validate_ost_uniqueness(titles)
        # Should detect these as near-duplicates
        assert len(dupes) == 1
        canonical = next(iter(dupes))
        assert dupes[canonical] == [0, 1]

    def test_leading_trailing_whitespace_near_duplicate(self):
        from gomoveshift.pipeline.ost_validator import validate_ost_uniqueness

        titles = [
            "Seaside Road",
            "  Seaside Road  ",
            "Other Place",
        ]
        dupes = validate_ost_uniqueness(titles)
        assert len(dupes) == 1

    def test_empty_strings_ignored(self):
        """Empty/whitespace-only titles are not considered duplicates."""
        from gomoveshift.pipeline.ost_validator import validate_ost_uniqueness

        titles = ["", "", "Real Title", "   "]
        dupes = validate_ost_uniqueness(titles)
        assert dupes == {}

    def test_single_title_no_duplicate(self):
        from gomoveshift.pipeline.ost_validator import validate_ost_uniqueness

        titles = ["Only One Title"]
        dupes = validate_ost_uniqueness(titles)
        assert dupes == {}

    def test_empty_list(self):
        from gomoveshift.pipeline.ost_validator import validate_ost_uniqueness

        dupes = validate_ost_uniqueness([])
        assert dupes == {}

    def test_three_way_duplicate(self):
        from gomoveshift.pipeline.ost_validator import validate_ost_uniqueness

        titles = ["Same Title", "Different", "Same Title", "Same Title"]
        dupes = validate_ost_uniqueness(titles)
        assert "Same Title" in dupes
        assert dupes["Same Title"] == [0, 2, 3]

    def test_case_preserved_in_key(self):
        """The canonical key uses the first occurrence's original text."""
        from gomoveshift.pipeline.ost_validator import validate_ost_uniqueness

        titles = ["Stocks Creek Road, SC", "Stocks Creek Road, SC"]
        dupes = validate_ost_uniqueness(titles)
        assert "Stocks Creek Road, SC" in dupes

    def test_tab_whitespace_normalized(self):
        """Tabs and other whitespace characters are collapsed."""
        from gomoveshift.pipeline.ost_validator import validate_ost_uniqueness

        titles = [
            "Stocks Creek\tRoad",
            "Stocks Creek Road",
        ]
        dupes = validate_ost_uniqueness(titles)
        assert len(dupes) == 1
