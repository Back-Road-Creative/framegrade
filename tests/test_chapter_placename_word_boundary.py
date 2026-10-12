"""Tourism-keyword rejection must match on word boundaries, not substrings.

Regression guard: a bare substring test silently excluded
legitimate settlements whose names merely *contain* a tourism keyword
("Centerville" -> "center", "Farmington" -> "farm", "Banks" -> "bank"),
pushing the chapter/title ladder onto a farther settlement or a bare county.
The keyword must only reject when it appears as a whole word/token.
"""

from gomoveshift.pipeline.chapters import is_valid_chapter_placename


class TestChapterPlacenameWordBoundary:
    def test_settlements_with_embedded_keyword_pass(self):
        """Towns that merely contain a keyword as a substring are valid."""
        good_names = [
            "Centerville",  # embeds "center"
            "Farmington",  # embeds "farm"
            "Banks",  # embeds "bank"
            "Bankston",  # embeds "bank"
            "Shopiere",  # embeds "shop"
            "Storey",  # embeds "store"
        ]
        for name in good_names:
            assert is_valid_chapter_placename(name), (
                f"'{name}' embeds a keyword but is a real settlement; must pass"
            )

    def test_whole_word_keyword_still_rejected(self):
        """A keyword appearing as a whole word/token still rejects the name."""
        bad_names = [
            "Visitor Center",  # "visitor" + "center" as whole words
            "Smith Farm",  # "farm" as a whole word
            "Food Bank",  # "bank" as a whole word
            "Corner Store",  # "store" as a whole word
        ]
        for name in bad_names:
            assert not is_valid_chapter_placename(name), (
                f"'{name}' contains a whole-word tourism keyword; must be rejected"
            )
