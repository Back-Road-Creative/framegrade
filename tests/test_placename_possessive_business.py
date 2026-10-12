"""Placename validator: possessive bypasses and settlement false-rejects.

Regression guard for two label-validation defects.

Possessive bypass: the possessive guard only fired on an ASCII ``'s`` followed by a space,
so a curly apostrophe ("Murphy’s") and a name-final possessive with no
trailing space ("Murphy's", "O'Connor's") slipped through as valid placenames;
and the tourism-keyword set lacked common pub/business words
(bar/inn/lodge/tavern/resort/distillery), letting business names reach chapter
titles, endpoints, and via-roads.

Settlement false-reject: several tourism keywords (bank/center/centre/dollar/golf) collide with the
whole names of real settlements (Red Bank, Center Point, Centre, Golf,
Dollar Bay), evicting a true endpoint. A settlement allowlist checked before the
keyword loop must let those through.
"""

from gomoveshift.pipeline.chapters import is_valid_chapter_placename


class TestPossessiveGuard:
    def test_curly_apostrophe_possessive_rejected(self):
        """A U+2019 curly-apostrophe possessive is a business name, not a place."""
        assert not is_valid_chapter_placename("Murphy’s")

    def test_name_final_possessive_rejected(self):
        """A name ending in "'s" (no trailing space) is possessive/business."""
        for name in ("Murphy's", "O'Connor's", "Dickie's"):
            assert not is_valid_chapter_placename(name), (
                f"'{name}' is a name-final possessive; must be rejected"
            )

    def test_geographic_possessive_still_passes(self):
        """A possessive ending in a geographic feature is a real place."""
        for name in ("Lady's Island", "Lady’s Island", "Devil's Lake"):
            assert is_valid_chapter_placename(name), (
                f"'{name}' ends in a geographic feature; must pass"
            )


class TestBusinessKeywords:
    def test_pub_and_lodging_suffixes_rejected(self):
        """Pub/lodging/distillery business words reject the name."""
        bad_names = [
            "Dingle Distillery",
            "Sea View Inn",
            "Foxford Lodge",
            "The Anchor Tavern",
            "Ashford Resort",
            "Harbor Bar",
        ]
        for name in bad_names:
            assert not is_valid_chapter_placename(name), (
                f"'{name}' is a business name; must be rejected"
            )


class TestSettlementAllowlist:
    def test_keyword_colliding_settlements_accepted(self):
        """Real settlements whose whole name embeds a keyword must pass."""
        good_names = [
            "Red Bank",  # bank
            "Center Point",  # center
            "Center",  # center
            "Centre",  # centre
            "Golf",  # golf
            "Dollar Bay",  # dollar
            "Bar Harbor",  # bar (new keyword)
        ]
        for name in good_names:
            assert is_valid_chapter_placename(name), f"'{name}' is a real settlement; must pass"

    def test_allowlist_is_exact_full_name_only(self):
        """The allowlist must not exempt a business that merely contains a
        settlement token — a keyword phrase still rejects."""
        assert not is_valid_chapter_placename("Ring of Kerry Golf")
        assert not is_valid_chapter_placename("Red Bank Diner")
