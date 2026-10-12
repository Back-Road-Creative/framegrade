"""``_fit_shorts_title`` fits an overlong shorts title region-preservingly.

The shorts hard-fit path historically dropped trailing WORDS one at a time,
so "Dingle Peninsula toward Gortmore, County Kerry" degraded to
"Dingle Peninsula toward Gortmore, County" — a dangling "County" fragment.
The fix drops the whole trailing comma clause (", County Kerry") first, keeping
the distinguishing town, and only word/char-chops as a last resort.
"""

from __future__ import annotations

from gomoveshift.pipeline.day_pipeline import _fit_shorts_title


def test_fit_drops_county_clause_not_word():
    """Over-budget "Town, County X" keeps the town, drops the whole county
    clause cleanly — never a dangling "County"."""
    result = _fit_shorts_title("Dingle Peninsula toward Gortmore, County Kerry")
    assert result == "Dingle Peninsula toward Gortmore"
    assert "County" not in result
    assert not result.endswith(",")


def test_fit_single_long_word_still_char_chops():
    """A single unspaceable over-budget word still degrades (last-resort
    char-chop) — the region-clause pass never blocks the existing fallback."""
    result = _fit_shorts_title("Supercalifragilisticexpialidociousplacename")
    assert 0 < len(result) <= 40
