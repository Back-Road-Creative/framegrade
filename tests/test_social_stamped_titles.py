"""Drive-path short titles are date-stamped: ``"{Mon} {D}, {k}th stretch: {body}"``.

The stamp carries the whole first-3-word prefix and is derived from the
short's own day key (date) and its position in that day's sidecar, so titles
are unique by construction instead of by a shrinking frame pool. The place slot
is the short's own ``location_short`` and only when its place is clip-GPS
verified; the legacy <=6-frame composer serves callers with no facts.
"""

from __future__ import annotations

import re

import pytest

from gomoveshift.models import ResolvedLocation
from gomoveshift.pipeline import longform_metadata as lm
from gomoveshift.pipeline import social_generator as sg
from gomoveshift.pipeline.social_generator import (
    STAMPED_BODY_TEMPLATES,
    STAMPED_PLACE_FREE_TEMPLATES,
    SocialGenerator,
    TitleFacts,
    assert_day_title_stamps,
    assert_title_prefixes_unique,
    day_title_ordinals,
    ordinal_suffix,
    title_stamp,
)
from gomoveshift.publishing import anti_spam_gate
from gomoveshift.publishing.anti_spam_gate import shared_prefix_groups, title_prefix_key


@pytest.fixture(autouse=True)
def _no_published_titles(monkeypatch):
    monkeypatch.setattr(lm, "load_published_titles", lambda: set())


LABEL = "Wythe County, VA"
REGIONS = ["Wythe County, Virginia"]


def _gen(lf: int = 1, location: str = LABEL, **kw) -> SocialGenerator:
    return SocialGenerator(location=location, project_name="drive", series_part=lf, **kw)


def _facts(date: str, ordinals, secs: int = 41, place_ok: bool = True) -> list[TitleFacts]:
    return [TitleFacts(date=date, ordinal=k, secs=secs, place_ok=place_ok) for k in ordinals]


def _all(gen, facts, label: str = LABEL):
    n = len(facts)
    return gen.generate_all(
        num_shorts=n,
        per_short_locations=[label] * n,
        per_short_regions=[REGIONS] * n,
        per_short_title_facts=facts,
    )


def test_ordinal_suffixes():
    expected = {
        1: "1st",
        2: "2nd",
        3: "3rd",
        4: "4th",
        11: "11th",
        12: "12th",
        13: "13th",
        21: "21st",
        22: "22nd",
        23: "23rd",
        101: "101st",
        111: "111th",
        112: "112th",
    }
    for n, text in expected.items():
        assert f"{n}{ordinal_suffix(n)}" == text
        assert anti_spam_gate._WORD_RE.findall(text) == [text]


def test_stamp_prefix_is_month_day_ordinal():
    stamp = title_stamp("2026-09-14", 3)
    assert stamp == "Sep 14, 3rd stretch"
    assert title_prefix_key(stamp + ": Wythe County") == ("sep", "14", "3rd")


def test_stamp_refuses_a_bad_date():
    for bad in ("2026-9-14x", "not-a-date", "2026-13-01", ""):
        with pytest.raises(ValueError):
            title_stamp(bad, 1)


def test_day_ordinals_order_by_start_then_rank():
    entries = [
        {"rank": 3, "start_sec": 50.0},
        {"rank": 1, "start_sec": 200.0},
        {"rank": 2, "start_sec": 50.0},
    ]
    assert day_title_ordinals(entries) == {2: 1, 3: 2, 1: 3}


@pytest.mark.parametrize(
    "entries",
    [
        [{"rank": 1, "start_sec": 1.0}, {"rank": 1, "start_sec": 2.0}],
        [{"rank": 1}],
        [{"rank": 1, "start_sec": "soon"}],
        [{"start_sec": 1.0}],
    ],
)
def test_day_ordinals_refuse_duplicate_rank_or_missing_start(entries):
    with pytest.raises(ValueError):
        day_title_ordinals(entries)


def test_assert_day_title_stamps_accepts_a_clean_day_and_refuses_a_bad_one():
    assert_day_title_stamps("2026-09-14", [{"rank": 1, "start_sec": 1.0}])
    with pytest.raises(ValueError):
        assert_day_title_stamps("2026-09-14", [{"rank": 1}])


def test_prefix_assert_names_every_colliding_key():
    with pytest.raises(RuntimeError, match=r"phase D.*'a'.*'b'"):
        assert_title_prefixes_unique(
            {"a": "Sep 14, 1st stretch: x", "b": "Sep 14, 1st stretch: y", "c": "Other title here"},
            where="phase D",
        )
    assert_title_prefixes_unique({"a": "Sep 14, 1st stretch: x"}, where="phase D")


def test_three_lfs_fourteen_each_no_shared_prefix():
    lf1 = _all(_gen(1), _facts("2026-09-13", range(1, 15)))
    lf2 = _all(_gen(2), _facts("2026-09-14", [1, 2, 3, 5, 6, 8, 9, 10, 12, 13, 15, 16, 17, 20]))
    lf3 = _all(
        _gen(3), _facts("2026-09-14", [4, 7, 11, 14, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28])
    )
    posts = lf1 + lf2 + lf3
    assert len(posts) == 42
    yt = [p.youtube_title for p in posts]
    assert shared_prefix_groups(yt) == {}
    assert len(set(yt)) == 42


def test_title_names_only_short_label():
    parent = SocialGenerator(
        location="Fancy Gap to Hillsville via I-77, VA",
        project_name="p",
        series_part=1,
        road_name="I-77",
        nearest_poi="Hillsville Flea Market",
        county="Carroll County",
        resolved_location=ResolvedLocation(destination="Hillsville Flea Market", country_code="US"),
    )
    (post,) = parent.generate_all(
        num_shorts=1,
        per_short_locations=[LABEL],
        per_short_regions=[REGIONS],
        per_short_title_facts=_facts("2026-09-14", [1]),
    )
    for title in (post.youtube_title, post.tiktok_title):
        assert "Wythe County" in title
        for banned in ("Fancy", "Hillsville", "I-77", "Flea", "VA", "Virginia", "Carroll"):
            assert banned not in title, (banned, title)


def test_unverified_place_gets_place_free_body():
    gen = _gen()
    for k in range(1, 8):
        (post,) = gen.generate_all(
            num_shorts=1,
            per_short_locations=[LABEL],
            per_short_regions=[REGIONS],
            per_short_title_facts=_facts("2026-09-14", [k], place_ok=False),
        )
        for title in (post.youtube_title, post.tiktok_title):
            assert "Wythe" not in title and "County" not in title, title
            assert title.startswith(title_stamp("2026-09-14", k) + ": ")


def test_place_free_body_when_generator_has_no_location_short():
    fact = TitleFacts(date="2026-09-14", ordinal=2, secs=30, place_ok=True)
    title = _gen(location="").generate_post(1, title_facts=fact).youtube_title
    assert title == "Sep 14, 2nd stretch: 30 seconds on a Monday"


def test_ledger_walk_changes_body_not_stamp(monkeypatch):
    gen = _gen()
    fact = TitleFacts(date="2026-09-14", ordinal=4, secs=41, place_ok=True)
    first = gen.generate_post(1, title_facts=fact).youtube_title
    monkeypatch.setattr(lm, "load_published_titles", lambda: {first})
    second = gen.generate_post(1, title_facts=fact).youtube_title
    assert second != first
    stamp = title_stamp("2026-09-14", 4) + ": "
    assert first.startswith(stamp) and second.startswith(stamp)


def test_truncation_keeps_stamp():
    fact = TitleFacts(date="2026-09-14", ordinal=12, secs=41, place_ok=True)
    gen = _gen(location="A" * 120 + ", VA")
    for suffix, max_len in (("titles", 100), ("tiktok", 90)):
        title = gen._compose_title(1, suffix, max_len, facts=fact)
        assert len(title) == max_len
        assert title.startswith("Sep 14, 12th stretch: ")


def test_deterministic_across_runs():
    facts = _facts("2026-09-14", range(1, 9))
    a = [p.youtube_title for p in _all(_gen(), facts)]
    b = [p.youtube_title for p in _all(_gen(), facts)]
    assert a == b


def test_per_short_title_facts_length_is_checked():
    with pytest.raises(ValueError, match="per_short_title_facts"):
        _gen().generate_all(num_shorts=2, per_short_title_facts=_facts("2026-09-14", [1]))


def test_none_entry_uses_the_legacy_composer_for_that_short():
    (post,) = _gen().generate_all(num_shorts=1, per_short_title_facts=[None])
    assert not re.match(r"^[A-Z][a-z]{2} \d+, \d+\w+ stretch", post.youtube_title)


def test_no_facts_is_byte_identical_to_legacy_composition():
    gen = _gen(terrain_type="coastal")
    legacy = [gen.generate_post(i).youtube_title for i in range(1, 4)]
    via_all = [p.youtube_title for p in _gen(terrain_type="coastal").generate_all(num_shorts=3)]
    assert legacy == via_all
    assert not any("stretch" in t for t in legacy)


def test_hand_title_still_wins_over_facts(tmp_path):
    import yaml

    titles = tmp_path / "titles.yaml"
    titles.write_text(yaml.safe_dump({1: {"title": "My own hand title"}}))
    gen = _gen(titles_file=titles)
    post = gen.generate_post(1, title_facts=_facts("2026-09-14", [1])[0])
    assert post.youtube_title == "My own hand title"


def test_stamped_frames_static_rules():
    bodies = STAMPED_BODY_TEMPLATES + STAMPED_PLACE_FREE_TEMPLATES
    prepositions = r"\b(?:in|near|to|into|through|toward|towards|at|of)\s+$"
    for tmpl in bodies:
        for m in re.finditer(r"\{location\}", tmpl):
            assert not re.search(prepositions, tmpl[: m.start()], re.I), tmpl
        assert not re.search(r"\{weekday\}\s+drive", tmpl, re.I), tmpl
        assert not sg._DIRECTION_FRAMING_RE.search(
            tmpl.format(location="Wythe", secs=1, weekday="Monday")
        )
    assert all("{location}" not in t for t in STAMPED_PLACE_FREE_TEMPLATES)
    counties = {w.lower() for w in re.findall(r"[A-Za-z]+", LABEL)}
    names = {m.lower() for m in sg._MONTHS} | {d.lower() for d in sg._WEEKDAYS}
    assert not (names & counties)
