"""A drive Short's title names the place its own clip measured, never a stamp.

The final-text review backs one cited place per title (a town held for every
retained second, or a county the clip voted) and no road, so the title is that
place and nothing else: no "Nth stretch", no "N seconds", no weekday. Two
Shorts that would get the same title take the next measured place; only when no
measured place is left does the one Short fall back to its stamped title.
"""

from __future__ import annotations

import logging
import re

import pytest

from gomoveshift.pipeline import longform_metadata as lm
from gomoveshift.pipeline import social_generator as sg
from gomoveshift.pipeline.social_generator import SocialGenerator, TitleFacts, title_stamp
from gomoveshift.pipeline.title_decision import short_place_facts
from gomoveshift.publishing.anti_spam_gate import shared_prefix_groups

LABEL = "Polk County, NC"
REGIONS = ["Polk County, North Carolina"]
DATE = "2026-02-24"
BANNED = re.compile(
    r"stretch|seconds|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday", re.I
)


@pytest.fixture(autouse=True)
def _no_published_titles(monkeypatch):
    monkeypatch.setattr(lm, "load_published_titles", lambda: set())
    # Meta ranking and the title snapshot read Waypost modules (caption_ranking,
    # title_evidence) that not every environment has; titles do not depend on them.
    monkeypatch.setattr(SocialGenerator, "_meta_ranked", lambda self, platform, order: order)
    monkeypatch.setattr(sg, "shipped_title_snapshot", lambda: None)


def _route(*towns, retained=38.0):
    return {
        "retained_seconds": retained,
        "containment_basis": "retained_gps",
        "places": [
            {"name": n, "evidence_id": f"ev-{n}", "required": True, "retained_seconds": retained}
            for n in towns
        ],
    }


def _gen():
    return SocialGenerator(
        location=LABEL, project_name="drive", series_part=1, spooled_texts={}, experiments=()
    )


def _run(routes, roads=None, place_ok=True, date=DATE):
    n = len(routes)
    return _gen().generate_all(
        num_shorts=n,
        per_short_locations=[LABEL] * n,
        per_short_regions=[REGIONS] * n,
        per_short_roads=roads or [[] for _ in routes],
        per_short_route_records=routes,
        per_short_title_facts=[TitleFacts(date, k, 38, place_ok) for k in range(1, n + 1)],
    )


def _titles(posts):
    return [t for p in posts for t in (p.youtube_title, p.tiktok_title)]


def test_title_is_the_most_specific_measured_town_not_a_stamp():
    (post,) = _run([_route("Tryon")])
    assert post.youtube_title == "Tryon"
    assert post.tiktok_title == "Tryon"
    assert short_place_facts(
        post.youtube_title,
        "Tryon",
        REGIONS,
        segment_id="s",
        towns={"Tryon": "ev-Tryon"},
    )


def test_no_town_names_the_voted_county():
    (post,) = _run([None])
    assert post.youtube_title == "Polk County"


def test_no_title_carries_a_stamp_length_or_weekday():
    routes = [_route("Tryon"), _route("Saluda"), None, _route("Columbus"), _route("Landrum")]
    for title in _titles(_run(routes)):
        assert not BANNED.search(title), title


def test_a_road_is_never_in_a_title():
    posts = _run([_route("Tryon"), None], roads=[["US-64"], ["US-64", "NC-108"]])
    for title in _titles(posts):
        assert "US-64" not in title and "NC-108" not in title, title


def test_a_town_the_review_would_refuse_is_never_named():
    partial = _route("Saluda", retained=38.0)
    partial["places"][0]["retained_seconds"] = 12.0
    (post,) = _run([partial])
    assert "Saluda" not in post.youtube_title
    assert post.youtube_title == "Polk County"


def test_batch_titles_are_unique_and_pass_the_prefix_gate():
    routes = [_route("Tryon", "Saluda"), _route("Tryon", "Saluda"), _route("Saluda")]
    posts = _run(routes)
    titles = [p.youtube_title for p in posts]
    assert len(set(titles)) == len(titles)
    assert shared_prefix_groups(titles) == {}
    assert titles[:2] == ["Tryon", "Saluda"]


def test_same_place_twice_takes_the_next_measured_place():
    posts = _run([_route("Tryon"), _route("Tryon"), _route("Tryon")])
    assert [p.youtube_title for p in posts[:2]] == ["Tryon", "Polk County"]


def test_with_no_measured_place_left_one_short_falls_back_to_its_stamp(caplog):
    caplog.set_level(logging.INFO, logger="gomoveshift.pipeline.social_generator")
    posts = _run([_route("Tryon")] * 3)
    last = posts[2].youtube_title
    assert last.startswith(title_stamp(DATE, 3) + ": ")
    assert len({p.youtube_title for p in posts}) == 3
    assert any("stamped title" in r.message for r in caplog.records if r.levelno >= logging.INFO)


def test_a_published_title_is_never_repeated(monkeypatch):
    monkeypatch.setattr(lm, "load_published_titles", lambda: {"Tryon"})
    (post,) = _run([_route("Tryon")])
    assert post.youtube_title == "Polk County"


def test_unverified_place_keeps_the_place_free_stamped_title():
    (post,) = _run([_route("Tryon")], place_ok=False)
    assert post.youtube_title.startswith(title_stamp(DATE, 1) + ": ")
    assert "Tryon" not in post.youtube_title and "Polk" not in post.youtube_title


def test_the_title_body_arm_is_not_assigned_to_a_fact_title():
    gen = _gen()
    assert "yt_title_body" not in gen._shorts_arm_factors(1, TitleFacts(DATE, 1, 38, True))
    assert sg.SHORTS_TITLE_BODY_ARMS  # the experiment's definition is untouched


def test_a_place_a_spooled_title_already_opens_with_is_skipped():
    # the queue gate refuses a title whose opener key a spooled youtube text holds
    gen = SocialGenerator(
        location=LABEL,
        project_name="drive",
        series_part=1,
        spooled_texts={"youtube": ["Tryon"]},
        experiments=(),
    )
    (post,) = gen.generate_all(
        num_shorts=1,
        per_short_locations=[LABEL],
        per_short_regions=[REGIONS],
        per_short_route_records=[_route("Tryon")],
        per_short_title_facts=[TitleFacts(DATE, 1, 38, True)],
    )
    assert post.youtube_title == "Polk County"


def test_shared_used_titles_keep_titles_unique_across_calls():
    used: dict = {}
    titles = []
    for lf in (1, 2):
        (post,) = _gen().generate_all(
            num_shorts=1,
            per_short_locations=[LABEL],
            per_short_regions=[REGIONS],
            per_short_route_records=[_route("Tryon")],
            per_short_title_facts=[TitleFacts(DATE, lf, 38, True)],
            used_titles=used,
        )
        titles.append(post.youtube_title)
    assert titles[0] == "Tryon" and titles[1] == "Polk County"
