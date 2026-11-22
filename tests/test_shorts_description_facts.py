"""A Short's YouTube description states measured facts only.

It opens with the place its own fact title used ("{place} — POV drive."), in the
channel voice, never with a camera, vehicle or body lead ("Dashcam POV", "Hands on the
wheel") and never with a "Pt.N" stamp after a unique measured place. A Short with no
measured title place falls back to leads that are true of every Short.
"""

from __future__ import annotations

import re

import pytest
from test_publishing_anti_spam_gate import _record, _yt

from gomoveshift.pipeline import longform_metadata as lm
from gomoveshift.pipeline import social_generator as sg
from gomoveshift.pipeline.social_generator import SocialGenerator, TitleFacts
from gomoveshift.pipeline.validation.content_gates import find_grammar_issues
from gomoveshift.publishing.anti_spam_gate import check_batch, text_opener_key

LABEL = "Polk County, NC"
REGIONS = ["Polk County, North Carolina"]
DATE = "2026-02-24"
TOWNS = ["Tryon", "Saluda", "Columbus", "Mill Spring", "Landrum", "Lynn", "Sunny View"]
BANNED = re.compile(
    r"dashcam|seat belt|wheel|windshield|hood|\bcab\b|dashboard|front-seat|driver's-seat"
    r"|ride along|eyes forward|road ahead",
    re.I,
)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(lm, "load_published_titles", lambda: set())
    monkeypatch.setattr(SocialGenerator, "_meta_ranked", lambda self, platform, order: order)
    monkeypatch.setattr(sg, "shipped_title_snapshot", lambda: None)


def _route(*towns):
    return {
        "retained_seconds": 38.0,
        "containment_basis": "retained_gps",
        "places": [
            {"name": n, "evidence_id": f"ev-{n}", "required": True, "retained_seconds": 38.0}
            for n in towns
        ],
    }


def _gen(project="drive"):
    return SocialGenerator(
        location=LABEL, project_name=project, series_part=1, spooled_texts={}, experiments=()
    )


def _run(routes, place_ok=True, gen=None):
    n = len(routes)
    return (gen or _gen()).generate_all(
        num_shorts=n,
        per_short_locations=[LABEL] * n,
        per_short_regions=[REGIONS] * n,
        per_short_roads=[[] for _ in routes],
        per_short_route_records=routes,
        per_short_title_facts=[TitleFacts(DATE, k, 38, place_ok) for k in range(1, n + 1)],
    )


def _batch(n=len(TOWNS)):
    return _run([_route(t) for t in TOWNS[:n]])


def test_description_opens_with_the_titles_measured_place():
    posts = _batch()
    for post in posts:
        assert post.youtube_title in TOWNS
        assert post.youtube_description.startswith(f"{post.youtube_title} — POV drive."), (
            post.youtube_description
        )


def test_no_description_carries_a_camera_vehicle_or_body_lead():
    for post in [*_batch(), *_run([None] * 7, place_ok=False)]:
        assert not BANNED.search(post.youtube_description), post.youtube_description
    gen = _gen()
    for i in range(1, 15):
        assert not BANNED.search(gen._build_youtube_description(i))
        assert not BANNED.search(gen._build_youtube_description(i, interactive=True))
    assert not [lead for lead in sg.YOUTUBE_DESCRIPTION_LEADS if BANNED.search(lead)]


def test_unique_measured_place_needs_no_part_stamp():
    for post in _batch():
        assert "Pt." not in post.youtube_description, post.youtube_description


def test_a_batch_at_one_place_has_distinct_openers_and_passes_check_batch():
    posts = _batch()
    descs = [p.youtube_description for p in posts]
    assert len({text_opener_key(d) for d in descs}) == len(descs), descs
    records = [
        _record(i, youtube=_yt(p.youtube_title, p.youtube_description)) for i, p in enumerate(posts)
    ]
    assert check_batch(records, per_location_cap=None) == []


def test_fallback_without_a_measured_place_still_varies_and_passes_the_grammar_gate():
    posts = _run([None] * 7, place_ok=False)
    descs = [p.youtube_description for p in posts]
    assert len({text_opener_key(d) for d in descs}) == len(descs), descs
    assert all(not find_grammar_issues(d) for d in descs), descs
    assert all("— POV drive." not in d for d in descs)


def test_interactive_description_still_withholds_the_place():
    gen = _gen()
    descs = [gen._build_youtube_description(i, interactive=True) for i in range(1, 8)]
    for d in descs:
        head = d.split("\n\n")[0]
        assert "Polk" not in head and "this route" in head
        assert d.endswith(f"\U0001f4cd {gen.social_location_full}")
    assert len({text_opener_key(d) for d in descs}) == len(descs), descs
