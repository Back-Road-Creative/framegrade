"""The Shorts experiment arms change the Short's text, deterministically, and
every variant passes the gates the legacy text passes. All data is synthetic."""

from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_publishing_plan_builder import _build  # noqa: E402

from gomoveshift.pipeline import longform_metadata as lm  # noqa: E402
from gomoveshift.pipeline.content_validator import ContentValidator  # noqa: E402
from gomoveshift.pipeline.drive_phases.phase_d import _serialize_publish_plan  # noqa: E402
from gomoveshift.pipeline.social_generator import (  # noqa: E402
    GENERIC_HASHTAGS,
    STAMPED_BODY_TEMPLATES,
    SocialGenerator,
    TitleFacts,
)
from gomoveshift.publishing.anti_spam_gate import shared_prefix_groups  # noqa: E402
from gomoveshift.publishing.experiments import (  # noqa: E402
    SHORTS_ARMS,
    SHORTS_FACTORS,
    SHORTS_HASHTAG_SET_ARMS,
    SHORTS_TITLE_BODY_ARMS,
    assign_shorts_arms,
)
from gomoveshift.publishing.plan_gates import (  # noqa: E402
    assert_plan_text_grammar,
    assert_shorts_metadata_in_region,
)
from gomoveshift.publishing.plan_io import load_publish_plan  # noqa: E402
from gomoveshift.publishing.youtube_adapter import short_review_refusal  # noqa: E402

LABEL = "Wythe County, Virginia"
REGIONS = ["Wythe County, Virginia"]
DATE = "2026-09-14"


@pytest.fixture(autouse=True)
def _no_published_titles(monkeypatch):
    monkeypatch.setattr(lm, "load_published_titles", lambda: set())


def _gen(**kw) -> SocialGenerator:
    return SocialGenerator(location=LABEL, project_name="drive", series_part=1, **kw)


def _facts(n=8, place_ok=True):
    return [TitleFacts(date=DATE, ordinal=k, secs=41, place_ok=place_ok) for k in range(1, n + 1)]


def _all(gen, facts, **kw):
    n = len(facts)
    return gen.generate_all(
        num_shorts=n,
        per_short_locations=[LABEL] * n,
        per_short_regions=[REGIONS] * n,
        per_short_title_facts=facts,
        **kw,
    )


def _dict(post):
    out = {k: v for k, v in post.to_dict().items() if k != "generated_at"}
    # The title decision stamps wall-clock evidence time; it is not post text.
    out["title_decision"] = {
        k: v for k, v in (out.get("title_decision") or {}).items() if k != "evidence_as_of"
    }
    return out


def _rendered(post):
    return (post.youtube_title, post.youtube_description, post.youtube_tags, post.captions)


# --- tracked config ----------------------------------------------------------


def test_arm_config_is_one_source_of_truth():
    assert tuple(SHORTS_ARMS) == SHORTS_FACTORS
    assert SHORTS_ARMS["yt_title_body"] == tuple(SHORTS_TITLE_BODY_ARMS)
    assert SHORTS_ARMS["yt_hashtag_set"] == tuple(SHORTS_HASHTAG_SET_ARMS)
    assert all(len(v) >= 2 for v in SHORTS_ARMS.values())


def test_every_title_body_is_an_existing_generator_body():
    for bodies in SHORTS_TITLE_BODY_ARMS.values():
        assert set(bodies) <= set(STAMPED_BODY_TEMPLATES)
    assert len({b for v in SHORTS_TITLE_BODY_ARMS.values() for b in v}) == sum(
        len(v) for v in SHORTS_TITLE_BODY_ARMS.values()
    ), "arms must not share a body"


def test_every_hashtag_is_an_existing_generic_hashtag():
    for tags in SHORTS_HASHTAG_SET_ARMS.values():
        assert set(tags) <= set(GENERIC_HASHTAGS)
    assert len(set(SHORTS_HASHTAG_SET_ARMS.values())) == len(SHORTS_HASHTAG_SET_ARMS)


# --- assignment --------------------------------------------------------------


def test_assignment_is_deterministic_and_stratified():
    def run():
        history: dict = {}
        return [assign_shorts_arms(("p", "slot1"), k, history, SHORTS_FACTORS) for k in range(1, 9)]

    first = run()
    assert first == run()
    for factor, arms in SHORTS_ARMS.items():
        counts = [sum(1 for a in first if a[factor] == arm) for arm in arms]
        assert max(counts) - min(counts) <= 1, factor


def test_strata_are_balanced_independently():
    history: dict = {}
    for slot in ("a", "b"):
        for k in range(2):
            assign_shorts_arms(("p", slot), k, history, ["yt_hashtag_set"])
    for slot in ("a", "b"):
        assert sorted(history[("yt_hashtag_set", "p", slot)]) == sorted(
            SHORTS_ARMS["yt_hashtag_set"]
        )


def test_generate_all_assigns_and_stores_arms_deterministically():
    posts = _all(_gen(), _facts(), shorts_arms_enabled=True)
    again = _all(_gen(), _facts(), shorts_arms_enabled=True)
    assert [p.experiment_arms for p in posts] == [p.experiment_arms for p in again]
    assert [_rendered(p) for p in posts] == [_rendered(p) for p in again]
    for post in posts:
        assert set(SHORTS_FACTORS) <= set(post.experiment_arms)
        for factor in SHORTS_FACTORS:
            assert post.experiment_arms[factor] in SHORTS_ARMS[factor]


# --- rendering ---------------------------------------------------------------


def _by_arm(posts, factor):
    out: dict = {}
    for post in posts:
        out.setdefault(post.experiment_arms[factor], []).append(post)
    return out


def test_title_body_arms_change_the_title_as_specified():
    posts = _all(_gen(), _facts(), shorts_arms_enabled=True)
    by_arm = _by_arm(posts, "yt_title_body")
    assert set(by_arm) == set(SHORTS_TITLE_BODY_ARMS)
    for arm, members in by_arm.items():
        for post in members:
            body = post.youtube_title.split(": ", 1)[1].split(" #", 1)[0]
            assert body.startswith("Wythe County") != (arm == "clip_led"), (arm, body)
            assert ("seconds" in body) == (arm == "clip_led"), (arm, body)


def test_hashtag_set_arms_change_the_description_as_specified():
    posts = _all(_gen(), _facts(), shorts_arms_enabled=True)
    by_arm = _by_arm(posts, "yt_hashtag_set")
    assert set(by_arm) == set(SHORTS_HASHTAG_SET_ARMS)
    for arm, members in by_arm.items():
        for post in members:
            tags = re.findall(r"#\w+", post.youtube_description)
            assert tags[0] == "#Shorts"
            for tag in SHORTS_HASHTAG_SET_ARMS[arm]:
                assert tag in tags
            for other, other_tags in SHORTS_HASHTAG_SET_ARMS.items():
                if other != arm:
                    assert not set(other_tags) & set(tags) - set(SHORTS_HASHTAG_SET_ARMS[arm])


def test_no_arm_renders_byte_identical_to_legacy():
    legacy = _all(_gen(), _facts())
    flagged_off = _all(_gen(), _facts(), shorts_arms_enabled=False)
    assert [_dict(p) for p in legacy] == [_dict(p) for p in flagged_off]
    assert all(not set(SHORTS_FACTORS) & set(p.experiment_arms) for p in legacy)


def test_unknown_arm_renders_as_legacy_and_is_not_stored():
    legacy = _gen().generate_post(3, title_facts=_facts()[2])
    for arms in (
        {"yt_title_body": "nope", "yt_hashtag_set": "nope"},
        {"not_a_factor": "place_led"},
    ):
        post = _gen().generate_post(3, title_facts=_facts()[2], shorts_arms=arms)
        assert _dict(post) == _dict(legacy)


def test_pov_hashtag_arm_equals_the_legacy_description():
    legacy = _gen()._build_youtube_description(2)
    assert _gen()._build_youtube_description(2, hashtag_arm="pov") == legacy
    assert _gen()._build_youtube_description(2, hashtag_arm="road") != legacy


def test_a_factor_the_text_cannot_carry_is_not_assigned(tmp_path):
    # No verified place: no title-body arm, hashtag arm still rendered.
    posts = _all(_gen(), _facts(place_ok=False), shorts_arms_enabled=False)
    gen = _gen()
    assert gen._shorts_arm_factors(1, TitleFacts(DATE, 1, 41, False)) == ["yt_hashtag_set"]
    assert gen._shorts_arm_factors(1, None) == ["yt_hashtag_set"]
    assert gen._shorts_arm_factors(1, TitleFacts(DATE, 1, 41, True)) == list(SHORTS_FACTORS)
    assert posts


# --- every variant passes the existing gates ---------------------------------


def _route_record():
    from gomoveshift.pipeline.route_claims import RouteEvidence, _Bin

    bins = [_Bin(60.0, "Wythe County", "Virginia", None)]
    return RouteEvidence(
        retained_seconds=60.0, gaps=[], places=[], unsettled_seconds=0.0, bins=bins,
        boundary_proofs={}, windows_key="k",
    ).record()  # fmt: skip


def test_every_variant_passes_the_existing_gates():
    posts = _all(_gen(), _facts(), shorts_arms_enabled=True)
    legacy = _all(_gen(), _facts())
    assert {p.experiment_arms["yt_title_body"] for p in posts} == set(SHORTS_TITLE_BODY_ARMS)
    assert {p.experiment_arms["yt_hashtag_set"] for p in posts} == set(SHORTS_HASHTAG_SET_ARMS)
    validator = ContentValidator()
    for post in posts:
        assert len(post.youtube_title) <= 100
        validator.enforce(
            titles=[post.youtube_title, post.tiktok_title],
            captions=[c for c in (post.caption, post.first_comment) if c],
            descriptions=[post.youtube_description],
        )
    assert_plan_text_grammar({1: posts})
    assert_shorts_metadata_in_region({1: posts})
    assert not shared_prefix_groups([p.youtube_title for p in posts])
    assert not shared_prefix_groups([p.youtube_title for p in legacy])

    plan = _build(
        posts=posts,
        parent_recording_date_by_short_index={p.short_index: DATE for p in posts},
    )
    for record in plan.shorts:
        record = dataclasses.replace(record, route_record=_route_record())
        assert short_review_refusal(record) is None, record.payloads.youtube.title


# --- persistence -------------------------------------------------------------


def test_arms_are_persisted_on_the_plan_and_survive_the_sidecar(tmp_path):
    posts = _all(_gen(), _facts(4), shorts_arms_enabled=True)
    plan = _build(
        posts=posts,
        parent_recording_date_by_short_index={p.short_index: DATE for p in posts},
    )
    assert [r.experiment_arms for r in plan.shorts] == [p.experiment_arms for p in posts]
    path = tmp_path / "p_publish_plan.json"
    path.write_text(_serialize_publish_plan(plan), encoding="utf-8")
    loaded = load_publish_plan(path)
    assert [r.experiment_arms for r in loaded.shorts] == [p.experiment_arms for p in posts]
