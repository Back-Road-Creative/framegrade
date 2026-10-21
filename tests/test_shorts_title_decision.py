"""U5 (Shorts): the Shorts title goes through Waypost's title decision, and
the FINAL YouTube payload (after the hashtag append) can only carry it.

Waypost's ``select_title`` has no single-place Short template yet, so every
Short records a ``facts_unavailable`` decision that keeps the composed
baseline; the guarantee tested here is that no later path swaps it.
"""

import pytest

from gomoveshift.pipeline.social_generator import SocialGenerator, SocialPost
from gomoveshift.pipeline.title_decision import (
    FACTS_UNAVAILABLE,
    TitleDecisionMismatchError,
)
from test_publishing_plan_builder import _build, _post


def _generated_post():
    gen = SocialGenerator(location="County Kerry, Ireland", project_name="u5-shorts", series_part=1)
    return gen.generate_post(short_index=1)


def _decision(title):
    return {
        "format": "short",
        "chosen_index": 0,
        "chosen_title": title,
        "reason": FACTS_UNAVAILABLE,
        "candidates": [],
    }


def test_generated_short_carries_its_title_decision():
    post = _generated_post()
    decision = post.title_decision
    assert decision["format"] == "short"
    assert decision["reason"] == FACTS_UNAVAILABLE
    assert decision["chosen_title"] == post.youtube_title


def test_decision_survives_the_sidecar_round_trip():
    post = _generated_post()
    again = SocialPost(**post.to_dict())
    assert again.title_decision == post.title_decision


def test_pre_u5_sidecar_without_a_decision_still_loads():
    data = _generated_post().to_dict()
    data.pop("title_decision")
    assert SocialPost(**data).title_decision is None


def test_final_payload_title_is_the_decided_title_plus_hashtags_only():
    title = "County Kerry Drive #1"
    record = _build(posts=[_post(title_decision=_decision(title))]).shorts[0]
    final = record.payloads.youtube.title
    assert final.startswith(title)
    rest = final[len(title) :].split()
    assert all(word.startswith("#") for word in rest)


def test_a_replaced_short_title_is_refused_before_the_plan_exists():
    post = _post(
        youtube_title="Some Other Title", title_decision=_decision("County Kerry Drive #1")
    )
    with pytest.raises(TitleDecisionMismatchError):
        _build(posts=[post])


# ─── U5c: a Short whose place is its own clip-GPS vote goes to Waypost as a
# single-place TitleFacts (waypost#4), so the decision carries that evidence.

from gomoveshift.pipeline.social_generator import TitleFacts as StampFacts  # noqa: E402
from gomoveshift.pipeline.title_decision import short_place_facts  # noqa: E402


def _stamped_short(place_ok=True, regions=("Washington County, PA",)):
    gen = SocialGenerator(location="Washington County, PA", project_name="u5c", series_part=1)
    variant = gen._per_short_variant("Washington County, PA", list(regions))
    facts = StampFacts(date="2026-03-04", ordinal=1, secs=42, place_ok=place_ok)
    return variant.generate_post(short_index=1, title_facts=facts)


def test_place_backed_by_the_clip_vote_becomes_a_single_place_fact():
    facts = short_place_facts(
        "Mar 4, 1st stretch: Washington County",
        "Washington County",
        ["Washington County, PA"],
        segment_id="short001",
    )
    assert facts.format == "short"
    assert facts.place.name == "Washington County"
    assert facts.place.evidence_id
    assert facts.origin is None and facts.destination is None and facts.region is None


def test_place_absent_from_the_clip_vote_is_not_a_fact():
    assert (
        short_place_facts(
            "Mar 4, 1st stretch: Greene County",
            "Greene County",
            ["Washington County, PA"],
            segment_id="short001",
        )
        is None
    )


def test_a_named_short_records_its_place_evidence():
    post = _stamped_short()
    decision = post.title_decision
    assert decision["reason"] != FACTS_UNAVAILABLE
    assert decision["chosen_title"] == post.youtube_title
    baseline = decision["candidates"][0]
    assert baseline["template_id"] == "place"
    assert len(baseline["claim_evidence"]) == 1


def test_a_place_free_short_stays_facts_unavailable():
    assert _stamped_short(place_ok=False).title_decision["reason"] == FACTS_UNAVAILABLE


def test_generate_all_hands_each_gps_short_its_own_place_evidence():
    gen = SocialGenerator(location="Washington County, PA", project_name="u5c", series_part=1)
    facts = [StampFacts(date="2026-03-04", ordinal=n, secs=40, place_ok=True) for n in (1, 2)]
    posts = gen.generate_all(
        num_shorts=2,
        per_short_title_facts=facts,
        per_short_locations=["Washington County, PA", "Washington County, PA"],
        per_short_regions=[["Washington County, PA"], ["Washington County, PA"]],
    )
    ids = [p.title_decision["candidates"][0]["claim_evidence"] for p in posts]
    assert all(len(i) == 1 for i in ids)
    assert ids[0] != ids[1]
