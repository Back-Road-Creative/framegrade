"""Long-form YouTube titles explore 1 upload in 4 (Waypost ``title-policy-3``).

Only the chosen title earns reach evidence, so while the baseline always ships
an alternative can never reach Waypost's ``min_n``. The pre-render approval
(``approve_retained_title``) hands Waypost a stable ``explore_key`` (the
baseline title, the same on every re-render and resume) and Waypost picks the
uploads that explore. Shorts never do. An explored title then passes the same
gates as the baseline: the GPS endpoint check and the final-text review.
Every place name here is invented."""

from __future__ import annotations

from seo_manager.analytics.title_selection import Place, TitleFacts, explores

from gomoveshift.pipeline.final_text_record import build_final_text_review
from gomoveshift.pipeline.location_evidence import refuse_unsupported_title
from gomoveshift.pipeline.title_decision import approve_retained_title, decide_title

from test_final_text_record import OST, _location, _plan_lf, _route, _state
from w9_drive_text import attach_drive_description


def _facts(baseline: str, fmt: str = "longform") -> TitleFacts:
    return TitleFacts(
        format=fmt,
        baseline_title=baseline,
        origin=Place("Brindlemoor", "seg-1:gps"),
        destination=Place("Quillhaven", "seg-9:gps"),
        region=Place("Nordale", "seg-1:geocode"),
        suffix=" in 4K",
    )


def _baselines(explore: bool, count: int = 3) -> list[str]:
    titles = (f"Brindlemoor to Quillhaven, Nordale (Part {i}) in 4K" for i in range(200))
    return [t for t in titles if explores(t, 4) is explore][:count]


def _decide(baseline: str, fmt: str = "longform", **kw) -> dict:
    return decide_title(fmt, baseline, None, suffix=" in 4K", facts=_facts(baseline, fmt), **kw)


def test_an_exploring_longform_ships_its_alternative():
    for baseline in _baselines(True):
        decision = _decide(baseline, explore_key=baseline)
        assert decision["reason"] == "exploration"
        assert decision["chosen_index"] != 0
        assert decision["chosen_title"] == "Nordale Drive: Brindlemoor to Quillhaven in 4K"
        assert decision["policy_version"] == "title-policy-3"


def test_other_longforms_keep_their_baseline():
    for baseline in _baselines(False):
        decision = _decide(baseline, explore_key=baseline)
        assert decision["reason"] == "insufficient_evidence"
        assert decision["chosen_title"] == baseline


def test_no_explore_key_never_explores():
    for baseline in _baselines(True):
        decision = _decide(baseline)
        assert decision["reason"] == "insufficient_evidence"
        assert decision["chosen_title"] == baseline


def test_the_same_upload_always_gets_the_same_decision():
    baseline = _baselines(True, 1)[0]
    first = _decide(baseline, explore_key=baseline)
    again = _decide(baseline, explore_key=baseline)
    assert first["chosen_title"] == again["chosen_title"]
    assert first["reason"] == again["reason"] == "exploration"


def test_shorts_never_explore():
    for baseline in _baselines(True):
        facts = TitleFacts(
            format="short",
            baseline_title=baseline,
            place=Place("Nordale", "seg-1:geocode"),
            highlight=Place("Ashtarn Ridge", "seg-4:gps"),
        )
        decision = decide_title("short", baseline, None, suffix="", facts=facts, explore_key=baseline)
        assert decision["reason"] == "insufficient_evidence"
        assert decision["chosen_title"] == baseline


# The pre-render approval of the final-text fixture's drive ("Harrow to Pellmere,
# Westmark in 4K"), whose baseline falls in the explored quarter.


def _approved() -> dict:
    return approve_retained_title(OST, _route(), _location())


def test_the_pre_render_approval_explores_by_its_baseline():
    decision = _approved()
    assert explores(decision["baseline_title"], 4)
    assert decision["reason"] == "exploration"
    assert decision["chosen_title"] == "Westmark Drive: Harrow to Pellmere in 4K"


def test_an_explored_title_passes_the_gps_endpoint_check():
    refuse_unsupported_title(_approved()["chosen_title"], _location(), lf_index=1)


def _review(decision: dict) -> dict:
    state = _state()
    meta = {"title": decision["chosen_title"], "title_decision": decision}
    attach_drive_description(state, meta)
    return build_final_text_review(state, _plan_lf(), meta)


def test_the_final_text_review_agrees_with_the_approved_explored_title():
    review = _review(_state().per_lf_pre_render_title_decisions[1])
    assert review["verdict"] == "agree", review


def test_an_explored_title_no_pre_render_approval_chose_still_disagrees():
    baseline = _approved()["baseline_title"]
    decision = decide_title("longform", baseline, _location(), suffix=" in 4K", explore_key=baseline)
    assert decision["reason"] == "exploration"
    review = _review(decision)
    assert review["verdict"] == "disagree"
    assert "title_changed_after_approval" in review["reasons"]
