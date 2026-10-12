"""
TDD tests for the objective policy declaration and refusal gate.

An objective policy declares what "better" means for one (platform, format)
pair: primary outcome, guardrails, minimum sample/exposure, maximum evidence
age. Evaluation against a candidate observation must REFUSE -- not default,
not warn-and-continue -- whenever any required declaration is missing or a
candidate fails a hard requirement. Refusal carries a machine-readable
reason naming which requirement failed.
"""

from datetime import datetime, timedelta, timezone

import pytest

from gomoveshift.tracking.objective_policy import (
    YOUTUBE_SHORT_OBJECTIVE_POLICY,
    CandidateObservation,
    Decision,
    GuardrailSpec,
    ObjectivePolicy,
    Refusal,
    evaluate,
)


def _base_policy(**overrides):
    """A fully-specified policy, used as a baseline that individual tests
    knock a single field out of to prove that field's own refusal path."""
    defaults = dict(
        platform="youtube",
        format="short",
        primary_outcome_name="median_views_at_7d",
        primary_outcome_unit="views",
        primary_outcome_window_days=7,
        primary_outcome_estimator="median",
        min_useful_improvement=0.10,
        guardrails=(
            GuardrailSpec(
                name="likes_plus_comments_over_views",
                max_acceptable_degradation=0.05,
            ),
        ),
        min_exposure=100,
        min_independent_samples=30,
        max_evidence_age_days=30,
    )
    defaults.update(overrides)
    return ObjectivePolicy(**defaults)


def _fresh_evidence():
    return datetime.now(timezone.utc) - timedelta(days=1)


def _base_candidate(**overrides):
    defaults = dict(
        primary_outcome_value=120.0,
        guardrail_values={"likes_plus_comments_over_views": 0.08},
        exposure=150,
        independent_samples=40,
        evidence_as_of=_fresh_evidence(),
    )
    defaults.update(overrides)
    return CandidateObservation(**defaults)


def test_refuses_when_min_useful_improvement_unset():
    policy = _base_policy(min_useful_improvement=None)
    candidate = _base_candidate()

    result = evaluate(policy, candidate)

    assert isinstance(result, Refusal)
    assert result.reason == "MIN_USEFUL_IMPROVEMENT_UNSET"


def test_refuses_when_guardrail_degradation_margin_unset():
    policy = _base_policy(
        guardrails=(
            GuardrailSpec(
                name="likes_plus_comments_over_views",
                max_acceptable_degradation=None,
            ),
        )
    )
    candidate = _base_candidate()

    result = evaluate(policy, candidate)

    assert isinstance(result, Refusal)
    assert result.reason == "GUARDRAIL_DEGRADATION_UNSET"
    assert "likes_plus_comments_over_views" in result.detail


def test_refuses_when_min_exposure_unset():
    policy = _base_policy(min_exposure=None)
    candidate = _base_candidate()

    result = evaluate(policy, candidate)

    assert isinstance(result, Refusal)
    assert result.reason == "MIN_EXPOSURE_UNSET"


def test_refuses_when_min_independent_samples_unset():
    policy = _base_policy(min_independent_samples=None)
    candidate = _base_candidate()

    result = evaluate(policy, candidate)

    assert isinstance(result, Refusal)
    assert result.reason == "MIN_INDEPENDENT_SAMPLES_UNSET"


def test_refuses_when_max_evidence_age_unset():
    policy = _base_policy(max_evidence_age_days=None)
    candidate = _base_candidate()

    result = evaluate(policy, candidate)

    assert isinstance(result, Refusal)
    assert result.reason == "MAX_EVIDENCE_AGE_UNSET"


def test_refuses_when_mandatory_guardrail_metric_is_none():
    policy = _base_policy()
    candidate = _base_candidate(guardrail_values={"likes_plus_comments_over_views": None})

    result = evaluate(policy, candidate)

    assert isinstance(result, Refusal)
    assert result.reason == "GUARDRAIL_UNMEASURED"
    assert "likes_plus_comments_over_views" in result.detail


def test_refuses_when_evidence_is_stale():
    policy = _base_policy()
    candidate = _base_candidate(evidence_as_of=datetime.now(timezone.utc) - timedelta(days=31))

    result = evaluate(policy, candidate)

    assert isinstance(result, Refusal)
    assert result.reason == "SOURCE_STALE"


def test_refuses_when_samples_below_minimum():
    policy = _base_policy()
    candidate = _base_candidate(independent_samples=29)

    result = evaluate(policy, candidate)

    assert isinstance(result, Refusal)
    assert result.reason == "FAMILY_TOO_SMALL"


def test_refuses_when_exposure_below_minimum():
    policy = _base_policy()
    candidate = _base_candidate(exposure=99)

    result = evaluate(policy, candidate)

    assert isinstance(result, Refusal)
    assert result.reason == "FAMILY_TOO_SMALL"


def test_fully_specified_policy_can_decide():
    policy = _base_policy()
    candidate = _base_candidate()

    result = evaluate(policy, candidate)

    assert isinstance(result, Decision)
    assert result.primary_outcome_value == 120.0


def test_youtube_shorts_policy_refuses_today_for_unset_threshold():
    candidate = _base_candidate()

    result = evaluate(YOUTUBE_SHORT_OBJECTIVE_POLICY, candidate)

    assert isinstance(result, Refusal)
    assert result.reason == "MIN_USEFUL_IMPROVEMENT_UNSET"


def test_youtube_shorts_policy_leaves_ungrounded_fields_unset():
    policy = YOUTUBE_SHORT_OBJECTIVE_POLICY
    assert policy.min_useful_improvement is None
    assert policy.min_exposure is None
    assert policy.max_evidence_age_days is None
    assert all(g.max_acceptable_degradation is None for g in policy.guardrails)
    # The lone exception: the proposed shadow default.
    assert policy.min_independent_samples == 30


def test_youtube_shorts_policy_guardrail_is_not_shared_engagement_rate():
    # The YouTube guardrail must be its own declared metric
    # (likes+comments over views), never the schema's shared
    # ``engagement_rate`` property, which is structurally always None for
    # YouTube because the Data API never reports `shares`.
    names = {g.name for g in YOUTUBE_SHORT_OBJECTIVE_POLICY.guardrails}
    assert "engagement_rate" not in names
    assert "likes_plus_comments_over_views" in names


def test_objective_policy_is_frozen():
    policy = _base_policy()
    with pytest.raises(Exception):
        policy.min_useful_improvement = 0.2


def test_evaluate_has_no_literal_confidence_or_success_bypass():
    # Robust-by-construction: the evaluation entry point signature accepts
    # only a policy and a candidate observation -- no caller-suppliable
    # confidence/success flag exists to short-circuit the gate.
    import inspect

    params = list(inspect.signature(evaluate).parameters)
    assert params == ["policy", "candidate"]
