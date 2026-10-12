"""
Objective policy declaration and refusal gate.

Each account/format policy must declare what "better" means before any
evidence is compared against it: one primary outcome (its unit/window/
estimator and the minimum practically useful improvement), mandatory
guardrail outcomes with acceptable degradation margins, minimum exposure/
independent-sample counts, and a maximum evidence age. A policy that has
not made all of those declarations is not usable -- ``evaluate`` refuses
rather than silently defaulting or warning and continuing.

This module builds the typed declaration and the refusal gate ONLY. It
invents no product thresholds: ``min_useful_improvement``, guardrail
degradation margins, ``min_exposure``, ``min_independent_samples`` and
``max_evidence_age_days`` are all the operator's to set -- from measured baseline
spread, historical exposure, or an explicitly qualified statistical
method -- not guessed here. Until each is set, every evaluation against a
policy that leaves any of them ``None`` refuses -- that is the intended,
correct behaviour, not a bug to work around.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Union


@dataclass(frozen=True)
class GuardrailSpec:
    """One mandatory guardrail outcome and its acceptable degradation.

    ``max_acceptable_degradation`` is a fraction (e.g. 0.05 == the
    guardrail may fall by up to 5% and still be acceptable). It is
    declared here, not inferred from a candidate. Optional and defaults
    to ``None`` for the same reason as ``ObjectivePolicy.
    min_useful_improvement``: this is a product value someone must choose
    and cite, not a number this module invents. A guardrail left at
    ``None`` cannot pass ``evaluate`` -- it refuses instead of silently
    picking a degradation margin.
    """

    name: str
    max_acceptable_degradation: Optional[float] = None


@dataclass(frozen=True)
class ObjectivePolicy:
    """A single account/format objective policy.

    ``min_useful_improvement``, ``min_exposure``, ``min_independent_samples``
    and ``max_evidence_age_days`` are all Optional and default to ``None``
    on purpose: none of these is this module's to invent. Each is the operator's
    product value to set (from measured baseline spread, historical
    exposure, or an explicitly qualified statistical method), not a
    guessed default. A policy left with any of them at ``None`` is a
    valid, fully-typed declaration -- it just cannot pass ``evaluate``
    yet, and refuses instead of silently proceeding. The whole thesis of
    this module is that an under-specified policy is unusable, whichever
    field is missing.
    """

    platform: str
    format: str
    primary_outcome_name: str
    primary_outcome_unit: str
    primary_outcome_window_days: int
    primary_outcome_estimator: str
    guardrails: Tuple[GuardrailSpec, ...]
    min_exposure: Optional[int] = None
    min_independent_samples: Optional[int] = None
    max_evidence_age_days: Optional[int] = None
    min_useful_improvement: Optional[float] = None


@dataclass(frozen=True)
class CandidateObservation:
    """One candidate observation to evaluate against an ObjectivePolicy.

    Deliberately carries no confidence score, success flag, or any other
    field a caller could use to bypass the gate -- ``evaluate`` derives
    its decision solely from comparing this against the policy's
    declarations.
    """

    primary_outcome_value: float
    guardrail_values: Dict[str, Optional[float]]
    exposure: int
    independent_samples: int
    evidence_as_of: datetime


@dataclass(frozen=True)
class Refusal:
    """A typed, machine-readable refusal. Never a default, never a warning."""

    reason: str
    detail: str


@dataclass(frozen=True)
class Decision:
    """A typed decision produced only when every requirement is satisfied."""

    primary_outcome_value: float
    guardrail_values: Dict[str, Optional[float]]


def evaluate(policy: ObjectivePolicy, candidate: CandidateObservation) -> Union[Decision, Refusal]:
    """Evaluate ``candidate`` against ``policy``, or refuse.

    Refuses -- never defaults, never warns-and-continues -- when any
    required declaration is unset (``None``):
      * ``policy.min_useful_improvement``;
      * a guardrail's ``max_acceptable_degradation``;
      * ``policy.min_exposure``;
      * ``policy.min_independent_samples``;
      * ``policy.max_evidence_age_days``.

    Each unset declaration gets its own distinct reason -- an
    under-specified policy is unusable whichever field is missing, which
    is this module's whole thesis. Only once every declaration is present
    does ``evaluate`` compare the candidate against them (guardrail
    measured, evidence freshness, sample/exposure sufficiency).

    Signature is intentionally exactly ``(policy, candidate)`` -- there is
    no literal-confidence or success-flag parameter a caller could pass to
    bypass this gate.
    """
    if policy.min_useful_improvement is None:
        return Refusal(
            reason="MIN_USEFUL_IMPROVEMENT_UNSET",
            detail=(
                "policy for "
                f"{policy.platform}/{policy.format} declares no minimum "
                "practically useful improvement; this is the operator's "
                "product value to set from measured baseline spread, not a "
                "default"
            ),
        )

    for guardrail in policy.guardrails:
        if guardrail.max_acceptable_degradation is None:
            return Refusal(
                reason="GUARDRAIL_DEGRADATION_UNSET",
                detail=(
                    f"guardrail '{guardrail.name}' declares no acceptable "
                    "degradation margin; this is a product value to set, "
                    "not a default"
                ),
            )

    if policy.min_exposure is None:
        return Refusal(
            reason="MIN_EXPOSURE_UNSET",
            detail=(f"policy for {policy.platform}/{policy.format} declares no minimum exposure"),
        )

    if policy.min_independent_samples is None:
        return Refusal(
            reason="MIN_INDEPENDENT_SAMPLES_UNSET",
            detail=(
                f"policy for {policy.platform}/{policy.format} declares "
                "no minimum independent sample count"
            ),
        )

    if policy.max_evidence_age_days is None:
        return Refusal(
            reason="MAX_EVIDENCE_AGE_UNSET",
            detail=(
                f"policy for {policy.platform}/{policy.format} declares no maximum evidence age"
            ),
        )

    for guardrail in policy.guardrails:
        value = candidate.guardrail_values.get(guardrail.name)
        if value is None:
            return Refusal(
                reason="GUARDRAIL_UNMEASURED",
                detail=(f"mandatory guardrail '{guardrail.name}' has no measured value"),
            )

    now = datetime.now(timezone.utc)
    evidence_as_of = candidate.evidence_as_of
    if evidence_as_of.tzinfo is None:
        evidence_as_of = evidence_as_of.replace(tzinfo=timezone.utc)
    evidence_age_days = (now - evidence_as_of).total_seconds() / 86400.0
    if evidence_age_days > policy.max_evidence_age_days:
        return Refusal(
            reason="SOURCE_STALE",
            detail=(
                f"evidence is {evidence_age_days:.1f} days old, exceeds "
                f"the declared maximum of {policy.max_evidence_age_days}"
            ),
        )

    if (
        candidate.independent_samples < policy.min_independent_samples
        or candidate.exposure < policy.min_exposure
    ):
        return Refusal(
            reason="FAMILY_TOO_SMALL",
            detail=(
                f"{candidate.independent_samples} independent samples / "
                f"{candidate.exposure} exposure below the declared "
                f"minimum of {policy.min_independent_samples} / "
                f"{policy.min_exposure}"
            ),
        )

    return Decision(
        primary_outcome_value=candidate.primary_outcome_value,
        guardrail_values=dict(candidate.guardrail_values),
    )


# One concrete policy instance for YouTube shorts (primary outcome
# = median views at a fixed age of 7 days). The guardrail is its own
# declared metric -- likes+comments over views -- and is deliberately NOT
# ``PlatformMetrics.engagement_rate`` (see ``schema.PlatformMetrics``): that property returns
# ``None`` whenever any of views/likes/comments/shares is missing, and
# ``YouTubeCollector`` can never supply ``shares`` at all
# ("Not available via Data API -- unreported, not zero"), so YouTube's
# shared engagement_rate is structurally None forever. Using it as a
# guardrail here would refuse every candidate for a data-availability
# reason that has nothing to do with this policy.
#
# Every declaration below is left ``None`` -- and evaluating any candidate
# against this policy today therefore refuses -- EXCEPT
# min_independent_samples, with the single exception explained at its own
# line. None of these is this module's number to invent:
#   * min_useful_improvement: no product value is chosen; it
#     must be measured from baseline spread, not guessed.
#   * guardrail max_acceptable_degradation: no product value chosen.
#   * min_exposure: not grounded in any measurement.
#   * max_evidence_age_days: not grounded; the nearest real number
#     for this policy is a trailing 180-day lookback, not a smaller
#     invented figure.
YOUTUBE_SHORT_OBJECTIVE_POLICY = ObjectivePolicy(
    platform="youtube",
    format="short",
    primary_outcome_name="median_views_at_7d",
    primary_outcome_unit="views",
    primary_outcome_window_days=7,
    primary_outcome_estimator="median",
    guardrails=(
        GuardrailSpec(
            name="likes_plus_comments_over_views",
            max_acceptable_degradation=None,
        ),
    ),
    min_exposure=None,
    # A proposed shadow default of 30 independent reference clusters: a nominal
    # minimum, not a power calculation and not a validated operating threshold.
    min_independent_samples=30,
    max_evidence_age_days=None,
    min_useful_improvement=None,
)
