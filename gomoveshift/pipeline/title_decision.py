"""GMS consumes Waypost's title decision (title-accuracy program U5).

GMS supplies verified facts and ships whatever Waypost's ``select_title``
decides. The facts are the route-only baseline title and the places a title
may name: its two endpoints and its region slot, each carrying the
``evidence_id`` of the GPS record that puts the footage there. A claim the
record does not back gives no facts, and the baseline ships with the reason
``facts_unavailable`` rather than a guessed place.

The decision record rides with the metadata, and :func:`check_final_title`
refuses an upload whose title differs from it (and
:func:`require_title_decision` one with no decision at all), so a format or resume path can
never silently replace the selected title.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional, Union

from seo_manager.analytics.title_selection import (
    DEFAULT_POLICY,
    EvidenceSnapshot,
    Place,
    TitleFacts,
    TitlePolicy,
    record_to_dict,
    select_title,
)

from gomoveshift.pipeline.location_evidence import LocationEvidence, _evidence_id, _state_matches
from gomoveshift.pipeline.validation.content_gates import title_endpoints

FACTS_UNAVAILABLE = "facts_unavailable"
_ENTERED = ("in", "through")


class TitleDecisionMismatchError(RuntimeError):
    """The title about to ship is not the one Waypost's decision selected."""


class TitleDecisionMissingError(RuntimeError):
    """A title is about to ship with no Waypost decision behind it."""


class PreRenderTitleError(RuntimeError):
    """A Waypost title cannot be approved for the retained route before encode."""


def route_fingerprint(record: dict) -> str:
    """Stable digest of the retained route record used for title approval."""
    body = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _as_record(record) -> Optional[LocationEvidence]:
    if record is None or isinstance(record, LocationEvidence):
        return record
    return LocationEvidence.from_dict(record) if record else None


def _region_slot(baseline_title: str, suffix: str) -> Optional[str]:
    route = baseline_title[: -len(suffix)] if suffix and baseline_title.endswith(suffix) else ""
    parts = [p.strip() for p in route.split(",")]
    return parts[-1] if len(parts) >= 2 and parts[-1] else None


def _region_backed(slot: str, place) -> Optional[str]:
    """The region name ``place``'s own record backs for ``slot``, if any."""
    if _state_matches(place.state, slot):
        return place.state
    if place.county and place.county.strip().lower() == slot.strip().lower():
        return place.county
    return None


def title_facts(fmt: str, baseline_title: str, record, *, suffix: str) -> Optional[TitleFacts]:
    """Waypost facts for ``baseline_title`` from ``record``, or ``None`` when
    the record does not back both endpoints and the region slot."""
    rec = _as_record(record)
    ends = title_endpoints(baseline_title)
    region = _region_slot(baseline_title, suffix)
    if rec is None or ends is None or region is None:
        return None
    places = []
    for name in ends:
        hit = next(
            (
                p
                for p in rec.places
                if p.name.lower() == name.lower()
                and p.scope in _ENTERED
                and _region_backed(region, p)
            ),
            None,
        )
        if hit is None:
            return None
        places.append(hit)
    origin, destination = places
    return TitleFacts(
        format=fmt,
        baseline_title=baseline_title,
        origin=Place(origin.name, origin.evidence_id),
        destination=Place(destination.name, destination.evidence_id),
        region=Place(str(_region_backed(region, destination)), destination.evidence_id),
        suffix=suffix,
    )


def short_place_facts(
    baseline_title: str, place: str, regions, *, segment_id: str
) -> Optional[TitleFacts]:
    """Single-place Waypost facts for a Short whose title names ``place``,
    or ``None`` unless ``place`` is in the title and one of the clip's own
    GPS region votes (``"County, State"``) backs it."""
    if not place or place.lower() not in baseline_title.lower():
        return None
    for region in regions or ():
        county, _, state = (part.strip() for part in region.partition(","))
        if place.strip().lower() in (county.lower(), region.strip().lower()):
            evidence_id = _evidence_id(segment_id, place, county, state, "clip_gps_vote")
            return TitleFacts(
                format="short", baseline_title=baseline_title, place=Place(place, evidence_id)
            )
    return None


def decide_title(
    fmt: str,
    baseline_title: str,
    record,
    *,
    suffix: str,
    snapshot: Optional[EvidenceSnapshot] = None,
    policy: Optional[TitlePolicy] = None,
    facts: Optional[TitleFacts] = None,
) -> dict:
    """The decision record for one upload (JSON-safe). No snapshot means no
    observations: the baseline ships as ``insufficient_evidence``."""
    policy = policy or DEFAULT_POLICY
    facts = facts or title_facts(fmt, baseline_title, record, suffix=suffix)
    if facts is None:
        return {
            "policy_version": policy.version,
            "format": fmt,
            "chosen_index": 0,
            "chosen_title": baseline_title,
            "reason": FACTS_UNAVAILABLE,
            "candidates": [],
        }
    if snapshot is None:
        snapshot = EvidenceSnapshot(as_of=datetime.now(timezone.utc).isoformat())
    # PipelineState is persisted as JSON. Normalize Waypost's tuple-valued
    # candidates here so an in-memory decision and its resumed copy agree.
    return json.loads(
        json.dumps(record_to_dict(select_title(facts, snapshot, policy)), allow_nan=False)
    )


def _retained_title_issues(title: str, route_evidence) -> list[str]:
    """Base title issues plus the zero-tolerance place gate (every retained second
    must lie in a place the title names)."""
    from .route_claims import place_text_issues

    issues = _title_route_issues(title, route_evidence)
    return issues + [i for i in place_text_issues(title, route_evidence) if i not in issues]


def _title_route_issues(title: str, route_evidence) -> list[str]:
    """Check the places a title names against the LF's retained route.

    Waypost ranks supplied title facts, while GMS owns the footage facts. An
    SEO template can omit a material intermediate place even when its endpoint
    facts are valid. A generic county or state title is allowed only when the
    existing retained-route region gate supports that whole region.
    """
    import re

    from .geo_constants import US_STATE_ABBREVS
    from .route_claims import REGION_MIN_COVERAGE, _region_ok

    required = list(route_evidence.required)

    def names(place):
        return re.search(rf"(?<!\w){re.escape(place.name)}(?!\w)", title, re.IGNORECASE) is not None

    named = [p for p in route_evidence.places if names(p)]
    state_names = set(US_STATE_ABBREVS) | {b.state for b in route_evidence.bins if b.state}
    matches = []
    for state in state_names:
        for hit in re.finditer(rf"(?<!\w){re.escape(state)}(?!\w)", title, re.IGNORECASE):
            matches.append((hit.start(), hit.end(), state))
        abbrev = US_STATE_ABBREVS.get(state)
        if abbrev:
            for hit in re.finditer(rf"(?<!\w){abbrev}(?!\w)", title):
                matches.append((hit.start(), hit.end(), state))
    # "West Virginia" contains "Virginia"; the longer state name owns that span.
    claimed_states = {
        state
        for start, end, state in matches
        if not any(
            (other_start <= start and end <= other_end) and (other_end - other_start > end - start)
            for other_start, other_end, _other_state in matches
        )
    }

    def state_key(state):
        return US_STATE_ABBREVS.get(state, state).casefold()

    if claimed_states and any(
        b.state and state_key(b.state) not in {state_key(s) for s in claimed_states}
        for b in route_evidence.bins
    ):
        return [f"title claims {sorted(claimed_states)!r} but retained GPS samples lie outside"]
    # A title naming no place outside a region's own name is a region title,
    # even when a town shares that name ("Beaufort & Colleton Counties").
    for level in ("county", "counties", "state"):
        claim = route_evidence.region_claim(level)
        region = claim and re.compile(rf"(?<!\w){re.escape(claim['name'])}(?!\w)", re.IGNORECASE)
        if not region or not region.search(title):
            continue
        rest = region.sub(" ", title)
        if any(
            re.search(rf"(?<!\w){re.escape(p.name)}(?!\w)", rest, re.IGNORECASE)
            for p in route_evidence.places
        ):
            continue
        why = _region_ok(claim, route_evidence, REGION_MIN_COVERAGE)
        if not why:
            return []
        return [f"region {claim['name']!r}: {reason}" for reason in why]
    issues = [
        f"omits {p.name!r} ({p.retained_seconds:.0f} s of retained footage)"
        for p in required
        if not names(p)
    ]
    return issues or (["no retained place or supported region in title"] if not named else [])


def _region_only_title_facts(label: str, baseline: str, route_evidence, suffix: str):
    """Offer Waypost a region shape only for the route gate's exact award."""
    from .final_text_record import _region_id
    from .geo_constants import US_STATE_ABBREVS
    from .route_claims import REGION_MIN_COVERAGE, _region_ok

    for level in ("county", "counties", "state"):
        claim = route_evidence.region_claim(level)
        if not claim or _region_ok(claim, route_evidence, REGION_MIN_COVERAGE):
            continue
        name = claim["name"]
        if level != "state":
            state_claim = route_evidence.region_claim("state")
            if not state_claim or _region_ok(state_claim, route_evidence, REGION_MIN_COVERAGE):
                continue
            state = state_claim["name"]
            if claim.get("state") and claim["state"] != state:
                continue
            labels = {f"{name}, {state}", f"{name}, {US_STATE_ABBREVS.get(state, state)}"}
            evidence_name = name
        else:
            labels = {name, US_STATE_ABBREVS.get(name, name)}
            evidence_name = label
        if label.casefold() not in {item.casefold() for item in labels}:
            continue
        return TitleFacts(
            format="longform",
            baseline_title=baseline,
            region=Place(label, _region_id(level, evidence_name)),
            suffix=suffix,
        )
    return None


def approve_retained_title(
    label: str,
    route_evidence,
    record,
    *,
    resolved=None,
    title_part: Optional[tuple] = None,
    snapshot: Optional[EvidenceSnapshot] = None,
    facts: Optional[TitleFacts] = None,
) -> dict:
    """Choose and approve one LF title before rendering commits OST pixels.

    The baseline is assembled by the same formatter Phase D uses. If Waypost's
    winner omits a required retained place, the approved baseline wins with an
    explicit reason. An unsafe baseline has no fallback and refuses the encode.
    """
    from .final_text_record import route_record
    from .longform_metadata import assemble_final_title
    from .title_builder import append_full_drive_part_suffix

    baseline = assemble_final_title(label, resolved=resolved)
    if title_part:
        baseline = append_full_drive_part_suffix(baseline, *title_part)
    baseline_issues = _retained_title_issues(baseline, route_evidence)
    if baseline_issues:
        raise PreRenderTitleError(
            f"pre-render baseline {baseline!r} is unsupported: {'; '.join(baseline_issues)}"
        )
    cut = baseline.rfind(" in 4K")
    suffix = baseline[cut:] if cut >= 0 else ""
    if facts is None:
        facts = _region_only_title_facts(label, baseline, route_evidence, suffix)
    decision = decide_title(
        "longform",
        baseline,
        record,
        suffix=suffix,
        snapshot=snapshot,
        facts=facts,
    )
    decision["baseline_title"] = baseline
    decision["input_label"] = label
    decision["route_fingerprint"] = route_fingerprint(route_record(route_evidence))
    issues = _retained_title_issues(decision["chosen_title"], route_evidence)
    if issues:
        decision["geo_rejected_index"] = decision["chosen_index"]
        decision["geo_rejected_reasons"] = issues
        for candidate in decision.get("candidates", []):
            if candidate.get("index") == decision["chosen_index"]:
                candidate["excluded_reason"] = "retained_route"
        decision["chosen_index"] = 0
        decision["chosen_title"] = baseline
        decision["reason"] = "retained_route_excluded_winner"
    return decision


def check_final_title(title: str, metadata: Union[dict, None]) -> None:
    """Refuse ``title`` when ``metadata`` carries a decision that chose another."""
    decision = (metadata or {}).get("title_decision") if isinstance(metadata, dict) else None
    if not decision:
        return
    chosen = decision.get("chosen_title")
    if title != chosen:
        raise TitleDecisionMismatchError(
            f"upload title {title!r} is not the decided title {chosen!r} "
            f"(reason {decision.get('reason')!r}); a format or resume path "
            "replaced Waypost's selection"
        )


def require_title_decision(title: str, metadata: Union[dict, None]) -> None:
    """Refuse ``title`` unless ``metadata`` carries a decision that chose it.

    Every path that sends a new title to YouTube calls this upstream of the
    upload call, so a title that never went through Waypost's decision cannot
    ship; ``check_final_title`` keeps the mismatch refusal."""
    decision = metadata.get("title_decision") if isinstance(metadata, dict) else None
    if not decision:
        raise TitleDecisionMissingError(
            f"upload title {title!r} has no Waypost title decision "
            "(metadata['title_decision'] is missing); refusing to send a "
            "title that never went through the location-evidence decision"
        )
    check_final_title(title, metadata)
