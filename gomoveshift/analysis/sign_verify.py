"""Compare road-sign evidence against an LF's resolved location.

Sign reading is **advisory evidence that validates the existing
``ResolvedLocation``, never a new location authority**. Nothing here writes,
mutates, or returns a location: every function reports agree / no-signal /
conflict and stops. The location cascade in ``gps_enrichment`` /
``title_builder`` remains the only thing that decides where a video was shot.

Why the conflict rule is deliberately timid
-------------------------------------------
Signs are noisy. A billboard, a business name ("Example Feed & Seed"), or a
county sign glimpsed a quarter-mile BEFORE the line all read as location
claims. A gate that flags on any of those burns operator trust and trains
people to click through it — at which point the one real desync it exists to
catch gets clicked through too. So this module inherits the SAME conservatism
as the existing far-road gate:

* a county/state claim contradicts GPS only when at least
  ``min_corroborating_hits`` DISTINCT hits assert the same thing — one noisy
  sign never contradicts a quorum-gated GPS vote;
* town and route claims are corroboration-only in this release; they can
  raise confidence, never a conflict;
* silence is NOT agreement — an OCR pass that read nothing yields
  ``STATUS_NO_SIGNAL``, never ``STATUS_CORROBORATES``;
* an evidence-free ``ResolvedLocation`` (GPS-less, legacy, or quorum-degraded:
  no ``regions`` AND no ``vote``) is a hard no-op. There is nothing to
  contradict, so nothing may be flagged.

``reasons`` lines quote the raw OCR text behind each finding so a flagged
conflict is judgeable by a human without re-running the pass.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from ..models import ResolvedLocation
from ..pipeline.title_builder import _hyphenate_route_ref, _road_ref_members
from ..pipeline.validation.content_gates import _county_names, _place_tokens
from .sign_evidence import (
    STATUS_CONFLICTS,
    STATUS_CORROBORATES,
    STATUS_NO_SIGNAL,
    SignClaim,
    SignHit,
    SignVerdict,
    hits_by_date_from_state,
)

# ``SignClaim.kind`` values — the ``LocationType`` enum members
# ``COUNTY_BOUNDARY`` / ``STATE_BOUNDARY`` / ``TOWN_ENTRY`` / ``ROUTE_MARKER``
# in ``gomoveshift.models``. Referenced by literal value (not by importing the
# enum) so this module stays decoupled from the enum's rollout: the claims
# graph is JSON-native and stores these strings anyway.
KIND_COUNTY = "county_boundary"
KIND_STATE = "state_boundary"
KIND_TOWN = "town_entry"
KIND_ROUTE = "route_marker"

# A region / vote key is ``"County, State"`` on the persisted model and
# ``"County|State"`` on the ``dominant_location`` adapter's output; split on
# either so one comparator serves both the encode and --label-dry-run paths.
_REGION_SPLIT_RE = re.compile(r"[|,]")
_COUNTY_FILLER_RE = re.compile(r"\bcounty\b", re.IGNORECASE)


def _bare_place(text: Any) -> str:
    """Lowercased place name with the ``County`` filler dropped and whitespace
    collapsed — ``"Example County"`` and ``"County Example"`` both →
    ``"example"``. Mirrors the filler-drop in
    ``pipeline.validation.content_gates._place_tokens`` (imported above and used
    for the token-subset axis), but preserves multi-word order so
    ``"Prince George's"`` stays one comparable name."""
    if not isinstance(text, str):
        return ""
    return " ".join(_COUNTY_FILLER_RE.sub(" ", text).lower().split())


def _region_labels(resolved: ResolvedLocation) -> list[str]:
    """Every region / vote key on the model, de-duplicated in order."""
    out: list[str] = []
    for label in list(resolved.regions or []) + list((resolved.vote or {}).keys()):
        if isinstance(label, str) and label.strip() and label not in out:
            out.append(label)
    return out


def _resolved_counties(resolved: ResolvedLocation) -> set[str]:
    """Bare county names the trace actually sampled.

    Two axes, union'd: ``content_gates._county_names`` (the "X County" /
    "County X" regex — the discriminating identity the far-road gate uses) and
    the bare leading field of each region key, which catches a geocoder that
    returned a county WITHOUT the ``County`` filler (Irish counties, US
    independent cities).
    """
    names: set[str] = set()
    for label in _region_labels(resolved):
        names |= _county_names(label)
        head = _REGION_SPLIT_RE.split(label)[0]
        bare = _bare_place(head)
        if bare:
            names.add(bare)
    dom = _bare_place(resolved.dominant_county)
    if dom:
        names.add(dom)
    return {n for n in names if n}


def _resolved_states(resolved: ResolvedLocation) -> set[str]:
    """Lowercased state names: ``dominant_state`` plus the trailing field of
    every region / vote key (``"Example County, South Carolina"`` →
    ``"south carolina"``)."""
    names: set[str] = set()
    dom = _bare_place(resolved.dominant_state)
    if dom:
        names.add(dom)
    for label in _region_labels(resolved):
        parts = _REGION_SPLIT_RE.split(label)
        for tail in parts[1:]:
            bare = _bare_place(tail)
            if bare:
                names.add(bare)
    return names


def _resolved_place_tokens(resolved: ResolvedLocation) -> set[str]:
    """Town-level name tokens across every place surface on the model —
    ``ordered_place_sequence``, ``start_label``, ``end_label``, ``destination``.
    Tokenised with ``content_gates._place_tokens`` so a claim matches whether or
    not the surface carries region filler ("Clifden" vs "Clifden, County
    Galway")."""
    tokens: set[str] = set()
    surfaces: list[Any] = list(resolved.ordered_place_sequence or [])
    surfaces += [resolved.start_label, resolved.end_label, resolved.destination]
    for s in surfaces:
        if isinstance(s, str) and s.strip():
            tokens |= _place_tokens(s)
    return tokens


def _resolved_route_members(resolved: ResolvedLocation) -> set[str]:
    """Upper-cased hyphenated route members from ``road`` + ``road_refs``.

    Uses ``title_builder._road_ref_members`` so an OSM concurrency string
    (``"I 81;US 11"``) compares MEMBER-WISE — an I-81 sign must corroborate a
    trace whose raw ref bundles I-81 with US-11.
    """
    members: set[str] = set()
    for raw in [resolved.road] + list(resolved.road_refs or []):
        if isinstance(raw, str) and raw.strip():
            members |= {m.upper() for m in _road_ref_members(raw) if m}
    return members


def _coerce_resolved(value: Any) -> Optional[ResolvedLocation]:
    """Pass through a ``ResolvedLocation``; rehydrate a raw state dict; else
    ``None``. The upload-manifest lesson: never let a consumer meet a plain
    dict where it does attribute access."""
    if isinstance(value, ResolvedLocation):
        return value
    if isinstance(value, dict):
        allowed = set(ResolvedLocation.__dataclass_fields__)
        return ResolvedLocation(**{k: v for k, v in value.items() if k in allowed})
    return None


def _latlon(value: Any) -> Optional[tuple[float, float]]:
    """``(lat, lon)`` from a 2-sequence, else ``None``."""
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return None
    return None


# ── Bucketing ─────────────────────────────────────────────────────────


def bucket_hits_by_lf(hits_by_date: Any, drive_plan: Any) -> dict[int, list[SignHit]]:
    """Assign each sign hit to the LF whose segments contain it.

    Mirrors the slicing idiom of
    ``pipeline.gps_enrichment._build_gps_points_by_lf``: walk every LF's
    segments and match on the segment's GLOBAL ``source_file_index`` plus its
    ``[start_time, end_time]`` span. A ``SignHit`` carries exactly those
    coordinates (``episode_index`` + ``t_local``) by design, so the assignment
    is an exact span test, not a reconstruction of a day-master offset.

    IMPORTANT — same constraint as ``_build_gps_points_by_lf``: this must be
    called against an UNPARTITIONED plan. ``source_file_index`` is a global
    episode index only BEFORE ``_partition_lf_plans_by_date`` remaps it to
    day-local; after the partition the episode match below points at the wrong
    episode (or out of range) and silently drops or MISFILES every hit — which
    is precisely the class of bug this feature exists to catch.

    ``hits_by_date`` may arrive as raw JSON dicts straight off
    ``state.sign_hits_by_date``; it is normalised through
    ``hits_by_date_from_state``. A hit matching no segment is dropped (an
    unassignable hit is evidence about nothing). Returns ``{}`` when the plan
    is missing or has no ``regular_lfs``.

    Returns ``{lf_index: [SignHit, ...]}`` — only LFs with at least one hit.
    """
    if drive_plan is None or not getattr(drive_plan, "regular_lfs", None):
        return {}

    normalised = hits_by_date_from_state(hits_by_date)
    all_hits: list[SignHit] = []
    for _date in sorted(normalised):
        all_hits.extend(normalised[_date] or [])
    if not all_hits:
        return {}

    result: dict[int, list[SignHit]] = {}
    for lf in drive_plan.regular_lfs:
        lf_idx = int(getattr(lf, "index", 0))
        segments = getattr(lf, "segments", None) or []
        matched: list[SignHit] = []
        for hit in all_hits:
            for seg in segments:
                if int(getattr(seg, "source_file_index", -1)) != int(hit.episode_index):
                    continue
                start = float(getattr(seg, "start_time", 0.0))
                end = float(getattr(seg, "end_time", 0.0))
                if start <= float(hit.t_local) <= end:
                    matched.append(hit)
                    break
        if matched:
            result[lf_idx] = matched
    return result


# ── dominant_location adapter ─────────────────────────────────────────


def resolved_from_dominant(dom: Any) -> ResolvedLocation:
    """Build a ``ResolvedLocation`` from a ``title_builder.dominant_location``
    result so the ``--label-dry-run`` path — which holds a ``DominantLocation``,
    not a persisted model — feeds the SAME comparator as the encode path. Two
    comparators would be two behaviours, and the dry run is the pre-encode gate.

    This is an ADAPTER, not a resolver: it never geocodes, never votes, and its
    output is used only as a comparison target. Tolerates ``None`` (returns an
    empty model, which :func:`verify_lf` treats as a no-op).

    Mapping: ``dominant_county`` ← ``county``, ``dominant_state`` ← ``state``,
    ``vote`` ← ``votes``, ``regions`` ← the voted county keys, ``road`` ←
    ``road``, ``road_refs`` ← ``road_refs``. The vote keys keep their
    ``"County|State"`` producer shape; the comparator splits on ``|`` or ``,``
    so both shapes read identically.
    """
    rl = ResolvedLocation()
    if dom is None:
        return rl
    rl.dominant_county = getattr(dom, "county", None)
    rl.dominant_state = getattr(dom, "state", None)
    votes = getattr(dom, "votes", None) or {}
    rl.vote = dict(votes)
    rl.regions = list(votes.keys())
    rl.road = getattr(dom, "road", None)
    rl.road_refs = list(getattr(dom, "road_refs", None) or [])
    return rl


# ── Comparison ────────────────────────────────────────────────────────


def _claim_bearing(hits: list[SignHit]) -> list[tuple[int, SignHit, list[SignClaim]]]:
    """``[(hit_position, hit, claims)]`` for hits carrying at least one usable
    claim. Position is the dedup key for the corroboration floor — the floor
    counts DISTINCT HITS, so one sign asserting the same county twice is one
    voice, not two."""
    out: list[tuple[int, SignHit, list[SignClaim]]] = []
    for pos, hit in enumerate(hits):
        claims = [
            c
            for c in hit.parsed_claims()
            if isinstance(c.kind, str) and c.kind and isinstance(c.value, str) and c.value.strip()
        ]
        if claims:
            out.append((pos, hit, claims))
    return out


def _group_by_name(
    claim_hits: list[tuple[int, SignHit, list[SignClaim]]], kind: str
) -> dict[str, dict[str, Any]]:
    """``{bare_name: {"display", "hits": set[int], "texts": [raw, ...]}}`` for
    one claim kind."""
    groups: dict[str, dict[str, Any]] = {}
    for pos, hit, claims in claim_hits:
        for claim in claims:
            if claim.kind != kind:
                continue
            key = _bare_place(claim.value)
            if not key:
                continue
            entry = groups.setdefault(key, {"display": claim.value, "hits": set(), "texts": []})
            entry["hits"].add(pos)
            text = (claim.raw_text or hit.raw_text or "").strip()
            if text and text not in entry["texts"]:
                entry["texts"].append(text)
    return groups


def _quoted(texts: list[str]) -> str:
    return "; ".join(f'"{t}"' for t in texts)


def verify_lf(
    lf_index: Any,
    hits: Any,
    resolved: Any,
    *,
    min_corroborating_hits: int = 2,
    geocode_fn: Optional[Callable[[str], Any]] = None,
) -> SignVerdict:
    """Compare one LF's sign hits against its ``ResolvedLocation``.

    Advisory only — returns a :class:`SignVerdict`, never a location.

    Rules (see the module docstring for WHY they are this timid):

    * **county** claims corroborate on a bare-name match against ``regions`` /
      ``vote``; a county in ZERO regions conflicts only once
      ``min_corroborating_hits`` DISTINCT hits assert it.
    * **state** claims: same shape, against ``dominant_state`` plus the state
      field of each region key, same floor.
    * **town** claims: corroboration-only, NEVER a conflict in this release.
      Matches ``ordered_place_sequence`` / ``start_label`` / ``end_label`` /
      ``destination``. When ``geocode_fn`` is supplied and the name matches
      nothing, the town is forward-geocoded and a distance line is recorded as
      advisory context — still never a conflict. ``geocode_fn`` defaults to
      ``None`` (no network); a ``None`` return (degraded geocoder) is a skip.
    * **route** claims: corroboration-only, member-wise against ``road`` /
      ``road_refs``.
    * no claims at all → ``STATUS_NO_SIGNAL``. Silence is not agreement.
    * status precedence: any conflict → ``STATUS_CONFLICTS``; else any
      corroboration → ``STATUS_CORROBORATES``; else ``STATUS_NO_SIGNAL``.

    ``confidence`` is the share of claim-bearing hits supporting the reported
    status. ``corroborating_counties`` / ``conflicting_counties`` hold COUNTY
    names only (per the field names); a state-level finding is carried in
    ``reasons``.

    A ``resolved`` that is missing, or has neither ``regions`` nor ``vote``
    (GPS-less / legacy / quorum-degraded), is a no-op: ``STATUS_NO_SIGNAL``
    with a reason. Never flag a conflict against an empty model.
    """
    lf_idx = int(lf_index)
    typed_hits = [SignHit.from_dict(h) for h in (hits or [])]
    verdict = SignVerdict(lf_index=lf_idx, hit_count=len(typed_hits))

    model = _coerce_resolved(resolved)
    if model is None or (not model.regions and not model.vote):
        verdict.status = STATUS_NO_SIGNAL
        verdict.reasons.append(
            f"LF {lf_idx}: nothing to verify against — the resolved location carries no "
            f"sampled regions or county vote (GPS-less, legacy, or quorum-degraded LF); "
            f"{len(typed_hits)} sign hit(s) left uncompared"
        )
        return verdict

    claim_hits = _claim_bearing(typed_hits)
    if not claim_hits:
        verdict.status = STATUS_NO_SIGNAL
        verdict.reasons.append(
            f"LF {lf_idx}: no legible sign named a place across {len(typed_hits)} hit(s) — "
            f"no signal (this is NOT agreement with the GPS-derived location)"
        )
        return verdict

    counties = _resolved_counties(model)
    states = _resolved_states(model)
    place_tokens = _resolved_place_tokens(model)
    route_members = _resolved_route_members(model)
    region_display = _region_labels(model)
    floor = max(int(min_corroborating_hits), 1)

    corroborating_hits: set[int] = set()
    conflicting_hits: set[int] = set()

    # ── county claims ──
    for name, entry in sorted(_group_by_name(claim_hits, KIND_COUNTY).items()):
        display = entry["display"]
        n_hits = len(entry["hits"])
        if name in counties:
            verdict.corroborating_counties.append(display)
            corroborating_hits |= entry["hits"]
            verdict.reasons.append(
                f"LF {lf_idx}: county sign(s) {_quoted(entry['texts'])} name {display!r}, "
                f"which IS in the sampled regions {region_display} — corroborates"
            )
        elif n_hits >= floor:
            verdict.conflicting_counties.append(display)
            conflicting_hits |= entry["hits"]
            verdict.reasons.append(
                f"LF {lf_idx}: {n_hits} distinct sign(s) {_quoted(entry['texts'])} name county "
                f"{display!r}, which is in ZERO sampled regions {region_display} — CONFLICT"
            )
        else:
            verdict.reasons.append(
                f"LF {lf_idx}: {n_hits} sign(s) {_quoted(entry['texts'])} name county "
                f"{display!r} (not in {region_display}), below the {floor}-hit corroboration "
                f"floor — advisory only, not a conflict"
            )

    # ── state claims (same floor) ──
    for name, entry in sorted(_group_by_name(claim_hits, KIND_STATE).items()):
        display = entry["display"]
        n_hits = len(entry["hits"])
        if name in states:
            corroborating_hits |= entry["hits"]
            verdict.reasons.append(
                f"LF {lf_idx}: state sign(s) {_quoted(entry['texts'])} name {display!r}, "
                f"matching the resolved state — corroborates"
            )
        elif n_hits >= floor:
            conflicting_hits |= entry["hits"]
            verdict.reasons.append(
                f"LF {lf_idx}: {n_hits} distinct sign(s) {_quoted(entry['texts'])} name state "
                f"{display!r}, which the trace never sampled "
                f"({sorted(states) or 'no resolved state'}) — CONFLICT"
            )
        else:
            verdict.reasons.append(
                f"LF {lf_idx}: {n_hits} sign(s) {_quoted(entry['texts'])} name state "
                f"{display!r} (unsampled), below the {floor}-hit corroboration floor — "
                f"advisory only, not a conflict"
            )

    # ── town claims: corroboration-only, never a conflict ──
    for name, entry in sorted(_group_by_name(claim_hits, KIND_TOWN).items()):
        display = entry["display"]
        claim_tokens = {t for t in _place_tokens(display) if t}
        if claim_tokens and claim_tokens <= place_tokens:
            corroborating_hits |= entry["hits"]
            verdict.reasons.append(
                f"LF {lf_idx}: town sign(s) {_quoted(entry['texts'])} name {display!r}, "
                f"which is on the resolved route — corroborates"
            )
            continue
        verdict.reasons.append(
            f"LF {lf_idx}: town sign(s) {_quoted(entry['texts'])} name {display!r}, not on the "
            f"resolved route — advisory only (town claims never conflict)"
        )
        if geocode_fn is None:
            continue
        try:
            coords = _latlon(geocode_fn(display))
        except Exception as exc:  # noqa: BLE001 — a geocoder failure is never a verdict
            verdict.reasons.append(
                f"LF {lf_idx}: forward geocode of {display!r} failed ({exc}) — skipped"
            )
            continue
        if coords is None:
            continue  # degraded geocoder: a skip, never a failure
        anchor = _latlon(model.endpoints.get("start")) or _latlon(model.endpoints.get("end"))
        for pos, hit, _claims in claim_hits:
            if pos in entry["hits"] and hit.lat is not None and hit.lon is not None:
                anchor = (float(hit.lat), float(hit.lon))
                break
        if anchor is None:
            continue
        from ..pipeline.title_builder import _haversine_km_simple

        km = _haversine_km_simple(anchor[0], anchor[1], coords[0], coords[1])
        verdict.reasons.append(
            f"LF {lf_idx}: town {display!r} forward-geocodes ~{km:.0f} km from where its sign "
            f"was read — advisory context only"
        )

    # ── route claims: corroboration-only, never a conflict ──
    for _name, entry in sorted(_group_by_name(claim_hits, KIND_ROUTE).items()):
        display = entry["display"]
        ref = _hyphenate_route_ref(display.strip()).upper()
        if ref and ref in route_members:
            corroborating_hits |= entry["hits"]
            verdict.reasons.append(
                f"LF {lf_idx}: route sign(s) {_quoted(entry['texts'])} name {display!r}, "
                f"carried by the resolved road {sorted(route_members)} — corroborates"
            )
        else:
            verdict.reasons.append(
                f"LF {lf_idx}: route sign(s) {_quoted(entry['texts'])} name {display!r}, not in "
                f"the resolved road refs {sorted(route_members) or '[]'} — advisory only "
                f"(route claims never conflict)"
            )

    total_claim_hits = len(claim_hits)
    if conflicting_hits:
        verdict.status = STATUS_CONFLICTS
        supporting = len(conflicting_hits)
    elif corroborating_hits:
        verdict.status = STATUS_CORROBORATES
        supporting = len(corroborating_hits)
    else:
        verdict.status = STATUS_NO_SIGNAL
        supporting = 0
        verdict.reasons.append(
            f"LF {lf_idx}: {total_claim_hits} claim-bearing hit(s) neither matched nor "
            f"contradicted the resolved location — no signal"
        )
    verdict.confidence = round(supporting / total_claim_hits, 4) if total_claim_hits else 0.0
    return verdict


def verify_all(
    hits_by_date: Any,
    drive_plan: Any,
    resolved_by_lf: Any,
    *,
    min_corroborating_hits: int = 2,
    geocode_fn: Optional[Callable[[str], Any]] = None,
) -> dict[int, SignVerdict]:
    """Bucket every hit, then verify each LF that HAS a resolved model.

    ``resolved_by_lf`` is ``{lf_index: ResolvedLocation}`` — the shape of
    ``state.per_lf_resolved_location``; raw dicts as values are rehydrated. An
    LF with no entry is skipped entirely (no model, nothing to verify against),
    so it never appears in the result.

    Same UNPARTITIONED-plan constraint as :func:`bucket_hits_by_lf`.

    Returns ``{lf_index: SignVerdict}``.
    """
    buckets = bucket_hits_by_lf(hits_by_date, drive_plan)
    by_lf: dict[Any, Any] = dict(resolved_by_lf or {})

    verdicts: dict[int, SignVerdict] = {}
    for lf in getattr(drive_plan, "regular_lfs", None) or []:
        lf_idx = int(getattr(lf, "index", 0))
        raw = by_lf.get(lf_idx, by_lf.get(str(lf_idx)))
        model = _coerce_resolved(raw)
        if model is None:
            continue
        verdicts[lf_idx] = verify_lf(
            lf_idx,
            buckets.get(lf_idx, []),
            model,
            min_corroborating_hits=min_corroborating_hits,
            geocode_fn=geocode_fn,
        )
    return verdicts
