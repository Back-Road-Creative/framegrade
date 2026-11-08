"""Derive an LF's label from its road signs, for an LF the camera never located.

This is the ONE place road signs may become a location authority, and only
for an LF with **zero GPS fixes**. Everywhere else (``sign_verify``) signs
stay advisory: they grade the ``ResolvedLocation`` the GPS chain produced and
never write a location. A GPS-less LF has no such chain — until now its only
label came from an operator (``--per-lf-location`` / ``--placeless-label``),
typed from memory and measured by nothing. The signs in the footage are the
only evidence of where it was; this module turns them into a label under rules
that fail closed.

The rules (each one a refusal, with a reason the dry-run can print)
------------------------------------------------------------------
* The LF has NO GPS fix. One fix means the GPS chain owns the label; signs
  then verify it, they never replace it (``gps_fix_count`` must be 0).
* At least ``min_corroborating_hits`` sign hits from DISTINCT seconds name the
  same county (``config.drive.sign_verify_min_corroborating_hits``, default
  2 — the same floor ``sign_verify`` needs before a sign may contradict GPS).
  Two reads of one frame are one observation.
* No OTHER county is named by any hit in the LF. A county-line sign inside the
  LF means it spans two counties, and a single label would be wrong for one
  of them; the operator decides that one.
* Exactly one state is named. The parser (``ocr.parse_sign_claims``) emits a
  state claim only beside a place claim or a boundary cue, so a destination on
  a guide sign ("Winchester / West Virginia / Exit 310") never supplies it.
* The label is the county shape only — ``"Colleton County, SC"``, the same
  form ``route_claims.region_label`` gives a measured county. Never a town,
  never "A to B": a town-entry sign places one moment, not the LF.

What comes out is a :class:`SignLabel`: the label plus the evidence record
(hit count, the distinct seconds, the raw sign texts) that the wiring stores
as the LF's tier evidence, so a provenance sidecar and the OST system review
can see WHY the LF says what it says. The consumer that writes it into the
per-LF override map is the Phase B/C wiring; this module only derives.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .sign_evidence import SignHit

KIND_COUNTY = "county_boundary"
KIND_STATE = "state_boundary"

#: ``SignLabel.source`` / the tier-evidence source marker a wired label carries.
SIGN_LABEL_SOURCE = "sign_label"


@dataclass
class SignLabel:
    """A county-level label derived from an LF's road signs, with its evidence."""

    lf_index: int
    label: str
    county: str
    state: str
    hit_count: int
    hit_seconds: list[float] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    source: str = SIGN_LABEL_SOURCE

    def to_dict(self) -> dict:
        return asdict(self)


def _typed_hits(hits: Any) -> list[SignHit]:
    out: list[SignHit] = []
    for h in hits or []:
        if isinstance(h, SignHit):
            out.append(h)
        elif isinstance(h, dict):
            out.append(SignHit.from_dict(h))
    return out


def _state_abbrev(state: str) -> Optional[str]:
    from ..pipeline.geo_constants import US_STATE_ABBREVS

    by_lower = {name.lower(): abbr for name, abbr in US_STATE_ABBREVS.items()}
    return by_lower.get(state.strip().lower())


def _group(hits: list[SignHit], kind: str) -> dict[str, dict[str, Any]]:
    """``{name: {"seconds": set[int], "texts": [raw, ...]}}`` for one claim kind."""
    groups: dict[str, dict[str, Any]] = {}
    for hit in hits:
        for claim in hit.parsed_claims():
            if claim.kind != kind or not claim.value.strip():
                continue
            entry = groups.setdefault(claim.value.strip(), {"seconds": set(), "texts": []})
            entry["seconds"].add(int(float(hit.t_local)))
            text = (claim.raw_text or hit.raw_text or "").strip()
            if text and text not in entry["texts"]:
                entry["texts"].append(text)
    return groups


def _quoted(texts: list[str]) -> str:
    return "; ".join(f'"{t}"' for t in texts)


def explain_sign_label(
    lf_index: Any,
    hits: Any,
    *,
    min_corroborating_hits: int = 2,
    gps_fix_count: int = 0,
) -> tuple[Optional[SignLabel], list[str]]:
    """Derive the label and say why — ``(SignLabel | None, reasons)``.

    Every reason line starts with ``LF <idx>:`` so the dry-run can print it
    beside the LF it is about. A refusal names the rule that refused.
    """
    lf_idx = int(lf_index)
    reasons: list[str] = []
    typed = _typed_hits(hits)
    floor = max(int(min_corroborating_hits), 1)

    if int(gps_fix_count or 0) > 0:
        reasons.append(
            f"LF {lf_idx}: has {int(gps_fix_count)} GPS fix(es); the GPS chain owns its label and "
            f"signs only verify it — no sign-derived label"
        )
        return None, reasons
    if not typed:
        reasons.append(f"LF {lf_idx}: no sign hits — no sign-derived label")
        return None, reasons

    counties = _group(typed, KIND_COUNTY)
    states = _group(typed, KIND_STATE)

    if not counties:
        reasons.append(
            f"LF {lf_idx}: {len(typed)} sign hit(s) but no county named — a label needs a county"
        )
        return None, reasons
    if len(counties) > 1:
        named = ", ".join(
            f"{name!r} ({len(e['seconds'])} second(s), {_quoted(e['texts'])})"
            for name, e in sorted(counties.items())
        )
        reasons.append(
            f"LF {lf_idx}: signs name more than one county — {named}; the LF spans a county line, "
            f"so no single label is right for it"
        )
        return None, reasons

    ((county, entry),) = counties.items()
    n_seconds = len(entry["seconds"])
    if n_seconds < floor:
        reasons.append(
            f"LF {lf_idx}: county {county!r} named in {n_seconds} distinct second(s) "
            f"({_quoted(entry['texts'])}), below the {floor}-hit floor — no sign-derived label"
        )
        return None, reasons

    if not states:
        reasons.append(
            f"LF {lf_idx}: county {county!r} named but no state — a county label needs its state"
        )
        return None, reasons
    if len(states) > 1:
        named = ", ".join(f"{name!r} ({_quoted(e['texts'])})" for name, e in sorted(states.items()))
        reasons.append(
            f"LF {lf_idx}: signs name more than one state — {named}; no sign-derived label"
        )
        return None, reasons

    ((state, state_entry),) = states.items()
    abbrev = _state_abbrev(state)
    if not abbrev:
        reasons.append(
            f"LF {lf_idx}: state {state!r} is not a US state the label table knows — no sign-derived label"
        )
        return None, reasons

    label = f"{county} County, {abbrev}"
    texts = list(entry["texts"])
    for t in state_entry["texts"]:
        if t not in texts:
            texts.append(t)
    result = SignLabel(
        lf_index=lf_idx,
        label=label,
        county=county,
        state=state,
        hit_count=n_seconds,
        hit_seconds=sorted(float(h.t_local) for h in _county_hits(typed, county)),
        texts=texts,
    )
    reasons.append(
        f"LF {lf_idx}: {label!r} from {n_seconds} distinct second(s) of county sign(s) "
        f"{_quoted(entry['texts'])} and state sign(s) {_quoted(state_entry['texts'])}"
    )
    return result, reasons


def _county_hits(hits: list[SignHit], county: str) -> list[SignHit]:
    """The hits that name ``county``, one per distinct second (first wins)."""
    seen: set[int] = set()
    out: list[SignHit] = []
    for hit in sorted(hits, key=lambda h: float(h.t_local)):
        if any(c.kind == KIND_COUNTY and c.value.strip() == county for c in hit.parsed_claims()):
            sec = int(float(hit.t_local))
            if sec in seen:
                continue
            seen.add(sec)
            out.append(hit)
    return out


def derive_sign_label(
    lf_index: Any,
    hits: Any,
    *,
    min_corroborating_hits: int = 2,
    gps_fix_count: int = 0,
) -> Optional[SignLabel]:
    """The :class:`SignLabel` for one LF, or ``None`` when any rule refuses."""
    label, _reasons = explain_sign_label(
        lf_index,
        hits,
        min_corroborating_hits=min_corroborating_hits,
        gps_fix_count=gps_fix_count,
    )
    return label


def derive_sign_labels(
    hits_by_lf: Any,
    *,
    gps_points_by_lf: Any = None,
    min_corroborating_hits: int = 2,
) -> dict[int, SignLabel]:
    """``{lf_index: SignLabel}`` for every LF whose signs pass the rules.

    ``hits_by_lf`` is ``sign_verify.bucket_hits_by_lf``'s output;
    ``gps_points_by_lf`` is the per-LF GPS point map the enrichment already
    builds, so an LF with any fix is refused here rather than by the caller.
    """
    out: dict[int, SignLabel] = {}
    for lf_idx, hits in (hits_by_lf or {}).items():
        fixes = len((gps_points_by_lf or {}).get(lf_idx) or [])
        label = derive_sign_label(
            lf_idx, hits, min_corroborating_hits=min_corroborating_hits, gps_fix_count=fixes
        )
        if label is not None:
            out[int(lf_idx)] = label
    return out
