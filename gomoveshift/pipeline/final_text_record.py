"""The final-text record (G2): one long-form LF's visible text, bound to its
evidence, reviewed by Waypost and re-verified before upload.

Phase BC persists each LF's retained-route evidence (:func:`route_record`).
Phase D builds Waypost's ``final-text-record-1`` from it and stores the review
in the LF's metadata (:func:`build_final_text_review`). The upload path calls
:func:`check_final_text` from GMS's own ``upload_video``, just before
``videos().insert``; :func:`final_text_context` recomputes the source and plan
fingerprints and the visible text from current state and refuses anything the
review does not cover exactly, or a review whose verdict is not ``agree``.

The record carries only place names, seconds and ids — no fix, coordinate or
trace. ``approval_hash`` is the day's OST audit hash and only when that audit
passed against it, so an unapproved or empty marker never yields one. Passing
is not an approval: the OST confirmation gate still runs before the burn.

GPS-less LFs (no location-evidence record) have nothing to review here; their
operator label stays under the existing gates.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Optional

from seo_manager.analytics.drive_templates import LAYOUT as DRIVE_LAYOUT
from seo_manager.analytics.final_text_review import (
    RECORD_VERSION,
    FinalTextPolicy,
    FinalTextReviewRefused,
    review_final_text,
    review_to_dict,
    verify_review,
)
from seo_manager.analytics.final_text_surfaces import compose_description, compose_tags

from .geo_constants import US_STATE_ABBREVS
from .route_claims import (
    BOUNDARY_REVIEW_VERSION,
    MIN_PLACE_S,
    REGION_MIN_COVERAGE,
    recheck_place_text,
)


def boundary_review_available() -> bool:
    """The pinned Waypost reviewer must understand bounded region evidence."""
    from seo_manager.analytics import final_text_review

    return getattr(final_text_review, "BOUNDARY_PROOF_VERSION", None) == BOUNDARY_REVIEW_VERSION


def short_clip_place_names(regions, roads) -> list:
    """The county names (from ``"County, State"`` votes) and GPS road names of
    one Short's own clip, in that order. Nothing is inherited from the LF."""
    counties = [c for label in regions or () if (c := str(label).partition(",")[0].strip())]
    return counties + [str(r).strip() for r in roads or () if str(r).strip()]


def reviewable_short_tags(tags, regions, roads) -> list:
    """``tags`` limited to what Waypost can review for a Short: its tag templates,
    the clip's county names and the clip's own GPS road names; the clip's names
    are added when absent. Every other tag (filler, LF-wide road or POI) is
    dropped for a Short only; a long-form tag list never passes through here."""
    from seo_manager.analytics.final_text_surfaces import all_templates

    templates = {text for tid, text in all_templates().items() if tid.startswith("tag.")}
    names = short_clip_place_names(regions, roads)
    kept = [t for t in tags if t in templates or t in names]
    return kept + [n for n in dict.fromkeys(names) if n not in kept]


logger = logging.getLogger(__name__)

#: The GMS route-claim policy agreed with Waypost: a place needs a minute of
#: retained footage; a region stands in for places only with full coverage.
GMS_TEXT_POLICY = FinalTextPolicy("gms-route-claims-1", MIN_PLACE_S, REGION_MIN_COVERAGE)

_REGION_LEVELS = ("county", "state")

#: The description layout every GMS drive upload (long-form, superlong part) is
#: reviewed under; Waypost's id, never a literal here.
DESCRIPTION_LAYOUT = DRIVE_LAYOUT

#: On: the description and tags being sent must equal the reviewed
#: surfaces byte for byte (``stale:description`` / ``stale:tags``). Off, they
#: are carried to the check and ignored.
BIND_DESCRIPTION_TAGS = True


def reviewed_text(metadata, description, tags):
    """``(description, tags)`` to send: Waypost's composed text when the
    metadata carries drive entries, else the given text unchanged (a GPS-less
    LF has no review). Robust by construction: the sent bytes are the reviewed
    bytes, never a second build of the same text."""
    entries = (metadata or {}).get("final_text_description")
    if not entries:
        return description, tags
    return compose_description(entries), compose_tags(entries)


class FinalTextReviewError(RuntimeError):
    """A long-form upload whose visible text no agreeing review covers."""

    def __init__(self, lf_index, reasons, detail: str = ""):
        self.lf_index = lf_index
        self.reasons = list(reasons)
        super().__init__(
            f"LF {lf_index}: final-text review refused ({', '.join(self.reasons)})"
            + (f": {detail}" if detail else "")
            + "; next: fix the label or footage and re-run Phase D through "
            "`gomoveshift health heal` / resume, which rebuilds the review"
        )


class PublicationGateError(FinalTextReviewError):
    """Every YouTube insert is held while the final-text byte-bind is off."""

    def __init__(self, message: str):
        RuntimeError.__init__(self, message)
        self.lf_index = None
        self.reasons = ["final_text_bind_off"]


def route_record(ev) -> dict:
    """The persisted per-LF route evidence: the sampled record plus its region
    claims (``None`` levels dropped)."""
    claims = (ev.region_claim(level) for level in _REGION_LEVELS)
    return {"record": ev.record(), "region_claims": [c for c in claims if c]}


def _digest(prefix: str, payload) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return prefix + hashlib.sha256(body.encode("utf-8")).hexdigest()


def source_fingerprint(state) -> Optional[str]:
    """The drive's source footage, by file identity (sorted paths)."""
    files = sorted(str(p) for p in (getattr(state, "episode_files", None) or []))
    return _digest("src-", files) if files else None


def plan_fingerprint(lf) -> Optional[str]:
    """The LF's retained segments: source file and media range, to the ms."""
    segments = getattr(lf, "segments", None) if lf is not None else None
    if not segments:
        return None
    rows = [
        [
            int(getattr(s, "source_file_index", -1)),
            round(float(getattr(s, "start_time", 0.0) or 0.0), 3),
            round(float(getattr(s, "end_time", 0.0) or 0.0), 3),
        ]
        for s in segments
    ]
    return _digest("plan-", {"lf": int(getattr(lf, "index", 0)), "segments": rows})


def _per_lf(state, field: str, idx: int):
    table = getattr(state, field, None) or {}
    return table.get(idx, table.get(str(idx)))


def visible_text(state, lf_idx: int, metadata: Optional[dict]) -> dict:
    """``{surface: text}`` the upload shows besides the title: the burned OST
    and the thumbnail, which Phase D draws from the same text."""
    ost = str(_per_lf(state, "per_lf_ost_text", lf_idx) or "").strip()
    thumb = ost or str((metadata or {}).get("title") or "").strip()
    out = {}
    if ost:
        out["ost"] = ost
    if thumb:
        out["thumbnail"] = thumb
    return out


def needs_review(state, lf_idx: int) -> bool:
    """An LF with GPS location evidence or route evidence ships only with a review."""
    return bool(
        _per_lf(state, "per_lf_location_evidence", lf_idx)
        or _per_lf(state, "per_lf_route_record", lf_idx)
    )


def _approval_hash(state, date: Optional[str]) -> Optional[str]:
    if not date:
        return None
    passed = (getattr(state, "ost_audit_passed", None) or {}).get(date) is True
    digest = (getattr(state, "ost_audit_hash", None) or {}).get(date)
    return str(digest) if passed and digest else None


def _region_id(level: str, name: str) -> str:
    return "region-" + hashlib.sha1(f"{level}|{name}".encode()).hexdigest()[:16]


def _named(text: str, value: str) -> bool:
    """``value`` appears in ``text`` as whole words ("SC" is not in "Scenic")."""
    pattern = rf"(?<!\w){re.escape(value)}(?!\w)"
    return bool(value) and re.search(pattern, text or "", re.IGNORECASE) is not None


def _evaluated_decision(decision: dict, title: str, route_places, regions, location_places):
    """``decision`` with its chosen claims mapped onto retained-route evidence.

    Each claim id from the location-evidence record becomes the route place of
    the same name, county and state; one the retained route never resolved
    stays as-is, so Waypost refuses it as ``evidence_missing``. Every route
    place or region the title names is added as a claim. A decision with no
    candidates gets one for its chosen title, so the claims are checked rather
    than refused wholesale."""
    by_key = {(p["name"].casefold(), p["county"], p["state"]): p for p in route_places}
    loc_by_id = {p.get("evidence_id"): p for p in location_places}
    out = json.loads(json.dumps(decision or {}))
    candidates = out.get("candidates") or []
    index = out.get("chosen_index")
    if not candidates:
        index = 0
        candidates = [{"index": 0, "title": out.get("chosen_title"), "claim_evidence": []}]
        out["candidates"], out["chosen_index"] = candidates, index
    if not (isinstance(index, int) and 0 <= index < len(candidates)):
        return out
    claims = []
    for claim in candidates[index].get("claim_evidence") or []:
        place = loc_by_id.get(claim)
        hit = None
        if place:
            key = (str(place.get("name", "")).casefold(), place.get("county"), place.get("state"))
            hit = by_key.get(key)
        claims.append(hit["evidence_id"] if hit else claim)
    claims += [p["evidence_id"] for p in route_places if _named(title, p["name"])]
    claims += [r["evidence_id"] for r in regions if _named(title, r["value"])]
    candidates[index]["claim_evidence"] = list(dict.fromkeys(claims))
    return out


def build_final_text_review(state, lf, metadata: dict) -> Optional[dict]:
    """Waypost's review of LF ``lf``'s final text, as its persisted dict, or
    ``None`` when the LF has no GPS evidence to review against."""
    idx = int(getattr(lf, "index", 0))
    if not needs_review(state, idx):
        return None
    entry = _per_lf(state, "per_lf_route_record", idx) or {}
    record = entry.get("record") or {}
    route_places = list(record.get("places") or [])
    date = str(_per_lf(state, "per_lf_recording_date", idx) or "") or None
    decision = (metadata or {}).get("title_decision") or {}
    title = str(decision.get("chosen_title") or (metadata or {}).get("title") or "")
    location = _per_lf(state, "per_lf_location_evidence", idx) or {}

    evidence = [
        {
            "evidence_id": p["evidence_id"],
            "kind": "location",
            "value": p["name"],
            "source": record.get("containment_basis") or "retained_gps",
        }
        for p in route_places
    ]
    gaps = [
        {
            "start_s": g["media_start_s"],
            "end_s": g["media_start_s"] + g["length_s"],
            "segment_id": g["segment"],
        }
        for g in record.get("gaps") or []
    ]
    regions = []
    for claim in entry.get("region_claims") or []:
        # A US state the title abbreviates ("SC") is the same measured claim,
        # so the abbreviation is its own region item with the same coverage.
        names = [claim["name"], US_STATE_ABBREVS.get(claim["name"])]
        for name in filter(None, names):
            rid = _region_id(claim["level"], name)
            regions.append({"evidence_id": rid, "value": name, "claim": claim})
            evidence.append(
                {
                    "evidence_id": rid,
                    "kind": "region",
                    "value": name,
                    "source": claim.get("containment_basis") or "retained_gps",
                }
            )
    date_id = f"date-{date}" if date else None
    if date:
        evidence.append(
            {
                "evidence_id": date_id,
                "kind": "date",
                "value": date,
                "source": "per_lf_recording_date",
            }
        )

    def ids_in(text):
        named = [p["evidence_id"] for p in route_places if _named(text, p["name"])]
        return named + [r["evidence_id"] for r in regions if _named(text, r["value"])]

    candidates = decision.get("candidates") or [{}]
    approved = {
        "record_version": RECORD_VERSION,
        "approval_hash": _approval_hash(state, date),
        "title": candidates[0].get("title") or decision.get("chosen_title"),
        "recording_date": date,
        "date_evidence_id": date_id,
        "source_fingerprint": source_fingerprint(state),
        "plan_fingerprint": plan_fingerprint(lf),
        "route_places": [
            {
                "evidence_id": p["evidence_id"],
                "retained_seconds": p["retained_seconds"],
                "required": p["required"],
            }
            for p in route_places
        ],
        "region_claims": [
            {
                "evidence_id": r["evidence_id"],
                "contains": r["claim"]["required_place_ids"],
                "coverage": r["claim"]["coverage"],
                "gaps": gaps,
                "samples_outside": r["claim"]["samples_outside"],
                **(
                    {"boundary_proof": r["claim"]["boundary_proof"]}
                    if r["claim"].get("boundary_proof")
                    else {}
                ),
            }
            for r in regions
        ],
        "visible": [
            {
                "surface": s,
                "kind": "place_line",
                "text": t,
                "evidence_ids": ids_in(t),
                "scope": "whole",
            }
            for s, t in visible_text(state, idx, metadata).items()
        ]
        # Waypost-built description and tag entries, once a generator supplies
        # them; none today, so a drive review lacks its required layout blocks.
        + list((metadata or {}).get("final_text_description") or []),
        "description_layout": DESCRIPTION_LAYOUT,
    }
    evaluated = _evaluated_decision(
        decision, title, route_places, regions, location.get("places") or []
    )
    review = review_final_text(approved, evaluated, evidence, GMS_TEXT_POLICY)
    level = logging.INFO if review.verdict == "agree" else logging.ERROR
    logger.log(
        level,
        "FINAL-TEXT: LF %s verdict=%s reasons=%s",
        idx,
        review.verdict,
        ", ".join(review.reasons) or "none",
    )
    return review_to_dict(review)


def final_text_context(state, lf_idx: int, lf, metadata) -> Optional[dict]:
    """What the upload will send, recomputed from current state, for
    :func:`check_final_text`; ``None`` for an LF with no GPS evidence."""
    if not needs_review(state, lf_idx):
        return None
    if not _per_lf(state, "per_lf_route_record", lf_idx):
        raise FinalTextReviewError(
            lf_idx, ["route_record_missing"], "state has no retained-route record for this LF"
        )
    meta = metadata if isinstance(metadata, dict) else {}
    return {
        "route_record": (_per_lf(state, "per_lf_route_record", lf_idx) or {}).get("record"),
        "lf_index": lf_idx,
        "visible": visible_text(state, lf_idx, meta),
        "source_fingerprint": source_fingerprint(state) or "",
        "plan_fingerprint": plan_fingerprint(lf) or "",
        "policy_version": str((meta.get("title_decision") or {}).get("policy_version") or ""),
    }


def _place_gate(context: dict, lf_idx, title, description, tags) -> None:
    """Zero tolerance before upload: every text that names a place must hold every
    retained second (title, description and the tags as one text)."""
    record = context.get("route_record")
    if not record and not context.get("place_record_required"):
        return
    texts = {"title": title, "description": description, "tags": ", ".join(tags or [])}
    issues = [
        f"{surface}: {issue}"
        for surface, text in texts.items()
        if text
        for issue in recheck_place_text(record or {}, text)
    ]
    if issues:
        logger.error("FINAL-TEXT E: LF %s place gate: %s", lf_idx, "; ".join(issues))
        raise FinalTextReviewError(lf_idx, ["place_outside_text"], "; ".join(issues))


def check_final_text(
    review,
    context: Optional[dict],
    title: str,
    description: Optional[str] = None,
    tags: Optional[list] = None,
) -> None:
    """Refuse unless ``review`` agrees and covers exactly ``title`` and the
    ``context`` (from :func:`final_text_context`) being sent now.

    ``description`` (exact bytes) and ``tags`` are carried here for the
    byte-bind; they are compared only while :data:`BIND_DESCRIPTION_TAGS` is on.
    The review must carry the layout ``context["description_layout"]`` names,
    :data:`DESCRIPTION_LAYOUT` when the key is absent: a drive review made
    without it is always refused. Only a Short's context names ``None``."""
    lf_idx = (context or {}).get("lf_index")
    if not context:
        raise FinalTextReviewError(lf_idx, ["context_missing"], "no upload context to verify")
    layout = context.get("description_layout", DESCRIPTION_LAYOUT)
    if not isinstance(review, dict) or (
        layout is not None and review.get("description_layout") != layout
    ):
        raise FinalTextReviewError(
            lf_idx, ["layout_missing"], f"review lacks description_layout={layout}"
        )
    surfaces = review.get("surfaces") or []
    # Bind off: the reviewed description and tags are compared to themselves, so
    # their surfaces drop out of the ``visible`` comparison without binding what
    # is sent. Bind on: the exact bytes being sent are compared.
    bound = (
        {"description": description, "tags": list(tags or [])}
        if BIND_DESCRIPTION_TAGS
        else {"description": compose_description(surfaces), "tags": compose_tags(surfaces)}
    )
    try:
        verify_review(
            review,
            final_title=title,
            visible=context["visible"],
            source_fingerprint=context["source_fingerprint"],
            plan_fingerprint=context["plan_fingerprint"],
            policy_version=context["policy_version"],
            text_policy=GMS_TEXT_POLICY,
            **bound,
        )
    except FinalTextReviewRefused as exc:
        logger.error("FINAL-TEXT E: LF %s upload refused: %s", lf_idx, exc)
        raise FinalTextReviewError(lf_idx, exc.reasons, str(exc)) from exc
    # Last: a text the review and byte-bind accept must still name only places that
    # hold every retained second.
    surfaces_now = review.get("surfaces") if isinstance(review, dict) else None
    if surfaces_now:
        # What ships is the reviewed text (bind on: the byte-bind refuses any other).
        description, tags = compose_description(surfaces_now), compose_tags(surfaces_now)
    _place_gate(context, lf_idx, title, description, tags)
