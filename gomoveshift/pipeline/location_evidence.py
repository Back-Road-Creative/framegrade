"""The location-evidence record (title-accuracy program U1).

One :class:`LocationEvidence` per footage segment, built ONLY from that
segment's own GPS fixes (a short gets its clip window, never the whole
journey). Each place the trace entered carries a scope, the method and
provenance that produced it, and a stable ``evidence_id`` — the key Waypost's
``Place(name, evidence_id)`` passes through, and the reason a place without
one is refused on both sides.

Scopes: ``in`` (the segment starts or ends there), ``through`` (a fix
resolved to it in between), ``near`` (it geocodes close to the trace but no
fix ever resolved to it — never enough for a title claim).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from gomoveshift.pipeline import gps_quality as gq

logger = logging.getLogger(__name__)

NEAR_KM = 10.0
_UNRESOLVED = frozenset({"", "none", "null", "unknown"})


def _slot(value) -> Optional[str]:
    text = str(value if value is not None else "").strip()
    return None if text.lower() in _UNRESOLVED else text


@dataclass(frozen=True)
class PlaceEvidence:
    name: str
    county: Optional[str]
    state: Optional[str]
    scope: str
    method: str
    provenance: str
    segment_id: str
    evidence_id: str


@dataclass
class LocationEvidence:
    segment_id: str
    provenance: str
    method: str
    places: list[PlaceEvidence] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    trace: list = field(default_factory=list, compare=False, repr=False)

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "provenance": self.provenance,
            "method": self.method,
            "places": [asdict(p) for p in self.places],
            "conflicts": list(self.conflicts),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LocationEvidence":
        return cls(
            segment_id=data["segment_id"],
            provenance=data["provenance"],
            method=data["method"],
            places=[PlaceEvidence(**p) for p in data.get("places") or []],
            conflicts=list(data.get("conflicts") or []),
        )

    def place(self, name: str, state: Optional[str] = None) -> Optional[PlaceEvidence]:
        """The one entered place called ``name`` (in ``state`` when given). Two
        same-name places with no state to tell them apart is a conflict, never
        a guess."""
        want = name.strip().lower()
        hits = [p for p in self.places if p.name.lower() == want]
        if state:
            hits = [p for p in hits if _state_matches(p.state, state)]
        if len(hits) > 1:
            note = f"same-name place {name!r} in {sorted({str(p.state) for p in hits})}"
            if note not in self.conflicts:
                self.conflicts.append(note)
            return None
        return hits[0] if hits else None

    def scope_of(
        self, name: str, state: Optional[str] = None, geocode_fn: Optional[Callable] = None
    ) -> Optional[str]:
        hit = self.place(name, state)
        if hit is not None:
            return hit.scope
        if callable(geocode_fn) and self.trace:
            try:
                coord = geocode_fn(name)
            except Exception:  # noqa: BLE001 — a geocoder failure is no evidence
                coord = None
            if coord:
                dist = gq.nearest_trace_distance_km(
                    float(coord[0]), float(coord[1]), self.trace, float("-inf"), float("inf")
                )
                if dist is not None and dist <= NEAR_KM:
                    return "near"
        return None


_STATE_ABBREVS: dict[str, str] = {}


def _state_matches(state: Optional[str], want: str) -> bool:
    if not state:
        return False
    if not _STATE_ABBREVS:
        from gomoveshift.pipeline.geo_constants import US_STATE_ABBREVS

        _STATE_ABBREVS.update({k.lower(): v.lower() for k, v in US_STATE_ABBREVS.items()})
    s, w = state.strip().lower(), want.strip().lower()
    return s == w or _STATE_ABBREVS.get(s) == w


def _evidence_id(segment_id: str, name: str, county, state, method: str) -> str:
    key = "|".join([segment_id, name.lower(), str(county or ""), str(state or ""), method])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def build_location_evidence(
    points,
    *,
    geocoder,
    segment_id: str,
    window: Optional[tuple[float, float]] = None,
    provenance: str = "gps",
) -> LocationEvidence:
    """The record for ONE segment: every in-window fix reverse-geocoded, places
    kept in travel order (consecutively deduped). A fix whose town is
    unresolved contributes no place."""
    method = "reverse_geocode"
    w0, w1 = window if window else (float("-inf"), float("inf"))
    coords = sorted(gq._in_window_coords(points, w0, w1))
    rec = LocationEvidence(segment_id=segment_id, provenance=provenance, method=method)
    rec.trace = [{"lat": la, "lon": lo, "time": t} for t, la, lo in coords]
    seen: list[tuple] = []
    for _t, lat, lon in coords:
        geo = geocoder.reverse_geocode_detailed(lat, lon) or {}
        town = _slot(geo.get("town"))
        if town is None:
            continue
        key = (town, _slot(geo.get("county")), _slot(geo.get("state")))
        if not seen or seen[-1] != key:
            seen.append(key)
    ordered = list(dict.fromkeys(seen))
    for i, (town, county, state) in enumerate(ordered):
        scope = "in" if i in (0, len(ordered) - 1) else "through"
        rec.places.append(
            PlaceEvidence(
                name=town,
                county=county,
                state=state,
                scope=scope,
                method=method,
                provenance=provenance,
                segment_id=segment_id,
                evidence_id=_evidence_id(segment_id, town, county, state, method),
            )
        )
    return rec


def title_claim_issues(
    label: str, record: Optional[LocationEvidence], *, geocode_fn: Optional[Callable] = None
) -> list[str]:
    """Why ``label``'s place claims are unsupported by ``record`` (empty = ok).

    An "A to B" label needs both endpoints entered (scope in/through) in that
    travel order; a place only ``near`` the trace is not a claim basis."""
    if record is None:
        return ["no location evidence record"]
    from gomoveshift.pipeline.validation.content_gates import _TITLE_ENDPOINTS_RE

    m = _TITLE_ENDPOINTS_RE.match(label or "")
    if not m:
        return []
    # The trailing slot disambiguates same-name places only when it IS a state
    # the record knows ("Birch, VA"); "County Cork" is a county, not a filter.
    tail = (label or "").rsplit(",", 1)
    state = tail[1].strip() if len(tail) == 2 else None
    if state and not any(_state_matches(p.state, state) for p in record.places):
        state = None
    issues: list[str] = []
    order: list[int] = []
    for role in ("start", "end"):
        name = m.group(role).strip()
        hit = record.place(name, state)
        if hit is None:
            scope = record.scope_of(name, state, geocode_fn=geocode_fn)
            why = f"only {scope} the trace" if scope else "never entered by the trace"
            issues.append(f"{role} {name!r} is {why}")
            continue
        order.append(record.places.index(hit))
    if len(order) == 2 and order[0] >= order[1]:
        issues.append(
            f"endpoints {m.group('start').strip()!r} → {m.group('end').strip()!r} "
            "run against the travel order"
        )
    return issues


class TitleEvidenceError(RuntimeError):
    """A long-form title names places its location-evidence record does not back."""


def refuse_unsupported_title(label: str, record, *, lf_index) -> None:
    """Raise :class:`TitleEvidenceError` (and log ``TITLE-EVIDENCE E``) when
    ``label`` makes an "A to B" claim ``record`` does not back. No record behind
    such a claim is a refusal, never a skip; a label with no A-to-B claim has
    nothing to check. ``record`` may be the dict form persisted to state."""
    from gomoveshift.pipeline.validation.content_gates import _TITLE_ENDPOINTS_RE

    if not _TITLE_ENDPOINTS_RE.match(label or ""):
        return
    if record is not None and not isinstance(record, LocationEvidence):
        record = LocationEvidence.from_dict(record) if record else None
    issues = title_claim_issues(label, record)
    if not issues:
        return
    why = "; ".join(issues)
    logger.error("TITLE-EVIDENCE E: LF %s title %r refused: %s", lf_index, label, why)
    raise TitleEvidenceError(f"LF {lf_index} title {label!r} is not backed by its GPS: {why}")
