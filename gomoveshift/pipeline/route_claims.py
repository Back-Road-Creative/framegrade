"""Retained-footage GPS claims: what a long-form label may say about where it was.

A label claims only what the GPS of the footage the video actually
keeps proves, at matching specificity. Excluded footage supplies no claim; a
multi-place route gets a label naming every place it spends real time in, or a
broader region that provably contains the whole retained route; otherwise one
actionable refusal. Nothing is interpolated across a GPS gap and no endpoint is
invented.

Measurement is per retained window (one LF segment) in media seconds. Fixes are
reverse-geocoded one per ``bin_s`` bin, so a place visited for less than one bin
can be missed; the place floor (``min_place_s``) is set well above the bin.

Gap rule: a valid fix is finite, in range, not 0,0, and
not a repeat of an earlier timestamp (the trackpoints carry no DOP or fix type to
filter on). A gap is a fix-to-fix interval longer than ``GAP_S`` (exactly 5.0 s
is not a gap), measured from the nearest valid fix on each side even when that
fix lies outside the window, so a dropout a cut falls inside still counts; only
its retained overlap counts toward ``gap_seconds``. The evidence is sampled:
it never proves containment (``containment_proven`` is always false). A separate
Census polygon check can record bounded containment under explicit travel and
GPS-error assumptions; the sampled fields and gaps remain unchanged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from gomoveshift.pipeline.geo_constants import US_STATE_ABBREVS
from gomoveshift.pipeline.location_evidence import _evidence_id

#: A stretch with no valid fix longer than this is a gap (not interpolated).
GAP_S = 5.0
#: One reverse-geocode per bin of retained time.
BIN_S = 30.0
#: Retained time in a place at or above this makes the place required.
MIN_PLACE_S = 60.0
#: A region label must cover this share of retained time with GPS.
REGION_MIN_COVERAGE = 1.0
#: A Short adds no number: it bins at ``GAP_S``, and any
#: place, gap or unsettled time at all counts -- there is no floor.
_SHORT_FLOOR_S = 0.0
_METHOD = "retained_reverse_geocode"
#: The timeline the record measures on: the phase-A parse of
#: each episode's ``.gpx`` held in state, not a fresh raw-GPX read -- a verifier
#: re-measuring from raw GPX sees sub-second per-fix shifts and a different gap count.
TIMELINE = "state.gps_per_episode"
#: Where fix times come from; a verifier re-measures from the same source.
TIMESTAMP_SOURCE = (
    "state.gps_per_episode trackpoint timestamp minus the episode's effective offset "
    "(media seconds)"
)
CONTAINMENT_BASIS = "sampled_reverse_geocode"
BOUNDARY_REVIEW_VERSION = "tiger2026-bounded-v1"
_CONTAINMENT_WORDS = re.compile(
    r"\bverified in\b|\bcontained\b|\bcontainment proven\b|\bproven (?:to be )?(?:in|within)\b",
    re.IGNORECASE,
)
#: Settlement fields of the detailed geocode, most significant first (the
#: title builder's ladder; ``name`` is POI-prone and never read).
_SETTLEMENT_KEYS = ("city", "town", "village", "hamlet", "locality", "suburb")
_LEVELS = ("county", "state")

logger = logging.getLogger(__name__)


class RouteClaimRefusedError(RuntimeError):
    """No label the retained footage's GPS supports could be chosen."""


@dataclass
class RouteWindow:
    """One retained segment: ``[start_s, end_s)`` and its ``(t, lat, lon)`` fixes.

    ``prev_fix_s`` / ``next_fix_s`` are the nearest valid fixes of the same
    file outside the window (media seconds), or ``None`` when the file has none
    on that side; the window edge is then the bound."""

    segment_id: str
    start_s: float
    end_s: float
    fixes: list = field(default_factory=list)
    source: str = ""
    prev_fix_s: Optional[float] = None
    next_fix_s: Optional[float] = None
    # The nearest source fixes outside a retained cut, with coordinates. Only
    # boundary reachability uses them; place votes never count excluded footage.
    prev_fix: Optional[tuple] = None
    next_fix: Optional[tuple] = None


@dataclass
class RouteGap:
    """A fix-to-fix interval over ``GAP_S``; ``[start_s, end_s)`` is its retained part."""

    segment_id: str
    source: str
    fix_before_s: float
    fix_after_s: float
    start_s: float
    end_s: float

    @property
    def interval_s(self) -> float:
        return self.fix_after_s - self.fix_before_s

    @property
    def seconds(self) -> float:
        return self.end_s - self.start_s

    def describe(self) -> str:
        where = f"{self.segment_id} {self.source}".strip()
        return f"{where} @{self.start_s:.1f} s +{self.seconds:.1f} s"


@dataclass
class RoutePlace:
    name: str
    county: Optional[str]
    state: Optional[str]
    retained_seconds: float
    required: bool
    evidence_id: str


@dataclass
class _Bin:
    seconds: float
    county: Optional[str]
    state: Optional[str]
    town: Optional[str] = None
    where: str = ""


def retained_windows_key(windows) -> str:
    """Deterministic key of the exact retained seconds: ``(segment_id, source,
    start_s, end_s)`` per window, rounded to ms. A boundary proof carries the key
    of the windows it checked and covers evidence of those windows only."""
    rows = [
        [w.segment_id, w.source, round(float(w.start_s), 3), round(float(w.end_s), 3)]
        for w in windows
    ]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


@dataclass
class RouteEvidence:
    retained_seconds: float
    gaps: list  # [RouteGap]
    places: list  # [RoutePlace] in first-visit order
    #: Retained seconds with a valid fix but no settlement (or a failed lookup).
    unsettled_seconds: float
    bins: list = field(default_factory=list, repr=False)
    invalid_fixes: int = 0
    duplicate_fixes: int = 0
    tick_s: Optional[float] = None
    boundary_notes: list = field(default_factory=list)
    boundary_proofs: dict = field(default_factory=dict)
    windows_key: str = ""

    @classmethod
    def from_record(cls, record: dict) -> "RouteEvidence":
        """Rebuild the evidence from a persisted :meth:`record` (offline re-check).
        A legacy record with no ``bins`` yields no bins, so every second is unresolved."""
        gaps = [
            RouteGap(
                g["segment"],
                g.get("source", ""),
                g["fix_before_s"],
                g["fix_after_s"],
                g["media_start_s"],
                g["media_start_s"] + g["length_s"],
            )
            for g in record.get("gaps") or []
        ]
        places = [
            RoutePlace(
                p["name"],
                p.get("county"),
                p.get("state"),
                p["retained_seconds"],
                bool(p.get("required")),
                p.get("evidence_id", ""),
            )
            for p in record.get("places") or []
        ]
        bins = [
            _Bin(b["seconds"], b.get("county"), b.get("state"), b.get("town"))
            for b in record.get("bins") or []
        ]
        return cls(
            record.get("retained_seconds", 0.0),
            gaps,
            places,
            record.get("unsettled_seconds", 0.0),
            bins,
            boundary_proofs=dict(record.get("boundary_proofs") or {}),
            windows_key=record.get("windows_key") or "",
        )

    @property
    def gap_seconds(self) -> float:
        return sum(g.seconds for g in self.gaps)

    @property
    def coverage(self) -> float:
        if self.retained_seconds <= 0:
            return 0.0
        return max(0.0, 1.0 - self.gap_seconds / self.retained_seconds)

    @property
    def required(self) -> list:
        return [p for p in self.places if p.required]

    def unknown_summary(self) -> str:
        return (
            f"coverage {self.coverage:.2%} ({len(self.gaps)} gap(s), gap_seconds "
            f"{self.gap_seconds:.1f}, unsettled_seconds {self.unsettled_seconds:.1f})"
        )

    def record(self) -> dict:
        """The sampled evidence, JSON-ready. Never a containment proof."""
        return {
            "timeline": TIMELINE,
            "timestamp_source": TIMESTAMP_SOURCE,
            "tick_s": self.tick_s,
            "gap_threshold_s": GAP_S,
            "gap_rule": "fix-to-fix interval > gap_threshold_s; retained overlap only",
            "retained_seconds": round(self.retained_seconds, 3),
            "coverage_pct": round(self.coverage * 100, 2),
            "gap_seconds": round(self.gap_seconds, 3),
            "unsettled_seconds": round(self.unsettled_seconds, 3),
            "gaps": [
                {
                    "segment": g.segment_id,
                    "source": g.source,
                    "media_start_s": round(g.start_s, 3),
                    "length_s": round(g.seconds, 3),
                    "fix_before_s": round(g.fix_before_s, 3),
                    "fix_after_s": round(g.fix_after_s, 3),
                    "interval_s": round(g.interval_s, 3),
                }
                for g in self.gaps
            ],
            "boundary_notes": list(self.boundary_notes),
            "boundary_proofs": dict(self.boundary_proofs),
            "windows_key": self.windows_key,
            "bins": self._bin_record(),
            "invalid_fixes": self.invalid_fixes,
            "duplicate_fixes": self.duplicate_fixes,
            "fix_quality_fields": "none in source (no DOP or fix type on the trackpoints)",
            "samples": len(self.bins),
            "places": [
                {
                    "name": p.name,
                    "county": p.county,
                    "state": p.state,
                    "retained_seconds": round(p.retained_seconds, 3),
                    "required": p.required,
                    "evidence_id": p.evidence_id,
                }
                for p in self.places
            ],
            "containment_proven": False,
            "containment_basis": CONTAINMENT_BASIS,
        }

    def _bin_record(self) -> list:
        """Retained seconds per resolved (town, county, state), names and seconds only."""
        totals: dict = {}
        for b in self.bins:
            key = (b.town, b.county, b.state)
            totals[key] = totals.get(key, 0.0) + b.seconds
        return [
            {"town": t, "county": c, "state": st, "seconds": round(sec, 3)}
            for (t, c, st), sec in totals.items()
        ]

    def region_claim(self, level: str) -> Optional[dict]:
        """The ``level`` region ("county"/"state") most retained time is in, with
        the containment evidence a region label needs; ``None`` if none known."""
        votes = Counter()
        for b in self.bins:
            name = getattr(b, level)
            if name:
                votes[name] += b.seconds
        if not votes:
            return None
        name = votes.most_common(1)[0][0]
        inside = sum(1 for b in self.bins if getattr(b, level) == name)
        other = sum(1 for b in self.bins if getattr(b, level) and getattr(b, level) != name)
        unresolved = len(self.bins) - inside - other
        proof = self.boundary_proofs.get(level)
        # A proof covers only the exact retained seconds it checked.
        proved = bool(
            proof
            and proof.get("region") == name
            and self.windows_key
            and proof.get("windows_key") == self.windows_key
        )
        # A pinned polygon proof checks every retained fix and span, so bins
        # whose lookup resolved nothing are proven inside; without one they
        # stay "outside" (nothing shows they are not).
        outside = other if proved else other + unresolved
        claim = {
            "level": level,
            "name": name,
            "required_place_ids": [
                p.evidence_id for p in self.required if getattr(p, level) == name
            ],
            "coverage": self.coverage,
            "gaps": [g.describe() for g in self.gaps],
            "samples_inside": inside,
            "samples_outside": outside,
            "samples_other": other,
            "windows_key": self.windows_key,
            "samples_unresolved": unresolved,
            "containment_proven": False,
            "containment_basis": CONTAINMENT_BASIS,
        }
        if level == "county":
            states = Counter(b.state for b in self.bins if b.county == name and b.state)
            claim["state"] = states.most_common(1)[0][0] if states else None
        if proved:
            claim["boundary_proof"] = proof
        return claim


def _valid(lat, lon) -> bool:
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return False
    return abs(lat) <= 90 and abs(lon) <= 180 and (lat, lon) != (0.0, 0.0)


#: Float slack on the gap boundary: 5.0 s computed from two floats is still 5.0 s.
_EPS = 1e-9


def _window_gaps(w: RouteWindow, fixes, gap_s: float, tick_s: Optional[float], notes) -> list:
    lo = w.prev_fix_s if w.prev_fix_s is not None and w.prev_fix_s < w.start_s else w.start_s
    hi = w.next_fix_s if w.next_fix_s is not None and w.next_fix_s >= w.end_s else w.end_s
    times = [lo, *(t for t, _la, _lo in fixes), hi]
    gaps = []
    for a, b in zip(times, times[1:]):
        start, end = max(a, w.start_s), min(b, w.end_s)
        if end <= start:
            continue
        interval = b - a
        if tick_s and abs(interval - gap_s) <= tick_s + _EPS:
            kind = "counted" if interval > gap_s + _EPS else "not counted"
            notes.append(
                f"{w.segment_id} {w.source}".strip()
                + f" interval {interval:.3f} s at {a:.3f} is within one tick "
                f"({tick_s:g} s) of {gap_s:g} s: {kind}"
            )
        if interval > gap_s + _EPS:
            gaps.append(RouteGap(w.segment_id, w.source, float(a), float(b), start, end))
    return gaps


def _covered(a: float, b: float, gaps) -> float:
    lost = sum(max(0.0, min(b, g.end_s) - max(a, g.start_s)) for g in gaps)
    return max(0.0, (b - a) - lost)


def _clean_fixes(w: RouteWindow) -> tuple:
    """``(fixes, invalid, duplicates)``: valid in-window fixes, sorted, one per timestamp."""
    fixes, invalid, dup, seen = [], 0, 0, set()
    for t, la, lo in w.fixes:
        if not (w.start_s <= float(t) < w.end_s):
            continue
        if not _valid(la, lo):
            invalid += 1
            continue
        if float(t) in seen:
            dup += 1
            continue
        seen.add(float(t))
        fixes.append((float(t), float(la), float(lo)))
    fixes.sort()
    return fixes, invalid, dup


def _tick(cleaned) -> Optional[float]:
    deltas = [b[0] - a[0] for fixes in cleaned for a, b in zip(fixes, fixes[1:])]
    deltas = [d for d in deltas if d > 0]
    return round(statistics.median(deltas), 3) if deltas else None


def _geocode(geocoder, lat, lon) -> dict:
    try:
        return geocoder.reverse_geocode_detailed(lat, lon) or {}
    except Exception:  # noqa: BLE001 — a failed lookup is unresolved time, never a place
        return {}


def _slot(value) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None


def measure_route(
    windows,
    geocoder,
    *,
    bin_s: float = BIN_S,
    gap_s: float = GAP_S,
    min_place_s: float = MIN_PLACE_S,
) -> RouteEvidence:
    """Where the retained footage was, measured from its own fixes only."""
    windows = list(windows)
    cleaned = [_clean_fixes(w) for w in windows]
    tick_s = _tick([c[0] for c in cleaned])
    retained = 0.0
    gaps: list = []
    notes: list = []
    bins: list = []
    seconds: dict = {}
    first: dict = {}
    unresolved = 0.0
    for w, (fixes, _inv, _dup) in zip(windows, cleaned):
        retained += max(0.0, w.end_s - w.start_s)
        wgaps = _window_gaps(w, fixes, gap_s, tick_s, notes)
        gaps.extend(wgaps)
        a = w.start_s
        i = 0
        while a < w.end_s:
            b = min(a + bin_s, w.end_s)
            span = _covered(a, b, wgaps)
            while i < len(fixes) and fixes[i][0] < a:
                i += 1
            if span > 0:
                rep = fixes[i] if i < len(fixes) and fixes[i][0] < b else None
                geo = _geocode(geocoder, rep[1], rep[2]) if rep else {}
                town = next((v for v in map(_slot, map(geo.get, _SETTLEMENT_KEYS)) if v), None)
                county, state = _slot(geo.get("county")), _slot(geo.get("state"))
                bins.append(_Bin(span, county, state, town, f"{w.segment_id} @{a:.0f} s"))
                if town is None:
                    unresolved += span
                else:
                    key = (town, county, state)
                    seconds[key] = seconds.get(key, 0.0) + span
                    first.setdefault(key, w.segment_id)
            a = b
    places = [
        RoutePlace(
            name=town,
            county=county,
            state=state,
            retained_seconds=secs,
            required=secs >= min_place_s,
            evidence_id=_evidence_id(first[(town, county, state)], town, county, state, _METHOD),
        )
        for (town, county, state), secs in seconds.items()
    ]
    return RouteEvidence(
        retained,
        gaps,
        places,
        unresolved,
        bins,
        invalid_fixes=sum(c[1] for c in cleaned),
        duplicate_fixes=sum(c[2] for c in cleaned),
        tick_s=tick_s,
        boundary_notes=notes,
        windows_key=retained_windows_key(windows),
    )


def containment_wording_issues(text: str) -> list:
    """Phrases in ``text`` that assert containment; sampled evidence proves none."""
    return [m.group(0) for m in _CONTAINMENT_WORDS.finditer(text or "")]


def _names(label: str, name: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(name)}(?!\w)", label, re.IGNORECASE) is not None


_CLAUSE = re.compile(r"\s+(?:via\s.*|—.*|loop)$", re.IGNORECASE)
_JOIN = re.compile(r"\s+(?:to|and|&)\s+", re.IGNORECASE)


def _named_places(label: str) -> list:
    """Place names a label's leading clause claims ("A to B via R, ST" → A, B)."""
    head = _CLAUSE.sub("", (label or "").split(",")[0].strip())
    return [t.strip() for t in _JOIN.split(head) if t.strip()]


def _state_suffix_issues(label: str, ev: RouteEvidence) -> list[str]:
    """Check an explicit route state against observed retained samples.

    A label like ``A and B, VA`` is a Virginia claim even when both named
    towns were entered. The place-list check alone cannot catch footage in
    another state. An unrecognized trailing slot may be a country or road,
    so only known state names and abbreviations (or a sampled state name)
    are treated as state claims here.
    """
    if "," not in label:
        return []
    tail = label.rsplit(",", 1)[1].strip().casefold()
    by_name = {name.casefold(): name for name in US_STATE_ABBREVS}
    by_abbrev = {abbr.casefold(): name for name, abbr in US_STATE_ABBREVS.items()}
    sampled = {b.state.casefold(): b.state for b in ev.bins if b.state}
    target = by_name.get(tail) or by_abbrev.get(tail) or sampled.get(tail)
    if target is None:
        return []
    target_key = US_STATE_ABBREVS.get(target, target).casefold()
    outside = sum(
        1
        for b in ev.bins
        if b.state and US_STATE_ABBREVS.get(b.state, b.state).casefold() != target_key
    )
    return [f"{outside} retained sample(s) outside claimed state {target!r}"] if outside else []


def _region_ok(claim: Optional[dict], ev: RouteEvidence, min_coverage: float) -> list:
    if claim is None:
        return ["no region is known for the retained footage"]
    proof = claim.get("boundary_proof") or {}
    if proof:
        from .boundary_proof import pinned_boundary_proof

    if proof and pinned_boundary_proof(proof, claim):
        if claim.get("samples_other"):
            return [
                f"{claim['samples_other']} retained sample(s) geocoded outside "
                f"{claim['name']} despite the boundary proof (zero tolerance)"
            ]
        if claim["level"] == "county" and claim.get("state"):
            state_claim = ev.region_claim("state")
            if state_claim is None or state_claim["name"] != claim["state"]:
                return ["county state has no matching retained-route claim"]
            return _region_ok(state_claim, ev, min_coverage)
        return []
    why = []
    missing = {p.evidence_id for p in ev.required} - set(claim["required_place_ids"])
    if missing:
        why.append(f"{len(missing)} required place(s) lie outside {claim['name']}")
    if claim["samples_outside"]:
        why.append(f"{claim['samples_outside']} retained sample(s) outside or unresolved")
    if claim["gaps"]:
        why.append(f"{len(claim['gaps'])} GPS gap(s): {'; '.join(claim['gaps'])}")
    if claim["coverage"] < min_coverage:
        why.append(f"coverage {claim['coverage']:.0%} < {min_coverage:.0%}")
    return why


def _state_key(state: Optional[str]) -> str:
    return US_STATE_ABBREVS.get(state, state or "").casefold()


_URL = re.compile(r"(?:https?://|www\.)\S+|(?<![\w@])[\w-]+(?:\.[\w-]+)+/\S*")


def _named_states(text: str, ev: RouteEvidence) -> set:
    """State keys the text names: full names (longest first, so "West Virginia"
    owns its span) and upper-case abbreviations."""
    full = sorted(set(US_STATE_ABBREVS) | {b.state for b in ev.bins if b.state}, key=len)
    # A link names no place: youtu.be/...-ID-1 is not Idaho.
    rest, found = _URL.sub(" ", text or ""), set()
    for name in reversed(full):
        pattern = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE)
        if pattern.search(rest):
            found.add(_state_key(name))
            rest = pattern.sub(" ", rest)
    for name, abbr in US_STATE_ABBREVS.items():
        if re.search(rf"(?<!\w){abbr}(?!\w)", rest):
            found.add(_state_key(name))
    return found


def place_text_issues(text: str, ev: RouteEvidence) -> list:
    """Zero tolerance: every retained second must lie in a place
    ``text`` names. A bin is covered when its settlement, county or state is named
    (a settlement or county in an unnamed state is not, when the text names states);
    an unresolved bin or GPS gap is covered only by a named county/state whose pinned
    proof covers exactly these windows. Text naming no known place has no issue."""
    towns = {p.name for p in ev.places} | {b.town for b in ev.bins if b.town}
    counties = {b.county for b in ev.bins if b.county}
    t_named = {t for t in towns if _names(text or "", t)}
    c_named = {
        c
        for c in counties
        if _names(text or "", c) or _names(text or "", re.sub(r"\s+County$", "", c, flags=re.I))
    }
    s_named = _named_states(text or "", ev)
    if not (t_named or c_named or s_named):
        return []
    from .boundary_proof import pinned_boundary_proof

    proof_covers = False
    for level in _LEVELS:
        claim = ev.region_claim(level)
        if not claim or not claim.get("boundary_proof"):
            continue
        named = (
            claim["name"] in c_named if level == "county" else _state_key(claim["name"]) in s_named
        )
        if named and pinned_boundary_proof(claim["boundary_proof"], claim):
            proof_covers = True
    names = ", ".join(sorted(t_named | c_named | {k.title() for k in s_named}))
    outside: dict = {}
    for b in ev.bins:
        sk = _state_key(b.state) if b.state else None
        state_ok = not s_named or sk is None
        if sk in s_named or (
            state_ok and ((b.town and b.town in t_named) or (b.county and b.county in c_named))
        ):
            continue
        label = b.town or b.county or b.state
        if not label and proof_covers:
            continue
        key = label or "an unresolved place (no county/settlement/state)"
        sec, where = outside.get(key, (0.0, b.where))
        outside[key] = (sec + b.seconds, where)
    issues = [
        f"{sec:.0f} s of retained footage in {key} ({where}) is outside the place(s) the "
        f"text names ({names})"
        for key, (sec, where) in outside.items()
    ]
    if ev.gap_seconds > _EPS and not proof_covers:
        issues.append(
            f"{ev.gap_seconds:.0f} s of retained footage is a GPS gap "
            f"({'; '.join(g.describe() for g in ev.gaps)}) and no pinned boundary proof "
            f"of a place the text names ({names}) covers it"
        )
    return issues


def recheck_place_text(record: dict, text: str) -> list:
    """Re-check ``text`` against a persisted route ``record`` (``RouteEvidence.record()``,
    or ``route_record(ev)["record"]``), e.g. a finished render's ``per_lf_route_record``.
    Empty = every retained second is inside a place the text names. A record with no
    ``bins`` (made before this gate) cannot be checked and returns an issue."""
    if "bins" not in (record or {}):
        return ["route record has no 'bins' (measured before the zero-tolerance gate); re-measure"]
    return place_text_issues(text, RouteEvidence.from_record(record))


def region_label(claim: dict) -> str:
    name = claim["name"]
    if claim["level"] == "state":
        return name
    state = claim.get("state")
    return f"{name}, {US_STATE_ABBREVS.get(state, state)}" if state else name


def label_issues(
    label: str,
    ev: RouteEvidence,
    *,
    min_coverage: float = REGION_MIN_COVERAGE,
    unknown_floor_s: float = MIN_PLACE_S,
) -> list:
    """Why ``label`` over-claims the retained route (empty = supported).
    ``unknown_floor_s``: gap + unsettled time at or above this is an issue."""
    head = (label or "").split(",")[0].strip().lower()
    for level in _LEVELS:
        claim = ev.region_claim(level)
        if claim and claim["name"].lower() == head:
            why = _region_ok(claim, ev, min_coverage)
            return [f"region {claim['name']!r}: {w}" for w in why]
    known = {p.name.lower() for p in ev.places}
    issues = [
        f"names {n!r}, which no retained GPS fix resolves to"
        for n in _named_places(label)
        if n.lower() not in known
    ]
    issues += [
        f"omits {p.name!r} ({p.retained_seconds:.0f} s of retained footage)"
        for p in ev.required
        if not _names(label or "", p.name)
    ]
    issues += _state_suffix_issues(label, ev)
    issues += place_text_issues(label, ev)
    unknown = ev.gap_seconds + ev.unsettled_seconds
    if unknown >= unknown_floor_s and unknown > _EPS:
        issues.append(
            f"{unknown:.0f} s of retained footage has no resolved place (gap_seconds "
            f"{ev.gap_seconds:.1f}, unsettled_seconds {ev.unsettled_seconds:.1f})"
        )
    issues += [
        f"asserts containment ({w!r}); the evidence is sampled"
        for w in containment_wording_issues(label)
    ]
    return issues


def _region_candidates(ev: RouteEvidence) -> list:
    out = []
    for level in _LEVELS:
        claim = ev.region_claim(level)
        if claim is None:
            continue
        if level == "county":
            states = Counter(b.state for b in ev.bins if b.county == claim["name"] and b.state)
            claim["state"] = states.most_common(1)[0][0] if states else None
        out.append((claim, region_label(claim)))
    return out


def select_supported_label(
    label: str,
    ev: RouteEvidence,
    *,
    operator: bool = False,
    min_coverage: float = REGION_MIN_COVERAGE,
    unknown_floor_s: float = MIN_PLACE_S,
) -> tuple:
    """``(label, note)``: ``label`` if the retained GPS supports it, else the
    narrowest region that provably contains the whole retained route. An
    operator label is checked, never replaced. Raises
    :class:`RouteClaimRefusedError` with every reason when nothing qualifies."""
    issues = label_issues(label, ev, min_coverage=min_coverage, unknown_floor_s=unknown_floor_s)
    if not issues:
        return label, ""
    tried = []
    if not operator:
        for claim, candidate in _region_candidates(ev):
            why = _region_ok(claim, ev, min_coverage) or place_text_issues(candidate, ev)
            if not why:
                return candidate, f"{label!r} {'; '.join(issues)} — using {candidate!r}"
            tried.append(f"{candidate!r}: {'; '.join(why)}")
    places = ", ".join(f"{p.name} {p.retained_seconds:.0f} s" for p in ev.places) or "none"
    lines = [
        f"{'operator ' if operator else ''}label {label!r} is not supported by the retained GPS:",
        *(f"  - {i}" for i in issues),
        *(f"  - fallback {t}" for t in tried),
        f"  places in retained footage: {places}; {ev.unknown_summary()}",
        "  next: pass --per-lf-location 'N=<label naming every required place>' "
        "(still checked) or leave it for the human OST review",
    ]
    raise RouteClaimRefusedError("\n".join(lines))


def short_route_window(rank, start_s: float, end_s: float, day_points) -> RouteWindow:
    """One Short's retained window ``[start_s, end_s)`` over the day's ``(t, lat, lon)``
    fixes, with the nearest valid fix on each side so a dropout the clip edge falls
    inside still counts as a gap."""
    return _day_window(f"short{rank}", start_s, end_s, day_points)


def _day_window(segment_id: str, start_s: float, end_s: float, day_points) -> RouteWindow:
    valid = sorted(
        (float(t), la, lo) for t, la, lo in (day_points or []) if _valid(la, lo)
    )  # fmt: skip
    before = [p[0] for p in valid if p[0] < start_s]
    after = [p[0] for p in valid if p[0] >= end_s]
    return RouteWindow(
        segment_id=segment_id,
        start_s=float(start_s),
        end_s=float(end_s),
        fixes=[p for p in valid if start_s <= p[0] < end_s],
        prev_fix_s=before[-1] if before else None,
        next_fix_s=after[0] if after else None,
    )


def make_short_route_fn(geocoder):
    """``check(day_points, short_range, text) -> (label, note)`` for one GPS-bearing
    Short: the long-form rule at Short scale (one sample per ``GAP_S``, no floor).
    Raises :class:`RouteClaimRefusedError` when no supported label exists."""

    def check(day_points, short_range, text):
        w = short_route_window(
            short_range.rank, float(short_range.start_sec), float(short_range.end_sec), day_points
        )
        ev = measure_route([w], geocoder, bin_s=GAP_S, min_place_s=_SHORT_FLOOR_S)
        return select_supported_label(text, ev, unknown_floor_s=_SHORT_FLOOR_S)

    check.geocoder = geocoder
    return check


def make_superlong_route_fn(geocoder):
    """``check(part_index, ranges, day_points, text) -> (label, note, ev)`` for one
    superlong part: the long-form rule (``BIN_S``, ``MIN_PLACE_S``, ``GAP_S``, region
    fallback) over that part's OWN retained ``ranges`` (day-master seconds), never
    the day's. Raises :class:`RouteClaimRefusedError` when no supported label exists.
    Check the part's lower-thirds against ``ev`` with :func:`lower_third_issues`."""

    def check(part_index, ranges, day_points, text):
        if geocoder is None:
            raise RouteClaimRefusedError(
                f"label {text!r} cannot be checked — no geocoder is available to resolve "
                "its retained GPS; next: restore the geocoder cache or network"
            )
        windows = [
            _day_window(f"part{part_index}.{i}", float(a), float(b), day_points)
            for i, (a, b) in enumerate(ranges or [], 1)
        ]
        ev = measure_route(windows, geocoder)
        label, note = select_supported_label(text, ev)
        return label, note, ev

    check.geocoder = geocoder
    return check


def lower_third_issues(lower_thirds, ev: RouteEvidence) -> list:
    """Lower-thirds naming a place no retained fix of ``ev`` resolves to
    (``place_not_on_route``). A lower-third is never broadened: it is refused."""
    known = {p.name.lower() for p in ev.places}
    return [
        f"lower-third {lbl!r} at {float(t):.0f} s names {n!r}, which no retained GPS fix "
        "of this part resolves to (place_not_on_route)"
        for lbl, t in (lower_thirds or [])
        for n in _named_places(lbl)
        if n.lower() not in known
    ]


def route_claim_outcome(
    idx: int, requested: str, ev: Optional["RouteEvidence"], *, chosen: Optional[str], reason: str
) -> dict:
    """One LF's claim decision: ``ev.record()`` verbatim (``None`` fields when nothing
    could be measured) plus the requested label, the label kept (``None`` = refused)
    and why. Never asserts containment; carries names and seconds, never a fix."""
    rec = ev.record() if ev is not None else {
        "containment_proven": False, "containment_basis": CONTAINMENT_BASIS,
    }  # fmt: skip
    return {
        "lf_index": idx, "label_requested": requested, "label": chosen,
        "refused": chosen is None, "reason": reason, **rec,
    }  # fmt: skip


def enforce_route_claims(
    ost_text_by_date: dict,
    windows_by_lf: dict,
    geocoder,
    *,
    overrides: Optional[dict] = None,
    outcomes: Optional[dict] = None,
    boundary_prover=None,
    require_review_support: bool = False,
    review_support_available: bool = False,
    recording_dates: Optional[dict] = None,
) -> dict:
    """Check every LF's burn text against its retained GPS, in place.

    A label the retained footage does not support is replaced by its broader
    supported region (operator labels are checked, never replaced). Every LF
    that still has no supported label is collected into ONE
    :class:`RouteClaimRefusedError`, raised after all LFs are measured. An LF
    with no GPS window has nothing to measure here (the GPS-less rules apply).
    Returns ``{lf_index: RouteEvidence}`` for the measured LFs. ``outcomes``,
    when given, is filled per checked LF -- before any raise -- with
    :func:`route_claim_outcome`, the record a dry run writes.
    """
    outcomes = {} if outcomes is None else outcomes
    overrides = {k: v for k, v in (overrides or {}).items() if str(v or "").strip()}
    measured: dict = {}
    refusals: list = []
    for _date, day in ost_text_by_date.items():
        for idx, label in list(day.items()):
            windows = windows_by_lf.get(idx)
            if not windows:
                continue
            if geocoder is None:
                why = (
                    f"label {label!r} cannot be checked — no geocoder is available "
                    "to resolve its retained GPS; next: restore the geocoder cache or network"
                )
                refusals.append(f"LF {idx}: {why}")
                outcomes[idx] = route_claim_outcome(idx, label, None, chosen=None, reason=why)
                continue
            ev = measure_route(windows, geocoder)
            measured[idx] = ev
            needs_boundary = bool(boundary_prover and label_issues(label, ev))
            recorded_on = (recording_dates or {}).get(idx, _date)
            year_mismatch = needs_boundary and not str(recorded_on).startswith("2026-")
            review_missing = (
                needs_boundary
                and not year_mismatch
                and require_review_support
                and not review_support_available
            )
            if needs_boundary and not (year_mismatch or review_missing):
                for claim, _candidate in _region_candidates(ev):
                    if _region_ok(claim, ev, REGION_MIN_COVERAGE):
                        proof = boundary_prover.prove(windows, claim)
                        if proof.get("verified") is True:
                            proof["recording_date"] = str(recorded_on)
                            ev.boundary_proofs[claim["level"]] = proof
            try:
                chosen, note = select_supported_label(label, ev, operator=idx in overrides)
            except RouteClaimRefusedError as exc:
                reason = str(exc)
                if year_mismatch:
                    reason += "\n  - Census 2026 boundary year does not match the recording date"
                if review_missing:
                    reason += (
                        "\n  - Waypost bounded-region review unavailable; refusing before encode"
                    )
                refusals.append(f"LF {idx}: {reason}")
                outcomes[idx] = route_claim_outcome(idx, label, ev, chosen=None, reason=reason)
                continue
            outcomes[idx] = route_claim_outcome(idx, label, ev, chosen=chosen, reason=note)
            if note:
                logger.warning("ROUTE-CLAIM: LF %s %s", idx, note)
                day[idx] = chosen
    if refusals:
        for r in refusals:
            logger.error("ROUTE-CLAIM E: %s", r.splitlines()[0])
        raise RouteClaimRefusedError(
            f"{len(refusals)} long-form label(s) claim more than the retained GPS proves:\n"
            + "\n".join(refusals)
        )
    return measured
