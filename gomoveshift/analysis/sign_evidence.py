"""Shared value types for road-sign GPS verification.

Road signs are the only *independent* evidence source this pipeline has for
"where was this footage shot". Every other location surface descends from one
chain — GPS trace -> geocoder -> ``ResolvedLocation`` — so when the trace and
the geocoder agree with each other but are both wrong (drift, a mislabel, a
degraded reverse-geocode), nothing downstream can tell. A physical sign in the
frame can.

Sign reading is therefore **advisory evidence that validates the existing
``ResolvedLocation``, never a new location authority**: nothing in this module
or its consumers ever writes a location. The flow is three decoupled halves
joined by the two dataclasses here:

``SignHit``
    One OCR'd sign observation, recorded during Phase A on raw (pre-blur)
    footage. Carries BOTH clocks: ``episode_index`` + ``t_local`` (the
    authoritative pair — the same ``(source_file_index, start_time..end_time)``
    coordinates an LF's segments are expressed in, so bucketing a hit into an
    LF is an exact span test rather than a reconstruction) and ``ts_day`` (the
    day-master second, for day-level surfaces and logs). ``t_local`` is the
    bucketing key by design: reconstructing a day-master offset at compare time
    is exactly the misfiling class this feature exists to catch.

``SignClaim``
    A location assertion parsed out of one hit's raw text ("Example County",
    "Welcome to Clifden", "I-95"). ``kind`` holds a ``LocationType`` *value*
    (plain ``str``) so the whole graph stays JSON-native.

``SignVerdict``
    The per-LF comparison outcome. Advisory by default (printed at the
    ``--label-dry-run`` gate, WARN on conflict); ``--sign-verify-strict``
    escalates a corroborated conflict to a pre-encode hard fail.

All three are plain dataclasses with JSON-native fields ONLY, so they round-trip
through ``PipelineState``'s serializer (``dataclasses.asdict`` out,
``cls(**value)`` in) with no custom codec — the same contract
``ResolvedLocation`` follows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# ``SignVerdict.status`` values. Deliberately three-valued: "no signal" (no
# legible sign named a place) is NOT agreement, and must never be reported as
# corroboration — a silent OCR failure would otherwise read as evidence.
STATUS_CORROBORATES = "corroborates"
STATUS_NO_SIGNAL = "no-signal"
STATUS_CONFLICTS = "conflicts"


@dataclass
class SignClaim:
    """One location assertion parsed from a sign's OCR text.

    Fields
    ------
    kind:
        A ``LocationType`` value (``"county_boundary"``, ``"state_boundary"``,
        ``"town_entry"``, ``"route_marker"``). Stored as the plain string so the
        dataclass stays JSON-native.
    value:
        The parsed place/route name, title-cased for a place ("Example",
        "South Carolina", "Clifden") or normalised hyphenated form for a route
        ("I-95", "US-17"). County claims carry the BARE county name — no
        "County" filler — so they compare directly against
        ``ResolvedLocation.regions`` / ``vote`` members.
    raw_text:
        The OCR text the claim was extracted from, kept for operator review of
        a flagged conflict (a noisy sign is judged by its raw text, not its
        parse).
    """

    kind: str
    value: str
    raw_text: str = ""


@dataclass
class SignHit:
    """One sign observation: OCR'd text plus where and when it was seen.

    Fields
    ------
    episode_index:
        GLOBAL episode index (an index into ``state.episode_files`` /
        ``state.gps_per_episode``) — the same index space as a plan segment's
        ``source_file_index`` BEFORE ``_partition_lf_plans_by_date`` remaps it
        to day-local. Consumers that bucket by segment span must therefore run
        against an unpartitioned plan (which is what ``--label-dry-run`` has).
    t_local:
        Seconds from the start of that episode — comparable to a segment's
        ``start_time`` / ``end_time``.
    ts_day:
        Seconds from the start of the day master (the clock ``day_scores`` rows
        and the stop mask use), derived at record time from the day's episode
        ordering.
    date:
        ``YYYY-MM-DD`` of the day the hit belongs to.
    raw_text / ocr_conf:
        Combined OCR text for the sign ROI and its mean per-token confidence.
    lat / lon:
        The GPS fix nearest the hit, when the episode had a trace — the
        coordinate the sign was READ at (not a claim about where the sign
        points). ``None`` on a GPS-less episode.
    claims:
        Parsed ``SignClaim``s, stored as plain dicts so the hit stays
        JSON-native; use :meth:`parsed_claims` for typed access.
    """

    episode_index: int
    t_local: float
    ts_day: float
    date: str
    raw_text: str
    ocr_conf: float = 0.0
    lat: Optional[float] = None
    lon: Optional[float] = None
    claims: list = field(default_factory=list)

    def parsed_claims(self) -> list[SignClaim]:
        """Typed view of ``claims`` (tolerates already-typed entries)."""
        out: list[SignClaim] = []
        for c in self.claims or []:
            if isinstance(c, SignClaim):
                out.append(c)
            elif isinstance(c, dict):
                out.append(
                    SignClaim(
                        kind=str(c.get("kind", "")),
                        value=str(c.get("value", "")),
                        raw_text=str(c.get("raw_text", "")),
                    )
                )
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Any) -> "SignHit":
        """Rehydrate from a JSON-loaded dict (pass-through for an instance)."""
        if isinstance(d, cls):
            return d
        d = dict(d or {})
        claims = [c if isinstance(c, dict) else asdict(c) for c in (d.get("claims") or [])]
        return cls(
            episode_index=int(d.get("episode_index", -1)),
            t_local=float(d.get("t_local", 0.0)),
            ts_day=float(d.get("ts_day", 0.0)),
            date=str(d.get("date", "")),
            raw_text=str(d.get("raw_text", "")),
            ocr_conf=float(d.get("ocr_conf", 0.0)),
            lat=None if d.get("lat") is None else float(d["lat"]),
            lon=None if d.get("lon") is None else float(d["lon"]),
            claims=claims,
        )


@dataclass
class SignVerdict:
    """Outcome of comparing one LF's sign hits against its resolved location.

    ``status`` is one of ``STATUS_CORROBORATES`` / ``STATUS_NO_SIGNAL`` /
    ``STATUS_CONFLICTS``; ``reasons`` carries operator-readable lines (each
    naming the sign text behind it) for the dry-run print and the strict-gate
    error message.
    """

    lf_index: int
    status: str = STATUS_NO_SIGNAL
    confidence: float = 0.0
    reasons: list = field(default_factory=list)
    hit_count: int = 0
    corroborating_counties: list = field(default_factory=list)
    conflicting_counties: list = field(default_factory=list)

    @property
    def conflicts(self) -> bool:
        return self.status == STATUS_CONFLICTS

    def to_dict(self) -> dict:
        return asdict(self)


def hits_by_date_to_dicts(hits_by_date: dict) -> dict:
    """``{date: [SignHit,...]}`` -> ``{date: [dict,...]}`` for state persistence."""
    return {
        str(date): [h.to_dict() if isinstance(h, SignHit) else dict(h) for h in (hits or [])]
        for date, hits in (hits_by_date or {}).items()
    }


def hits_by_date_from_state(raw: Any) -> dict:
    """``state.sign_hits_by_date`` (JSON dicts) -> ``{date: [SignHit,...]}``.

    Tolerates ``None``, already-typed hits, and a non-dict (returns ``{}``) —
    a consumer must never see a raw dict
    where it expects attribute access.
    """
    if not isinstance(raw, dict):
        return {}
    return {str(date): [SignHit.from_dict(h) for h in (hits or [])] for date, hits in raw.items()}
