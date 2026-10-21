"""The system review that may stand in for a person at the OST burn
gates, opt-in per run.

Phase BC's route-claim check (``route_claims.enforce_route_claims``) measures
every LF's burn text against its retained GPS. :func:`record_system_reviews`
keeps what it decided, per day, in ``state.ost_system_review[date]``: the
exact text checked, each LF's verdict (``agree`` or ``refused``) and the
refusal reason. The record is sealed (``digest``) and carries its writer and
version.

:func:`system_review_passes` is the fourth pass signal at both OST gates
(``day_pipeline.run_day_pipeline`` and ``ost_audit_sidecar.lf_gate_or_raise``).
It passes only when ``--ost-system-review`` was given on THIS run and the
day's record is sealed by this writer, covers exactly the gate's LFs, names no
refused claim, agrees on every LF and hashes the exact current burn text. It
is read from state only: no file on disk satisfies it, and it writes no
``.confirmed`` marker. The gates still check staleness first, so a label
changed after a pass reopens the gate as before.

The record holds labels, verdicts and reasons -- never a fix or coordinate.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

WRITER = "gms-route-claims"
RECORD_VERSION = "ost-system-review-1"
AGREE = "agree"
REFUSED = "refused"
#: Per-invocation opt-in (never persisted; see ``drive_state_model``).
OPT_IN_FIELD = "ost_system_review_this_run"


def _field(state, key: str):
    if state is None:
        return None
    if isinstance(state, dict):
        return state.get(key)
    return getattr(state, key, None)


def ost_text_hash(date: str, text_by_lf: dict) -> str:
    """Hash of one day's burn text: the date and each LF's exact text."""
    rows = sorted([int(idx), str(text)] for idx, text in text_by_lf.items())
    body = json.dumps({"date": date, "lfs": rows}, sort_keys=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _digest(record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "digest"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def seal(record: dict) -> dict:
    """``record`` with its ``digest`` set over every other field."""
    return {**record, "digest": _digest(record)}


def record_system_reviews(state, ost_text_by_date: dict, outcomes: dict) -> None:
    """Write each day's record from the route-claim ``outcomes``
    (``{lf_index: route_claim_outcome}``) and the burn text in
    ``ost_text_by_date``. An LF with no outcome (no retained GPS to measure)
    gets no review, so its day can only pass through a person."""
    records = dict(_field(state, "ost_system_review") or {})
    for date, day in (ost_text_by_date or {}).items():
        reviews = {}
        for idx, text in day.items():
            outcome = outcomes.get(idx)
            if outcome is None:
                continue
            refused = bool(outcome.get("refused"))
            reviews[str(int(idx))] = {
                "ost": str(text),
                "verdict": REFUSED if refused else AGREE,
                "refused": refused,
                "label_requested": outcome.get("label_requested"),
                "reason": outcome.get("reason") or "",
            }
        if not reviews:
            records.pop(date, None)
            continue
        texts = {int(i): r["ost"] for i, r in reviews.items()}
        records[date] = seal(
            {
                "writer": WRITER,
                "record_version": RECORD_VERSION,
                "date": date,
                "lf_indices": sorted(texts),
                "reviews": reviews,
                "ost_hash": ost_text_hash(date, texts),
            }
        )
    if isinstance(state, dict):
        state["ost_system_review"] = records
    else:
        state.ost_system_review = records


def opted_in(state) -> bool:
    return bool(_field(state, OPT_IN_FIELD))


def _refusal(record, date: str, lf_indices) -> Optional[str]:
    """Why ``record`` does not pass for this gate, or ``None``."""
    if not isinstance(record, dict):
        return "no system review record for this day"
    if record.get("writer") != WRITER or record.get("record_version") != RECORD_VERSION:
        return "record has no system provenance"
    if record.get("digest") != _digest(record):
        return "record seal does not match its contents"
    if record.get("date") != date:
        return f"record is for {record.get('date')!r}, not {date}"
    if record.get("lf_indices") != sorted(int(i) for i in lf_indices):
        return f"record covers LFs {record.get('lf_indices')}, gate has {sorted(lf_indices)}"
    reviews = record.get("reviews")
    if not isinstance(reviews, dict):
        return "record reviews are malformed"
    refused = [
        f"LF {i} {r.get('label_requested') or r.get('ost')!r} refused: {r.get('reason')}"
        for i, r in sorted(reviews.items())
        if isinstance(r, dict) and r.get("refused")
    ]
    if refused:
        return "route claim refused -- " + "; ".join(refused)
    for i, r in sorted(reviews.items()):
        if not isinstance(r, dict) or r.get("verdict") != AGREE:
            return f"LF {i} verdict is {(r or {}).get('verdict') if isinstance(r, dict) else r!r}"
    return None


def system_review_passes(state, date: str, text_by_lf: dict) -> tuple[bool, str]:
    """The fourth OST-gate signal: ``(passes, reason)``. ``text_by_lf`` is the
    gate's own resolved burn text per LF (the audit payload's ``ost_text``)."""
    if not opted_in(state):
        return False, ""
    record = (_field(state, "ost_system_review") or {}).get(date)
    why = _refusal(record, date, text_by_lf)
    if why is None and record["ost_hash"] != ost_text_hash(date, text_by_lf):
        why = "burn text differs from the text the system reviewed"
    if why:
        return False, f"system review did not pass: {why}"
    return True, f"system review record {record['ost_hash']} ({WRITER})"


# ── The same signal at the Shorts and superlong burn gates ────────────
#: Each surface keeps its own sealed record in its own state field, under its
#: own record version, so one surface's record never passes another's gate.
SURFACE_FIELDS = {
    "shorts": "shorts_ost_system_review",
    "superlong": "superlong_ost_system_review",
}
SURFACE_VERSIONS = {
    "shorts": "ost-system-review-shorts-1",
    "superlong": "ost-system-review-superlong-1",
}
#: A Short's text with this source was inherited from its parent LF or the
#: drive, not derived from the clip's own retained GPS (``day_pipeline``).
INHERITED_SOURCE = "lf_label"
_ENTRY_NAMES = {"shorts": "short", "superlong": "part"}


def _entry_id(surface: str, entry: dict) -> int:
    return int(entry["rank"] if surface == "shorts" else entry["part_index"])


def _identity(surface: str, entry: dict) -> dict:
    """What the surface burns for one entry: its identity and exact text."""
    if surface == "shorts":
        return {
            "rank": int(entry["rank"]),
            "start_sec": float(entry["start_sec"]),
            "end_sec": float(entry["end_sec"]),
            "ost_text": str(entry["ost_text"]),
        }
    return {
        "part_index": int(entry["part_index"]),
        "part_count": int(entry["part_count"]),
        "ost_text": str(entry["ost_text"]),
        "lower_thirds": [
            [str(lt["label"]), float(lt["time_sec"])] for lt in (entry.get("lower_thirds") or [])
        ],
    }


def surface_text_hash(surface: str, date: str, entries) -> str:
    """Hash of one surface's burn for a day: the surface, the date and each
    entry's identity and exact burned text (superlong: its lower-thirds too)."""
    rows = sorted((_identity(surface, e) for e in entries), key=lambda r: _entry_id(surface, r))
    body = json.dumps({"surface": surface, "date": date, "entries": rows}, sort_keys=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def record_surface_reviews(state, surface: str, date: str, entries, outcomes: dict) -> None:
    """Write ``surface``'s record for ``date`` from the route-claim ``outcomes``
    (``{entry_id: {"refused", "reason", "label_requested", "ost_text"}}``)
    measured over each entry's own retained GPS. An entry with no outcome, or
    whose text is not the text that was measured, gets no review, so its day
    can only pass through a person."""
    field = SURFACE_FIELDS[surface]
    records = dict(_field(state, field) or {})
    reviews, reviewed = {}, []
    for entry in entries or []:
        outcome = outcomes.get(_entry_id(surface, entry))
        if outcome is None or outcome.get("ost_text", entry["ost_text"]) != entry["ost_text"]:
            continue
        refused = bool(outcome.get("refused"))
        reviewed.append(entry)
        reviews[str(_entry_id(surface, entry))] = {
            **_identity(surface, entry),
            "place_source": entry.get("place_source"),
            "verdict": REFUSED if refused else AGREE,
            "refused": refused,
            "label_requested": outcome.get("label_requested"),
            "reason": outcome.get("reason") or "",
        }
    if reviews:
        records[date] = seal(
            {
                "writer": WRITER,
                "record_version": SURFACE_VERSIONS[surface],
                "surface": surface,
                "date": date,
                "entry_ids": sorted(_entry_id(surface, e) for e in reviewed),
                "reviews": reviews,
                "text_hash": surface_text_hash(surface, date, reviewed),
            }
        )
    else:
        records.pop(date, None)
    if isinstance(state, dict):
        state[field] = records
    else:
        setattr(state, field, records)


def _surface_refusal(record, surface: str, date: str, entries) -> Optional[str]:
    """Why ``record`` does not pass ``surface``'s gate for ``entries``, or ``None``."""
    name = _ENTRY_NAMES[surface]
    if not isinstance(record, dict):
        return f"no {surface} system review record for this day"
    if (
        record.get("writer") != WRITER
        or record.get("record_version") != SURFACE_VERSIONS[surface]
        or record.get("surface") != surface
    ):
        return f"record has no {surface} system provenance"
    if record.get("digest") != _digest(record):
        return "record seal does not match its contents"
    if record.get("date") != date:
        return f"record is for {record.get('date')!r}, not {date}"
    inherited = [
        f"{name} {_entry_id(surface, e)} {e.get('ost_text')!r} text is inherited "
        f"({INHERITED_SOURCE}), not measured on its own clip"
        for e in sorted(entries, key=lambda e: _entry_id(surface, e))
        if e.get("place_source") == INHERITED_SOURCE
    ]
    if inherited:
        return "; ".join(inherited)
    gate_ids = sorted(_entry_id(surface, e) for e in entries)
    if record.get("entry_ids") != gate_ids:
        return f"record covers {name}s {record.get('entry_ids')}, gate has {gate_ids}"
    reviews = record.get("reviews")
    if not isinstance(reviews, dict):
        return "record reviews are malformed"
    refused = [
        f"{name} {i} {r.get('label_requested') or r.get('ost_text')!r} refused: {r.get('reason')}"
        for i, r in sorted(reviews.items())
        if isinstance(r, dict) and r.get("refused")
    ]
    if refused:
        return "route claim refused -- " + "; ".join(refused)
    for i, r in sorted(reviews.items()):
        if not isinstance(r, dict) or r.get("verdict") != AGREE:
            return (
                f"{name} {i} verdict is {(r or {}).get('verdict') if isinstance(r, dict) else r!r}"
            )
    return None


def surface_review_passes(state, surface: str, date: str, entries) -> tuple[bool, str]:
    """The fourth signal at the Shorts / superlong gate: ``(passes, reason)``.
    ``entries`` are the gate's own entries (what it is about to burn)."""
    if not opted_in(state):
        return False, ""
    entries = list(entries or [])
    record = (_field(state, SURFACE_FIELDS[surface]) or {}).get(date)
    why = _surface_refusal(record, surface, date, entries)
    if why is None and record["text_hash"] != surface_text_hash(surface, date, entries):
        why = "burn text differs from the text the system reviewed"
    if why:
        return False, f"system review did not pass: {why}"
    return True, f"system review record {record['text_hash']} ({WRITER})"
