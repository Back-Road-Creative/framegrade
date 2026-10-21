"""Per-LF provenance sidecar (additive operator artifact).

Written NEXT TO ``route_stats.json`` by the same Phase BC call site, it records
— per long-form (LF) — WHERE each video's facts came from. Like
``route_stats_sidecar.py`` it is **best-effort and never gates**: everything is
read from state earlier stages populated; anything unavailable at write time is
``null`` with a reason in the LF's ``null_reasons`` map. Never raises, never
changes label/route logic.

Schema of ``lf_provenance.json`` (``PROVENANCE_SCHEMA_VERSION``); top level
``{schema_version, generated_at, code_sha, lfs: [<lf>...]}`` (``code_sha`` is
the git revision of the code that wrote the file, ``null`` with a top-level
``null_reasons.code_sha`` when no git metadata is reachable), each ``<lf>``::

    lf_index          int
    burned_ost_text   str|null     # canonical per_lf_ost_text — what was BURNED
    source            {gpx_path: <abs>|null,          # single-rep GPX
                       episode_paths: [<abs>...]|null} # every episode spanned
    footage_window    {start_sec: float|null, end_sec: float|null}  # in-video s
    fix_density_pct   float|null   # 0-100 locked-GPS share, from the pre-encode ladder award
    endpoints         {start: {lat,lon,label}|null, end: {lat,lon,label}|null}
    road_vote         {member: count, ...}|null  # ResolvedLocation.road_votes tally
    road_coverage_measured  bool   # always false: no retained-time road coverage is measured
    road_vote_basis   str          # "reverse_geocode_votes_untimed": how road_vote was counted
    awarded_tier      str|null     # from the pre-encode ladder award
    tier_evidence     dict|list|null             # from the pre-encode ladder award
    cut_from_master_fingerprint  str|null  # per_day_completed[date].master_fingerprint
    master_attestation_verified  bool|null # does it match the day's attestation? see below
    null_reasons      {<field>: <why null>, ...}

``cut_from_master_fingerprint`` is the identity — cheap ``size:mtime_ns``,
never a full hash of the 60 GB master — of the day master THIS LF was cut
from, read from ``per_day_completed[date]`` (populated once at encode time by
``day_master_encoder.build_master_record``). ``master_attestation_verified``
is always ``null`` out of ``build_lf_provenance_payload`` alone (it is pure —
state only, no filesystem access); ``write_lf_provenance_sidecar`` cross-checks
the claimed fingerprint against the ``master_fingerprint`` stamped into that
day's ``day_{date}_privacy_attestation.json`` and fills in ``true``/``false``.
Absent evidence (no recording date, no per-day record, no attestation on
disk, or an attestation written before this field existed) is its own state —
``null`` with a reason — and never reads as verified. This records the
mismatch; it decides no enforcement policy (that is a publication-time call).

``burned_ost_text`` (the single burn source, ``per_lf_ost_text``) and
``road_vote`` (``ResolvedLocation.road_votes``, threaded from ``build_lf_title``
— the only road-vote source) are read from persisted state, so the record can
prove what was actually burned. ``fix_density_pct`` / ``awarded_tier`` /
``tier_evidence`` come from ``per_lf_gps_coverage_pct`` / ``per_lf_awarded_tier``
/ ``per_lf_tier_evidence``, recorded by the pre-encode specificity-ladder
demotion pass; an LF that pass never evaluated (no GPS evidence, or an
authoritative operator override) is null-with-reason. ``fix_density_pct`` is
the share of the LF's duration with a locked fix, not retained-time coverage;
that is the route-claims record's ``coverage_pct`` (``route_claims.py``). The
state field keeps its old name so saved runs resume.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..health.store import code_sha
from .cache import load_cached_json

logger = logging.getLogger(__name__)

LF_PROVENANCE_FILENAME = "lf_provenance.json"
#: The shorts analogue, written beside it. A separate file rather than a
#: key inside ``lf_provenance.json`` because a short is not a property of a
#: long-form: a day's shorts outnumber its LFs several times over, and reading
#: "which master did THIS short come from" should not mean walking an LF list.
SHORTS_PROVENANCE_FILENAME = "shorts_provenance.json"
PROVENANCE_SCHEMA_VERSION = 1
ROAD_VOTE_BASIS = "reverse_geocode_votes_untimed"


def build_lf_provenance_payload(state) -> dict:
    """Aggregate per-LF provenance; missing inputs degrade to null + reason."""
    gpx_path: dict = state.per_lf_gpx_path or {}
    episode_paths: dict = state.per_lf_gpx_episode_paths or {}
    resolved: dict = state.per_lf_resolved_location or {}
    titles: dict = state.per_lf_titles or {}
    ost_text: dict = state.per_lf_ost_text or {}
    coverage: dict = state.per_lf_gps_coverage_pct or {}
    tiers: dict = state.per_lf_awarded_tier or {}
    tier_evidence: dict = state.per_lf_tier_evidence or {}
    recording_dates: dict = getattr(state, "per_lf_recording_date", None) or {}
    per_day_completed: dict = getattr(state, "per_day_completed", None) or {}

    indices: set[int] = set()
    for src in (gpx_path, episode_paths, resolved, titles, ost_text):
        for k in src or {}:
            try:
                indices.add(int(k))
            except (TypeError, ValueError):
                continue

    lfs: list[dict] = []
    for idx in sorted(indices):
        reasons: dict[str, str] = {}
        loc = _lookup(resolved, idx)
        road_vote = _build_road_vote(loc, reasons)
        coverage_pct = _coerce_float(coverage, idx)
        if coverage_pct is None:
            reasons["fix_density_pct"] = (
                "no locked-GPS coverage recorded; the pre-encode "
                "specificity-ladder pass did not evaluate this LF"
            )
        awarded_tier = _lookup(tiers, idx)
        if awarded_tier is None:
            reasons["awarded_tier"] = (
                "no tier recorded; the pre-encode specificity-ladder pass did not evaluate this LF"
            )
        lfs.append(
            {
                "lf_index": idx,
                "burned_ost_text": _build_burned_ost_text(ost_text, idx, reasons),
                "source": _build_source(gpx_path, episode_paths, idx, reasons),
                "footage_window": _build_window(episode_paths, idx, reasons),
                "fix_density_pct": coverage_pct,
                "endpoints": _build_endpoints(resolved, idx, reasons),
                "road_vote": road_vote,
                # The tally counts untimed reverse-geocode samples. It is
                # not retained-time coverage, so it backs no "via <road>" claim.
                "road_coverage_measured": False,
                "road_vote_basis": ROAD_VOTE_BASIS,
                "awarded_tier": str(awarded_tier) if awarded_tier is not None else None,
                "tier_evidence": _lookup(tier_evidence, idx),
                "cut_from_master_fingerprint": _build_cut_from_master_fingerprint(
                    recording_dates, per_day_completed, idx, reasons
                ),
                # Disk-based cross-check against the actual attestation is done
                # by write_lf_provenance_sidecar (this function is pure/state-
                # only) — default null-with-reason until that runs.
                "master_attestation_verified": None,
                "null_reasons": reasons,
            }
        )
        if "cut_from_master_fingerprint" not in reasons:
            reasons.setdefault(
                "master_attestation_verified",
                "not yet cross-checked against the day's attestation "
                "(build_lf_provenance_payload has no filesystem access)",
            )

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "lfs": lfs,
    }


def _stamp_code_sha(payload: dict) -> None:
    """Record which code revision wrote ``payload`` (null-with-reason if unknown)."""
    sha = code_sha(Path(__file__).resolve().parent)
    payload["code_sha"] = sha
    if sha is None:
        payload.setdefault("null_reasons", {})["code_sha"] = (
            "no git metadata reachable from the package"
        )


def write_lf_provenance_sidecar(output_base: Path, state) -> Optional[Path]:
    """Write ``<output_base>/lf_provenance.json``; best-effort, never raises.

    ``None`` when there is no per-LF data or the write fails (logged WARNING).

    Also cross-checks each LF's ``cut_from_master_fingerprint`` against
    the ``master_fingerprint`` actually stamped into that day's privacy
    attestation on disk — the one filesystem-aware step
    ``build_lf_provenance_payload`` can't do on its own. Records a
    match/mismatch/absent verdict; decides no enforcement policy.
    """
    payload = build_lf_provenance_payload(state)
    if not payload.get("lfs"):
        logger.debug("lf_provenance_sidecar: no per-LF state to serialise (skipping write)")
        return None
    output_base = Path(output_base)
    stages_base = Path(getattr(state, "stages_base", None) or (output_base / "_stages"))
    recording_dates: dict = getattr(state, "per_lf_recording_date", None) or {}
    for lf in payload["lfs"]:
        date = _lookup(recording_dates, lf["lf_index"])
        _attach_master_attestation_verdict(lf, stages_base, date)
    _stamp_code_sha(payload)
    try:
        output_base.mkdir(parents=True, exist_ok=True)
        path = output_base / LF_PROVENANCE_FILENAME
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        logger.info("  Wrote LF provenance sidecar: %s", path)
        return path
    except OSError as exc:
        logger.warning("lf_provenance_sidecar: write failed at %s: %s", output_base, exc)
        return None


def _short_day_and_rank(path_str, reasons: dict, fallback_date):
    """The day and rank a short's OWN path claims.

    Rendered shorts live at ``<stages>/day_<DATE>/shorts/short_<rank>.mp4`` --
    the same shape ``phase_d_scheduling._per_short_sidecar_labels`` reads rank
    from. The path is DIRECT evidence of which day's master this clip was cut
    from; the LF's ``per_lf_recording_date`` is an indirect lookup. Preferring
    the lookup is how a short ends up attributed to the wrong day's master, so
    the path wins and the LF date is only a named fallback.
    """
    try:
        path = Path(path_str)
        day_name = path.parent.parent.name
        rank = int(path.stem.split("_")[1])
    except (TypeError, ValueError, IndexError, AttributeError):
        day_name, rank = "", None
    if day_name.startswith("day_") and day_name[len("day_") :]:
        return day_name[len("day_") :], rank
    if fallback_date:
        reasons["day"] = (
            f"the short's path {str(path_str)!r} is not the expected "
            "<stages>/day_<DATE>/shorts/short_<rank>.mp4 shape; fell back to "
            "its long-form's recording date"
        )
        return fallback_date, rank
    reasons["day"] = (
        f"the short's path {str(path_str)!r} names no day and its long-form has "
        "no recording date; the master it was cut from cannot be identified"
    )
    return None, rank


def build_shorts_provenance_payload(state) -> dict:
    """Per-short cut-from-master identity, in the LF sidecar's own shape.

    Long-forms could prove which day master they came from and whether that
    master's privacy attestation agrees; shorts could not -- the surface with
    the largest audience per file, shipping to four platforms through a CSV,
    had no way to say. This closes that through the SAME
    ``attach_master_identity_to_record`` the thumbnail entries use, rather than
    a second comparison that would drift.

    **Records; never gates.** Whether a mismatch should hold a delivery is a
    product decision, and this makes none: every absent input is
    null-with-reason, never ``verified``.
    """
    shorts_per_lf: dict = getattr(state, "shorts_per_lf", None) or {}
    recording_dates: dict = getattr(state, "per_lf_recording_date", None) or {}
    per_day_completed: dict = getattr(state, "per_day_completed", None) or {}
    stages_base = Path(getattr(state, "stages_base", None) or ".")

    entries: list[dict] = []
    for raw_idx in sorted(shorts_per_lf, key=lambda k: int(k)):
        idx = int(raw_idx)
        for short in shorts_per_lf[raw_idx] or []:
            reasons: dict[str, str] = {}
            day, rank = _short_day_and_rank(short, reasons, _lookup(recording_dates, idx))
            record: dict = {
                "lf_index": idx,
                "rank": rank,
                "path": str(short),
                "day": day,
                "null_reasons": reasons,
            }
            attach_master_identity_to_record(
                record,
                date=day,
                per_day_completed=per_day_completed,
                stages_base=stages_base,
            )
            entries.append(record)

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "shorts": entries,
    }


def write_shorts_provenance_sidecar(output_base: Path, state) -> Optional[Path]:
    """Write ``<output_base>/shorts_provenance.json``; best-effort, never raises.

    ``None`` when the run rendered no shorts -- an empty file would be a claim
    about nothing -- or when the write fails (logged WARNING). By the time this
    runs the day master is encoded and the shorts exist, so a record that could
    abort the run would be a record deciding an outcome.
    """
    payload = build_shorts_provenance_payload(state)
    if not payload.get("shorts"):
        logger.debug("shorts_provenance_sidecar: no shorts to serialise (skipping write)")
        return None
    _stamp_code_sha(payload)
    output_base = Path(output_base)
    try:
        output_base.mkdir(parents=True, exist_ok=True)
        path = output_base / SHORTS_PROVENANCE_FILENAME
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        logger.info("  Wrote shorts provenance sidecar: %s", path)
        return path
    except OSError as exc:
        logger.warning("shorts_provenance_sidecar: write failed at %s: %s", output_base, exc)
        return None


def _build_burned_ost_text(ost_text: dict, idx: int, reasons: dict) -> Optional[str]:
    # The canonical single burn source (``per_lf_ost_text``) — so the record can
    # prove WHAT was actually burned for this LF, not just where facts came from.
    v = _lookup(ost_text, idx)
    if isinstance(v, str) and v.strip():
        return v
    reasons["burned_ost_text"] = "no per-LF burned OST text (per_lf_ost_text) recorded for this LF"
    return None


def attach_master_identity_to_record(
    record: dict, *, date, per_day_completed: Optional[dict], stages_base: Path
) -> None:
    """Populate ``cut_from_master_fingerprint`` / ``master_attestation_
    verified`` on any plain-dict record — shorts-manifest entries, thumbnail
    upload entries — reusing the SAME pure fingerprint lookup and disk-based
    attestation cross-check ``write_lf_provenance_sidecar`` runs for LFs,
    rather than a second, drifting copy of the comparison. Best-effort and
    never gates: mutates ``record`` in place, merging into any existing
    ``null_reasons`` map.
    """
    reasons = record.setdefault("null_reasons", {})
    recording_dates = {0: date} if date else {}
    fp = _build_cut_from_master_fingerprint(recording_dates, per_day_completed or {}, 0, reasons)
    record["cut_from_master_fingerprint"] = fp
    lf_like = {
        "cut_from_master_fingerprint": fp,
        "null_reasons": reasons,
        "master_attestation_verified": None,
    }
    _attach_master_attestation_verdict(lf_like, Path(stages_base), date)
    record["master_attestation_verified"] = lf_like["master_attestation_verified"]


def _attach_master_attestation_verdict(lf: dict, stages_base: Path, date) -> None:
    """Fill in ``lf["master_attestation_verified"]`` by reading the
    day's privacy attestation off disk — the one step ``build_lf_provenance_
    payload`` can't do (it is pure). Absent evidence at any step (no claimed
    fingerprint, no date, no attestation file, an attestation predating this
    field) stays ``null`` with a reason; never reads as verified."""
    from .day_master_encoder import PRIVACY_ATTESTATION_NAME

    reasons = lf["null_reasons"]
    claimed_fp = lf.get("cut_from_master_fingerprint")
    if not claimed_fp:
        return  # reason already recorded by _build_cut_from_master_fingerprint
    if not date:
        reasons["master_attestation_verified"] = (
            "no recording date recorded for this LF; cannot locate its day's attestation"
        )
        return
    attestation_path = stages_base / f"day_{date}" / PRIVACY_ATTESTATION_NAME.format(date=date)
    doc = load_cached_json(attestation_path)
    if doc is None:
        reasons["master_attestation_verified"] = (
            f"no privacy attestation found at {attestation_path} for this LF's day"
        )
        return
    attested_fp = doc.get("master_fingerprint")
    if not attested_fp:
        reasons["master_attestation_verified"] = (
            "the day's privacy attestation carries no master_fingerprint (written "
            "before this field existed, or before the master was promoted)"
        )
        return
    verified = attested_fp == claimed_fp
    lf["master_attestation_verified"] = verified
    if verified:
        reasons.pop("master_attestation_verified", None)
    else:
        reasons["master_attestation_verified"] = (
            f"claimed master fingerprint {claimed_fp!r} does not match the attested "
            f"master fingerprint {attested_fp!r} for day {date!r}"
        )


def _build_cut_from_master_fingerprint(
    recording_dates: dict, per_day_completed: dict, idx: int, reasons: dict
) -> Optional[str]:
    date = _lookup(recording_dates, idx)
    if not date:
        reasons["cut_from_master_fingerprint"] = (
            "no recording date recorded for this LF (per_lf_recording_date); "
            "cannot identify which day master it was cut from"
        )
        return None
    record = _lookup(per_day_completed, date) if isinstance(per_day_completed, dict) else None
    fp = (record or {}).get("master_fingerprint") if isinstance(record, dict) else None
    if not fp:
        reasons["cut_from_master_fingerprint"] = (
            f"no master_fingerprint recorded in per_day_completed for day {date!r}"
        )
        return None
    return str(fp)


def _build_road_vote(loc, reasons: dict) -> Optional[dict]:
    # The persisted ResolvedLocation tally (threaded from build_lf_title) is the
    # only road-vote source there has ever been a writer for.
    model_votes = _attr(loc, "road_votes") if loc is not None else None
    if isinstance(model_votes, dict) and model_votes:
        return dict(model_votes)
    reasons["road_vote"] = "no road vote recorded (ResolvedLocation.road_votes empty)"
    return None


def _build_source(gpx_path: dict, episode_paths: dict, idx: int, reasons: dict) -> dict:
    single = _lookup(gpx_path, idx)
    single = str(single) if single else None
    ep_map = _lookup(episode_paths, idx)
    eps = sorted(str(p) for p in ep_map) if isinstance(ep_map, dict) and ep_map else None
    if single is None and eps is None:
        reasons["source"] = "no per-LF GPX recorded for this LF"
    return {"gpx_path": single, "episode_paths": eps}


def _build_window(episode_paths: dict, idx: int, reasons: dict) -> dict:
    ep_map = _lookup(episode_paths, idx)
    starts: list[float] = []
    ends: list[float] = []
    if isinstance(ep_map, dict):
        for ranges in ep_map.values():
            for rng in ranges or ():
                try:
                    starts.append(float(rng[0]))
                    ends.append(float(rng[1]))
                except (TypeError, ValueError, IndexError):
                    continue
    if not starts:
        reasons["footage_window"] = "no per-episode footage ranges recorded for this LF"
        return {"start_sec": None, "end_sec": None}
    return {"start_sec": min(starts), "end_sec": max(ends)}


def _build_endpoints(resolved: dict, idx: int, reasons: dict) -> dict:
    loc = _lookup(resolved, idx)
    endpoints = (_attr(loc, "endpoints") or {}) if loc is not None else {}
    out = {
        "start": _point(endpoints.get("start"), _attr(loc, "start_label")),
        "end": _point(endpoints.get("end"), _attr(loc, "end_label")),
    }
    if out["start"] is None and out["end"] is None:
        reasons["endpoints"] = "no ResolvedLocation endpoints for this LF"
    return out


def _point(coord, label) -> Optional[dict]:
    try:
        lat, lon = float(coord[0]), float(coord[1])
    except (TypeError, ValueError, IndexError):
        return None
    return {"lat": lat, "lon": lon, "label": str(label) if label else None}


def _attr(obj, name):
    # Read ``name`` from a dataclass instance OR a plain dict (JSON resume).
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _lookup(mapping: dict, idx: int):
    # Int-or-str-keyed lookup (JSON round-trips int keys to str).
    for k in (idx, str(idx)):
        if k in mapping:
            return mapping[k]
    return None


def _coerce_float(mapping: dict, idx: int) -> Optional[float]:
    v = _lookup(mapping, idx)
    try:
        return round(float(v), 1) if v is not None else None
    except (TypeError, ValueError):
        return None
