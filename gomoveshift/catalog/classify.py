"""Title-accuracy catalog classification (U7 phase 5): GPS evidence + A-E class.

For each inventory row this builds the video's own segment GPS location-evidence
record, then classifies its LIVE title against that record. Read-only: no YouTube
write exists in this module.

Classes (plan 2026-10-14, phase 5):

* ``A`` the live title is positively refuted: its endpoints run against the
  travel order, or a named place is located (forward geocode) farther than
  ``NEAR_KM`` from every GPS fix. A place merely absent from the record is not
  refutation -- the reverse geocode names a town for few fixes -- so an
  unlocatable or near-the-trace place is ``E``.
* ``B`` the live title names an endpoint the trace only passed ``through``: a
  claim more specific than the record supports.
* ``D`` every place claim passes the same check ``retitle_apply`` applies.
* ``E`` unresolved; the reason names the missing evidence.
* ``C`` (performance alternatives) is never emitted; it is out of scope here.

A Short is judged against its own clip, not an LF trace: its entry in the
drive's ``shorts_ost_audit_<date>.json`` (the place its own GPS fixes voted and
their regions), joined by the global short index in ``pipeline_state.json``.
A title naming that place or a region is ``D``; a title naming none of them is
``E`` with ``drift`` set, listed in ``worklist-shorts-drift.csv`` for review.
It is never ``A``: the audit keeps place names, not fixes, so nothing here can
refute the title's place, and no drift row reaches ``retitle_apply``.

Places come ONLY from the record, never from a title or YAML field. A/B rows
carry a suggested title from ``retitle_worklist.compose_suggested_title`` that
also passes ``retitle_apply``'s gates; with none, the row is ``E``.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional

CLASS_COLUMNS: List[str] = [
    "video_id",
    "kind",
    "project",
    "class",
    "reason",
    "live_title",
    "suggested_title",
]
#: Shorts whose title names none of their clip's places, for human review.
SHORTS_DRIFT_COLUMNS: List[str] = [
    "video_id",
    "project",
    "short_index",
    "live_title",
    "burned_place",
    "clip_regions",
    "reason",
]

PointsFor = Callable[[str, Any], Optional[list]]
#: ``short_for(project, short_index)``: the Short's clip audit entry, or ``None``.
ShortFor = Callable[[str, Any], Optional[Dict[str, Any]]]
_STATE_CODE_RE = re.compile(r"\s*,\s*([A-Z]{2})\b")

#: Forward geocoder: a place query ("Dunmore, VA") to ``(lat, lon)`` or ``None``.
Forward = Callable[[str], Optional[tuple]]

_SERVICE_ROOT = Path(__file__).resolve().parents[2]


def _tools():
    """``tools.retitle_apply`` / ``tools.retitle_worklist`` (service-root modules)."""
    if str(_SERVICE_ROOT) not in sys.path:
        sys.path.insert(0, str(_SERVICE_ROOT))
    from tools import retitle_apply, retitle_worklist

    return retitle_apply, retitle_worklist


#: Share of the video's length the GPS trace must span before its places can
#: judge the title: a partial trace cannot show that a named endpoint is missing.
TRACE_COVERAGE_MIN = 0.8

#: Speed bound for what the trace's unobserved time could cover: a place within
#: ``NEAR_KM`` plus that reach of the trace is never refuted (class ``E``).
REACH_KMH = 120.0


def _coverage_issue(points: list, duration: Any) -> Optional[str]:
    """Why the trace is too short to judge the title, or ``None`` when it covers it."""
    from gomoveshift.pipeline.gps_quality import usable_fix_times

    try:
        seconds = float(str(duration).strip())
    except (TypeError, ValueError):
        seconds = 0.0
    if seconds <= 0:
        return "no video duration to check GPS coverage against"
    times = usable_fix_times(points)
    span = times[-1] - times[0] if len(times) >= 2 else 0.0
    if span < TRACE_COVERAGE_MIN * seconds:
        return f"GPS covers {span:.0f} s of a {seconds:.0f} s video"
    return None


def _lf_int(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _project_points(stages_dir: Path) -> Dict[int, list]:
    """``{lf_index: [trackpoints]}`` for one project from its persisted state.

    The same chain a resumed drive uses: ``pipeline_state.json`` -> the GPX paths
    kept in ``state.gpx_files`` -> ``rebuild_gps_per_episode`` -> the per-LF slice
    ``_build_gps_points_by_lf`` cuts from the drive plan."""
    from gomoveshift.pipeline.drive_state_model import PipelineState
    from gomoveshift.pipeline.gps_enrichment import (
        _build_gps_points_by_lf,
        rebuild_gps_per_episode,
    )
    from gomoveshift.pipeline.state_manager import load_state

    raw = load_state(Path(stages_dir) / "pipeline_state.json")
    if not raw:
        return {}
    state = PipelineState.from_dict(raw)
    if state.drive_plan is None or not state.gpx_files:
        return {}
    try:
        return _build_gps_points_by_lf(state.drive_plan, rebuild_gps_per_episode(state.gpx_files))
    except Exception:  # noqa: BLE001 — an unreadable trace is "no GPS", classed E
        return {}


def load_lf_points(stages_dir_for: Callable[[str], Path]) -> PointsFor:
    """``points_for(project, lf_index)`` over each project's persisted state
    (loaded once per project). ``None`` when the project, LF or GPS is absent."""
    cache: Dict[str, Dict[int, list]] = {}

    def points_for(project: str, lf_index: Any) -> Optional[list]:
        if project not in cache:
            cache[project] = _project_points(stages_dir_for(project))
        idx = _lf_int(lf_index)
        return (cache[project].get(idx) if idx is not None else None) or None

    return points_for


def _day_relative(path: str) -> str:
    """``day_<date>/shorts/short_NN.mp4``: a short's path below its stages dir."""
    return "/".join(Path(path).parts[-3:])


def _project_shorts(stages_dir: Path) -> Dict[int, Dict[str, Any]]:
    """``{global_short_index: audit entry}`` for one project.

    ``state.short_global_index`` maps each rendered short's path to the stable
    global index the queue's ``short_index`` carries; each
    ``shorts_ost_audit_<date>.json`` entry names its short by that path. Paths
    are matched below the stages dir so a moved cache still joins."""
    from gomoveshift.pipeline.state_manager import load_state

    stages_dir = Path(stages_dir)
    raw = load_state(stages_dir / "pipeline_state.json") or {}
    by_rel = {
        _day_relative(path): idx
        for path, idx in (raw.get("short_global_index") or {}).items()
        if _lf_int(idx) is not None
    }
    out: Dict[int, Dict[str, Any]] = {}
    for audit in sorted(stages_dir.glob("shorts_ost_audit_*.json")):
        try:
            entries = json.loads(audit.read_text()).get("shorts") or []
        except (OSError, ValueError):
            continue
        for e in entries:
            idx = by_rel.get(_day_relative(str(e.get("path") or "")))
            if idx is not None:
                out[int(idx)] = e
    return out


def load_short_evidence(stages_dir_for: Callable[[str], Path]) -> ShortFor:
    """``short_for(project, short_index)`` over each project's shorts audit
    (loaded once per project). ``None`` when the project or short is absent."""
    cache: Dict[str, Dict[int, Dict[str, Any]]] = {}

    def short_for(project: str, short_index: Any) -> Optional[Dict[str, Any]]:
        if project not in cache:
            cache[project] = _project_shorts(stages_dir_for(project))
        idx = _lf_int(short_index)
        return cache[project].get(idx) if idx is not None else None

    return short_for


def _squash(text: str) -> str:
    """Lower-case letters and digits only, so ``#SevenMileFord`` meets
    ``Seven Mile Ford``."""
    return re.sub(r"[^0-9a-z]", "", text.lower())


def _classify_short(row: Dict[str, Any], short_for: ShortFor) -> Dict[str, Any]:
    result = _judge_short(row, short_for)
    result["short_index"] = row.get("short_index", "")
    return result


def _judge_short(row: Dict[str, Any], short_for: ShortFor) -> Dict[str, Any]:
    project, idx = (row.get("project") or "").strip(), _lf_int(row.get("short_index"))
    if not project:
        return _result(row, "E", "no project joined to this short (no segment evidence)")
    if idx is None:
        return _result(row, "E", f"project {project} has no short_index for this short")
    entry = short_for(project, idx)
    if entry is None:
        return _result(row, "E", f"no shorts OST audit entry for {project} short {idx}")
    burned = str(entry.get("ost_text") or "").strip()
    regions = [str(r) for r in entry.get("regions") or [] if str(r).strip()]
    record = {
        "ost_text": burned,
        "regions": regions,
        "place_source": entry.get("place_source") or "lf_label",
        "start_sec": entry.get("start_sec"),
        "end_sec": entry.get("end_sec"),
    }
    if record["place_source"] != "clip_gps":
        return _result(
            row,
            "E",
            f"burned place {burned!r} was inherited from the parent LF, not voted "
            "by the clip's own GPS",
            record,
        )
    title = _squash(row.get("title") or "")
    places = [p.split(",")[0].strip() for p in [burned, *regions] if p.split(",")[0].strip()]
    named = [p for p in places if _squash(p) and _squash(p) in title]
    if named:
        return _result(row, "D", f"title names the clip's place: {', '.join(named)}", record)
    result = _result(
        row,
        "E",
        f"title names none of the clip's places (burned {burned!r}; clip regions: "
        f"{'; '.join(regions) or 'none'})",
        record,
    )
    result["drift"] = True
    return result


def _build_record(points: list, lf_index: int, geocoder: Any):
    """The segment's record via the live pipeline's own builder (it samples the
    trace the way Phase D's U5 does); ``None`` when no place was entered."""
    from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
        _build_location_evidence_per_lf,
    )
    from gomoveshift.pipeline.location_evidence import LocationEvidence

    built = _build_location_evidence_per_lf(
        [SimpleNamespace(index=lf_index)], {lf_index: points}, geocoder
    )
    record = built.get(lf_index)
    if not record or not LocationEvidence.from_dict(record).places:
        return None
    return record


def _through_endpoints(title: str, record) -> List[str]:
    from gomoveshift.pipeline.validation.content_gates import _TITLE_ENDPOINTS_RE

    m = _TITLE_ENDPOINTS_RE.match(title or "")
    if not m:
        return []
    out = []
    for role in ("start", "end"):
        name = m.group(role).strip()
        hit = record.place(name)
        if hit is not None and hit.scope == "through":
            out.append(f"{role} {name!r} was only passed through")
    return out


def _unobserved_reach_km(points: list, duration: Any) -> float:
    """How far the vehicle could have gone while no fix was recorded: the video
    time past the trace's span plus its largest interior gap, at ``REACH_KMH``.
    Fix times are wall-clock, so WHERE the missing stretch sits is unknown."""
    from gomoveshift.pipeline.gps_quality import usable_fix_times

    times = usable_fix_times(points)
    span = times[-1] - times[0] if len(times) >= 2 else 0.0
    gap = max((b - a for a, b in zip(times, times[1:])), default=0.0)
    unobserved = max(0.0, float(duration) - span) + gap
    return unobserved * REACH_KMH / 3600.0


def _located_region_issue(name: str, code: Optional[str], coord, rec, geocoder: Any):
    """Why the forward lookup's match is not evidence about the title's place
    (``None`` = it lies in the state the title names, or one the trace entered)."""
    from gomoveshift.pipeline.location_evidence import _state_matches

    try:
        state = (geocoder.reverse_geocode_detailed(coord[0], coord[1]) or {}).get("state")
    except Exception as exc:  # noqa: BLE001 — an unchecked region is not evidence
        return f"{name!r}: could not check the lookup's region: {exc}"
    wanted = [code] if code else [p.state for p in rec.places if p.state]
    if any(_state_matches(state, w) for w in wanted):
        return None
    return (
        f"{name!r}: the lookup's match is in {state or 'an unknown region'}, not "
        f"{code or 'a state the trace entered'}; the name is ambiguous"
    )


def _locate_missing_endpoints(
    title: str, rec, points: list, forward: Optional[Forward], geocoder: Any, duration: Any
):
    """``(unrefuted, refuted)`` for the endpoints missing from ``rec``: why each
    is NOT shown wrong, and the measured distance of each that lies beyond
    ``NEAR_KM`` plus the unobserved reach of every GPS fix (positively refuted)."""
    from gomoveshift.pipeline.gps_quality import nearest_trace_distance_km
    from gomoveshift.pipeline.location_evidence import NEAR_KM

    reach = _unobserved_reach_km(points, duration)
    from gomoveshift.pipeline.validation.content_gates import _TITLE_ENDPOINTS_RE

    m = _TITLE_ENDPOINTS_RE.match(title or "")
    # The state code the title puts after an endpoint ("Birch, VA"), never the
    # tail after it ("VA in 4K"); a start with none borrows the end's.
    codes = {
        role: (_STATE_CODE_RE.match(title, m.end(role)) or [None, None])[1]
        for role in ("start", "end")
    }
    why: List[str] = []
    far: List[str] = []
    for role in ("start", "end"):
        name = m.group(role).strip()
        if rec.place(name) is not None:
            continue
        try:
            code = codes[role] or codes["end"]
            coord = forward(f"{name}, {code}" if code else name) if forward else None
        except Exception as exc:  # noqa: BLE001 — one lookup failure leaves one row unresolved
            why.append(f"{role} {name!r}: forward geocode failed: {exc}")
            continue
        if not coord:
            why.append(f"{role}: cannot locate {name!r} to test it against the trace")
            continue
        region = _located_region_issue(name, code, coord, rec, geocoder)
        if region:
            why.append(f"{role} {region}")
            continue
        km = nearest_trace_distance_km(coord[0], coord[1], points, float("-inf"), float("inf"))
        if km is None or km <= NEAR_KM:
            why.append(
                f"{role} {name!r} lies within {NEAR_KM:.0f} km of the trace; the reverse "
                "geocode names no town there"
            )
        elif km <= NEAR_KM + reach:
            why.append(
                f"{role} {name!r} is {km:.0f} km from the trace, within reach of its "
                f"unobserved time ({reach:.0f} km at {REACH_KMH:.0f} km/h)"
            )
        else:
            far.append(f"{role} {name!r} is {km:.0f} km from the trace")
    return why, far


def _result(row: Dict[str, Any], cls: str, reason: str, record=None, suggested: str = ""):
    return {
        "video_id": row["video_id"],
        "kind": row.get("kind", ""),
        "project": row.get("project", ""),
        "class": cls,
        "reason": reason,
        "live_title": row.get("title", ""),
        "suggested_title": suggested,
        "record": record,
    }


def _classify(
    row: Dict[str, Any],
    points_for: PointsFor,
    geocoder: Any,
    forward: Optional[Forward],
    short_for: Optional[ShortFor] = None,
) -> Dict[str, Any]:
    from gomoveshift.pipeline.location_evidence import LocationEvidence, title_claim_issues
    from gomoveshift.pipeline.validation.content_gates import _TITLE_ENDPOINTS_RE

    project, lf = (row.get("project") or "").strip(), _lf_int(row.get("lf_index"))
    if row.get("kind") != "long":
        if short_for is None:
            return _result(row, "E", "short: its clip window is not resolved by this tool")
        return _classify_short(row, short_for)
    if not project:
        return _result(row, "E", "no project joined to this video (no segment evidence)")
    if lf is None:
        return _result(row, "E", f"project {project} has no lf_index for this video")
    points = points_for(project, lf)
    if not points:
        return _result(row, "E", f"no GPS points for {project} LF {lf}")
    short_trace = _coverage_issue(points, row.get("duration_s"))
    if short_trace:
        return _result(row, "E", f"{project} LF {lf}: {short_trace}")
    try:
        record = _build_record(points, lf, geocoder)
    except Exception as exc:  # noqa: BLE001 — one row's lookup failure must not end the batch
        return _result(row, "E", f"geocoder failed for {project} LF {lf}: {exc}")
    if record is None:
        return _result(row, "E", f"GPS for {project} LF {lf} resolved no geocoded place")

    title = row.get("title", "")
    rec = LocationEvidence.from_dict(record)
    if not _TITLE_ENDPOINTS_RE.match(title or ""):
        return _result(row, "E", "live title makes no 'A to B' claim the record can verify", record)
    issues = title_claim_issues(title, rec)
    if issues:
        unrefuted, far = _locate_missing_endpoints(
            title, rec, points, forward, geocoder, row.get("duration_s")
        )
        if unrefuted:
            return _result(
                row, "E", f"{'; '.join(unrefuted)} (record: {'; '.join(issues)})", record
            )
        cls, reason = "A", "; ".join(issues + far)
    else:
        through = _through_endpoints(title, rec)
        if not through:
            return _result(row, "D", "every place claim is supported by the GPS record", record)
        cls, reason = "B", "; ".join(through)

    ra, rw = _tools()
    suggested = rw.compose_suggested_title({"location_evidence": record})
    probe = {"suggested_title": suggested, "location_evidence": json.dumps(record)}
    if suggested == "needs-human" or ra.evaluate_row(probe) or ra.evidence_issues(probe):
        return _result(row, "E", f"no supportable replacement title ({cls}: {reason})", record)
    return _result(row, cls, reason, record, suggested)


def classify_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    points_for: PointsFor,
    geocoder: Any,
    forward: Optional[Forward] = None,
    short_for: Optional[ShortFor] = None,
) -> List[Dict[str, Any]]:
    """One classified dict per inventory row, ordered by ``video_id``.

    ``forward`` locates a title place the record lacks; without it such a
    title is ``E`` (cannot be refuted), never ``A``. ``short_for`` supplies a
    Short's clip audit entry; without it every Short stays ``E``."""
    ordered = sorted(rows, key=lambda r: r["video_id"])
    return [_classify(r, points_for, geocoder, forward, short_for) for r in ordered]


def write_outputs(results: List[Dict[str, Any]], out_dir: Path) -> Dict[str, int]:
    """``evidence.jsonl``, ``classification.csv``, ``worklist-AB.csv`` and
    ``worklist-shorts-drift.csv`` under ``out_dir``; returns ``{class: count}``."""
    _ra, rw = _tools()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = sorted(results, key=lambda r: r["video_id"])
    with (out / "evidence.jsonl").open("w") as fh:
        for r in results:
            if r["record"] is not None:
                fh.write(json.dumps({"video_id": r["video_id"], "record": r["record"]}) + "\n")
    with (out / "classification.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CLASS_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    work = [
        {
            "video_id": r["video_id"],
            "current_title": r["live_title"],
            "reason": f"{r['class']}: {r['reason']}",
            "suggested_title": r["suggested_title"],
            "platform": "youtube",
            "location_evidence": json.dumps(r["record"]),
        }
        for r in results
        if r["class"] in ("A", "B")
    ]
    rw.write_csv(work, out / "worklist-AB.csv")
    with (out / "worklist-shorts-drift.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SHORTS_DRIFT_COLUMNS)
        w.writeheader()
        for r in results:
            if r.get("drift"):
                rec = r["record"] or {}
                w.writerow(
                    {
                        "video_id": r["video_id"],
                        "project": r["project"],
                        "short_index": r.get("short_index", ""),
                        "live_title": r["live_title"],
                        "burned_place": rec.get("ost_text", ""),
                        "clip_regions": "; ".join(rec.get("regions") or []),
                        "reason": r["reason"],
                    }
                )
    counts: Dict[str, int] = {}
    for r in results:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    return dict(sorted(counts.items()))
