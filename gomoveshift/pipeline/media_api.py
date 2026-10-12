"""Typed API around the day render: ``RunSpec`` in, ``Deliverables`` out.

Phase D (publishing) will consume a ``Deliverables`` object instead of
reading day state. ``RunSpec`` mirrors ``run_day_pipeline``'s parameters
one-for-one (a test pins the two together); ``render_day`` runs the render and
lifts the ``DayArtifacts`` it returns into a ``Deliverables``.
"""

from __future__ import annotations

import re
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Optional

from .day_pipeline import DayArtifacts


@dataclass(kw_only=True)
class RunSpec:
    """Every input ``run_day_pipeline`` takes, with the same defaults."""

    day_batch: Any
    day_scores: Any
    output_dir: Path
    location: str
    config: Any = None
    lf_plans: list
    per_lf_locations: Optional[dict] = None
    per_lf_ost_text: Optional[dict] = None
    gps_points_by_date: Optional[dict] = None
    short_ost_text_fn: Optional[Callable] = None
    short_regions_fn: Optional[Callable] = None
    state: Any = None


@dataclass
class Measurements:
    """Per-output measurements the engine knows at render time.

    Keyed by output path string. The shape is the contract: nothing sets
    ``sharpen_amount`` or ``upscale_provenance`` yet, and the day pipeline
    holds no per-output source-picture score, so all three are empty today.
    """

    sharpen_amount: dict[str, float] = field(default_factory=dict)
    source_picture_scores: dict[str, float] = field(default_factory=dict)
    upscale_provenance: dict[str, str] = field(default_factory=dict)


@dataclass
class Deliverables(DayArtifacts):
    """``DayArtifacts`` plus the sidecars written and the measurements taken."""

    sidecars: dict[str, Path] = field(default_factory=dict)
    measurements: Measurements = field(default_factory=Measurements)


_SIDECAR_STEMS = ("ost_audit", "shorts_ost_audit")


def run_day_pipeline(**kwargs: Any) -> Optional[DayArtifacts]:
    """Late-bound delegate so the render entry point stays patchable once."""
    from . import day_pipeline

    return day_pipeline.run_day_pipeline(**kwargs)


def _locate_sidecars(output_dir: Path, date: str) -> dict[str, Path]:
    out_dir = Path(output_dir)
    found: dict[str, Path] = {}
    for stem in _SIDECAR_STEMS:
        name = f"{stem}_{date}.json"
        for base in (out_dir / f"day_{date}", out_dir):
            if (base / name).is_file():
                found[stem] = base / name
                break
    return found


def render_day(spec: RunSpec) -> Optional[Deliverables]:
    """Render one day; ``None`` (a soft failure) stays ``None``."""
    artifacts = run_day_pipeline(
        day_batch=spec.day_batch,
        day_scores=spec.day_scores,
        output_dir=spec.output_dir,
        location=spec.location,
        config=spec.config,
        lf_plans=spec.lf_plans,
        per_lf_locations=spec.per_lf_locations,
        per_lf_ost_text=spec.per_lf_ost_text,
        gps_points_by_date=spec.gps_points_by_date,
        short_ost_text_fn=spec.short_ost_text_fn,
        short_regions_fn=spec.short_regions_fn,
        state=spec.state,
    )
    if artifacts is None:
        return None
    return lift_artifacts(artifacts, spec.output_dir)


def _field_default(f: Any) -> Any:
    if f.default is not MISSING:
        return f.default
    return f.default_factory() if f.default_factory is not MISSING else None


def lift_artifacts(artifacts: DayArtifacts, output_dir: Path) -> Deliverables:
    """``artifacts`` as ``Deliverables`` (unchanged if it already is one)."""
    if isinstance(artifacts, Deliverables):
        return artifacts
    base = {f.name: getattr(artifacts, f.name, _field_default(f)) for f in fields(DayArtifacts)}
    return Deliverables(
        **base,
        sidecars=_locate_sidecars(output_dir, artifacts.date),
        measurements=Measurements(),
    )


def _to_str(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def deliverables_to_state(d: Deliverables) -> dict:
    """JSON-safe record of ``d`` (paths as strings) for ``per_day_completed``."""
    return {
        "date": d.date,
        "master_path": _to_str(d.master_path),
        "regular_lf_paths": [str(p) for p in d.regular_lf_paths],
        "regular_lf_indices": list(d.regular_lf_indices),
        "short_paths": [str(p) for p in d.short_paths],
        "short_ranges": [
            asdict(r) if hasattr(r, "__dataclass_fields__") else r for r in d.short_ranges
        ],
        "master_record": d.master_record,
        "planned_shorts_count": d.planned_shorts_count,
        "shorts_target_in_force": d.shorts_target_in_force,
        "sidecars": {k: str(v) for k, v in d.sidecars.items()},
        "measurements": asdict(d.measurements),
    }


def deliverables_from_state(record: dict) -> Deliverables:
    """Inverse of :func:`deliverables_to_state`; paths come back as ``Path``."""
    from .short_planner import ShortRange

    master = record.get("master_path")
    return Deliverables(
        date=record["date"],
        master_path=Path(master) if master else None,
        regular_lf_paths=[Path(p) for p in record.get("regular_lf_paths") or []],
        regular_lf_indices=list(record.get("regular_lf_indices") or []),
        short_paths=[Path(p) for p in record.get("short_paths") or []],
        short_ranges=[
            ShortRange(**r) if isinstance(r, dict) else r for r in record.get("short_ranges") or []
        ],
        master_record=record.get("master_record"),
        planned_shorts_count=record.get("planned_shorts_count"),
        shorts_target_in_force=record.get("shorts_target_in_force"),
        sidecars={k: Path(v) for k, v in (record.get("sidecars") or {}).items()},
        measurements=Measurements(**(record.get("measurements") or {})),
    )


def _legacy_deliverables(lf_paths: list) -> list[Deliverables]:
    """State written before per-day records: group LF paths by ``day_<DATE>``."""
    by_date: dict[str, list] = {}
    for raw in lf_paths:
        path = Path(raw)
        day = path.parent.parent.name
        by_date.setdefault(day[len("day_") :] if day.startswith("day_") else "", []).append(path)
    out = []
    for date, paths in by_date.items():
        indices = []
        for pos, path in enumerate(paths):
            m = re.match(r"lf_(\d+)$", path.stem)
            indices.append(int(m.group(1)) if m else pos)
        out.append(
            Deliverables(
                date=date, master_path=None, regular_lf_paths=paths, regular_lf_indices=indices
            )
        )
    return out


def _artifact_date(item: Any) -> Optional[str]:
    return item.get("date") if isinstance(item, dict) else getattr(item, "date", None)


def deliverables_for_publish(state: Any) -> list[Deliverables]:
    """Per-day ``Deliverables`` Phase B/C persisted, in Phase B/C artifact order.

    The order is ``state.day_artifacts`` order (resume-skipped days first, then
    fresh days), NOT date order: ``phase_d_gates`` (``zip(drive_plan.regular_lfs,
    regular_lf_paths)``) and ``phase_d_metadata`` (two such zips) pair plans to
    files positionally, so a ``--days`` subset or a re-run of one middle day
    would otherwise pair an LF file with the wrong plan. Records whose date is
    not in ``day_artifacts`` follow, in date order; with no ``day_artifacts`` the
    whole list is in date order. Falls back to grouping ``state.regular_lf_paths``
    by day for state files that predate the per-day records.
    """
    per_day = getattr(state, "per_day_completed", None)
    records = {
        date: entry["deliverables"]
        for date, entry in (per_day.items() if isinstance(per_day, dict) else [])
        if isinstance(entry, dict) and isinstance(entry.get("deliverables"), dict)
    }
    if not records:
        return _legacy_deliverables(getattr(state, "regular_lf_paths", None) or [])
    position: dict[str, int] = {}
    for item in getattr(state, "day_artifacts", None) or []:
        position.setdefault(_artifact_date(item), len(position))
    order = sorted(records, key=lambda d: (position.get(d, len(position)), d))
    return [deliverables_from_state(records[d]) for d in order]


def publish_lf_paths(state: Any) -> list[str]:
    """Flat LF paths for Phase D, sorted by carried LF index (stable).

    Matches the ``skeleton_lfs`` / ``regular_lf_paths`` index sort Phase B/C
    applies, so ``zip(drive_plan.regular_lfs, ...)`` pairs stay aligned.
    """
    pairs = [
        (idx, str(path))
        for d in deliverables_for_publish(state)
        for idx, path in zip(d.regular_lf_indices, d.regular_lf_paths)
    ]
    return [path for _, path in sorted(pairs, key=lambda p: p[0])]
