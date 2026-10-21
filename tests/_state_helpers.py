"""Test fixture helper: construct ``PipelineState`` via the canonical path.

The pipeline runs on ``PipelineState`` (a dataclass) but historically tests
constructed state as a raw ``dict`` literal and relied on the dict-protocol
shim (``__getitem__`` / ``get`` / ``setdefault`` / ``update`` / ``__contains__``
/ ``keys`` / ``items`` / ``__iter__``) declared on ``PipelineState``. The shim is
being deleted. To unblock the shim deletion, every test fixture that
historically wrote ``state = {...}`` for a fixture passed to production code
must move to constructing a real ``PipelineState`` instance.

``make_state(**kwargs)`` is the single migration point. It accepts the same
field names the dataclass exposes and, for compound fields whose value is a
dataclass instance (``drive_plan``, ``ingest_plan``,
``drive_schedule``, ``upload_manifests``), it accepts EITHER a fully
constructed dataclass OR a plain dict — coercing the dict via the field's
``from_dict`` classmethod so callers don't have to import the compound type
to write a fixture.

The implementation deliberately does NOT route through
``PipelineState.from_dict``. That path round-trips JSON which is too
expensive for tests and loses type information on non-serializable fields
(``Path``, ``set``, numpy arrays). The thin dataclass-coercion pass here
keeps tests fast and explicit.

Example
-------

```
from _state_helpers import make_state
from gomoveshift.pipeline.uploader import UploadManifest

state = make_state(
    output_base=tmp_path,
    region="kerry",
    drive_plan={"regular_lfs": [...], "montage_peaks": [...], ...},
    upload_manifests={0: existing_manifest_instance},
)
```
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gomoveshift.models import DrivePlan
from gomoveshift.pipeline.drive_state_model import PipelineState
from gomoveshift.pipeline.ingest.gopro_parser import IngestPlan
from gomoveshift.pipeline.scheduler import DriveReleaseSchedule
from gomoveshift.pipeline.uploader import UploadManifest

# Singleton dataclass fields — value is a single dataclass instance.
# Mirror of ``drive_state_model._dataclass_fields_map`` but kept local
# to avoid importing a private helper.
_SINGLETON_DATACLASS_FIELDS: dict[str, type] = {
    "drive_plan": DrivePlan,
    "ingest_plan": IngestPlan,
    "drive_schedule": DriveReleaseSchedule,
}

# Dict-of-dataclass fields — value is ``dict[int, Dataclass]``.
# Mirror of ``drive_state_model._dict_value_dataclass_fields_map``.
_DICT_VALUE_DATACLASS_FIELDS: dict[str, type] = {
    "upload_manifests": UploadManifest,
}


def _coerce_singleton(value: Any, cls: type) -> Any:
    """Pass a dataclass instance through unchanged; coerce dict via ``from_dict``."""
    if value is None or isinstance(value, cls):
        return value
    if isinstance(value, dict):
        from_dict = getattr(cls, "from_dict", None)
        if callable(from_dict):
            return from_dict(value)
        return cls(**value)
    return value


def _coerce_dict_value(value: Any, cls: type) -> Any:
    """Walk an int-keyed dict, coercing each value to ``cls`` if dict-shaped."""
    if value is None or not isinstance(value, dict):
        return value
    out: dict[int, Any] = {}
    for k, v in value.items():
        try:
            int_k = int(k)
        except (ValueError, TypeError):
            int_k = k
        out[int_k] = _coerce_singleton(v, cls)
    return out


def make_state(**kwargs: Any) -> PipelineState:
    """Construct a ``PipelineState`` for test fixtures.

    All keyword arguments must match declared ``PipelineState`` field names.
    Compound dataclass fields accept EITHER an already-constructed instance
    OR a dict literal that gets coerced via the field's ``from_dict``.

    The resulting instance is identical to one produced by direct
    ``PipelineState(**kwargs)`` construction (same ``__post_init__`` runs),
    just with the dict-shape escape hatch for compound fields.
    """
    # ``make_state(ost_audit_confirmed=...)`` is the fixture analog of the
    # production ``--ost-audit-confirmed`` arg→state path, which sets the
    # non-persisted ``ost_audit_confirmed_this_run`` signal the OST gate reads
    # for a THIS-run bypass. A fixture that constructs confirmed state directly
    # (bypassing that entry point) is, by intent, a fresh-confirmed run — so
    # mirror the flag here unless the caller set it explicitly (a persisted-only
    # confirmation is expressed by passing ``ost_audit_confirmed_this_run=None``).
    if "ost_audit_confirmed" in kwargs and "ost_audit_confirmed_this_run" not in kwargs:
        kwargs["ost_audit_confirmed_this_run"] = kwargs["ost_audit_confirmed"]
    for field_name, cls in _SINGLETON_DATACLASS_FIELDS.items():
        if field_name in kwargs:
            kwargs[field_name] = _coerce_singleton(kwargs[field_name], cls)
    for field_name, cls in _DICT_VALUE_DATACLASS_FIELDS.items():
        if field_name in kwargs:
            kwargs[field_name] = _coerce_dict_value(kwargs[field_name], cls)
    return PipelineState(**kwargs)


def verified_shorts_per_lf(
    stages: Path,
    counts: dict[int, int],
    *,
    date: str = "2025-09-14",
    place: str = "Cork, Ireland",
    regions: tuple[str, ...] = ("County Cork",),
) -> dict[int, list[Path]]:
    """Rendered short paths whose day sidecar records a clip-GPS verified place.

    A publish run (``youtube_upload=True``) refuses every posted short whose
    place its own clip's GPS did not verify
    (``phase_d_scheduling.assert_all_places_verified``), so a publish-run
    fixture needs real ``<stages>/day_<date>/shorts/short_NN.mp4`` files backed
    by a ``shorts_ost_audit_<date>.json`` entry carrying ``ost_text``, clip-GPS
    ``regions`` and ``place_source="clip_gps"`` — not bare ``MagicMock`` shorts.
    ``counts`` is ``{lf_index: n_shorts}``; ranks run day-wide across the LFs.
    """
    shorts_dir = stages / f"day_{date}" / "shorts"
    shorts_dir.mkdir(parents=True, exist_ok=True)
    out: dict[int, list[Path]] = {}
    entries = []
    rank = 0
    for lf_index, n in counts.items():
        out[lf_index] = []
        for _ in range(n):
            rank += 1
            path = shorts_dir / f"short_{rank:02d}.mp4"
            path.write_bytes(b"y" * 1024)
            out[lf_index].append(path)
            entries.append(
                {
                    "rank": rank,
                    "parent_lf_index": lf_index,
                    "ost_text": place,
                    "regions": list(regions),
                    "place_source": "clip_gps",
                    "start_sec": float(rank * 60),
                    "end_sec": float(rank * 60 + 30),
                }
            )
    (stages / f"shorts_ost_audit_{date}.json").write_text(
        json.dumps({"date": date, "shorts": entries})
    )
    return out


def decided(metadata, title):
    """``metadata`` plus the Waypost title decision that chose ``title``."""
    from gomoveshift.pipeline.title_decision import decide_title

    return {**metadata, "title_decision": decide_title("longform", title, None, suffix="")}
