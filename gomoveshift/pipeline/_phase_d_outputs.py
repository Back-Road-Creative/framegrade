"""Phase D output-existence helpers — missing-output detection on resume.

Background: ``phase_d`` is invalidated when the persisted CLI fingerprint /
``PHASE_D_CODE_VERSION`` differs from the current invocation. This module
layers defense in depth on top: even when the fingerprint matches, verify the
canonical Phase D outputs still exist on disk. A partial failure, manual
``rm``, or filesystem corruption can leave ``state["phases_completed"]``
naming ``phase_d`` while the metadata JSONs / shorts CSV / thumbnails are
gone — the fingerprint check alone would still skip in that case.

Two pure helpers, no I/O beyond ``Path.exists``:

* :func:`expected_phase_d_outputs` — canonical list of files Phase D is
  expected to have written, derived from ``state``. Enumerates LFs from
  ``state["regular_lf_paths"]`` (the only state-recorded source of LF
  identity available on resume after a JSON round-trip), includes the
  shorts CSV only when ``state["shorts_per_lf"]`` is non-empty, and
  (when an optional ``config`` is passed) includes the first-party
  publish-plan JSON sidecar iff any ``DriveConfig.publishing_route_*``
  is ``"first_party"``.
* :func:`find_missing_phase_d_outputs` — subset of the expected list
  whose files are not on disk.

The orchestrator's ``_invalidate_phase_d_if_outputs_missing`` is the
sole consumer; tests pin the helpers directly so the failure mode is
construction-correct.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

_LF_INDEX_RE = re.compile(r"lf_(\d{1,3})\.mp4$", re.IGNORECASE)


def _lf_indices_from_state(state) -> list[int]:
    """Enumerate LF indices from ``state.regular_lf_paths``.

    The list shape is canonical: each entry is a path string ending in
    ``lf_NN.mp4`` (zero-padded; ``phase_bc_day_first`` names every final
    LF after ``lf_plan.index``). Indices are returned sorted, deduped.

    Falls back to enumerating from ``state.shorts_per_lf`` keys (when
    present) to cover any future state shape that records LF identity
    only via the shorts mapping. Both shapes coexist today.
    """
    indices: set[int] = set()
    paths = getattr(state, "regular_lf_paths", None)
    if paths:
        for entry in paths:
            if not entry:
                continue
            m = _LF_INDEX_RE.search(str(entry))
            if m:
                try:
                    indices.add(int(m.group(1)))
                except (TypeError, ValueError):
                    continue
    if not indices:
        shorts_map = getattr(state, "shorts_per_lf", None)
        if isinstance(shorts_map, Mapping):
            for k in shorts_map.keys():
                try:
                    indices.add(int(k))
                except (TypeError, ValueError):
                    continue
    return sorted(indices)


def _has_shorts(state) -> bool:
    """True iff ``state.shorts_per_lf`` describes any shorts.

    A non-empty mapping with at least one non-empty list of shorts
    implies Phase D wrote ``<project>_shorts_metricool.csv``. An empty /
    missing mapping means no shorts were planned for this run; the CSV
    is not expected and its absence must NOT trigger Phase D re-run.
    """
    shorts_map = getattr(state, "shorts_per_lf", None)
    if not isinstance(shorts_map, Mapping):
        return False
    for v in shorts_map.values():
        if v:
            return True
    return False


def _has_first_party_route(config) -> bool:
    """True iff any ``DriveConfig.publishing_route_*`` is ``"first_party"``.

    ``config`` is optional (``None`` — e.g. the pre-migration call
    sites, or tests pinning the helper without a config) and treated
    as "no first-party routes" so an all-off caller is unaffected.
    """
    drive = getattr(config, "drive", None)
    if drive is None:
        return False
    for field in (
        "publishing_route_youtube",
        "publishing_route_instagram",
        "publishing_route_facebook",
        "publishing_route_tiktok",
    ):
        if getattr(drive, field, "off") == "first_party":
            return True
    return False


def expected_phase_d_outputs(
    state,
    output_base: Path,
    project_name: str,
    *,
    config=None,
) -> list[Path]:
    """Return the canonical list of files Phase D is expected to have written.

    Includes:

    * ``<output_base>/<project_name>_lfNN_youtube.json`` for each LF in
      ``state["regular_lf_paths"]``.
    * ``<output_base>/thumbnails/lf_NN.jpg`` for each LF.
    * ``<output_base>/<project_name>_shorts_metricool.csv`` when
      ``state["shorts_per_lf"]`` describes any shorts. Written
      unconditionally regardless of publish routes — Metricool stays
      the live publisher during the first-party migration.
    * ``<output_base>/<project_name>_publish_plan.json`` iff ``config``
      is supplied and any ``DriveConfig.publishing_route_*`` is
      ``"first_party"``. Additive: never replaces the CSV above.
    """
    output_base = Path(output_base)
    expected: list[Path] = []
    indices = _lf_indices_from_state(state)
    thumbs_dir = output_base / "thumbnails"
    for idx in indices:
        expected.append(output_base / f"{project_name}_lf{idx:02d}_youtube.json")
        expected.append(thumbs_dir / f"lf_{idx:02d}.jpg")
    if _has_shorts(state):
        expected.append(output_base / f"{project_name}_shorts_metricool.csv")
    if _has_first_party_route(config):
        expected.append(output_base / f"{project_name}_publish_plan.json")
    return expected


def find_missing_phase_d_outputs(
    state,
    output_base: Path,
    project_name: str,
    *,
    config=None,
) -> list[Path]:
    """Return the subset of expected Phase D outputs not present on disk.

    Pure ``Path.exists`` checks — no content validation, no size check.
    A zero-byte file is treated as present; the caller owns content
    validation (``DoD validator`` already gates on size + format
    downstream).
    """
    return [
        p
        for p in expected_phase_d_outputs(state, output_base, project_name, config=config)
        if not p.exists()
    ]
