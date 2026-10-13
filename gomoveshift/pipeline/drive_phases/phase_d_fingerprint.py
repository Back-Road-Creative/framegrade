"""Phase D resume-fingerprint check: staleness invalidation + persistence.

Split out of ``pipeline/orchestrator_fingerprint.py`` so the engine module no
longer imports ``drive_phases.phase_d`` (media boundary). The pure input
builder ``build_phase_d_fingerprint_inputs`` stays in the engine module.
"""

from __future__ import annotations

import logging

from .._phase_d_fingerprint import (
    compute_phase_d_fingerprint,
    diff_fingerprint_inputs,
)
from ..orchestrator_fingerprint import build_phase_d_fingerprint_inputs
from .phase_d import PHASE_D_CODE_VERSION

logger = logging.getLogger("gomoveshift.pipeline.drive_orchestrator")


def invalidate_phase_d_if_fingerprint_stale(
    state,
    resumed_phases: list[str],
    stages_completed: list[str],
    *,
    config,
) -> list[str]:
    """If the persisted Phase D fingerprint is stale (or missing),
    drop ``phase_d`` from the resume set.

    Returns the (possibly mutated) ``resumed_phases`` list. Mutates
    ``stages_completed`` and ``state["phases_completed"]`` in place
    when invalidating, mirroring the upload-intent trigger block.

    Logs an INFO line naming which fields changed (or, on missing
    fingerprint, an explicit pre-fingerprint message).
    """
    if "phase_d" not in resumed_phases:
        return resumed_phases
    from .. import drive_phases  # local import to keep top-level light

    # Read PHASE_D_CODE_VERSION via module access so monkeypatching the
    # constant in tests takes effect without re-importing this module.
    code_version = int(getattr(drive_phases.phase_d, "PHASE_D_CODE_VERSION", PHASE_D_CODE_VERSION))
    new_inputs = build_phase_d_fingerprint_inputs(state, config=config)
    new_hash = compute_phase_d_fingerprint(new_inputs, code_version=code_version)
    old_hash = getattr(state, "phase_d_fingerprint", None)
    old_inputs = getattr(state, "phase_d_fingerprint_inputs", None) or {}
    # Normalise old_inputs: JSON round-trip turns int dict-keys into
    # str on the per_lf_locations field. Re-coerce so a no-change
    # invocation diffs as empty.
    if isinstance(old_inputs, dict) and "per_lf_locations" in old_inputs:
        from .._phase_d_fingerprint import _normalise_per_lf_locations

        old_inputs = dict(old_inputs)
        old_inputs["per_lf_locations"] = _normalise_per_lf_locations(
            old_inputs.get("per_lf_locations")
        )

    if old_hash is None:
        # Backward-compat: state file from a pre-fingerprint code
        # version. Invalidate so the operator gets fresh outputs
        # against the current generators (this is the case that
        # bites the in-flight Dingle state today).
        logger.info(
            "Phase D ran under a pre-fingerprint code version — re-running to refresh outputs."
        )
    elif old_hash == new_hash:
        return resumed_phases
    else:
        # Build a comparable old-inputs view that includes code_version
        # so the diff can name a version bump as the reason.
        old_inputs_for_diff = dict(old_inputs)
        if "code_version" not in old_inputs_for_diff:
            # Old inputs were stored without code_version embedded
            # (we persist the inputs dict separately from the hash);
            # to detect a code-version-only bump, synthesise the prior
            # code_version from the hash mismatch context.
            pass
        new_inputs_for_diff = dict(new_inputs)
        new_inputs_for_diff["code_version"] = code_version
        if "code_version" not in old_inputs_for_diff:
            old_inputs_for_diff["code_version"] = "<unknown>"
        changed = diff_fingerprint_inputs(old_inputs_for_diff, new_inputs_for_diff)
        logger.info(
            "Phase D fingerprint changed (CLI flags or code version) — "
            "re-running Phase D. Changed fields: %s. Old hash: %s. New hash: %s.",
            changed,
            old_hash,
            new_hash,
        )
    resumed_phases = [p for p in resumed_phases if p != "phase_d"]
    stages_completed[:] = [p for p in stages_completed if p != "phase_d"]
    if hasattr(state, "get"):
        saved_phases = getattr(state, "phases_completed", None) or []
        state.phases_completed = [p for p in saved_phases if p != "phase_d"]
    return resumed_phases


def persist_phase_d_fingerprint(state, *, config) -> None:
    """Persist the current Phase D fingerprint + inputs onto ``state``.

    Called from the phase loop immediately after ``phase_d`` is
    appended to ``stages_completed`` (i.e. after a successful run).
    The next ``self._save_state`` call writes the values to disk.
    """
    from .. import drive_phases  # local import — see invalidator

    code_version = int(getattr(drive_phases.phase_d, "PHASE_D_CODE_VERSION", PHASE_D_CODE_VERSION))
    inputs = build_phase_d_fingerprint_inputs(state, config=config)
    # Persist inputs alongside the hash so the next resume can
    # compute a precise diff naming which field changed.
    state.phase_d_fingerprint = compute_phase_d_fingerprint(inputs, code_version=code_version)
    inputs_for_persist = dict(inputs)
    inputs_for_persist["code_version"] = code_version
    state.phase_d_fingerprint_inputs = inputs_for_persist
