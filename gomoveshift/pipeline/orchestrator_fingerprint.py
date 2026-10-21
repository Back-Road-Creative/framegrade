"""Phase-D resume-fingerprint logic, extracted from
DrivePipelineOrchestrator.

* invalidate_phase_d_if_outputs_missing
* build_phase_d_fingerprint_inputs

The two functions that read ``PHASE_D_CODE_VERSION``
(``invalidate_phase_d_if_fingerprint_stale``, ``persist_phase_d_fingerprint``)
live in ``drive_phases/phase_d_fingerprint.py`` so this engine module imports
no Phase D code (media boundary).
"""

from __future__ import annotations

import logging
from pathlib import Path

from ._phase_d_fingerprint import extract_cli_inputs_from_args_and_state
from ._phase_d_outputs import find_missing_phase_d_outputs
from .drive_phases.phase_a import PHASE_A_CODE_VERSION
from .drive_phases.phase_bc_day_first import PHASE_BC_CODE_VERSION

logger = logging.getLogger("gomoveshift.pipeline.drive_orchestrator")

# Phases (other than phase_d, which carries its own richer fingerprint) whose
# checkpointed outputs must be re-derived when their code-rule version changes.
PHASE_CODE_VERSIONS: dict[str, int] = {
    "phase_a": PHASE_A_CODE_VERSION,
    "phase_bc_day_first": PHASE_BC_CODE_VERSION,
}

# Phases built from a phase's checkpointed output: re-running that phase means
# re-running these too. phase_d reads phase_bc's route record and plan, and its
# own fingerprint holds neither.
_REBUILT_WITH: dict[str, tuple[str, ...]] = {
    "phase_bc_day_first": ("phase_d",),
}


def persist_phase_code_version(state, phase_name: str) -> None:
    """Stamp the completed ``phase_name``'s current code version onto ``state``.

    Called from the phase loop right after a phase is appended to
    ``stages_completed`` (mirrors ``persist_phase_d_fingerprint``). No-op for
    phases without a tracked version. The next ``_save_state`` writes it to disk.
    """
    version = PHASE_CODE_VERSIONS.get(phase_name)
    if version is None:
        return
    versions = getattr(state, "phase_code_versions", None)
    versions = dict(versions) if isinstance(versions, dict) else {}
    versions[phase_name] = version
    state.phase_code_versions = versions


def invalidate_phases_if_code_version_stale(saved: dict, resumed_phases: list[str]) -> list[str]:
    """Drop any resumed phase whose STAMPED code version differs from the current.

    ``saved`` is the raw checkpoint dict (``_load_state`` output). A phase in
    ``resumed_phases`` whose ``phase_code_versions`` stamp differs from
    ``PHASE_CODE_VERSIONS`` is removed so it re-runs against the current
    generators — a post-fix run stamps the version, so any later rule-driven
    bump re-runs the affected phase instead of shipping pre-fix renders.

    Deliberately lenient on a MISSING stamp (a pre-versioning checkpoint):
    those are trusted here rather than force-re-run, so an unchanged resume
    isn't needlessly re-encoded. The concrete pre-fix artifacts a rule change
    produces are still caught downstream — the LF/short render fingerprints
    (segment windows + audio policy) and the cached-master encode-algo version
    both re-derive a stale render/master when phase_bc re-enters. A dropped
    phase also drops the phases in ``_REBUILT_WITH`` (phase_bc → phase_d).
    Returns the trimmed list; logs each drop.
    """
    saved_versions = (saved or {}).get("phase_code_versions") or {}
    if not isinstance(saved_versions, dict):
        return resumed_phases
    dropped: set[str] = set()
    for phase in resumed_phases:
        current = PHASE_CODE_VERSIONS.get(phase)
        stamped = saved_versions.get(phase)
        if current is not None and stamped is not None and stamped != current:
            logger.info(
                "Phase %s code version changed (checkpoint=%r current=%r) — "
                "re-running to re-derive its outputs against the current rules.",
                phase,
                stamped,
                current,
            )
            dropped.add(phase)
            dropped.update(_REBUILT_WITH.get(phase, ()))
    return [p for p in resumed_phases if p not in dropped]


def build_phase_d_fingerprint_inputs(state, *, config) -> dict:
    """Construct the canonical Phase D fingerprint inputs from
    the orchestrator's current ``state`` + ``config``.
    """
    # The run's long-form days: ``--lf-day`` (the CLI handler maps
    # ``mon/tue/.../sun`` → weekday-int into the config transport before
    # ``orchestrator.run()``), else Waypost's days -- read through the
    # ``long_form_days`` accessor, so a Waypost decision that moves the days
    # invalidates a stale checkpoint. Encode the list as a sorted
    # tuple-of-ints so a re-ordering of the same selection doesn't
    # spuriously invalidate the checkpoint.
    from .schedule_constants import long_form_days, long_form_time

    try:
        lf_day_value = sorted(int(d) for d in long_form_days(config))
        lf_day_repr: list[int] | None = list(lf_day_value) or None
    except (AttributeError, TypeError, ValueError):
        lf_day_repr = None
    # Project name: derived from source_dir.stem the same way ``run``
    # computes ``project_name`` so the fingerprint stays stable across
    # invocations of the same source.
    source_dir = getattr(state, "source_dir", None)
    if source_dir is not None:
        project_name = Path(source_dir).stem.replace(" ", "_").lower()
    else:
        project_name = ""
    # Publish-cadence flags reach Phase D via config (``--lf-time`` /
    # ``--shorts-times``) or state (``--lf-publish-date``). Include them so a
    # completed-phase_d resume with CORRECTED times re-runs instead of shipping
    # the old schedule.
    drive_cfg = getattr(config, "drive", None)
    # Build the args-like context. Using a dict keeps the indirection
    # explicit and lets ``extract_cli_inputs_from_args_and_state``
    # apply its own coercion / defaulting.
    args_like = {
        "lf_day": lf_day_repr,
        "publish": getattr(state, "publish_day", None),
        "start_date": getattr(state, "start_date", None),
        "shorts_target": getattr(state, "shorts_target", None),
        "shorts_per_day": getattr(state, "shorts_per_day", None),
        "per_lf_locations": getattr(state, "per_lf_overrides", None),
        "location": getattr(state, "location", None),
        "lf_time": long_form_time(config).strftime("%H:%M"),
        "shorts_times": getattr(drive_cfg, "shorts_publish_times", None),
        "lf_publish_date": getattr(state, "lf_publish_date", None),
    }
    return extract_cli_inputs_from_args_and_state(args_like, project_name=project_name)


def invalidate_phase_d_if_schedule_region_stale(
    state,
    resumed_phases: list[str],
    stages_completed: list[str],
) -> list[str]:
    """Drop ``phase_d`` from the resume set when the saved ``drive_schedule``'s
    region no longer matches the resolved ``state.region``.

    The fingerprint re-runs phase_d when CLI flags CHANGE, but a saved
    schedule that is wrong-but-flag-consistent (e.g. the
    region got silently overwritten and a later resume preserved that
    overwrite) is otherwise reused as-is — persisting a wrong publish schedule.
    Forcing phase_d to re-run makes ``_build_and_persist_drive_schedule``
    recompute against the resolved region. No-op when there is no saved
    schedule, the regions already match, or phase_d isn't in the resume set.
    Mutates ``stages_completed`` and ``state["phases_completed"]`` in place
    (mirrors the upload-intent / fingerprint invalidation idiom).
    """
    if "phase_d" not in resumed_phases:
        return resumed_phases
    ds = getattr(state, "drive_schedule", None)
    if ds is None:
        return resumed_phases
    ds_region = ds.get("region") if isinstance(ds, dict) else getattr(ds, "region", None)
    resolved = getattr(state, "region", None)
    if not ds_region or not resolved or ds_region == resolved:
        return resumed_phases
    logger.info(
        "Stale-schedule: drive_schedule.region=%r != resolved region=%r — "
        "dropping phase_d to recompute the publish schedule",
        ds_region,
        resolved,
    )
    out = [p for p in resumed_phases if p != "phase_d"]
    stages_completed[:] = [p for p in stages_completed if p != "phase_d"]
    saved = getattr(state, "phases_completed", None) or []
    state.phases_completed = [p for p in saved if p != "phase_d"]
    return out


def invalidate_phase_d_if_outputs_missing(
    state,
    resumed_phases: list[str],
    stages_completed: list[str],
    *,
    config=None,
) -> list[str]:
    """If any required Phase D output is missing on disk, drop
    ``phase_d`` from the resume set so it re-runs.

    Defense in depth on top of
    :func:`invalidate_phase_d_if_fingerprint_stale`. Even when the
    fingerprint matches, a manual ``rm`` / partial failure /
    filesystem corruption can leave ``state["phases_completed"]``
    naming ``phase_d`` while the metadata JSONs / shorts CSV /
    thumbnails are gone. Without this check, the orchestrator skips
    Phase D and ships nothing.

    Returns the (possibly mutated) ``resumed_phases`` list. Mutates
    ``stages_completed`` and ``state["phases_completed"]`` in place
    when invalidating, mirroring the fingerprint-stale block. Logs
    at WARNING (vs INFO for the fingerprint case) — a missing output
    is an unexpected disk-state divergence, not a routine CLI-flag
    change.
    """
    if "phase_d" not in resumed_phases:
        return resumed_phases
    # Output base + project name are derived the same way ``run``
    # builds them (and the same way Phase D writes its outputs):
    # ``state["output_base"]`` is the per-project dir; project base
    # is ``source_dir.stem.replace(" ", "_").lower()``.
    output_base_val = getattr(state, "output_base", None)
    source_dir_val = getattr(state, "source_dir", None)
    if output_base_val is None or source_dir_val is None:
        # No state-recorded location to verify against — leave the
        # resume set untouched. The fingerprint check + downstream
        # phase loop cover the no-state case.
        return resumed_phases
    output_base = Path(output_base_val)
    project_name = Path(source_dir_val).stem.replace(" ", "_").lower()
    missing = find_missing_phase_d_outputs(state, output_base, project_name, config=config)
    if not missing:
        return resumed_phases
    missing_str = ", ".join(str(p) for p in missing)
    logger.warning(
        "Phase D state present but output(s) missing: %s. Re-running Phase D.",
        missing_str,
    )
    resumed_phases = [p for p in resumed_phases if p != "phase_d"]
    stages_completed[:] = [p for p in stages_completed if p != "phase_d"]
    if hasattr(state, "get"):
        saved_phases = getattr(state, "phases_completed", None) or []
        state.phases_completed = [p for p in saved_phases if p != "phase_d"]
    return resumed_phases


def invalidate_phase_bc_if_superlong_incomplete(
    state,
    resumed_phases: list[str],
    stages_completed: list[str],
    *,
    stages_dir: Path,
) -> list[str]:
    """Re-enter ``phase_bc_day_first`` when this run wants superlong parts
    that are not on disk.

    ``phase_bc_day_first`` resumes as soon as its checkpoint and code version
    match, and NOTHING invalidates it when ``--superlong`` is newly supplied —
    ``superlong_enabled`` is a plain flag-registry row, in no fingerprint. A
    checkpointed phase_bc therefore never reaches ``run_day_pipeline``, the
    only code path that produces ``sl_NN.mp4``, and the run reports success
    with no superlong at all.

    Drops ``phase_d`` as well — not defensively, but because phase_d is what
    consumes and publishes the parts, and it declares phase_bc as its prereq.
    """
    from .drive_phases.phase_bc_day_first import day_superlong_complete

    if "phase_bc_day_first" not in resumed_phases:
        return resumed_phases
    if not bool(getattr(state, "superlong_enabled", None)):
        return resumed_phases
    if not stages_dir.is_dir():
        return resumed_phases

    for day_dir in sorted(p for p in stages_dir.glob("day_*") if p.is_dir()):
        date = day_dir.name[len("day_") :]
        if day_superlong_complete(date, day_dir, state):
            continue
        logger.warning(
            "Resume: --superlong is set but day %s has no complete superlong "
            "part set on disk — re-entering phase_bc_day_first (and phase_d). "
            "Superlong parts are produced only inside run_day_pipeline.",
            date,
        )
        resumed_phases = [p for p in resumed_phases if p not in ("phase_bc_day_first", "phase_d")]
        stages_completed[:] = [
            p for p in stages_completed if p not in ("phase_bc_day_first", "phase_d")
        ]
        saved = getattr(state, "phases_completed", None) or []
        state.phases_completed = [p for p in saved if p not in ("phase_bc_day_first", "phase_d")]
        return resumed_phases
    return resumed_phases
