"""Drive Pipeline Orchestrator — thin coordinator for multi-file drive processing.

The phases live in drive_phases/. This module is a
thin coordinator: build state, select manifest, loop phases, checkpoint.

Public API preserved:
  - DrivePipelineOrchestrator(config).run(**kwargs)
  - from gomoveshift.pipeline._drive_helpers import differentiate_duplicate_ost
"""

import json
import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Optional

from ..analysis.unified_scorer import UnifiedSegmentScorer
from ..config import Config, get_config, resolve_lf_mode
from ..day_layout import date_from_day_dir_name
from ..models import DrivePipelineResult
from ..publishing.title_kit import build_title_kit
from ._drive_state import load_drive_state, save_drive_state
from .auto_enqueue import auto_enqueue_shorts
from .drive_phases.phase_a import PhaseA
from .drive_phases.phase_b_preview import PhaseBPreview
from .drive_phases.phase_bc_day_first import PhaseBCDayFirst, _day_dir_has_finals
from .drive_phases.phase_d import PUBLISH_DAYS, PhaseD
from .drive_phases.phase_d_fingerprint import (
    invalidate_phase_d_if_fingerprint_stale,
    persist_phase_d_fingerprint,
)
from .drive_state_model import PER_LF_OVERRIDE_MIRROR_FIELDS, PipelineState
from .exceptions import PipelineValidationError
from .flag_registry import apply_cli_to_state, merge_on_resume
from .orchestrator_fingerprint import (
    invalidate_phase_bc_if_superlong_incomplete,
    invalidate_phase_d_if_outputs_missing,
    invalidate_phase_d_if_schedule_region_stale,
    invalidate_phases_if_code_version_stale,
    persist_phase_code_version,
)
from .orchestrator_preflight import run_preflight_and_prompt
from .ost_audit_sidecar import OstAuditPendingError
from .review_gate import ReviewFailedError, ReviewGate
from .state_invariants import assert_resume_state_coherent, resume_invariants
from .youtube_sidecar import load_sidecar, save_sidecar

# Tools package lives at repo root, not inside the gomoveshift package.
# Importing it here lets the orchestrator gate the run with the DoD
# validator without spawning a subprocess. It is NOT shipped in the wheel
# (pyproject packages only gomoveshift), so a non-editable install must
# still be able to import this module — the gate then fails loudly at
# call time instead of poisoning module import.
try:
    from tools.validate_pipeline_output import Mode as ValidationMode
    from tools.validate_pipeline_output import run_validation
except ModuleNotFoundError:
    _MISSING_TOOLS_MSG = (
        "DoD validation requires the repo checkout: tools/validate_pipeline_output.py "
        "is not part of the installed gomoveshift package. Run from a source checkout "
        "or install editable (pip install -e .)."
    )

    class _MissingToolsSentinel:
        def __getattr__(self, name):
            raise RuntimeError(_MISSING_TOOLS_MSG)

    ValidationMode = _MissingToolsSentinel()

    def run_validation(*args, **kwargs):
        raise RuntimeError(_MISSING_TOOLS_MSG)


logger = logging.getLogger(__name__)


# ─── SIGTERM/SIGHUP handler (lives on the orchestrator) ─────────────────
#
# Lives on the orchestrator so any future entry point that calls
# ``DrivePipelineOrchestrator.run()`` inherits the checkpoint-on-signal
# contract without having to remember to install the handler at its own
# argparse layer. Registration happens inside ``run()`` (defensively, via
# try/except ValueError so non-main-thread tests don't crash).
def _handle_sigterm(signum, frame):
    """Raise SystemExit so orchestrator try/finally blocks (state checkpoint,
    subprocess cleanup) run before the pipeline exits.

    Without this, SIGTERM from ``tmux kill-session`` or a system shutdown
    hard-terminates the interpreter, leaving half-written MP4s and an
    un-checkpointed ``pipeline_state.json`` that can't be resumed.
    """
    logger.warning("Signal %s received, shutting down drive pipeline...", signum)
    raise SystemExit(143)


# TODO: Delete this eager-import when
# mid-run .py merges are blocked by tooling. Until then, this is the cheap
# defense-in-depth against a stale
# sys.modules entry that would otherwise raise hours into a run.
# ─── Eager-import contract for state_invariants ──────────────────────────
#
# Mid-run merges of unrelated ``.py`` files can invalidate ``sys.modules``
# such that a deferred import of a ``state_invariants`` symbol later in the
# run raises ``ImportError`` mid-phase. That turns a code-merge accident
# into hours of lost compute. The remedy is to resolve every symbol the
# orchestrator (and its callees) need from ``state_invariants`` ONCE at
# orchestrator entry — before phase_a runs — so any ImportError surfaces
# at startup, not after 6 hours of master encode.
#
# Symbols listed below are the full public surface of
# ``gomoveshift.pipeline.state_invariants`` that the runtime touches:
#
#   * ``assert_state_invariants`` — Layer-1 write guard, called via
#     ``_drive_state.save_drive_state`` on every checkpoint.
#   * ``assert_resume_state_coherent`` — Layer-1.5 resume guard, called
#     once per ``run()`` after state load + auto-populate.
#   * ``StateInvariantError`` / ``StateCoherenceError`` — exception types
#     raised by the above. Resolved here so a stale ``sys.modules`` can't
#     hide the symbol behind a ``except`` that silently mismatches.
#   * ``STATE_INVARIANT_ERROR_PREFIX`` — grep-able log token. Resolving it
#     forces the constant's existence to be checked at startup.
#
# DO NOT remove this block without updating
# ``test_orchestrator_eager_imports.py``. The contract is: this
# function MUST be called from ``DrivePipelineOrchestrator.run`` before
# any phase work, and MUST raise on missing symbols rather than swallow.
def _eager_import_state_invariants() -> None:
    """Resolve every ``state_invariants`` symbol used at runtime.

    Called from :meth:`DrivePipelineOrchestrator.run` as its first
    statement, before phase_a or any state IO. If the
    ``gomoveshift.pipeline.state_invariants`` module has been invalidated
    in ``sys.modules`` (e.g. by a mid-run code merge), the import below
    will raise ``ImportError`` immediately — turning a runtime crash into
    a startup error.

    Raises
    ------
    ImportError
        If the ``state_invariants`` module or any of its referenced
        symbols cannot be resolved.
    """
    # The module-level ``from .state_invariants import …`` above already
    # resolved ``assert_resume_state_coherent`` at import time. This call
    # re-exercises the import path against the *current* ``sys.modules``
    # entry so that a mid-run module invalidation is caught here, not in
    # the deferred ``from .state_invariants import assert_state_invariants``
    # inside ``_drive_state.save_drive_state`` after hours of work.
    from . import state_invariants as _si

    # Touch every symbol the runtime depends on. ``getattr`` with no
    # default raises ``AttributeError`` (a subclass of ``Exception``) if
    # the symbol vanished — convert to ``ImportError`` so the contract
    # stays "import problems surface at startup".
    required = (
        "assert_state_invariants",
        "assert_resume_state_coherent",
        "resume_invariants",
        "StateInvariantError",
        "StateCoherenceError",
        "STATE_INVARIANT_ERROR_PREFIX",
    )
    missing: list[str] = []
    for name in required:
        if not hasattr(_si, name):
            missing.append(name)
    if missing:
        raise ImportError(
            "state_invariants module is missing required symbols "
            f"{missing!r} — sys.modules cache is stale (likely a mid-run "
            "code merge). Restart the pipeline from a clean Python "
            "interpreter."
        )


# Run-lock helpers extracted to orchestrator_lock.
# Re-exported here so callers and tests that import from drive_orchestrator
# still resolve. Design rationale for the lock lives at the top of
# ``orchestrator_lock.py`` (the "Orchestrator-entry concurrent-run lock" header).
from .orchestrator_lock import (  # noqa: E402, F401  (re-export)
    ConcurrentRunError,
    WorktreeLeaseError,
    _acquire_run_lock,
    _acquire_run_locks,
    _acquire_worktree_lease,
    _pid_is_alive,
    _raise_concurrent_run_error,
    _read_lock_metadata,
    _release_run_lock,
    _release_worktree_lease,
    _renew_worktree_lease,
    _write_lock_metadata,
    acquire_gpu_lock,
    release_gpu_lock,
)


def auto_populate_per_day_completed(state: dict, stages_dir: Path, *, fresh: bool) -> None:
    """Walk ``stages_dir`` for ``day_<YYYY-MM-DD>/`` dirs with finals on
    disk and add a recovery marker to ``state["per_day_completed"]``
    for each one not already recorded.

    When
    ``pipeline_state.json`` is wiped or its ``per_day_completed`` field
    is empty/corrupt but day directories with valid finals exist on
    disk, neither resume fast-path in ``phase_bc_day_first`` fires and
    the pipeline re-encodes from scratch (8+ hr loss). The recovery
    marker — ``{"master_path": None}`` — is exactly
    the shape fast-path #2 (finals-only reconstruction) expects: a
    truthy entry whose ``master_path`` is None falls through to the
    ``_day_dir_has_finals`` branch that rebuilds ``DayArtifacts`` from
    the on-disk LF + short outputs.

    Idempotent: existing entries are never overwritten. A missing /
    empty stages dir is a no-op (no exception). Dirs that don't match
    the canonical ``day_<YYYY-MM-DD>`` regex are ignored.

    ``fresh`` is keyword-only and required: a ``--fresh`` run re-encodes, so
    a prior run's finals are never recovered (otherwise a ``--fresh`` audit
    could reuse shorts that burned the wrong place, after ``_try_resume`` had
    already purged the state this hook then re-populated).
    """
    if fresh or not stages_dir.is_dir():
        return

    # Dual-shape read: PipelineState in production, plain dict in tests.
    if isinstance(state, dict):
        per_day = state.get("per_day_completed")
    else:
        per_day = getattr(state, "per_day_completed", None)
    if not isinstance(per_day, dict):
        per_day = {}

    for child in stages_dir.iterdir():
        if not child.is_dir():
            continue
        # The auto-populate hook matches by directory name alone, without
        # consulting drive_plan.json.
        date = date_from_day_dir_name(child.name)
        if date is None:
            continue
        if date in per_day:
            continue
        if not _day_dir_has_finals(child):
            continue
        logger.info(
            "  auto-populate: day %s has finals on disk but no "
            "per_day_completed entry — adding recovery marker",
            date,
        )
        per_day[date] = {"master_path": None}

    if isinstance(state, dict):
        state["per_day_completed"] = per_day
    else:
        state.per_day_completed = per_day


# ─── Per-phase artifact manifest ────────────────────────────────────────
# Invariant: an artifact listed here must be present (relative to
# stages_dir — the orchestrator's stages_base, typically
# ``output_base/_stages/``) for its phase to count as resumable. If
# ``phases_completed`` in the checkpoint names a phase but one of its
# listed artifacts is missing, ``_try_resume`` drops the phase from
# ``resumed_phases`` so it re-runs on this invocation (prevents silent
# downstream NoneAttributeError when a SIGKILL between state-save and
# artifact-save leaves the checkpoint ahead of the filesystem).
#
# Scope rules:
#   * Only include artifacts written by the phase itself at its own
#     "mark complete" write site (see drive_phases/phase_*.py).
#   * An empty list means "phase always re-validated via existing
#     in-phase gates" — use when no single top-level artifact cleanly
#     represents phase completion (e.g. phase_b_preview produces a
#     fan of per-episode preview clips, not one gate file).
#   * Paths are resolved relative to ``stages_dir``; artifacts written
#     outside ``stages_dir`` (e.g. per-LF mp4s under ``output_base``)
#     are validated by downstream phases' own existence checks, not
#     here.
#
# Drop-vs-raise: on missing artifact we DROP the phase from
# ``phases_completed`` so it re-runs. Phase re-runs are idempotent:
# phase_a re-ingests from source; phase_bc_day_first re-loads
# drive_plan.json or rebuilds from state; phase_d re-reads the
# ``youtube_uploads`` sidecar before any upload so re-runs never
# duplicate YouTube entries.


def full_drive_skeleton_indices(drive_cfg, plans) -> "list[int] | None":
    """gate_a skeleton LF indices for ``lf_selection_mode == "full_drive"``.

    Without balanced parts every day is exactly one LF, so ``[1]``. With
    ``full_drive_part_min_sec``/``full_drive_part_max_sec`` set the LF count is
    only known once phase_bc has planned: take it from ``plans`` (the drive
    plan's regular LFs) when present, else ``None`` (gate_a then prints the
    drive-wide label).
    """
    if (
        getattr(drive_cfg, "full_drive_part_min_sec", None) is None
        or getattr(drive_cfg, "full_drive_part_max_sec", None) is None
    ):
        return [1]
    if not plans:
        return None
    return [int(getattr(p, "index", i + 1)) for i, p in enumerate(plans)]


class PhaseSpec(NamedTuple):
    """One row of the drive-phase graph — the SINGLE source of truth from which
    execution order, ``--phase`` prereq validation, and resume artifact-gates
    all derive (no parallel hardcoded structures).

    * ``name``      — phase key (must be in ``_drive_state._KNOWN_PHASES``).
    * ``fn_attr``   — orchestrator method name; bound via ``getattr(self, ...)``
      at manifest-build time so tests can still patch the seam.
    * ``prereqs``   — phases that must be complete before ``--phase name`` runs.
    * ``artifacts`` — files (relative to ``stages_dir``) that must exist for the
      phase to count as resumable; see the artifact-manifest notes above.
    * ``in_manifest`` — whether the phase is part of the full-run execution
      order. ``phase_b_preview`` is False: it runs only in the ``--preview``
      branch, never in the A → BC → D manifest.
    """

    name: str
    fn_attr: str
    prereqs: tuple[str, ...]
    artifacts: tuple[str, ...]
    in_manifest: bool = True


PHASE_SPECS: tuple[PhaseSpec, ...] = (
    PhaseSpec("phase_a", "_phase_a_ingest_and_analyze", (), ("drive_scores.npz",)),
    # per-episode preview fan; no single gate, preview-branch only
    PhaseSpec("phase_b_preview", "_phase_b_preview", ("phase_a",), (), in_manifest=False),
    PhaseSpec(
        "phase_bc_day_first",
        "_phase_bc_day_first",
        ("phase_a",),
        ("drive_plan.json",),
    ),
    PhaseSpec(
        "phase_d",
        "_phase_d_metadata_and_publish",
        ("phase_a", "phase_bc_day_first"),
        ("drive_schedule.json",),
    ),
)

# Resume artifact-gate map, derived from the single phase-spec table.
PHASE_ARTIFACTS: dict[str, list[str]] = {s.name: list(s.artifacts) for s in PHASE_SPECS}


LF_POLICY_STAMP = "lf_policy.json"
# Phases that hold output built under an LF policy: the plan (phase_bc) and
# everything downstream of it.
_LF_POLICY_DEPENDENT_PHASES = ("phase_bc_day_first", "phase_d")


def write_lf_policy_stamp(stages_dir: Path, drive_cfg) -> None:
    """Record the LF policy (mode + part bounds) the saved plan was built under."""
    from ..config import lf_policy_snapshot

    path = Path(stages_dir) / LF_POLICY_STAMP
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lf_policy_snapshot(drive_cfg), sort_keys=True))


def reconcile_lf_policy_on_resume(stages_dir: Path, resumed_phases: list, drive_cfg) -> list:
    """Never let a resumed plan silently keep an old LF mode / part bounds.

    Neither ``pipeline_state.json`` nor the CLI<->state flag registry carries the
    LF mode or bounds: a resumed run always reads the CURRENT resolved config.
    What survives is the saved PLAN (``drive_plan.json``, a phase_bc artifact)
    built under whatever policy was live then. A stamp (``lf_policy.json``,
    written when phase_bc completes) records that policy. If it differs from the
    current one, phase_bc and what depends on it are dropped so they re-run under
    the current policy (``LF-POLICY-RESUME-MISMATCH``); with no stamp (a
    pre-stamp checkpoint) the plan is kept but named loudly
    (``LF-POLICY-RESUME-UNSTAMPED``).
    """
    from ..config import lf_policy_snapshot

    if "phase_bc_day_first" not in resumed_phases:
        return resumed_phases
    current = lf_policy_snapshot(drive_cfg)
    path = Path(stages_dir) / LF_POLICY_STAMP
    try:
        saved = json.loads(path.read_text())
    except (OSError, ValueError):
        logger.warning(
            "LF-POLICY-RESUME-UNSTAMPED: the saved plan has no recorded LF policy; "
            "keeping it, but it may have been built under a different mode/bounds "
            "than the current %s. Use --fresh to re-plan.",
            current,
        )
        return resumed_phases
    if saved == current:
        return resumed_phases
    logger.warning(
        "LF-POLICY-RESUME-MISMATCH: saved plan was built under %s but the current "
        "policy is %s -- re-running %s under the current policy.",
        saved,
        current,
        ", ".join(_LF_POLICY_DEPENDENT_PHASES),
    )
    return [p for p in resumed_phases if p not in _LF_POLICY_DEPENDENT_PHASES]


def preserve_per_lf_overrides_on_resume(state, cli_overrides: Optional[dict]) -> None:
    """Re-apply a non-empty CLI ``--per-lf-location`` override after the resume
    merge so operator intent wins over a saved-empty ``per_lf_overrides``.

    ``--per-lf-location`` seeds the FRESH initial state, but ``_try_resume``
    reloads the saved state (whose ``per_lf_overrides`` is empty from an original
    run that had none) OVER it — so an operator re-running a PAUSED drive to
    CORRECT a burned label saw the override silently dropped. A NON-EMPTY CLI
    override re-applies here (CLI wins); an empty/None override leaves the saved
    value untouched (saved wins) — the same saved-wins-unless-explicit contract
    as ``--region``. Mutates ``state`` in place.

    The override also rewrites each already-populated Phase-A-derived label map
    (``per_lf_locations`` / ``per_lf_ost_text`` / ``per_lf_titles``): a Phase-D-only
    resume never re-derives them, so without this the saved label still became
    the title (e.g. 'To Goleen, County Cork' hard-failed the
    metadata gate despite ``--per-lf-location '1=Goleen, County Cork'``). An
    unpopulated map is left alone — Phase A consumes ``per_lf_overrides`` itself.
    """
    if cli_overrides:
        state.per_lf_overrides = dict(cli_overrides)
        for field in PER_LF_OVERRIDE_MIRROR_FIELDS:
            derived = getattr(state, field, None)
            if derived:
                setattr(state, field, {**derived, **cli_overrides})


def reset_force_sidecar_reset_after_resume(state, cli_value: Optional[bool]) -> None:
    """Force ``state.force_sidecar_reset`` to THIS invocation's CLI intent,
    unconditionally, AFTER the resume merge.

    ``force_sidecar_reset`` is a one-shot operator recovery flag (wipes both
    upload sidecars), not a sticky setting. It is a ``FLAG_REGISTRY`` row
    (``SENTINEL_CLI_WINS``), so ``merge_on_resume``'s blanket saved-field
    copy re-imports whatever is on disk — including a poisoned ``True``
    written before this field was added to ``_UNSERIALIZED_FIELDS``
    (drive_state_model.py), since that exclusion only stops future writes,
    it does not filter reads. Without this call, that one poisoned
    checkpoint re-wipes both sidecars and re-uploads every published LF as
    a new public video on every later resume. Must run AFTER ``_try_resume``
    — placing it earlier (e.g. in ``_build_initial_state``, like the
    similar ``ost_audit_confirmed_this_run`` reset) would itself be
    clobbered by the merge, since that field isn't a registry row and
    ``force_sidecar_reset`` is. Mutates ``state`` in place.
    """
    state.force_sidecar_reset = True if cli_value else None


def reset_ost_opt_ins_after_resume(
    state, ost_audit_confirmed: Optional[bool], ost_system_review: Optional[bool]
) -> None:
    """Put the two per-run OST gate signals back to THIS invocation's flags,
    AFTER the resume merge.

    ``ost_audit_confirmed_this_run`` (``--ost-audit-confirmed``) and
    ``ost_system_review_this_run`` (``--ost-system-review``) are in
    ``_UNSERIALIZED_FIELDS``, which stops writes only: ``_try_resume`` still
    reads a legacy or edited ``pipeline_state.json``, so a saved ``true``
    would stand in for a flag this run never passed. Mutates ``state``.
    """
    state.ost_audit_confirmed_this_run = True if ost_audit_confirmed else None
    state.ost_system_review_this_run = True if ost_system_review else None


def run_phase_a_ost_gate(state) -> None:
    """OST uniqueness/validity gate after Phase A — fail fast before encoding.

    Gate-bypass hardening: the old inline ``if ep_locs:``
    guard silently no-opped when ``episode_locations`` was empty, and the gate
    only ran inside phase_a's execution branch, so a resumed run (phase_a
    checkpointed) skipped it entirely. Phase A always writes one
    ``episode_locations`` entry per episode ("" for GPS-less ones), so an
    EMPTY list while episodes exist is state drift → hard fail instead of a
    silent bypass. No episodes at all (nothing to encode) is a no-op.
    """
    ep_locs = getattr(state, "episode_locations", None) or []
    if not ep_locs:
        ep_files = getattr(state, "episode_files", None) or []
        if ep_files:
            raise RuntimeError(
                f"OST gate: state has {len(ep_files)} episode(s) but "
                "episode_locations is empty — phase_a state drift; refusing "
                "to encode without per-episode OST derivation."
            )
        return
    from .preflight import run_preflight_ost_check

    run_preflight_ost_check(ep_locs)


class DrivePipelineOrchestrator:
    """Coordinates the full Drive Pipeline from raw GoPro files to published content."""

    PUBLISH_DAYS = PUBLISH_DAYS

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self._phases = {
            "phase_a": PhaseA(self.config),
            "phase_b_preview": PhaseBPreview(self.config),
            "phase_bc_day_first": PhaseBCDayFirst(self.config, title_kit=build_title_kit()),
            "phase_d": PhaseD(self.config),
        }

    # -- Delegating methods (tests patch these) --

    def _phase_a_ingest_and_analyze(self, state: PipelineState) -> None:
        self._phases["phase_a"].run(state, labels_only=getattr(self, "_labels_only", False))

    def _phase_b_preview(self, state: PipelineState) -> None:
        self._phases["phase_b_preview"].run(state)

    def _phase_bc_day_first(self, state: PipelineState) -> None:
        phase = self._phases["phase_bc_day_first"]
        phase.run(
            state,
            probe_duration_fn=self._probe_duration,
            state_save_fn=getattr(phase, "_save_callback", None),
        )

    def _phase_d_metadata_and_publish(self, state: PipelineState, drive_plan=None) -> None:
        # The full-run path (_execute_phases) sets ``_pending_gate_cd`` so
        # Gate C/D runs INSIDE phase_d, ahead of its irreversible §2/§3 upload +
        # Metricool CSV blocks — a rejection then blocks publication entirely
        # instead of firing uselessly after the uploads. Direct unit-test calls
        # to this seam leave it unset → no gate (preserves the prior
        # isolated-phase behaviour where gate_cd lived only in the run loop).
        self._phases["phase_d"].run(
            state,
            drive_plan=drive_plan,
            pre_upload_gate=getattr(self, "_pending_gate_cd", None),
        )

    def _synthesize_ingest_plan_from_state(self, state):
        from .drive_phases.phase_bc_day_first import _synthesize_ingest_plan_from_state

        return _synthesize_ingest_plan_from_state(state)

    def _probe_duration(self, video_path):
        from .drive_phases.phase_bc_day_first import _probe_duration

        return _probe_duration(video_path)

    def _partition_shorts_slots(self, schedule, regular_lfs):
        from .drive_phases.phase_d import _partition_shorts_slots

        return _partition_shorts_slots(schedule, regular_lfs)

    # -- Phase append coherence guard --

    def _phase_produced_required_artifacts(self, phase_name: str, state) -> tuple[bool, str]:
        """Decide whether ``phase_name`` may be appended to ``phases_completed``.

        A phase that ran-and-returned but produced 0 artifacts is a silent
        failure: appending the phase locks in the failure across resume,
        because the orchestrator skips already-completed phases. Without
        this check, recovery means editing ``pipeline_state.json`` by hand.

        Rules (return ``(False, reason)`` to BLOCK the append):

        * ``phase_a`` — both ``episode_files`` and ``drive_scores`` empty /
          missing. Either populates means ingest+analyze did real work.
        * ``phase_bc_day_first`` — ``day_artifacts`` empty / missing. The
          per-day master is the single load-bearing output of BC; without
          it Phase D has nothing to slice.
        * ``phase_d`` — ``youtube_upload`` was True (operator opted in)
          AND ``youtube_uploads`` is empty. Phase D legitimately runs in
          skip-upload mode with no uploads recorded; the rule only fires
          when the operator asked for uploads and got none.

        Other phases (``phase_b_preview``, unknown / future phases) pass
        through unchanged. Returns ``(True, "")`` on pass.
        """

        def _g(key, default=None):
            # Tests pass plain dicts; production passes PipelineState.
            if isinstance(state, dict):
                return state.get(key, default)
            return getattr(state, key, default)

        if phase_name == "phase_a":
            episode_files = _g("episode_files") or []
            drive_scores = _g("drive_scores")
            # ``drive_scores`` may be a numpy ndarray — len() is safe;
            # bool() on an ndarray would raise ValueError.
            has_scores = drive_scores is not None
            if has_scores:
                try:
                    has_scores = len(drive_scores) > 0
                except TypeError:
                    has_scores = bool(drive_scores)
            if not episode_files and not has_scores:
                return (
                    False,
                    "phase_a produced empty episode_files AND empty drive_scores",
                )
            return (True, "")

        if phase_name == "phase_bc_day_first":
            day_artifacts = _g("day_artifacts") or []
            if not day_artifacts:
                return (
                    False,
                    "phase_bc_day_first produced 0 day_artifacts "
                    "(silent encode failure — NOT marking complete so resume retries)",
                )
            return (True, "")

        if phase_name == "phase_d":
            wants_upload = bool(_g("youtube_upload"))
            uploads = _g("youtube_uploads") or {}
            if wants_upload and not uploads:
                return (
                    False,
                    "phase_d ran with youtube_upload=True but youtube_uploads is empty "
                    "(no uploads recorded — NOT marking complete so resume retries)",
                )
            return (True, "")

        return (True, "")

    # -- State persistence --

    def _save_state(self, state_file: Path, state: PipelineState) -> None:
        # ``skip_invariants`` is the break-glass
        # for a latent Layer-1 invariant bug that would otherwise make
        # every save raise mid-run. Captured on the orchestrator at
        # ``run()`` entry so callback-driven saves (PhaseD, PhaseBC) pick
        # it up without re-threading the kwarg.
        save_drive_state(
            state_file,
            state,
            skip_invariants=getattr(self, "_skip_invariants", False),
        )
        phases = getattr(state, "phases_completed", None)
        # A labels-only run renders nothing; it must not move the recipe a heal resumes from.
        if phases and not getattr(self, "_labels_only", False):
            from ..health.resume import mark_phase  # noqa: PLC0415 -- never fails the save

            mark_phase(Path(state_file).parent, phases[-1])
        # Also mirror youtube_uploads into the durable sidecar so a future
        # state reset can't erase upload history.
        output_base = getattr(state, "output_base", None)
        uploads = getattr(state, "youtube_uploads", None)
        if output_base is not None and uploads is not None:
            try:
                sha_hex = save_sidecar(Path(output_base), uploads)
                state.youtube_uploads_sha = sha_hex
            except OSError as e:
                logger.warning("Failed to write youtube_uploads.json sidecar: %s", e)

    def _load_state(self, state_file: Path) -> Optional[PipelineState]:
        return load_drive_state(state_file)

    def _emit_pipeline_summary(
        self,
        *,
        source_dir: Path,
        state: Optional[dict],
        stages_completed: list[str],
        phase_defs_total: int,
        total_time: float,
    ) -> None:
        """Emit the PIPELINE_SUMMARY sentinel on any exit path.

        Load-bearing ``PIPELINE_SUMMARY`` prefix is grepped by operator
        dashboards — must fire exactly once per ``run()`` invocation,
        including early exits (dry-run, preflight decline, preview, DoD
        failure) and unhandled exceptions. Tolerates None/empty state and
        pre-phase-loop exits where ``phase_defs`` is not yet built.
        """
        s = state or {}
        n_lfs = len(s.get("regular_lf_paths", []) or [])
        shorts_per_lf = s.get("shorts_per_lf", {}) or {}
        n_shorts = sum(len(v or []) for v in shorts_per_lf.values())
        if s.get("skip_upload"):
            youtube_uploaded = "skipped"
        else:
            uploads = s.get("youtube_uploads") or {}
            youtube_uploaded = "true" if uploads else "false"
        logger.info(
            "PIPELINE_SUMMARY source=%s lfs=%d shorts=%d youtube_uploaded=%s "
            "phases=%d/%d duration=%.1fs",
            source_dir.name,
            n_lfs,
            n_shorts,
            youtube_uploaded,
            len(stages_completed),
            phase_defs_total,
            total_time,
        )

    # -- Main entry point --

    def run(
        self,
        source_dir: Path,
        output_base: Path,
        region: Optional[str] = None,
        publish_day: Optional[str] = None,
        location: str = "",
        fresh: bool = False,
        skip_upload: Optional[bool] = None,
        skip_drive_upload: Optional[bool] = None,
        skip_youtube_upload: Optional[bool] = None,
        skip_csv: bool = False,
        phase: Optional[str] = None,
        start_date: Optional[str] = None,
        lf_publish_date: Optional[str] = None,
        force_no_rebase: Optional[bool] = None,
        shorts_per_day: Optional[int] = None,
        shorts_target: Optional[int] = None,
        shorts_gap: Optional[float] = None,
        youtube_upload: Optional[bool] = None,
        keep_intermediates: Optional[bool] = None,
        keep_masters: Optional[bool] = None,
        allow_short_count: Optional[bool] = None,
        skip_shorts_quality_gate: Optional[bool] = None,
        no_ledger_reconcile: Optional[bool] = None,
        sign_verify: Optional[bool] = None,
        sign_verify_strict: Optional[bool] = None,
        force_sidecar_reset: Optional[bool] = None,
        allow_seo_degraded: Optional[bool] = None,
        dry_run: bool = False,
        force: bool = False,
        skip_invariants: bool = False,
        force_unlock: bool = False,
        per_lf_overrides: Optional[dict] = None,
        list_lfs: bool = False,
        label_dry_run: bool = False,
        ost_audit_confirmed: Optional[bool] = None,
        ost_system_review: Optional[bool] = None,
        gate_cd_confirmed: Optional[bool] = None,
        no_youtube_upload: Optional[bool] = None,
        skip_audio: Optional[bool] = None,
        lf_mode: Optional[str] = None,
        superlong_enabled: Optional[bool] = None,
        superlong_publish_day: Optional[str] = None,
        gps_captions: Optional[str] = None,
    ) -> DrivePipelineResult:
        """Execute the full Drive Pipeline (or a single phase).

        When the run completes all requested phases, the terminal DoD
        validator is invoked (unless ``config.drive.skip_validation`` is
        true). A failing validation raises :class:`PipelineValidationError`
        — that exception is NOT caught inside this method; callers (CLI,
        scripts) must handle it at the outermost boundary.

        Preflight execution plan
        ------------------------
        Before any phase work, :func:`preflight_plan.print_plan` is
        invoked with the assembled ``state`` and ``stages_dir``. The
        rendered plan enumerates each phase + per-day row with status
        (RUN / SKIP) and a reason. Gating:

        * ``dry_run=True`` → print the plan and return
          ``DrivePipelineResult(success=True, stages_completed=[])``
          without running any phase or touching disk.
        * ``force=True`` → print the plan and proceed, no prompt
          (tmux / non-interactive friendly).
        * ``force=False`` AND ``sys.stdin.isatty()`` → print the plan,
          prompt ``Proceed? [y/N]``. Any answer other than ``y``/``yes``
          (case-insensitive) aborts gracefully with ``success=True`` and
          no stages completed.
        * ``force=False`` AND no operator can see the prompt (stdin not a
          tty, or tty with no visible channel) → named refusal
          ``PREFLIGHT-NO-OPERATOR``; never auto-answered.

        """
        # Register SIGTERM/SIGHUP handlers before any I/O so
        # checkpoint-on-signal works for every entry point (cmd_drive +
        # future callers). Defensive: signal.signal() raises ValueError
        # when called from a non-main thread (e.g. some test runners), so
        # catch and log rather than crash a unit test that wraps run() in
        # a thread.
        try:
            signal.signal(signal.SIGTERM, _handle_sigterm)
            signal.signal(signal.SIGHUP, _handle_sigterm)
        except ValueError:
            logger.debug("signal handlers not registered (not main thread)")
        # Eager-import every ``state_invariants`` symbol the
        # runtime touches. Surfaces stale ``sys.modules`` (mid-run merge)
        # as an ImportError at startup, not after hours of phase work.
        # See the "Eager-import contract for state_invariants" comment block.
        _eager_import_state_invariants()
        start_time = time.time()
        # ISO-8601 anchor used by the DoD validator to assert metadata
        # freshness (mtime >= run_start). Captured once per invocation.
        self._run_start_iso = datetime.now(tz=timezone.utc).isoformat()
        self._run_start_is_fresh = True  # may be overridden by _try_resume
        # Break-glass for a latent Layer-1
        # invariant bug. Captured on the instance so callback-driven
        # saves (PhaseD, PhaseBC) pick it up without re-threading.
        self._skip_invariants = bool(skip_invariants)
        # Captured on the instance so _phase_a_ingest_and_analyze (called
        # positionally as phase_fn(state) from _execute_phases) can thread it
        # into PhaseA.run without widening every phase_fn's signature. See
        # "Shared GPU lock" below for what this changes.
        self._labels_only = bool(list_lfs or label_dry_run)
        if self._skip_invariants:
            logger.warning(
                "Drive pipeline started with --skip-invariants — Layer-1 "
                "state invariants will be bypassed on every save. "
                "BREAK-GLASS mode; never use in normal operation."
            )
        stages_completed: list[str] = []
        errors: list[str] = []
        timings: dict = {}
        # Tracks whether the OST audit gate raised
        # OstAuditPendingError during the phase loop. When True, the
        # success calc below short-circuits to True and the DoD validator
        # is skipped (no LFs encoded → nothing to validate).
        ost_audit_pending = False
        self.waiting_gate = None
        # Pre-initialize so the ``finally`` sentinel below can
        # reference these names on every exit path, including early
        # returns before the phase manifest is assembled.
        state: dict = {}
        phase_defs: list = []
        source_dir = Path(source_dir)
        if not source_dir.exists():
            self._emit_pipeline_summary(
                source_dir=source_dir,
                state=None,
                stages_completed=[],
                phase_defs_total=0,
                total_time=time.time() - start_time,
            )
            return DrivePipelineResult(
                success=False,
                source_dir=source_dir,
                output_dir=output_base,
                drive_plan=None,
                regular_lf_paths=[],
                shorts_per_lf={},
                stages_completed=[],
                errors=[f"Source directory not found: {source_dir}"],
                timings={},
            )
        output_base = Path(output_base)
        project_dir, stages_base = self._resolve_paths(source_dir, output_base)
        if publish_day is None:
            # No explicit --lf-day: derive the default from
            # the run's long_form_days (Waypost's earliest day) instead of a
            # stale "sat"/"wed" literal. The old USA "sat" (dow=5) is not in
            # Waypost's long-form days, so a USA drive without --lf-day
            # tripped state-invariant Rule 2. Mirrors scheduler's
            # min(long_form_days) source of truth. An explicit publish_day
            # (non-None) bypasses this and still wins.
            from .schedule_constants import default_publish_day

            publish_day = default_publish_day(self.config)
        # Worktree-code-tree lease: a live drive running from a disposable
        # worktree (a linked git worktree or a configured disposable
        # directory) holds a lease when a lease helper is configured
        # (``GMS_LEASE_BIN``), so a concurrent cleanup pass cannot delete
        # this run's code tree mid-run. No-op
        # for the main tree or when unconfigured. Refuses to start if another session holds it;
        # renewed at phase boundaries, released in the ``finally`` below.
        self._worktree_lease_handle = _acquire_worktree_lease()
        # Orchestrator-entry concurrent-run lock. Acquired AFTER
        # eager-import + path derivation, BEFORE any state IO or preflight,
        # so two operators against the same output dir fail-fast instead of
        # racing on day-master encodes. Released in the ``finally`` at the
        # bottom of run() — survives SIGTERM/SIGHUP/KeyboardInterrupt
        # because those handlers raise SystemExit through finally.
        # Dual lock: the project-dir lock is authoritative across differing
        # GMS_INTERMEDIATE_DIR (two runs on the same project_dir would otherwise
        # clobber the shared youtube_uploads.json sidecar / Metricool CSV and
        # double-book upload slots); the stages-base lock keeps the same-staging
        # refusal. Both released in the ``finally`` below.
        #
        # Both acquisitions below run BEFORE the ``finally`` at the bottom of
        # run() takes effect (that ``try`` starts after the GPU lock), so a
        # failure here needs its own rollback — otherwise a ConcurrentRunError
        # or GpuLockBusyError leaks the worktree lease for the full TTL and
        # blocks relaunch from this same worktree.
        try:
            run_lock_fd, run_lock_path, proj_lock_fd, proj_lock_path = _acquire_run_locks(
                project_dir, stages_base, output_base, force_unlock=force_unlock
            )
        except BaseException:
            _release_worktree_lease(self._worktree_lease_handle)
            raise
        # Shared GPU lock, taken after the run locks so a second operator is
        # refused for the right reason before we start waiting on hardware.
        # Held for the whole run: the encode stages are spread across day
        # masters, shorts, and OST passes, and a lock dropped between them
        # would let a model load into the gap. See orchestrator_lock.py.
        #
        # --label-dry-run / --list-lfs in full_drive mode never calls
        # acquire_gpu_lock at all: PhaseA.run(labels_only=True) skips the
        # GPU-bound scoring pass entirely for that mode (see phase_a.py), so
        # there is nothing here that would ever touch the GPU PROVIDED the
        # run also can't reach an encode stage — hence the ``preview``/
        # ``phase`` exclusions below. Without this, the pre-encode
        # label-verification gate was impossible whenever any other drive
        # held the lock — a --label-dry-run would wait out a full 3600s
        # GMS_GPU_LOCK_WAIT_S window before even starting.
        #
        # For any other labels_only run (best_moments), Phase A's LF
        # boundaries genuinely come from real composite scores, so scoring
        # still runs and still needs the lock — but a dry-run must never
        # SILENTLY wait out another run's hold, so it takes the lock
        # non-blocking (wait_s=0) and fails fast, naming the holder, instead.
        #
        # Two further exclusions, both able to reach an nvenc encode despite
        # ``labels_only`` being set: ``--preview`` runs ``_phase_b_preview``
        # (a real GPU encode) unconditionally before the labels-only
        # short-circuit ever fires; an explicit ``--phase`` filter bypasses
        # the phase_a short-circuit's own return entirely, falling through
        # to whatever single phase was requested.
        skip_gpu_lock = (
            self._labels_only
            and (resolve_lf_mode(self.config.drive) == "full_drive")
            and not self.config.drive.preview
            and phase is None
        )
        try:
            if skip_gpu_lock:
                gpu_lock_fd, gpu_lock_path = None, None
            elif self._labels_only:
                gpu_lock_fd, gpu_lock_path = acquire_gpu_lock(wait_s=0)
            else:
                gpu_lock_fd, gpu_lock_path = acquire_gpu_lock()
        except BaseException:
            _release_run_lock(run_lock_fd, run_lock_path)
            _release_run_lock(proj_lock_fd, proj_lock_path)
            _release_worktree_lease(self._worktree_lease_handle)
            raise
        try:
            state, upload_flag_explicit, skip_upload, youtube_upload = self._build_initial_state(
                source_dir=source_dir,
                project_dir=project_dir,
                stages_base=stages_base,
                region=region,
                publish_day=publish_day,
                location=location,
                fresh=fresh,
                skip_upload=skip_upload,
                skip_drive_upload=skip_drive_upload,
                skip_youtube_upload=skip_youtube_upload,
                youtube_upload=youtube_upload,
                skip_csv=skip_csv,
                start_date=start_date,
                lf_publish_date=lf_publish_date,
                force_no_rebase=force_no_rebase,
                shorts_per_day=shorts_per_day,
                shorts_target=shorts_target,
                shorts_gap=shorts_gap,
                keep_intermediates=keep_intermediates,
                keep_masters=keep_masters,
                allow_short_count=allow_short_count,
                skip_shorts_quality_gate=skip_shorts_quality_gate,
                no_ledger_reconcile=no_ledger_reconcile,
                sign_verify=sign_verify,
                sign_verify_strict=sign_verify_strict,
                force_sidecar_reset=force_sidecar_reset,
                allow_seo_degraded=allow_seo_degraded,
                per_lf_overrides=per_lf_overrides,
                ost_audit_confirmed=ost_audit_confirmed,
                ost_system_review=ost_system_review,
                gate_cd_confirmed=gate_cd_confirmed,
                no_youtube_upload=no_youtube_upload,
                skip_audio=skip_audio,
                lf_mode=lf_mode,
                superlong_enabled=superlong_enabled,
                superlong_publish_day=superlong_publish_day,
                gps_captions=gps_captions,
            )
            stages_dir = stages_base
            if str(output_base).startswith("/mnt/"):
                logger.info(
                    "Cross-mount output detected — using Linux FS for intermediates: %s",
                    stages_dir,
                )
            stages_dir.mkdir(parents=True, exist_ok=True)
            state_file = stages_dir / "pipeline_state.json"
            # Wire up PhaseD save callback so it can persist state after each upload.
            self._phases["phase_d"]._save_callback = lambda s: self._save_state(state_file, s)
            # Wire up PhaseBC save callback so per-day bookkeeping persists mid-phase.
            # Without this, a mid-phase crash (e.g. day N master encode fails after
            # days 0..N-1 landed) leaves ``per_day_completed`` empty on disk and the
            # next resume re-encodes already-done days.
            self._phases["phase_bc_day_first"]._save_callback = lambda s: self._save_state(
                state_file, s
            )
            resumed_phases = self._try_resume(
                state, state_file, stages_dir, stages_completed, fresh
            )
            # ── Region resume-preservation ──────────────────────────────────
            # ``--region`` is an argparse sentinel (default None): an ABSENT
            # flag must NOT overwrite a resumed run's saved region: that would
            # reset the publish schedule to the default region's cadence
            # (e.g. a USA drive's region silently becoming ireland).
            # ``_try_resume`` keeps
            # the saved region when this run's is None (the user_args
            # ``if val is not None`` filter); we resolve the still-None case
            # (fresh run, or no saved region) to the "ireland" default here,
            # AFTER the merge, so saved-wins is preserved and an explicit
            # ``--region usa`` still wins (and the fingerprint check
            # recomputes the schedule when it changes).
            if getattr(state, "region", None) is None:
                state.region = "ireland"
            # ── Per-LF override resume-preservation ──────────────────────────
            # ``--per-lf-location`` overrides reach run() as ``per_lf_overrides``
            # and seed the FRESH initial state (``_build_initial_state``), but
            # ``_try_resume`` reloads the saved state — whose ``per_lf_overrides``
            # is empty from an original run that had none — OVER them. Without
            # a re-apply, an operator re-running a PAUSED drive to CORRECT a burned
            # label would see the override silently dropped (a ``1=A to B``
            # override would never reach the OST cascade; the GPS-derived
            # label would burn instead). Re-apply a non-empty CLI override AFTER
            # the merge so operator intent wins on resume — same saved-wins-unless
            # -explicit pattern as ``--region`` above. An absent override ({})
            # leaves the saved value untouched. Authoritative (the
            # override string is burned verbatim downstream). Extracted for
            # unit-testability (mirrors the schedule-region-stale helper).
            preserve_per_lf_overrides_on_resume(state, per_lf_overrides)
            # ── force_sidecar_reset per-invocation reset (post-merge) ────────
            # See ``reset_force_sidecar_reset_after_resume`` docstring — must
            # run here, after ``_try_resume``'s ``merge_on_resume`` call.
            reset_force_sidecar_reset_after_resume(state, force_sidecar_reset)
            # Same for the per-run OST signals (see its docstring).
            reset_ost_opt_ins_after_resume(state, ost_audit_confirmed, ost_system_review)
            # ── Stale-schedule recompute on resume ──────────────────────────
            # The fingerprint re-runs phase_d when CLI FLAGS change, but a
            # saved ``drive_schedule`` whose region no longer matches the
            # resolved region (wrong-but-flag-consistent) is otherwise reused as-is. Drop phase_d
            # so it RECOMPUTES the schedule. Extracted for unit-testability.
            resumed_phases = invalidate_phase_d_if_schedule_region_stale(
                state, resumed_phases, stages_completed
            )
            # ── phase_d re-run trigger on explicit upload intent ─────────────
            # When the operator passes ``--youtube-upload`` (or otherwise
            # explicitly opts in to a Drive upload after a skip-upload audit),
            # a phase_d already in ``phases_completed`` MUST re-run — otherwise
            # the audit-then-publish workflow silently no-ops at the
            # orchestrator boundary (phase_d in ``resumed_phases`` → phase loop
            # ``continue`` → 0 uploads, ``phases=3/3`` reported, sidecar empty).
            #
            # Trigger predicate: ``youtube_upload=True`` AND phase_d is in
            # resumed_phases. The skip_drive_upload / skip_youtube_upload split
            # is for ``what to do once phase_d runs``, not ``whether to re-run``;
            # the re-run signal is ``the operator wants an upload step now``.
            # We do NOT trigger on a fresh run (phase_d not in resumed_phases
            # yet) and we do NOT trigger on a pure regen (no upload flag).
            if (
                getattr(state, "youtube_upload", None)
                and "phase_d" in resumed_phases
                and not (getattr(state, "skip_youtube_upload", None) or False)
            ):
                logger.info(
                    "re-running phase_d — youtube_upload=True with "
                    "phase_d already in phases_completed (audit→publish workflow)"
                )
                resumed_phases = [p for p in resumed_phases if p != "phase_d"]
                stages_completed[:] = [p for p in stages_completed if p != "phase_d"]
                saved_phases = getattr(state, "phases_completed", None) or []
                state.phases_completed = [p for p in saved_phases if p != "phase_d"]
            # ── Phase D fingerprint invalidation ─────────────────
            # Drop ``phase_d`` from the resume set when CLI flags or
            # PHASE_D_CODE_VERSION differ from the persisted fingerprint.
            # Eliminates the silent "Drive pipeline completed in 0.0s"
            # failure mode when CLI flags change between invocations.
            # Only fires on the default resume path; the explicit
            # ``--phase D`` single-phase invocation is unaffected.
            resumed_phases = invalidate_phase_d_if_fingerprint_stale(
                state, resumed_phases, stages_completed, config=self.config
            )
            # ── Phase D missing-output detection (defense in depth) ─
            # After the fingerprint check, verify the canonical Phase D
            # outputs exist on disk. A partial failure, manual ``rm``, or
            # filesystem corruption can leave ``state["phases_completed"]``
            # naming ``phase_d`` while the metadata JSONs / shorts CSV /
            # thumbnails are gone — the fingerprint check alone would
            # still skip in that case.
            resumed_phases = invalidate_phase_d_if_outputs_missing(
                state, resumed_phases, stages_completed, config=self.config
            )
            # ── superlong re-entry: --superlong's only producer lives inside
            # run_day_pipeline, which a checkpointed phase_bc never reaches.
            # Must run BEFORE auto_populate below, which would otherwise mark
            # the day complete on LFs+shorts alone.
            resumed_phases = invalidate_phase_bc_if_superlong_incomplete(
                state, resumed_phases, stages_completed, stages_dir=stages_dir
            )
            # ── Prevention: auto-populate per_day_completed from disk ──
            # If pipeline_state.json was wiped (or its per_day_completed field
            # was emptied / corrupted) but day_<date>/ directories with valid
            # finals already exist on disk, neither phase_bc_day_first
            # fast-path fires and the next phase invocation re-encodes from
            # scratch. Walking the stages dir here — BEFORE the phase loop
            # runs and BEFORE print_plan renders SKIP reasons — populates the
            # state so fast-path #2 (finals-only reconstruction) takes over.
            # Idempotent: existing entries are preserved. Skipped on --fresh.
            auto_populate_per_day_completed(state, stages_dir, fresh=fresh)
            # ── Prevention: cross-field state coherence check ─────
            # Runs immediately AFTER the auto-populate step so the coherence check validates the
            # populated state. Raises ``StateCoherenceError`` if
            # ``phases_completed`` claims a phase done while
            # ``per_day_completed`` is missing dates the phase was expected to
            # process. We do not auto-correct: the operator removes the
            # spurious phase entry (or wipes state) so the underlying
            # write-bug surfaces instead of getting masked.
            assert_resume_state_coherent(state)
            # ── resume_invariants — single post-load entry point ──
            # Runs only on the resume path (resumed_phases non-empty means
            # _try_resume found valid state on disk). Folds preflight
            # (disk/RAM/ffmpeg), sidecar integrity, OST audit hash drift,
            # and per-LF backstop completeness behind ONE call site so
            # resume cannot diverge from the fresh-run path. Fresh runs
            # skip this — preflight runs at phase_a entry below.
            if resumed_phases:
                resume_invariants(state, self.config)
            # On first run, persist run_start so resumed runs use the original
            # timestamp for freshness checks (metricool/youtube mtime >= run_start).
            if not getattr(state, "run_start", None):
                state.run_start = self._run_start_iso
                self._save_state(state_file, state)
            # Load the durable youtube_uploads sidecar and merge any records the
            # pipeline state doesn't already know about. The sidecar
            # survives state resets and fresh=True, preventing phantom duplicate
            # uploads when history is otherwise lost.
            sidecar_uploads = load_sidecar(project_dir)
            if sidecar_uploads:
                current = getattr(state, "youtube_uploads", None) or {}
                merged = dict(current)
                for key, rec in sidecar_uploads.items():
                    # State uses int LF index keys; sidecar persists them as str.
                    try:
                        int_key = int(key)
                    except (TypeError, ValueError):
                        int_key = key
                    if int_key not in merged and key not in merged:
                        merged[int_key] = rec
                state.youtube_uploads = merged
                # Safety default: if the sidecar has entries and the caller did
                # not explicitly opt into uploading, skip upload. The user can
                # re-run with youtube_upload=True to override.
                #
                # Trust ANY explicit upload
                # CLI flag — ``--youtube-upload``, ``--skip-youtube-upload``,
                # ``--skip-drive-upload``, ``--skip-upload``. If the operator was
                # explicit on the command line, do NOT auto-set
                # ``state["skip_upload"]``; the granular skip_drive_upload /
                # skip_youtube_upload flags above already encode their intent.
                if not upload_flag_explicit and not youtube_upload and not skip_upload:
                    state.skip_upload = True
                    # Keep the granular flags coherent with the coarse
                    # alias (matches the --skip-upload alias semantics in
                    # _build_initial_state). phase_d re-derives these via
                    # ``or skip_upload`` so this is consistency/observability,
                    # not a behavior change — any other reader of the granular
                    # state now sees the skip too.
                    state.skip_drive_upload = True
                    state.skip_youtube_upload = True
                    logger.warning(
                        "youtube_uploads.json has %d entries; defaulting skip_upload=True. "
                        "Pass youtube_upload=True to opt in.",
                        len(sidecar_uploads),
                    )

            early = run_preflight_and_prompt(
                state=state,
                stages_dir=stages_dir,
                dry_run=dry_run,
                force=force,
                source_dir=source_dir,
                project_dir=project_dir,
                start_time=start_time,
                stages_completed=stages_completed,
                config=self.config,
                emit_summary=self._emit_pipeline_summary,
                labels_only=self._labels_only,
            )
            if early is not None:
                return early
            phase_defs = self._select_phase_manifest(state_file)
            if phase:
                prereqs = self._get_prereqs()
                missing = [p for p in prereqs.get(phase, []) if p not in resumed_phases]
                if missing:
                    self._emit_pipeline_summary(
                        source_dir=source_dir,
                        state=state,
                        stages_completed=stages_completed,
                        phase_defs_total=len(phase_defs),
                        total_time=time.time() - start_time,
                    )
                    return DrivePipelineResult(
                        success=False,
                        source_dir=source_dir,
                        output_dir=project_dir,
                        drive_plan=None,
                        regular_lf_paths=[],
                        shorts_per_lf={},
                        stages_completed=[],
                        errors=[
                            "Cannot run %s: prerequisite phases not completed: "
                            "%s. Run the full pipeline first or "
                            "complete missing phases." % (phase, ", ".join(missing))
                        ],
                        timings={},
                    )
                phase_defs = [(n, fn) for n, fn in phase_defs if n == phase]
                resumed_phases = [p for p in resumed_phases if p != phase]
                logger.info("Single-phase mode: running %s only", phase)
            if self.config.drive.preview:
                if "phase_a" not in resumed_phases:
                    t0 = time.time()
                    self._phase_a_ingest_and_analyze(state)
                    timings["phase_a"] = time.time() - t0
                    ok, reason = self._phase_produced_required_artifacts("phase_a", state)
                    if ok:
                        stages_completed.append("phase_a")
                        state.phases_completed = list(stages_completed)
                    else:
                        logger.error(
                            "phase_a coherence guard tripped: %s — NOT appending to "
                            "phases_completed; resume will retry.",
                            reason,
                        )
                    self._save_state(state_file, state)
                try:
                    self._phase_b_preview(state)
                    # Wire the preview review gate (defined but historically
                    # never invoked). Automated check: preview clips exist. The
                    # human OST/colour confirmation is skipped under --no-review.
                    ReviewGate(interactive=not self.config.drive.no_review).gate_preview(
                        output_dir=project_dir
                    )
                except Exception as e:
                    logger.error("Preview encode failed: %s", e)
                    errors.append(str(e))
                    state.phases_completed = list(stages_completed)
                    self._save_state(state_file, state)
                    self._emit_pipeline_summary(
                        source_dir=source_dir,
                        state=state,
                        stages_completed=stages_completed,
                        phase_defs_total=len(phase_defs),
                        total_time=time.time() - start_time,
                    )
                    return DrivePipelineResult(
                        success=False,
                        source_dir=source_dir,
                        output_dir=project_dir,
                        drive_plan=None,
                        regular_lf_paths=[],
                        shorts_per_lf={},
                        stages_completed=stages_completed,
                        errors=errors,
                        timings=timings,
                    )
                logger.info("Preview mode complete -- re-run without --preview for full encode")
                self._emit_pipeline_summary(
                    source_dir=source_dir,
                    state=state,
                    stages_completed=stages_completed,
                    phase_defs_total=len(phase_defs),
                    total_time=time.time() - start_time,
                )
                return DrivePipelineResult(
                    success=True,
                    source_dir=source_dir,
                    output_dir=project_dir,
                    drive_plan=None,
                    regular_lf_paths=[],
                    shorts_per_lf={},
                    stages_completed=stages_completed,
                    errors=[],
                    timings=timings,
                )
            early_result, ost_audit_pending = self._execute_phases(
                state=state,
                state_file=state_file,
                phase_defs=phase_defs,
                resumed_phases=resumed_phases,
                source_dir=source_dir,
                project_dir=project_dir,
                location=location,
                list_lfs=list_lfs,
                label_dry_run=label_dry_run,
                start_time=start_time,
                stages_completed=stages_completed,
                errors=errors,
                timings=timings,
            )
            if early_result is not None:
                return early_result
            total_time = time.time() - start_time
            timings["total"] = total_time
            last_phase = phase or phase_defs[-1][0]
            success = len(errors) == 0 and last_phase in (
                getattr(state, "phases_completed", None) or []
            )
            # OST audit gate pending is a clean partial exit —
            # operator confirms sidecar then resumes. success=True with
            # phase_a in stages_completed but phase_bc/phase_d absent.
            if ost_audit_pending:
                success = True

            # Shorts are automatic: a successful publish run spools its own
            # plan for the hourly dispatch cron (pipeline/auto_enqueue.py).
            if success and not ost_audit_pending:
                if auto_enqueue_shorts(state, state_file, self.config, errors):
                    success = False

            self._run_dod_validator(
                state=state,
                output_base=output_base,
                stages_base=stages_base,
                project_dir=project_dir,
                success=success,
                ost_audit_pending=ost_audit_pending,
                source_dir=source_dir,
                stages_completed=stages_completed,
                phase_defs=phase_defs,
                total_time=total_time,
            )

            status = "completed" if success else "failed"
            logger.info(
                f"Drive pipeline {status} "
                f"in {total_time:.1f}s ({len(stages_completed)}/{len(phase_defs)} phases)"
            )

            # Structured PIPELINE_SUMMARY sentinel. See
            # ``_emit_pipeline_summary`` for field spec. The sentinel now fires
            # on every exit path (dry-run, preflight decline, preview,
            # DoD-failure, unhandled exception, happy path); each site calls
            # the helper directly so operators grep a consistent format.
            self._emit_pipeline_summary(
                source_dir=source_dir,
                state=state,
                stages_completed=stages_completed,
                phase_defs_total=len(phase_defs),
                total_time=total_time,
            )
            return DrivePipelineResult(
                success=success,
                source_dir=source_dir,
                output_dir=project_dir,
                drive_plan=getattr(state, "drive_plan", None),
                regular_lf_paths=(getattr(state, "regular_lf_paths", None) or []),
                shorts_per_lf=(getattr(state, "shorts_per_lf", None) or {}),
                stages_completed=stages_completed,
                errors=errors,
                timings=timings,
            )
        finally:
            release_gpu_lock(gpu_lock_fd, gpu_lock_path)
            _release_run_lock(run_lock_fd, run_lock_path)
            _release_run_lock(proj_lock_fd, proj_lock_path)
            _release_worktree_lease(self._worktree_lease_handle)

    def _purge_stale_stage_artifacts(self, state, stages_dir) -> None:
        """Delete a prior run's plan + checkpoint so ``--fresh`` can't inherit them.

        Robust-by-construction guard for the ``--fresh`` contract: Phase BC
        loads ``drive_plan.json`` from ``stages_dir`` (and the legacy
        ``output_base/_stages`` mirror) by path, so a stale plan left from a
        DIFFERENT source silently reassembles LFs from the wrong drive's
        segments. ``--fresh`` early-returns from resume, so without this the
        files survive. Wipe ``drive_plan.json`` + ``pipeline_state.json`` from
        both locations up front. Each deletion logs so an operator sees what a
        fresh run cleared.
        """
        mirror = Path(state.output_base) / "_stages"
        # Dedupe by resolved path so stages_dir == mirror collapses to one.
        targets = {
            (base / name).resolve(): base / name
            for base in (stages_dir, mirror)
            for name in ("drive_plan.json", "pipeline_state.json")
        }
        # Also purge any OST-audit ``*.confirmed`` markers: a leftover
        # confirmation from a prior run would silently confirm a fresh run's
        # never-reviewed burn text — ``confirmation_is_stale(None, x)`` is False
        # by design once ``--fresh`` wipes the state-persisted prior hash.
        for base in (Path(stages_dir), mirror):
            for confirmed in base.glob("*.confirmed"):
                targets.setdefault(confirmed.resolve(), confirmed)
        for path in targets.values():
            if path.exists():
                path.unlink()
                logger.info("  --fresh: cleared stale stage artifact %s", path)

    def _try_resume(self, state, state_file, stages_dir, stages_completed, fresh):
        resumed_phases = []
        if fresh:
            self._purge_stale_stage_artifacts(state, stages_dir)
            return resumed_phases
        saved = self._load_state(state_file)
        if not saved or "phases_completed" not in saved:
            return resumed_phases
        # Validate artifact manifest: drop any phase whose declared artifacts
        # are missing on disk so it re-runs cleanly. Without this gate, a
        # SIGKILL between state-save and artifact-save leaves phases_completed
        # ahead of the filesystem; downstream phases then crash on None far
        # from the root cause. See PHASE_ARTIFACTS docstring.
        raw_phases = saved["phases_completed"]
        resumed_phases = []
        dropped: list[tuple[str, list[str]]] = []
        # Artifact lookup: check stages_dir first, then the output_base/_stages/
        # fallback where older-version runs wrote Phase A outputs. Without the
        # fallback, a legacy drive_scores.npz at output_base/_stages/ triggers
        # a full Phase A rerun on every resume, even though the drive_scores reload
        # further down in this method knows how to load it from that fallback location.
        fallback_dir = Path(saved.get("output_base") or state.output_base) / "_stages"
        for phase_name in raw_phases:
            required = PHASE_ARTIFACTS.get(phase_name, [])
            missing = [
                name
                for name in required
                if not (stages_dir / name).exists() and not (fallback_dir / name).exists()
            ]
            if missing:
                dropped.append((phase_name, missing))
                continue
            resumed_phases.append(phase_name)
        if dropped:
            for phase_name, missing in dropped:
                logger.warning(
                    "  Resume artifact drift: %s marked complete in checkpoint "
                    "but missing artifact(s) %s under %s — re-running phase.",
                    phase_name,
                    missing,
                    stages_dir,
                )
            # Persist the trimmed list back into the loaded checkpoint so the
            # subsequent state.update(saved) below doesn't restore the stale
            # phases_completed; the next self._save_state will rewrite the
            # file with the validated list.
            saved["phases_completed"] = list(resumed_phases)
        # Code-version invalidation for phase_a / phase_bc (mirrors the phase_d
        # fingerprint gate below, in run()): a checkpoint whose stamped phase
        # code version differs from the current constant re-runs that phase so a
        # planner/audio/blur rule change re-derives its renders instead of
        # shipping the pre-fix artifact.
        code_valid = invalidate_phases_if_code_version_stale(saved, resumed_phases)
        if code_valid != resumed_phases:
            resumed_phases = code_valid
            saved["phases_completed"] = list(resumed_phases)
        # An explicit --lf-mode from the original run is persisted (flag registry);
        # a resume without the flag keeps that operator choice, sourced as such.
        if saved.get("lf_mode") and self.config.drive.lf_mode_source != "cli":
            self.config.drive.set_lf_mode(saved["lf_mode"], "resumed-state")
        lf_valid = reconcile_lf_policy_on_resume(stages_dir, resumed_phases, self.config.drive)
        if lf_valid != resumed_phases:
            resumed_phases = lf_valid
            saved["phases_completed"] = list(resumed_phases)
        # Single registry-driven merge path: replaces
        # the hand-wired ``user_args`` / ``schedule_overrides`` allowlists and
        # the resume-authoritative shorts_target/shorts_gap special case. CLI↔state
        # authority per flag is declared in ``flag_registry.FLAG_REGISTRY`` —
        # a new persisted flag is wired by ADDING A ROW THERE, never by
        # re-introducing a per-flag merge here.
        merge_on_resume(state, saved, self.config, log=logger)
        stages_completed.extend(resumed_phases)
        # Restore the original run_start so freshness checks compare against
        # the time metadata files were actually generated, not "now".
        if saved.get("run_start"):
            self._run_start_iso = saved["run_start"]
            self._run_start_is_fresh = False
        logger.info("Resuming from checkpoint: %s already done", resumed_phases)
        if "phase_a" in resumed_phases:
            scores_path = stages_dir / "drive_scores.npz"
            cached_meta: dict = {}
            if scores_path.exists():
                scorer = UnifiedSegmentScorer(weights=self.config.drive.scoring_weights)
                state.drive_scores, cached_meta = scorer.load_scores(scores_path)
                logger.info("Reloaded drive_scores from %s", scores_path.name)
            elif (fallback := Path(state.output_base) / "_stages" / "drive_scores.npz").exists():
                scorer = UnifiedSegmentScorer(weights=self.config.drive.scoring_weights)
                state.drive_scores, cached_meta = scorer.load_scores(fallback)
                logger.info("Reloaded drive_scores from fallback %s", fallback)
            # ``gps_per_episode`` is no longer persisted into state — rebuild
            # the per-episode trackpoint traces from the on-disk .gpx paths so
            # every downstream GPS consumer (per-LF trace, resolved location)
            # sees a real trace on resume, not a None.
            if state.gpx_files and not state.gps_per_episode:
                from .gps_enrichment import (
                    episodes_with_ingest_gps,
                    rebuild_gps_per_episode,
                )

                # A re-parse failure hard-fails ONLY for an episode that
                # carried GPS at ingest (a persisted resolved model attributes
                # to it); a genuinely GPS-less episode still degrades quietly.
                state.gps_per_episode = rebuild_gps_per_episode(
                    state.gpx_files, had_gps_indices=episodes_with_ingest_gps(state)
                )
                logger.info(
                    "Rebuilt gps_per_episode from %d gpx path(s) on resume",
                    len(state.gpx_files),
                )
            if (
                "file_boundaries" not in state
                and getattr(state, "file_durations", None) is not None
            ):
                offset = 0.0
                boundaries = []
                for d in state.file_durations:
                    boundaries.append((offset, offset + d))
                    offset += d
                state.file_boundaries = boundaries
                logger.info("Recomputed %d file_boundaries from file_durations", len(boundaries))
                # Make this resume-time
                # reconstruction observable. Whole-state recovery → use
                # the ``[-1]`` sentinel rather than enumerating each
                # boundary tuple.
                from .state_invariants import record_backstop_recovery

                record_backstop_recovery(
                    state,
                    "file_boundaries",
                    [-1],
                    f"legacy state file: reconstructed {len(boundaries)} "
                    "boundaries from file_durations (state predates "
                    "file_boundaries in the phase_a output)",
                )
            # The score_inputs validity key (episode file size+mtime) that
            # Phase A's own score-input gate checks was persisted into the .npz metadata
            # but is DISCARDED on this orchestrator reload path — so a re-ingest
            # since the checkpoint (new episode mtimes) reused stale scores and
            # phase_a never re-ran. Re-validate it here and drop phase_a so the
            # phase loop re-scores against the current episodes.
            episode_files = getattr(state, "episode_files", None)
            cached_key = cached_meta.get("score_inputs") if isinstance(cached_meta, dict) else None
            if episode_files and cached_key is not None:
                from .drive_phases.phase_a import _score_inputs_key

                if _score_inputs_key(episode_files) != cached_key:
                    logger.info(
                        "Phase A score_inputs changed since checkpoint (episodes "
                        "added/removed/re-ingested) — re-running Phase A instead of "
                        "reusing stale drive_scores."
                    )
                    resumed_phases = [p for p in resumed_phases if p != "phase_a"]
                    stages_completed[:] = [p for p in stages_completed if p != "phase_a"]
                    saved["phases_completed"] = [
                        p for p in saved.get("phases_completed", []) if p != "phase_a"
                    ]
                    if getattr(state, "phases_completed", None):
                        state.phases_completed = [
                            p for p in state.phases_completed if p != "phase_a"
                        ]
        # Legacy "phase_b" resume branch removed: the
        # non-day-first pipeline was the only producer of
        # phase_b_manifest.json + the only consumer of state.processed_files
        # threaded this way. load_drive_state now raises on stale
        # phase_b/phase_c state, so this branch was dead code.
        return resumed_phases

    def _execute_phases(
        self,
        state,
        state_file: Path,
        phase_defs: list,
        resumed_phases: list,
        source_dir: Path,
        project_dir: Path,
        location: str,
        list_lfs: bool,
        label_dry_run: bool,
        start_time: float,
        stages_completed: list,
        errors: list,
        timings: dict,
    ) -> tuple[Optional[DrivePipelineResult], bool]:
        """Iterate ``phase_defs`` and run each non-resumed phase.

        Mutates ``stages_completed``, ``errors``, and ``timings`` in place
        (the caller threads the same lists through downstream summary +
        DoD validator code paths). Returns ``(early_result, ost_audit_pending)``:

        * ``early_result`` is non-``None`` only on the ``--list-lfs``
          short-circuit after phase_a. The caller returns it unmodified.
        * ``ost_audit_pending`` is True when ``OstAuditPendingError`` fired
          mid-loop; the caller skips the DoD validator and forces
          ``success=True`` so the operator can confirm the sidecar and resume.
        """
        ost_audit_pending = False
        review = ReviewGate(interactive=not self.config.drive.no_review)
        # Gate C/D is injected INTO phase_d (before its uploads) rather
        # than run after phase_d finished. _phase_d_metadata_and_publish reads
        # this to wire PhaseD.run's pre_upload_gate.
        self._pending_gate_cd = lambda s: review.gate_cd(
            state=s, output_dir=project_dir, date_key="final"
        )

        def _print_labels_and_return() -> DrivePipelineResult:
            # Shared by the fresh-phase_a and resumed-phase_a branches below
            # so a checkpointed phase_a can't fall through to phase_bc/phase_d
            # (a real nvenc encode + upload) without ever printing labels —
            # otherwise a resumed --label-dry-run would silently run the
            # full encode because only the fresh-run branch short-circuited.
            from .drive_phases.lf_listing import print_label_dry_run, print_lf_listing

            refused = False
            if label_dry_run:
                refused = print_label_dry_run(self.config, state) is True
            else:
                print_lf_listing(self.config, state)
            self._emit_pipeline_summary(
                source_dir=source_dir,
                state=state,
                stages_completed=stages_completed,
                phase_defs_total=len(phase_defs),
                total_time=time.time() - start_time,
            )
            return DrivePipelineResult(
                success=not refused,
                source_dir=source_dir,
                output_dir=project_dir,
                drive_plan=None,
                regular_lf_paths=[],
                shorts_per_lf={},
                stages_completed=stages_completed,
                errors=["ROUTE-CLAIM REFUSED: see route_claims_dry_run.json in the stages dir"]
                if refused
                else [],
                timings=timings,
            )

        try:
            for phase_name, phase_fn in phase_defs:
                if phase_name in resumed_phases:
                    if phase_name == "phase_a":
                        # Re-run the OST gate on resume — a checkpointed
                        # phase_a used to skip it entirely.
                        run_phase_a_ost_gate(state)
                        if list_lfs or label_dry_run:
                            return _print_labels_and_return(), ost_audit_pending
                    continue
                t0 = time.time()
                logger.info(
                    "PHASE START %s (%d/%d)", phase_name, len(stages_completed) + 1, len(phase_defs)
                )
                phase_fn(state)
                timings[phase_name] = time.time() - t0
                logger.info("PHASE DONE %s in %.0fs", phase_name, timings[phase_name])
                ok, reason = self._phase_produced_required_artifacts(phase_name, state)
                if ok:
                    stages_completed.append(phase_name)
                    # Persist Phase D fingerprint before phases_completed
                    # so a half-saved state never carries a phases_completed
                    # entry without a matching fingerprint.
                    if phase_name == "phase_d":
                        persist_phase_d_fingerprint(state, config=self.config)
                    # Stamp phase_a/phase_bc code versions so a later resume can
                    # detect a planner/audio/blur rule change and re-run.
                    persist_phase_code_version(state, phase_name)
                    if phase_name == "phase_bc_day_first":
                        write_lf_policy_stamp(state_file.parent, self.config.drive)
                    state.phases_completed = list(stages_completed)
                else:
                    # Silent failures (e.g. BC ran but produced 0
                    # day_artifacts) must NOT be marked complete — that
                    # locks the failure in across resume. Persist the
                    # state (so any partial work is checkpointed) but do
                    # not extend phases_completed.
                    logger.error(
                        "%s coherence guard tripped: %s — NOT appending to "
                        "phases_completed; resume will retry.",
                        phase_name,
                        reason,
                    )
                self._save_state(state_file, state)
                # A multi-hour drive must not let the worktree lease's TTL
                # lapse mid-run — renew it at every phase boundary, the same
                # cadence state is checkpointed at.
                _renew_worktree_lease(self._worktree_lease_handle)
                if phase_name == "phase_a":
                    # --list-lfs short-circuits after
                    # discovery. Print the LF index → suggested-label
                    # table and return success without entering phase_bc.
                    if list_lfs or label_dry_run:
                        return _print_labels_and_return(), ost_audit_pending
                    # In full_drive mode without balanced parts there
                    # is exactly one LF (index 1) by construction, so we can
                    # pass the skeleton index list ahead of phase_bc. With
                    # parts, or for best_moments, the count isn't known until
                    # phase_bc plans — gate_a falls back to printing the
                    # drive-wide label in that case.
                    skeleton_lf_indices: list[int] | None = None
                    if resolve_lf_mode(self.config.drive) == "full_drive":
                        skeleton_lf_indices = full_drive_skeleton_indices(
                            self.config.drive,
                            getattr(getattr(state, "drive_plan", None), "regular_lfs", None),
                        )
                    review.gate_a(
                        state=state,
                        output_dir=project_dir,
                        location=location,
                        skeleton_lf_indices=skeleton_lf_indices,
                    )
                    # OST uniqueness/validity gate: fail fast before encoding.
                    # Empty-with-episodes hard-fails inside the helper.
                    run_phase_a_ost_gate(state)
                elif phase_name == "phase_bc_day_first":
                    review.gate_b(state=state, output_dir=project_dir)
                # Gate C/D no longer runs here (AFTER phase_d finished its
                # uploads). It is injected INTO phase_d ahead of the upload
                # blocks via _phase_d_metadata_and_publish's pre_upload_gate.
        except ReviewFailedError as e:
            logger.error("Review gate rejected: %s", e)
            errors.append(f"ReviewFailedError: {e}")
            state.phases_completed = list(stages_completed)
            self._save_state(state_file, state)
        except OstAuditPendingError as e:
            # The per-day OST audit gate fires before any
            # apply_ost re-encode commits pixels. Treat as a clean
            # exit with success=True so the operator can review the
            # sidecar JSON and resume — NOT an error condition.
            # phase_bc_day_first is intentionally NOT appended to
            # ``stages_completed`` so the next resume re-checks the
            # gate.
            logger.info("OST audit gate pending: %s", e)
            self.waiting_gate = str(e)
            state.phases_completed = list(stages_completed)
            self._save_state(state_file, state)
            ost_audit_pending = True
        except Exception as e:
            import traceback as _tb

            logger.error(f"Drive pipeline failed: {e}\n{_tb.format_exc()}")
            errors.append(str(e))
            state.phases_completed = list(stages_completed)
            self._save_state(state_file, state)
        return None, ost_audit_pending

    def _resolve_paths(self, source_dir: Path, output_base: Path) -> tuple[Path, Path]:
        """Derive ``(project_dir, stages_base)`` from ``source_dir`` + ``output_base``.

        Creates ``output_base`` and ``project_dir`` on disk. Avoids
        double-nesting when ``output_base`` already ends with the project
        name (e.g. ``--output-dir /data/out/myproject`` + source
        ``MyProject``). ``stages_base`` resolves to a Linux-FS intermediate
        cache so drvfs-rooted runs don't stall on cross-mount IO.
        """
        output_base.mkdir(parents=True, exist_ok=True)
        project_name = source_dir.stem.replace(" ", "_").lower()
        already_nested = output_base.name.lower() == project_name
        project_dir = output_base if already_nested else output_base / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        from ..paths import gms_intermediate_dir

        # gms_intermediate_dir appends project_name/_stages — pass the parent
        # when output_base already contains the project name.
        stages_base = gms_intermediate_dir(
            output_base.parent if already_nested else output_base,
            project_name,
            intermediate_folder=self.config.paths.intermediate_folder,
        )
        return project_dir, stages_base

    def _build_initial_state(
        self,
        *,
        source_dir: Path,
        project_dir: Path,
        stages_base: Path,
        region: str,
        publish_day: str,
        location: str,
        fresh: bool,
        skip_upload: Optional[bool],
        skip_drive_upload: Optional[bool],
        skip_youtube_upload: Optional[bool],
        youtube_upload: Optional[bool],
        skip_csv: bool,
        start_date: Optional[str],
        lf_publish_date: Optional[str],
        force_no_rebase: Optional[bool],
        shorts_per_day: Optional[int],
        shorts_target: Optional[int],
        shorts_gap: Optional[float],
        keep_intermediates: Optional[bool],
        keep_masters: Optional[bool],
        allow_short_count: Optional[bool],
        skip_shorts_quality_gate: Optional[bool] = None,
        force_sidecar_reset: Optional[bool],
        allow_seo_degraded: Optional[bool],
        per_lf_overrides: Optional[dict],
        ost_audit_confirmed: Optional[bool],
        ost_system_review: Optional[bool] = None,
        gate_cd_confirmed: Optional[bool] = None,
        no_youtube_upload: Optional[bool] = None,
        no_ledger_reconcile: Optional[bool] = None,
        sign_verify: Optional[bool] = None,
        sign_verify_strict: Optional[bool] = None,
        skip_audio: Optional[bool] = None,
        lf_mode: Optional[str] = None,
        superlong_enabled: Optional[bool] = None,
        superlong_publish_day: Optional[str] = None,
        gps_captions: Optional[str] = None,
    ) -> tuple[PipelineState, bool, bool, bool]:
        """Construct the fresh ``PipelineState`` + coerce CLI sentinels.

        Returns ``(state, upload_flag_explicit, skip_upload, youtube_upload)`` —
        the three coerced booleans are also surfaced for the sidecar
        auto-skip safety net further down ``run()``.

        Argparse-sentinel pattern: the
        four upload flags use ``None`` = "operator did NOT supply" and
        ``True``/``False`` = "operator EXPLICITLY supplied". Tracking
        ``upload_flag_explicit`` separately lets the sidecar safety net
        only fire when the CLI was silent.
        """
        upload_flag_explicit = any(
            flag is not None
            for flag in (skip_upload, skip_drive_upload, skip_youtube_upload, youtube_upload)
        )
        skip_upload = bool(skip_upload) if skip_upload is not None else False
        skip_drive_upload = bool(skip_drive_upload) if skip_drive_upload is not None else False
        skip_youtube_upload = (
            bool(skip_youtube_upload) if skip_youtube_upload is not None else False
        )
        youtube_upload = bool(youtube_upload) if youtube_upload is not None else False
        state = PipelineState(
            source_dir=source_dir,
            output_base=project_dir,
            region=region,
            publish_day=publish_day,
            location=location,
            fresh=fresh,
            skip_upload=skip_upload,
            skip_csv=skip_csv,
            stages_base=stages_base,
        )
        # Split skip-upload into Drive vs YouTube. ``--skip-upload``
        # remains a back-compat alias that sets BOTH.
        state.skip_drive_upload = bool(skip_drive_upload or skip_upload)
        state.skip_youtube_upload = bool(skip_youtube_upload or skip_upload)
        # Registry-driven sentinel writes: per-flag
        # apply semantics (sentinel vs always) are declared in
        # ``flag_registry.FLAG_REGISTRY`` — including the config.drive mirrors
        # for shorts_target/shorts_gap (resume-authoritative). force_no_rebase is a
        # sentinel row; no_youtube_upload is ALWAYS_CLI (concrete every run).
        apply_cli_to_state(
            state,
            {
                "start_date": start_date,
                "lf_publish_date": lf_publish_date,
                "force_no_rebase": force_no_rebase,
                "shorts_per_day": shorts_per_day,
                "shorts_target": shorts_target,
                "shorts_gap": shorts_gap,
                "youtube_upload": youtube_upload,
                "per_lf_overrides": per_lf_overrides,
                "keep_intermediates": keep_intermediates,
                "keep_masters": keep_masters,
                "allow_short_count": allow_short_count,
                "skip_shorts_quality_gate": skip_shorts_quality_gate,
                "no_ledger_reconcile": no_ledger_reconcile,
                "sign_verify": sign_verify,
                "sign_verify_strict": sign_verify_strict,
                "force_sidecar_reset": force_sidecar_reset,
                "allow_seo_degraded": allow_seo_degraded,
                "ost_audit_confirmed": ost_audit_confirmed,
                "gate_cd_confirmed": gate_cd_confirmed,
                "no_youtube_upload": no_youtube_upload,
                "skip_audio": skip_audio,
                "lf_mode": lf_mode,
                "superlong_enabled": superlong_enabled,
                "superlong_publish_day": superlong_publish_day,
                "gps_captions": gps_captions,
            },
            config=self.config,
        )
        # Record whether --ost-audit-confirmed was supplied on THIS run.
        # Non-persisted (see _UNSERIALIZED_FIELDS): only a live flag bypasses the
        # OST gate without a hash — a resume relies on the hash-bound markers.
        state.ost_audit_confirmed_this_run = True if ost_audit_confirmed else None
        # --ost-system-review is per-invocation too (never persisted).
        state.ost_system_review_this_run = True if ost_system_review else None
        return state, upload_flag_explicit, skip_upload, youtube_upload

    def _run_dod_validator(
        self,
        *,
        state,
        output_base: Path,
        stages_base: Path,
        project_dir: Path,
        success: bool,
        ost_audit_pending: bool,
        source_dir: Path,
        stages_completed: list,
        phase_defs: list,
        total_time: float,
    ) -> None:
        """Run the terminal DoD validator and raise on failure.

        The validator lives OUTSIDE the phase-loop try/except so
        ``PipelineValidationError`` propagates to the caller. If the phase
        loop already failed (``success=False``) we do not run the validator —
        the root cause is already visible. ``ost_audit_pending=True`` also
        skips the gate (no LFs encoded yet → nothing to validate).
        ``config.drive.skip_validation`` is honoured only on a skip-upload
        run; on a live upload run it raises (GMS-17c).

        Single-phase path (``--phase phase_d``) MUST run the validator —
        skipping it would silently pass the
        audit→publish workflow.
        """
        gate_applies = success and not ost_audit_pending
        if not gate_applies:
            return
        if self.config.drive.skip_validation:
            # GMS-17c: honoured only on a skip-upload (audit/preview) run. On a
            # live upload run a skipped check must never read as a PASS.
            if not getattr(state, "skip_upload", False):
                raise PipelineValidationError(
                    "--skip-validation refused: this run uploads (no --skip-upload), "
                    "and the DoD validator gates every live upload run"
                )
            logger.warning("DoD validation SKIPPED via --skip-validation (skip-upload run)")
            return
        project_slug = project_dir.name
        snapshot_dir = stages_base.parent / "_snapshot_pre_fix"
        expected_lf_min = len(getattr(state, "regular_lf_paths", None) or []) or 1
        schedule = getattr(state, "drive_schedule", None) or {}
        lf_slots = schedule.get("lf_slots") if isinstance(schedule, dict) else None
        expected_weeks = max(1, (len(lf_slots or []) + 6) // 7)
        # full_drive LFs may legitimately run up to
        # ``config.drive.full_drive_max_duration_sec`` (default 7200s);
        # the default 1800s cap is calibrated for best_moments highlights.
        # Without this branch, every full_drive run >30 min hard-fails the DoD
        # duration check even though the planner accepted the duration upstream.
        if resolve_lf_mode(self.config.drive) == "full_drive":
            max_lf_duration = float(self.config.drive.full_drive_max_duration_sec)
        else:
            max_lf_duration = 1800.0
        # Superlong parts get their OWN ceiling (never the regular-LF one).
        superlong_max_lf_duration = float(self.config.drive.superlong_part_max_duration_sec) * (
            1.0 + float(self.config.drive.superlong_split_grace)
        )
        try:
            report = run_validation(
                project=project_slug,
                output_base=output_base,
                snapshot_dir=snapshot_dir,
                state_dir=stages_base,
                run_start=self._run_start_iso,
                expected_lf_min=expected_lf_min,
                expected_weeks=expected_weeks,
                max_lf_duration=max_lf_duration,
                superlong_max_lf_duration=superlong_max_lf_duration,
                youtube_uploads_sha=getattr(state, "youtube_uploads_sha", None),
                mode=ValidationMode.AUTO,
                state=state,
            )
        except Exception as exc:
            logger.error("DoD validator crashed: %s", exc)
            self._emit_pipeline_summary(
                source_dir=source_dir,
                state=state,
                stages_completed=stages_completed,
                phase_defs_total=len(phase_defs),
                total_time=total_time,
            )
            raise PipelineValidationError(
                f"DoD validator crashed before producing a report: {exc}"
            ) from exc
        logger.info(
            "DoD validation: %d/%d passed",
            report.passed_count,
            len(report.checks),
        )
        if report.failed_count > 0:
            self._emit_pipeline_summary(
                source_dir=source_dir,
                state=state,
                stages_completed=stages_completed,
                phase_defs_total=len(phase_defs),
                total_time=total_time,
            )
            raise PipelineValidationError(
                f"DoD validation failed ({report.failed_count} check(s)):\n{report.format_table()}"
            )

    def _get_prereqs(self) -> dict[str, list[str]]:
        """Phase prerequisite map, derived from the single phase-spec table."""
        return {s.name: list(s.prereqs) for s in PHASE_SPECS}

    def _select_phase_manifest(self, state_file):
        """Ordered full-run execution manifest, derived from the phase-spec
        table (preview-only phases excluded). ``fn`` is bound here so tests can
        patch the delegating seams before run()."""
        return [(s.name, getattr(self, s.fn_attr)) for s in PHASE_SPECS if s.in_manifest]
