"""Drive pipeline state persistence — save/load for resume capability.

Extracted from DrivePipelineOrchestrator so the phase modules can
save checkpoints without importing the full orchestrator.
"""

import dataclasses
import json
import logging
from pathlib import Path
from typing import Optional

from ._atomic_write import atomic_write_json

logger = logging.getLogger(__name__)

_KNOWN_PHASES = frozenset(
    {
        "phase_a",
        "phase_b_preview",
        "phase_bc_day_first",
        "phase_d",
    }
)

# Legacy state files may carry phase names from the removed
# non-day-first pipeline (phase_b / phase_c / phase_cd_per_day). The
# silent-remap shim that used to fold these into
# ``phase_bc_day_first`` was removed: any state old enough to
# carry those names is almost certainly mismatched against the current
# encoder/schedule contracts, so we now raise rather than continue.
_STALE_LEGACY_PHASES = frozenset({"phase_b", "phase_c", "phase_cd_per_day"})


def _is_numpy_array(v) -> bool:
    """Detect numpy ndarrays without forcing a numpy import at module load."""
    return type(v).__module__ == "numpy" and type(v).__name__ == "ndarray"


def _serialize_value(v):
    """Recursively convert a value to a JSON-serializable form."""
    if isinstance(v, Path):
        return str(v)
    elif isinstance(v, (str, int, float, bool, type(None))):
        return v
    elif _is_numpy_array(v):
        # Tagged marker so from_dict can reconstruct dtype + shape. Without
        # this branch the fallback at the bottom calls str(v) and the array
        # round-trips as a lossy "[[1. 2.]\n [3. 4.]]" string.
        return {
            "__numpy__": True,
            "dtype": str(v.dtype),
            "shape": list(v.shape),
            "data": v.tolist(),
        }
    elif isinstance(v, dict):
        return _serialize_dict(v)
    elif isinstance(v, set):
        return sorted(_serialize_value(item) for item in v)
    elif isinstance(v, tuple):
        # Preserve tuples as lists so JSON can represent them. Consumers
        # that destructure (e.g. file_boundaries as (start, end)) accept
        # any 2-element sequence, so this round-trips correctly.
        return [_serialize_value(item) for item in v]
    elif isinstance(v, list):
        return [_serialize_value(item) for item in v]
    elif hasattr(v, "to_dict"):
        return _serialize_value(v.to_dict())
    elif dataclasses.is_dataclass(v) and not isinstance(v, type):
        return _serialize_dict(dataclasses.asdict(v))
    else:
        logger.debug("_serialize_value: converting %s to str", type(v).__name__)
        return str(v)


def _serialize_dict(d: dict) -> dict:
    """Recursively convert a dict to JSON-serializable form (string keys)."""
    return {str(k): _serialize_value(v) for k, v in d.items()}


def _serialize_state(state: dict) -> dict:
    """Convert state dict to JSON-serializable form."""
    return _serialize_dict(state)


def save_drive_state(state_file: Path, state, *, skip_invariants: bool = False) -> None:
    """Save drive pipeline state for resume capability.

    Accepts either a ``PipelineState`` dataclass or a plain ``dict``.

    Before any disk work, the state is validated against the Layer-1
    invariants in ``state_invariants.assert_state_invariants``. A failure
    raises ``StateInvariantError`` and the state file is left untouched —
    the last-valid state on disk survives, so a subsequent resume starts
    from known-good data rather than partially-corrupt output of the
    buggy phase that produced the invalid state.

    Break-glass parameter (``skip_invariants``):
    a latent invariant bug could make every ``save_drive_state`` call
    raise mid-run, killing an 8-hour encode with no recovery short of
    ``--fresh``. Passing ``skip_invariants=True`` bypasses the gate, logs
    a WARNING, and writes whatever state the caller handed in. Wired to
    the ``gomoveshift drive --skip-invariants`` CLI flag. Never used in
    normal operation; emergency-only.
    """
    from .drive_state_model import PipelineState
    from .state_invariants import assert_state_invariants

    # Layer-1 check: refuse to write state that violates type / shape /
    # cross-field invariants. This runs BEFORE mkdir / serialize / tempfile
    # so a bad state produces no side effects on disk. The break-glass
    # path skips the check and logs a WARNING so operators see the
    # bypass in the pipeline log file even when no human is watching.
    if skip_invariants:
        logger.warning(
            "save_drive_state: skip_invariants=True — bypassing Layer-1 "
            "invariant check (%s). BREAK-GLASS mode; never use in normal "
            "operation.",
            state_file,
        )
    else:
        assert_state_invariants(state)

    if isinstance(state, PipelineState):
        serializable = state.to_dict()
    else:
        serializable = _serialize_state(state)
    atomic_write_json(state_file, serializable)


# A recovered ``*.tmp`` is only adopted as pipeline state when it structurally
# matches the PipelineState schema — it must be a JSON object carrying at least
# one of these signature keys. Every serialized ``PipelineState`` has one: the
# orchestration flags below all have non-``None`` defaults, so ``to_dict``
# always emits them. Sibling ``atomic_write_json`` writers do not — e.g. an
# orphaned drive_schedule.json ``.tmp`` (``{region, lf_slots, shorts_slots,
# computed_at}``), a per-LF ``*_youtube.json`` metadata ``.tmp`` (``{title,
# description, tags, ...}``), or an upload manifest (``{project_name, files,
# ...}``) carry none. ``region`` is deliberately NOT a signature key —
# ``DriveReleaseSchedule.to_dict()`` also writes it, so keying on it would still
# adopt a drive_schedule orphan. Without this filter such an orphan parses
# cleanly and hydrates a near-default state (empty phases_completed /
# youtube_uploads / OST prior hash) on a WARNING instead of the designed
# CorruptStateError — a resume silently becomes a fresh-ish run.
_STATE_SIGNATURE_KEYS = frozenset(
    {
        "phases_completed",
        "publish_day",
        "fresh",
        "skip_upload",
        "skip_csv",
        "skip_youtube_upload",
        "skip_drive_upload",
        "youtube_upload",
    }
)


def _looks_like_pipeline_state(data) -> bool:
    """True when a recovered ``.tmp`` dict structurally matches PipelineState."""
    return isinstance(data, dict) and not _STATE_SIGNATURE_KEYS.isdisjoint(data)


def _recover_from_tmp(state_file: Path) -> Optional[dict]:
    """Try to load state from a schema-matching ``.tmp`` sibling file.

    Only a ``.tmp`` whose JSON structurally matches the PipelineState schema is
    adopted (see ``_STATE_SIGNATURE_KEYS``); an orphaned sibling-writer ``.tmp``
    left by a crashed ``atomic_write_json`` call (drive_schedule.json etc.) is
    skipped so it can't silently hydrate a near-default state. When no
    schema-matching sibling is found, returns ``None`` and the caller raises
    ``CorruptStateError`` rather than resuming from bogus state.
    """
    tmp_candidates = sorted(state_file.parent.glob("*.tmp"), reverse=True)
    for tmp in tmp_candidates:
        try:
            with open(tmp) as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError, OSError):
            continue
        if not _looks_like_pipeline_state(data):
            logger.warning(
                "Skipping %s during state recovery — not a PipelineState schema", tmp.name
            )
            continue
        logger.warning("Recovered drive state from %s", tmp.name)
        return data
    logger.warning("No valid .tmp sibling found for %s — returning None", state_file)
    return None


def load_drive_state(state_file: Path):
    """Load drive pipeline state for resuming.

    Returns a ``PipelineState`` dataclass (via ``from_dict``) when
    deserialization succeeds, or ``None`` when the file doesn't exist.

    If the main state file is corrupt, attempts recovery from a ``.tmp``
    sibling left by the atomic write pattern. If that recovery also
    fails, raises :class:`~gomoveshift.pipeline.exceptions.CorruptStateError`
    rather than silently falling back to a fresh run. Callers must handle
    this loudly (CLI: non-zero exit + banner, instructing the operator to
    pass ``--fresh`` if they genuinely want to discard the prior state).
    """
    from .drive_state_model import PipelineState
    from .exceptions import CorruptStateError

    if not state_file.exists():
        return None
    try:
        with open(state_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("Drive state file corrupt (%s): %s — attempting .tmp recovery", state_file, e)
        data = _recover_from_tmp(state_file)
        if data is None:
            raise CorruptStateError(state_file, str(e)) from e
    except OSError as e:
        logger.error("Failed to load state file %s: %s", state_file, e)
        raise CorruptStateError(state_file, str(e)) from e

    # JSON-normalization fix: dicts keyed by LF index are written with int
    # keys (e.g. ``state["per_lf_locations"] = {1: "Beaufort County, SC", ...}``)
    # but JSON converts int dict keys to strings on serialize. On reload,
    # code that does ``d.get(lf.index, fallback)`` with ``lf.index`` as int
    # misses every entry and silently falls back. This defeats per-LF
    # metadata + SocialGenerator location wiring and any other
    # int-keyed resume path.
    #
    # Normalize known int-keyed dicts back to int so callers don't need to
    # care whether they're in a fresh run or a resume.
    from .drive_state_model import _INT_KEY_DICT_FIELDS

    for key in _INT_KEY_DICT_FIELDS:
        val = data.get(key)
        if isinstance(val, dict):
            normalized = {}
            for k, v in val.items():
                if isinstance(k, str) and k.isdigit():
                    normalized[int(k)] = v
                else:
                    normalized[k] = v
            data[key] = normalized

    # Validate phases_completed -- must be a list of known phase names.
    # Legacy state files carrying non-day-first phase names
    # (phase_b / phase_c / phase_cd_per_day) now raise rather than being
    # silently remapped — see _STALE_LEGACY_PHASES.
    if "phases_completed" in data:
        raw = data["phases_completed"]
        if not isinstance(raw, list):
            logger.warning(
                "State file %s: phases_completed is %s (not a list) -- ignoring",
                state_file,
                type(raw).__name__,
            )
            data["phases_completed"] = []
        else:
            stale = [p for p in raw if p in _STALE_LEGACY_PHASES]
            if stale:
                raise RuntimeError(
                    f"Stale legacy state file {state_file}: "
                    f"phases_completed contains deleted phase(s) {sorted(set(stale))!r}. "
                    "These names came from the removed non-day-first pipeline "
                    "and are not safe to silently remap into the day-first checkpoint. "
                    "Delete pipeline_state.json and re-run (or pass --fresh)."
                )
            unknown = [p for p in raw if p not in _KNOWN_PHASES]
            if unknown:
                logger.warning(
                    "State file %s: unknown phase(s) %s -- filtering out",
                    state_file,
                    unknown,
                )
            data["phases_completed"] = [p for p in raw if p in _KNOWN_PHASES]

    return PipelineState.from_dict(data)
