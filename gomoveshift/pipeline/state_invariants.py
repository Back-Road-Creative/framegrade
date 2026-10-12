"""Layer-1 state invariants for the drive pipeline.

The goal: prevent bad data from entering persisted state. The int-vs-str
key issue is fixed on LOAD — this module is the equivalent guard at WRITE
time. A single entry point, ``assert_state_invariants(state)``, is called
from ``save_drive_state`` BEFORE JSON serialization. Any violation raises
``StateInvariantError`` and the file is left untouched on disk, so the
last-valid state survives a bad write attempt.

Rules (all raise ``StateInvariantError`` with a grep-able prefix on fail):

1. **Int-keyed dicts.** The following dicts must be keyed by ``int``, never
   by string (catches the bug class mitigated on load):

   - ``per_lf_locations``
   - ``per_lf_titles``
   - ``shorts_per_lf``
   - ``social_posts_per_lf``
   - ``metadata_per_lf``
   - ``upload_manifests``
   - ``youtube_uploads``

   ``per_day_completed`` is explicitly excluded — its keys are date strings
   (``"2024-01-01"``). It gets its own rule: all keys must be ``str``.

2. **publish_day domain.** When set, ``state["publish_day"]`` must be one of
   ``{mon,tue,wed,thu,fri,sat,sun}`` AND the corresponding weekday int must
   appear in ``LONG_FORM_DAYS`` from ``schedule_constants``. Empty / unset
   passes (orchestrator init can have blanks).

2b. **lf_publish_date on the LF day.** When set, ``state["lf_publish_date"]``
   must be a ``YYYY-MM-DD`` date whose weekday is in ``LONG_FORM_DAYS`` and
   equals ``publish_day`` (when that is set). A saved anchor survives a
   resume, so this refuses an off-day anchor at launch, not after encodes.

3. **phases_completed subset.** Every entry in ``state["phases_completed"]``
   must be one of ``{phase_a, phase_b_preview, phase_bc_day_first, phase_d}``.
   Empty list / None pass.

4. **Shape of episode_* arrays.** ``episode_files``, ``episode_dates``,
   ``file_durations``, ``file_boundaries`` must agree on length. Only
   checked when all four are present — partial state from early phases
   is fine.

5. **Cross-field (phase_bc_day_first).** When ``phase_bc_day_first`` is in
   ``phases_completed``, ``regular_lf_paths`` must be a ``list`` (possibly
   empty) and ``shorts_per_lf`` must be a ``dict`` (possibly empty). The
   empty-list / empty-dict path is legitimate: a zero-qualifying-footage
   drive hits the early-exit in ``phase_bc_day_first`` that sets
   ``regular_lf_paths=[]`` and ``shorts_per_lf={}``. A former "non-empty
   list" rule killed every resume of a zero-LF drive mid-run.

   What Rule 5 still catches: ``None`` (field was never populated — the
   phase either didn't reach setdefault or was reset incorrectly) and
   non-list/non-dict types (type bugs). That's the real invariant.

Error messages always start with ``STATE_INVARIANT_ERROR_PREFIX`` so
operators can ``grep`` pipeline logs for this class of failure.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .exceptions import (
    PhaseBCExitContractError,
    PhaseDEntryContractError,
    StateCoherenceError,
    StateInvariantError,
)
from .preflight import check_disk_space, check_ram, check_tools

logger = logging.getLogger(__name__)

# Stable grep-able prefix for every invariant violation message.
# Operators use this to filter pipeline logs. Do NOT change it without
# flagging the break.
STATE_INVARIANT_ERROR_PREFIX = "state invariant violated:"

# Dicts that must be keyed by int. Excludes ``per_day_completed`` (date-string
# keys) — see ``_STR_KEYED_DICTS``.
_INT_KEYED_DICTS: tuple[str, ...] = (
    "per_lf_locations",
    "per_lf_titles",
    "shorts_per_lf",
    "social_posts_per_lf",
    "metadata_per_lf",
    "upload_manifests",
    "youtube_uploads",
    "per_lf_recording_date",
    "per_lf_trim_ratio",
    "per_lf_gpx_path",
    "per_lf_resolved_location",
)

# Dicts that must be keyed by str. ``per_day_completed`` stores per-day
# master bookkeeping keyed by ``"YYYY-MM-DD"``. ``per_lf_state_recovered_from_disk``
# is the resume-time backstop observability dict — keys are field names like ``"per_lf_trim_ratio"``
# / ``"file_boundaries"``; values are ``list[int]`` of LF indices recovered
# (``[-1]`` sentinel = whole-state recovery).
_STR_KEYED_DICTS: tuple[str, ...] = ("per_day_completed", "per_lf_state_recovered_from_disk")

# Short day names → Python weekday int (Mon=0 … Sun=6). Sourced from
# ``scheduler.DAY_NAMES`` but re-declared here to keep state_invariants a
# leaf module (no back-imports into scheduler).
_DAY_NAME_TO_INT: dict[str, int] = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}

# The four phase names the orchestrator knows about. Matches the frozenset
# in ``_drive_state._KNOWN_PHASES`` — kept as a tuple here for error message
# composition (``sorted`` prints deterministically).
_KNOWN_PHASES: frozenset[str] = frozenset(
    {"phase_a", "phase_b_preview", "phase_bc_day_first", "phase_d"}
)

# The four episode_* arrays whose lengths must agree.
_EPISODE_SHAPE_FIELDS: tuple[str, ...] = (
    "episode_files",
    "episode_dates",
    "file_durations",
    "file_boundaries",
)


def _state_get(state: Any, key: str) -> Any:
    """Read ``key`` from a PipelineState dataclass OR a plain dict.

    ``save_drive_state`` accepts both shapes (the dataclass,
    plus legacy callers + tests that pass raw dicts directly), so this
    helper has to tolerate either. Returns ``None`` for absent
    attributes/keys to match the historical contract.
    """
    if state is None:
        return None
    if isinstance(state, dict):
        return state.get(key)
    return getattr(state, key, None)


def _raise(message: str) -> None:
    """Raise StateInvariantError with the standard grep-able prefix."""
    raise StateInvariantError(f"{STATE_INVARIANT_ERROR_PREFIX} {message}")


def _check_int_keyed_dicts(state: Any) -> None:
    """Rule 1: dicts in ``_INT_KEYED_DICTS`` must use ``int`` keys."""
    for key in _INT_KEYED_DICTS:
        val = _state_get(state, key)
        if val is None:
            continue
        if not isinstance(val, dict):
            _raise(f"{key} must be a dict or None, got {type(val).__name__}")
        for k in val:
            if not isinstance(k, int) or isinstance(k, bool):
                # bool is an int subclass in Python — treat it as non-int
                # for this rule since True/False aren't LF indices.
                _raise(f"{key} has non-int key {k!r} (type {type(k).__name__})")


def _check_str_keyed_dicts(state: Any) -> None:
    """Rule 1b: ``per_day_completed`` must use ``str`` keys (date strings)."""
    for key in _STR_KEYED_DICTS:
        val = _state_get(state, key)
        if val is None:
            continue
        if not isinstance(val, dict):
            _raise(f"{key} must be a dict or None, got {type(val).__name__}")
        for k in val:
            if not isinstance(k, str):
                _raise(f"{key} has non-str key {k!r} (type {type(k).__name__})")
    # Value-shape gate for the backstop observability dict: every value
    # must be ``list[int]`` (LF indices recovered, ``[-1]`` whole-state
    # sentinel).
    val = _state_get(state, "per_lf_state_recovered_from_disk")
    if isinstance(val, dict):
        for k, v in val.items():
            if not isinstance(v, list):
                _raise(
                    f"per_lf_state_recovered_from_disk[{k!r}] must be list[int], "
                    f"got {type(v).__name__}"
                )
            for item in v:
                if not isinstance(item, int) or isinstance(item, bool):
                    _raise(
                        f"per_lf_state_recovered_from_disk[{k!r}] contains "
                        f"non-int element {item!r} (type {type(item).__name__})"
                    )


# ─── backstop visibility helper ────────────────────────────────────────
#
# Resume-time backstops were silent — operators had no signal when a
# phase boundary got breached and recovery kicked in. This single helper
# is invoked from each fire-point (phase_bc trim_ratio + recording_date,
# drive_orchestrator file_boundaries)
# so the WARN format and the ``per_lf_state_recovered_from_disk`` state
# field stay in lockstep. No behavior change beyond observability.

BACKSTOP_FIRED_PREFIX = "backstop fired:"


def record_backstop_recovery(
    state: Any,
    field: str,
    lf_indices: list[int],
    reason: str,
) -> None:
    """Emit a WARN log + populate ``state["per_lf_state_recovered_from_disk"]``.

    ``state`` may be a ``PipelineState`` or a plain dict (both support
    ``setdefault`` / ``__setitem__`` via the same dict protocol). The
    field name is grep-able via ``BACKSTOP_FIRED_PREFIX`` — operators
    filter pipeline logs for that string to find every silent recovery.

    Sentinel: ``lf_indices=[-1]`` indicates a whole-state recovery
    (``file_boundaries`` reconstruction)
    rather than a per-LF gap fill.
    """
    logger.warning(
        "%s field=%s indices=%s reason=%s",
        BACKSTOP_FIRED_PREFIX,
        field,
        lf_indices,
        reason,
    )
    if state is None:
        return
    existing = _state_get(state, "per_lf_state_recovered_from_disk") or {}
    if not isinstance(existing, dict):
        existing = {}
    # Merge — multiple resumes in one process append rather than clobber.
    prior = list(existing.get(field, []))
    merged = sorted({*prior, *lf_indices})
    existing[field] = merged
    try:
        if isinstance(state, dict):
            state["per_lf_state_recovered_from_disk"] = existing
        else:
            state.per_lf_state_recovered_from_disk = existing
    except (TypeError, AttributeError):
        # Read-only state — skip silently; the WARN log is the contract.
        return


def _check_publish_day(state: Any) -> None:
    """Rule 2: publish_day ∈ day-name set AND corresponds to LONG_FORM_DAYS."""
    # Imported lazily so tests can monkeypatch ``schedule_constants.LONG_FORM_DAYS``.
    from . import schedule_constants

    publish_day = _state_get(state, "publish_day")
    if publish_day is None or publish_day == "":
        return
    if not isinstance(publish_day, str):
        _raise(
            f"publish_day must be str or empty, got {type(publish_day).__name__} {publish_day!r}"
        )
    pd_key = publish_day.lower()[:3]
    if pd_key not in _DAY_NAME_TO_INT:
        _raise(f"publish_day {publish_day!r} not in {{{','.join(sorted(_DAY_NAME_TO_INT))}}}")
    pd_int = _DAY_NAME_TO_INT[pd_key]
    long_form_days = schedule_constants.LONG_FORM_DAYS
    if pd_int not in long_form_days:
        _raise(
            f"publish_day {publish_day!r} (weekday={pd_int}) not in "
            f"LONG_FORM_DAYS={sorted(long_form_days)}"
        )


def lf_anchor_problem(
    anchor: Any, publish_day: Any, long_form_days: Any, flag: str = "--lf-publish-date"
) -> str | None:
    """Why ``anchor`` cannot be the run's LF date, or ``None`` when it can.

    The weekday decision is ``seo_manager.scheduling.slots.validate_anchor`` (the one
    place a date is judged against a cadence); this adds only the message the operator
    acts on. Shared by Rule 2b (state save) and the launch-time gate (``cmd_drive``),
    so a saved anchor and a CLI anchor are refused with the same words.
    """
    if anchor is None or anchor == "":
        return None
    label = "lf_publish_date" if flag == "--lf-publish-date" else flag
    if not isinstance(anchor, str):
        return f"{label} {anchor!r} is not a YYYY-MM-DD date"
    from datetime import datetime as _dt

    try:
        canonical = _dt.strptime(anchor, "%Y-%m-%d").strftime("%Y-%m-%d") == anchor
    except ValueError:
        canonical = False
    if not canonical:  # strptime takes "2026-10-9"; downstream isoformat parsers do not
        return f"{label} {anchor!r} is not a YYYY-MM-DD date"
    pd_int = None
    if isinstance(publish_day, str) and publish_day.lower()[:3] in _DAY_NAME_TO_INT:
        pd_int = _DAY_NAME_TO_INT[publish_day.lower()[:3]]
    allowed = frozenset(long_form_days)
    if pd_int is not None:
        allowed = allowed & {pd_int}
    from .seo_slots import anchor_problems

    problems = anchor_problems([anchor], allowed)
    if not problems:
        return None
    weekday = problems[0][1]
    if weekday is None:  # not a date at all (seo_manager's own parse)
        return f"{label} {anchor!r} is not a YYYY-MM-DD date"
    return (
        f"{label} {anchor!r} (weekday={weekday}) is off the run's LF day "
        f"(publish_day {publish_day!r}, LONG_FORM_DAYS={sorted(long_form_days)}); "
        f"a saved anchor survives a resume — re-run with {flag} "
        "<next free slot from 'gomoveshift schedule'> on that day"
    )


def _check_lf_publish_date(state: Any) -> None:
    """Rule 2b: lf_publish_date is YYYY-MM-DD on the run's LF day."""
    from . import schedule_constants

    problem = lf_anchor_problem(
        _state_get(state, "lf_publish_date"),
        _state_get(state, "publish_day"),
        schedule_constants.LONG_FORM_DAYS,
    )
    if problem:
        _raise(problem)


def _check_phases_completed(state: Any) -> None:
    """Rule 3: phases_completed ⊆ known phases."""
    phases = _state_get(state, "phases_completed")
    if phases is None:
        return
    if not isinstance(phases, list):
        _raise(f"phases_completed must be a list or None, got {type(phases).__name__}")
    unknown = [p for p in phases if p not in _KNOWN_PHASES]
    if unknown:
        _raise(
            f"phases_completed contains unknown phase(s) {unknown!r} "
            f"(known: {sorted(_KNOWN_PHASES)})"
        )


def _check_episode_shape(state: Any) -> None:
    """Rule 4: episode_* arrays agree on length when all four present."""
    values = {f: _state_get(state, f) for f in _EPISODE_SHAPE_FIELDS}
    if any(v is None for v in values.values()):
        return  # partial state — skip
    lengths = {f: len(v) for f, v in values.items()}
    if len(set(lengths.values())) != 1:
        _raise(f"episode_* arrays have mismatched length: {lengths!r} (all four must agree)")


def _check_phase_bc_cross_field(state: Any) -> None:
    """Rule 5: phase_bc_day_first in phases_completed → regular_lf_paths is a list.

    Accepts an empty list (zero-qualifying-footage drive — see
    the ``phase_bc_day_first`` early-exit). Rejects ``None`` (field never
    populated) and non-list types (type bug). A former "non-empty list"
    rule made every save of a zero-LF drive raise, killing the resume path.
    """
    phases = _state_get(state, "phases_completed") or []
    if "phase_bc_day_first" not in phases:
        return
    lf_paths = _state_get(state, "regular_lf_paths")
    if lf_paths is None:
        _raise(
            "phase_bc_day_first marked complete but regular_lf_paths is "
            "None (expected list, possibly empty for a zero-qualifying-"
            "footage drive)"
        )
    if not isinstance(lf_paths, list):
        _raise(
            "phase_bc_day_first marked complete but regular_lf_paths is "
            f"{type(lf_paths).__name__} (expected list, possibly empty)"
        )
    # shorts_per_lf is intentionally permissive. One could require
    # `shorts_per_lf populated`, but a short-light day can produce zero
    # renderable shorts for every LF (all segments below
    # `min_shorts_duration`). We require the field to at least exist as a
    # dict — a `None` here under completed phase_bc is a real bug. An
    # empty dict is fine.
    shorts = _state_get(state, "shorts_per_lf")
    if shorts is None:
        _raise(
            "phase_bc_day_first marked complete but shorts_per_lf is None "
            "(expected dict, possibly empty if every LF had zero renderable shorts)"
        )
    if not isinstance(shorts, dict):
        _raise(f"shorts_per_lf must be a dict, got {type(shorts).__name__}")


def assert_state_invariants(state: Any) -> None:
    """Raise StateInvariantError if ``state`` violates any Layer-1 invariant.

    Called from ``_drive_state.save_drive_state`` before JSON serialization.
    Accepts either a ``PipelineState`` dataclass or a raw ``dict`` — both
    support the same read interface used by ``_state_get``.

    Rules are checked in order; the first failure raises. We do not
    accumulate-and-report because a) write-time invariants should be cheap
    to satisfy (catch-and-fix, not a quality report), and b) the first
    failing rule is usually the root cause — downstream rules may depend on
    the failing field's shape, so reporting all of them would be noisy.
    """
    _check_int_keyed_dicts(state)
    _check_str_keyed_dicts(state)
    _check_publish_day(state)
    _check_lf_publish_date(state)
    _check_phases_completed(state)
    _check_episode_shape(state)
    _check_phase_bc_cross_field(state)


# ─── BC→D per-LF boundary contract ─────────────────────────────────────
#
# The per-field assertion family in ``phase_bc_day_first.py``
# (``_assert_per_lf_trim_ratio_complete`` et al.) catches one field at a
# time. The aggregators below wrap them into ONE named contract checked
# symmetrically at BC exit and D entry — operators see every missing
# per-LF field in a single error, and Phase D can read ``state["per_lf_*"]``
# with ``[]`` rather than ``.get(...) or {}`` because the contract
# guarantees presence by construction.

# Fields required for every rendered LF index, regardless of GPS availability.
# Each is populated by Phase BC before exit; Phase D consumes each downstream.
# Universal per-LF fields. The bool is the strict-mode flag: ``True`` means
# an empty mapping is itself a contract violation (the field must cover
# every rendered LF unconditionally — used for fields BC always populates
# in production). ``False`` means partial-coverage is the only failure mode
# (an entirely-empty mapping is tolerated, e.g. GPS-less drives / degenerate
# BC fast-paths that leave the upstream computation no input).
_BC_EXIT_UNIVERSAL_FIELDS: tuple[tuple[str, bool], ...] = (
    ("per_lf_recording_date", True),  # strict — BC raises on missing anyway
    ("per_lf_ost_text", False),
    ("per_lf_titles", False),
    ("per_lf_locations", False),
    ("per_lf_trim_ratio", False),
)

# Fields populated only on GPS-bearing drives — they key on
# ``per_lf_gpx_path`` (each LF with a non-empty GPX path must have an
# entry). GPS-less drives leave ``per_lf_gpx_path`` empty and these
# fields stay empty too; the contract no-ops accordingly.
_BC_EXIT_GPX_KEYED_FIELDS: tuple[str, ...] = (
    "per_lf_distance_km",
    "per_lf_elevation_gain_m",
    "per_lf_regions",
)


def _coerce_int_keys(mapping: Any) -> set[int]:
    have: set[int] = set()
    for k in mapping or {}:
        try:
            have.add(int(k))
        except (TypeError, ValueError):
            continue
    return have


def _missing_for_field(
    field_name: str,
    expected: set[int],
    mapping: Any,
    *,
    strict: bool = False,
) -> list[int]:
    """Return sorted LF indices in ``expected`` missing from ``mapping``.

    By default (``strict=False``), an entirely-empty mapping is tolerated
    (legitimate for GPS-less drives and degenerate BC fast-paths) — only
    a partially-populated mapping that fails to cover every expected
    index counts as a violation. ``strict=True`` enforces full coverage
    even when the mapping is empty; used for fields that BC unconditionally
    populates in production (``per_lf_recording_date``, ``per_lf_trim_ratio``
    when prerequisites are met).

    For ``per_lf_trim_ratio`` we additionally treat ``None`` values as
    missing (the historical sentinel for "unrecoverable source clip
    metadata" in the resume-from-disk path).
    """
    if not expected:
        return []
    if field_name == "per_lf_trim_ratio" and isinstance(mapping, dict):
        mapping = {k: v for k, v in mapping.items() if v is not None}
    have = _coerce_int_keys(mapping)
    if not have and not strict:
        # Entirely-empty mapping is tolerated for non-strict fields
        # (GPS-less drive / degenerate BC path / test fixture).
        return []
    return sorted(expected - have)


def _gpx_keyed_expected(per_lf_gpx_path: Any) -> set[int]:
    expected: set[int] = set()
    for k, v in (per_lf_gpx_path or {}).items():
        if not v:
            continue
        try:
            expected.add(int(k))
        except (TypeError, ValueError):
            continue
    return expected


def _gpx_keyed_per_field_missing(state: Any, gpx_expected: set[int]) -> dict[str, list[int]]:
    """Per-field coverage check for the GPX-keyed maps, EXCLUDING the indices
    the upstream gate legitimately omitted.

    ``state["per_lf_gpx_omitted"]`` (``{field: [lf_index, ...]}``) names LFs
    whose cleaned trace yielded no usable distance (stationary/over-filtered)
    or was plausibility-rejected — a deliberate omission, NOT missing data. We
    subtract them per-field so the aggregate boundary contract agrees with the
    per-field invariant (which already tolerated them); else it re-fails on the
    same LF. Shared by the BC-exit and D-entry aggregators so they stay
    symmetric.
    """
    omitted_map = _state_get(state, "per_lf_gpx_omitted") or {}
    out: dict[str, list[int]] = {}
    for field in _BC_EXIT_GPX_KEYED_FIELDS:
        omitted = _coerce_int_keys(omitted_map.get(field) or [])
        field_expected = gpx_expected - omitted
        missing = _missing_for_field(
            field, field_expected, _state_get(state, field), strict=bool(field_expected)
        )
        if missing:
            out[field] = missing
    return out


def _skeleton_expected_keys(skeleton_lfs: Any) -> set[int]:
    expected: set[int] = set()
    for lf in skeleton_lfs or []:
        idx = getattr(lf, "index", None)
        if idx is None:
            continue
        try:
            expected.add(int(idx))
        except (TypeError, ValueError):
            continue
    return expected


def _format_boundary_error(
    *,
    phase_label: str,
    n_lfs: int,
    per_field_missing: dict[str, list[int]],
) -> str:
    lines = [f"{phase_label}: per-LF boundary contract failed (rendered {n_lfs} LF(s))."]
    for field, missing in per_field_missing.items():
        lines.append(f"  - {field}: missing entries for LF indices {missing}")
    lines.append("Every required per_lf_* map must cover every rendered LF index.")
    return "\n".join(lines)


def assert_phase_bc_per_lf_complete(state: Any, skeleton_lfs: Any) -> None:
    """Phase BC exit aggregator: every per-LF map covers every rendered LF.

    Wraps the per-field assertion family into ONE named contract — a single
    raise of :class:`PhaseBCExitContractError` lists EVERY missing field
    with its missing LF indices, so operators repair the state file (or
    rerun the upstream phase) in one round.

    ``skeleton_lfs`` is the authoritative LF roster at this boundary
    (matches ``regular_lf_paths`` 1:1). GPS-less drives leave
    ``per_lf_gpx_path`` empty; the GPX-keyed sub-checks become vacuous.
    """
    expected = _skeleton_expected_keys(skeleton_lfs)
    if not expected:
        return  # zero-qualifying-footage drive: contract is vacuous.

    per_field_missing: dict[str, list[int]] = {}
    for field, strict in _BC_EXIT_UNIVERSAL_FIELDS:
        missing = _missing_for_field(field, expected, _state_get(state, field), strict=strict)
        if missing:
            per_field_missing[field] = missing

    gpx_expected = _gpx_keyed_expected(_state_get(state, "per_lf_gpx_path"))
    per_field_missing.update(_gpx_keyed_per_field_missing(state, gpx_expected))

    if per_field_missing:
        raise PhaseBCExitContractError(
            _format_boundary_error(
                phase_label="phase_bc_day_first exit",
                n_lfs=len(expected),
                per_field_missing=per_field_missing,
            )
        )


def assert_phase_d_per_lf_inputs_present(state: Any) -> None:
    """Phase D entry aggregator: symmetrical to BC exit, from D's POV.

    Reads the LF roster from ``state["drive_plan"].regular_lfs`` rather
    than ``skeleton_lfs`` (D doesn't carry the BC-internal name). Covers
    the resume-from-corrupted-state case where state on disk was written
    by an older revision and bypassed the BC-exit aggregator. Raises
    :class:`PhaseDEntryContractError` with the same single-shot aggregate
    framing as the BC counterpart.
    """
    drive_plan = _state_get(state, "drive_plan")
    if drive_plan is None:
        return  # empty/missing plan → upstream guards handle it
    regular_lfs = getattr(drive_plan, "regular_lfs", None)
    if not regular_lfs:
        return  # zero-LF drive — D's empty-plan branch handles it.

    expected = _skeleton_expected_keys(regular_lfs)
    if not expected:
        return

    per_field_missing: dict[str, list[int]] = {}
    for field, strict in _BC_EXIT_UNIVERSAL_FIELDS:
        missing = _missing_for_field(field, expected, _state_get(state, field), strict=strict)
        if missing:
            per_field_missing[field] = missing

    gpx_expected = _gpx_keyed_expected(_state_get(state, "per_lf_gpx_path"))
    per_field_missing.update(_gpx_keyed_per_field_missing(state, gpx_expected))

    if per_field_missing:
        raise PhaseDEntryContractError(
            _format_boundary_error(
                phase_label="phase_d entry",
                n_lfs=len(expected),
                per_field_missing=per_field_missing,
            )
        )


# ─── Layer-1.5: resume-time coherence ──────────────────────────────────
#
# ``assert_state_invariants`` runs on every save. The coherence rule below
# runs ONCE at orchestrator startup, after state load and after any
# auto-populate hook. It checks cross-field consistency that doesn't
# fit the per-save invariant pattern (mid-run saves can legitimately have
# phases-marked-complete without per_day_completed populated yet — the
# bookkeeping happens together at end-of-phase, not atomically with the
# phase-name append).
#
# Guards the case where ``phases_completed`` lists ``phase_bc_day_first`` but
# ``per_day_completed`` does not cover every day, which would skip Phase B+C
# on resume.


def _expected_dates_for_phase_bc(state: Any) -> list[str] | None:
    """Resolve the set of dates Phase BC was expected to process.

    Returns ``None`` (skip the check) when no source of truth is available —
    e.g. very early state before ``ingest_plan`` / ``episode_dates`` exist.
    The check only fires when there's a concrete plan to compare against;
    a missing plan is not itself drift.

    Resolution order (most authoritative first):

    1. ``state["episode_dates"]`` — populated by Phase A; one entry per
       processed file. We dedupe + drop falsy values.
    2. ``state["drive_plan"]`` — accepted as either a ``DrivePlan`` instance
       or a ``dict`` (resume from disk). The ``DrivePlan`` dataclass does
       not currently carry per-day dates as a first-class field, so we read
       ``getattr(..., "dates", None)`` first (forward-compat hook) and fall
       back to ``None`` if absent. We do NOT attempt to scrape dates from
       segment ``source_path`` strings — that's a heuristic that would
       silently drift from the truth.
    3. ``state["ingest_plan"]`` — last resort: walk episodes for
       ``creation_time`` / ``date`` keys. Mirrors ``day_grouper._episode_date_key``
       so the expected-dates set matches what Phase BC would actually
       process.

    Each branch returns ``None`` if it can't produce a non-empty list,
    causing the caller to skip the check rather than false-positive.
    """
    # 1. episode_dates (preferred — directly comparable to per_day_completed keys).
    ep_dates = _state_get(state, "episode_dates")
    if isinstance(ep_dates, list) and ep_dates:
        unique = sorted({d for d in ep_dates if isinstance(d, str) and d.strip()})
        if unique:
            return unique

    # 2. drive_plan — instance or dict.
    drive_plan = _state_get(state, "drive_plan")
    if drive_plan is not None:
        # Forward-compat: if a future DrivePlan grows a `.dates` accessor or a
        # serialized dict gains a "dates" key, honor it.
        plan_dates = None
        if hasattr(drive_plan, "dates"):
            plan_dates = getattr(drive_plan, "dates", None)
        elif isinstance(drive_plan, dict):
            plan_dates = drive_plan.get("dates")
        if isinstance(plan_dates, list) and plan_dates:
            unique = sorted({d for d in plan_dates if isinstance(d, str) and d.strip()})
            if unique:
                return unique

    # 3. ingest_plan — walk episodes.
    ingest_plan = _state_get(state, "ingest_plan")
    episodes = None
    if ingest_plan is not None:
        if hasattr(ingest_plan, "episodes"):
            episodes = getattr(ingest_plan, "episodes", None)
        elif isinstance(ingest_plan, dict):
            episodes = ingest_plan.get("episodes")
    if episodes:
        from datetime import datetime

        derived: set[str] = set()
        for ep in episodes:
            ct = (
                getattr(ep, "creation_time", None)
                if not isinstance(ep, dict)
                else ep.get("creation_time")
            )
            if isinstance(ct, datetime):
                derived.add(ct.strftime("%Y-%m-%d"))
                continue
            date_str = getattr(ep, "date", None) if not isinstance(ep, dict) else ep.get("date")
            if isinstance(date_str, str) and date_str.strip():
                derived.add(date_str.strip())
        if derived:
            return sorted(derived)

    return None


def assert_resume_state_coherent(state: Any) -> None:
    """Raise StateCoherenceError if loaded state has cross-field drift.

    Called from :meth:`DrivePipelineOrchestrator.run` immediately after
    ``_try_resume`` (and after any auto-populate hook). Fires once per run.

    Current rules:

    0. **phase_bc_day_first ⇒ day_artifacts is non-empty.**
       If BC is marked complete but ``day_artifacts`` is empty / missing,
       BC ran but produced 0 day-master encodes (the silent-failure mode
       of a run where every day failed mid-encode). Raises immediately with operator
       guidance — pre-empts the older per-date check below since with
       zero day_artifacts the per-day check is moot.

    1. **phase_bc_day_first ⇒ per_day_completed covers expected dates.**
       If ``phase_bc_day_first`` is in ``phases_completed``, every date the
       phase was expected to process must have a truthy entry in
       ``per_day_completed``. The expected date set is resolved by
       :func:`_expected_dates_for_phase_bc` — see its docstring for the
       resolution order. If no expected-date source is available, the check
       is a no-op (logged at DEBUG).

    Behavior on failure: raise :class:`StateCoherenceError` with the
    spurious phase, the missing dates, and a one-line operator hint. Do
    NOT silently auto-correct — the upstream bug that wrote the spurious
    phase needs to surface.
    """
    phases = _state_get(state, "phases_completed") or []
    if not isinstance(phases, list):
        # Type bug — defer to assert_state_invariants Rule 3 to catch it.
        # Coherence check only runs on shape-valid state.
        return

    if "phase_bc_day_first" not in phases:
        return

    expected = _expected_dates_for_phase_bc(state)

    # BC ∈ phases_completed and there WERE expected dates ⇒
    # at least one of (day_artifacts, per_day_completed) must show
    # completed work. The failure this catches: BC ran, every
    # day failed mid-encode, ``run_all_days`` returned []; the
    # orchestrator still appended BC to phases_completed; resume then
    # skipped BC and Phase D failed on missing LF artifacts.
    #
    # Both signals are accepted to stay backward-compatible with state
    # files written before ``day_artifacts`` was tracked:
    # ``per_day_completed`` is written by the same code path and is
    # sufficient evidence of completion. Only the *both empty* shape
    # combined with expected work raises here.
    #
    # Skipped intentionally when ``expected`` is empty/None — the
    # zero-qualifying-footage early-exit at ``phase_bc_day_first.py``
    # legitimately marks BC complete with empty day_artifacts AND empty
    # per_day_completed. See Rule 5 of ``assert_state_invariants``.
    if expected:
        day_artifacts = _state_get(state, "day_artifacts") or []
        per_day_completed_for_empty_check = _state_get(state, "per_day_completed") or {}
        if not day_artifacts and not per_day_completed_for_empty_check:
            raise StateCoherenceError(
                f"state coherence violated: 'phase_bc_day_first' in "
                f"phases_completed and {len(expected)} date(s) were expected "
                f"({expected!r}), but BOTH day_artifacts and per_day_completed "
                f"are empty. "
                f"Likely cause: phase_bc_day_first ran but produced 0 day "
                f"artifacts (e.g. NVENC mid-encode failure on every day) and "
                f"a prior revision of the orchestrator marked the phase "
                f"complete anyway. "
                f"To recover: remove 'phase_bc_day_first' from phases_completed "
                f"in the state file (or wipe state to force a clean resume)."
            )
    if not expected:
        logger.debug(
            "state coherence check: phase_bc_day_first marked complete but no "
            "expected-date source available (no episode_dates, drive_plan.dates, "
            "or ingest_plan.episodes). Skipping check."
        )
        return

    per_day = _state_get(state, "per_day_completed") or {}
    if not isinstance(per_day, dict):
        # Type bug — defer to assert_state_invariants Rule 1b to catch it.
        return

    missing = [d for d in expected if not per_day.get(d)]
    if not missing:
        return

    raise StateCoherenceError(
        f"state coherence violated: 'phase_bc_day_first' in phases_completed "
        f"but per_day_completed is missing {len(missing)} of {len(expected)} "
        f"expected date(s): {missing!r}. "
        f"Likely cause: a prior failed phase recorded itself as complete on "
        f"zero work. To recover: remove 'phase_bc_day_first' from "
        f"phases_completed in the state file (or wipe state to force a "
        f"clean resume)."
    )


# ─── resume_invariants — single post-load entry point ───────────────────
#
# The `if "phase_a" not in resumed_phases:` guard around disk/RAM preflight
# in ``drive_orchestrator.run`` silently exempts every resume from the
# resource gate, so a resume into Phase D would get no disk check.
#
# The fix is structural: ONE function — ``resume_invariants(state, config)``
# — runs immediately after ``_try_resume`` returns a non-empty resume set.
# It folds every cross-phase invariant the team has accreted as one-offs
# (disk, RAM, ffmpeg, sidecar integrity, OST audit hash drift, per-LF
# completeness) into a single gate so the resume path cannot diverge
# from the fresh-run path.
#
# This gate is assembly + a single call site, NOT new invariants. It
# prefers raising over silently filling per-field re-derivation backstops.


def _check_resume_preflight(state: Any) -> None:
    """Disk + RAM + ffmpeg/ffprobe + NVENC check on every resume.

    The legacy guard in ``drive_orchestrator`` skipped these when phase_a
    was already complete; this function exists to make sure a resume into
    a later phase still validates the environment. Disk targets are the
    same as ``run_preflight``: ≥50 GB at output_base, ≥30 GB at staging
    (when known), ≥2 GB RAM, ffmpeg+ffprobe on PATH.

    When ``phase_bc_day_first`` has not yet
    completed, also probe NVENC up-front. Driver resets between runs are
    common; without this gate the failure surfaced hours into encode
    instead of at resume entry. ``detect_nvenc`` raises
    ``NVENCUnavailableError`` (a ``RuntimeError`` subclass) on probe
    failure — we let it propagate; the message already states the
    recovery step.

    Raises ``RuntimeError`` on any failure (matches ``run_preflight``).
    """
    output_base = _state_get(state, "output_base")
    if output_base is not None:
        check_disk_space(Path(output_base), min_free_gb=50)
    stages_base = _state_get(state, "stages_base")
    if stages_base is not None:
        check_disk_space(Path(stages_base), min_free_gb=30)
    check_ram(min_free_gb=2.0)
    check_tools(("ffmpeg", "ffprobe"))

    # NVENC gate: only when encoding is still ahead of us. Once
    # phase_bc_day_first is in phases_completed every LF / short / master
    # is already on disk and no further hevc_nvenc invocation happens —
    # gating then would needlessly block a Phase-D-only resume on a
    # driver hiccup that doesn't matter.
    phases = _state_get(state, "phases_completed") or []
    if "phase_bc_day_first" in phases:
        return
    # Lazy import to keep state_invariants a leaf module relative to
    # color_filters (which transitively imports heavy ffmpeg utilities).
    from .color_filters import detect_nvenc

    detect_nvenc()


def _check_per_lf_completeness(state: Any) -> None:
    """Strict per-LF backstop completeness gate.

    The fields:

      * ``per_lf_recording_date`` — drives the ``📅 Filmed: <date>`` line
        in every YouTube description. Without it an LF can ship
        missing this line.
      * ``per_lf_trim_ratio`` — drives the title verb band
        (``Driving`` vs ``Highlights from``), the description hook, and
        the consolidated shorts CSV first-comment lead.

    Both fields are declared on ``PipelineState`` with load-time
    disk-derivation, implemented as ``PipelineState.__post_init__`` methods
    (``_migrate_per_lf_recording_date_from_paths``,
    ``_migrate_per_lf_trim_ratio_from_paths``) so every construction
    path applies them. Every legitimate state file therefore enters
    this gate populated — either by Phase BC's canonical writer or by
    the construction-time migration.

    Reaching this gate with empty state under completed
    ``phase_bc_day_first`` now means the on-disk LF artifacts are
    corrupted (no ``day_<DATE>/regular_lfs/lf_NN.mp4`` parse path) and
    the migration could not recover them — hard-fail is correct, the
    operator must re-run phase_bc to repopulate.

    A ``warn-and-pass`` mode for the fully-empty case no longer
    exists: the downstream in-phase backstops it relied on were deleted.
    """
    phases = _state_get(state, "phases_completed") or []
    if "phase_bc_day_first" not in phases:
        return

    regular_lf_paths = _state_get(state, "regular_lf_paths") or []
    if not regular_lf_paths:
        # Zero-qualifying-footage drive — nothing to validate.
        return

    expected_count = len(regular_lf_paths)
    # Derive expected LF index set from drive_plan when available
    # so a state whose dict happens to have the right LENGTH but offset
    # keys (e.g. {0,1,2} when plan says {1,2,3}) still fails fast. Falls
    # back to length-only check when drive_plan is absent (pre-phase-BC
    # zero-LF drives never reach here, so this is purely defensive).
    drive_plan = _state_get(state, "drive_plan")
    skeleton_lfs = getattr(drive_plan, "regular_lfs", None) if drive_plan is not None else None
    expected_indices: set[int] | None = None
    if skeleton_lfs:
        expected_indices = {
            int(getattr(lf, "index", -1))
            for lf in skeleton_lfs
            if getattr(lf, "index", None) is not None
        }
        expected_indices.discard(-1)

    for field in ("per_lf_recording_date", "per_lf_trim_ratio", "per_lf_gpx_path"):
        mapping = _state_get(state, field) or {}
        if not isinstance(mapping, dict):
            _raise(f"{field} must be a dict, got {type(mapping).__name__}")
        if expected_indices is not None:
            present = {int(k) for k in mapping}
            missing = expected_indices - present
            if missing:
                _raise(
                    f"phase_bc_day_first marked complete with "
                    f"{len(expected_indices)} regular LFs but {field} is "
                    f"missing index(es) {sorted(missing)}. "
                    f"Legacy state files migrate "
                    f"at PipelineState.from_dict time; reaching here with "
                    f"partial state means the on-disk LF artifacts (or "
                    f"file_durations for trim_ratio, or episode_dates for "
                    f"per_lf_gpx_path) cannot be recovered. "
                    f"To recover: re-run from phase_bc_day_first, or wipe "
                    f"state for a clean resume."
                )
        elif len(mapping) < expected_count:
            _raise(
                f"phase_bc_day_first marked complete with {expected_count} "
                f"regular_lf_paths but {field} has {len(mapping)} of "
                f"{expected_count} entries. "
                f"Legacy state files migrate at "
                f"PipelineState.from_dict time; reaching here with partial "
                f"or empty state means the on-disk LF artifacts (or "
                f"file_durations for trim_ratio) cannot be recovered. "
                f"To recover: re-run from phase_bc_day_first, or wipe state "
                f"for a clean resume."
            )


def _check_sidecar_integrity_on_resume(state: Any) -> None:
    """When ``phase_d`` is partially complete (in ``phases_completed`` or
    resumed mid-Phase-D), validate the on-disk youtube_uploads.json
    sidecar against ``drive_plan.regular_lfs``.

    The local-only gate lives in ``youtube_sidecar.assert_sidecar_integrity``
    — we reuse it. Imported lazily to avoid a module-level cycle.
    """
    phases = _state_get(state, "phases_completed") or []
    if "phase_d" not in phases:
        return
    drive_plan = _state_get(state, "drive_plan")
    output_base = _state_get(state, "output_base")
    if drive_plan is None or output_base is None:
        return
    if not getattr(drive_plan, "regular_lfs", None):
        return
    from .youtube_sidecar import assert_sidecar_integrity

    assert_sidecar_integrity(Path(output_base), drive_plan)


def _check_ost_audit_hash_drift(state: Any) -> None:
    """If ``state["ost_audit_passed"][date]`` is True, the on-disk
    ``ost_audit_<date>.json`` must still hash to the same value as
    ``state["ost_audit_hash"][date]``.

    Mirrors the gate in ``day_pipeline.run_day_pipeline``. The
    purpose here is to surface drift at resume time, before any further
    compute, instead of at the next per-day OST burn.

    A confirmation marker without a matching hash on disk is treated as
    a legitimate confirmation that pre-dated the hash tracking — the
    confirmation file path (``ost_audit_<date>.confirmed``) is sufficient
    on its own and we do NOT raise.
    """
    passed = _state_get(state, "ost_audit_passed") or {}
    stored_hashes = _state_get(state, "ost_audit_hash") or {}
    output_base = _state_get(state, "output_base")
    if not passed or output_base is None:
        return

    # Lazy import: ost_audit_sidecar pulls in day_pipeline machinery.
    from .ost_audit_sidecar import compute_ost_audit_hash

    output_base = Path(output_base)
    for date, was_passed in passed.items():
        if not was_passed:
            continue
        sidecar_path = output_base / f"ost_audit_{date}.json"
        if not sidecar_path.exists():
            continue  # nothing to hash; either a fresh confirmation file
            # path or the audit was never written. Either way the
            # day_pipeline gate will re-evaluate.
        try:
            payload = json.loads(sidecar_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            # A *confirmed* sidecar that we cannot read/parse can't have its
            # hash verified — "can't verify the confirmed hash" is drift, not
            # a free pass. Do NOT continue; force re-confirmation. (The
            # legitimately-absent case is handled above and still passes.)
            _raise(
                f"ost_audit confirmed sidecar for day {date} is unreadable: "
                f"{sidecar_path.name} — {type(exc).__name__}: {exc}. "
                f"The confirmed OST-audit hash cannot be re-verified at "
                f"resume, so the confirmation is treated as drift. "
                f"To recover: re-open {sidecar_path}, verify/repair the LF "
                f"text, then delete state['ost_audit_passed'][{date!r}] (or "
                f"re-run with --ost-audit-confirmed) so the gate re-confirms "
                f"against the on-disk content."
            )
        current_hash = compute_ost_audit_hash(payload)
        stored = stored_hashes.get(date)
        if stored is None:
            continue  # legacy confirmation pre-dating hash tracking
        if stored != current_hash:
            _raise(
                f"ost_audit_hash drift for day {date}: state stored "
                f"{stored!r} but {sidecar_path.name} now hashes to "
                f"{current_hash!r}. "
                f"Likely cause: per-LF labels were edited after the "
                f"OST audit was confirmed. "
                f"To recover: re-open {sidecar_path}, verify the LF "
                f"text is what you intend to burn, then delete "
                f"state['ost_audit_passed'][{date!r}] (or re-run with "
                f"--ost-audit-confirmed) so the gate re-confirms against "
                f"the new content."
            )


def resume_invariants(state: Any, config: Any = None) -> None:
    """Single post-load entry point invoked on every RESUME path.

    Fresh runs skip this — they hit ``run_preflight`` at phase_a entry
    and have nothing to reconcile. Resumed runs cannot otherwise reach
    these checks (phase_a's preflight is gated behind
    ``"phase_a" not in resumed_phases``, which is the very class of
    silent exemption this function exists to close).

    Order matters: write-time invariants first (cheap, raise first on
    obvious shape bugs), then resource preflight (cheap, catches the
    "out of disk" class), then sidecar / OST gates (slightly more
    work — JSON IO + hashing), then per-LF completeness last so a
    legitimately missing legacy state-file field surfaces with a clear
    cause-and-recovery message instead of getting drowned in a noisy
    earlier failure.

    Raises:
        StateInvariantError: any of the per-state invariants tripped.
        StateCoherenceError: cross-field drift detected.
        RuntimeError: preflight (disk / RAM / tools) failed.
        SidecarIntegrityError: youtube_uploads.json ↔ drive_plan mismatch.
    """
    # Cheap shape gates from this module — same set the save path runs.
    assert_state_invariants(state)
    assert_resume_state_coherent(state)
    # Resource preflight — a resume must not skip the disk check.
    _check_resume_preflight(state)
    # Sidecar integrity — youtube_uploads.json vs. drive_plan.regular_lfs.
    _check_sidecar_integrity_on_resume(state)
    # OST audit hash — operator-edited labels after confirmation.
    _check_ost_audit_hash_drift(state)
    # Per-LF backstop completeness.
    _check_per_lf_completeness(state)
