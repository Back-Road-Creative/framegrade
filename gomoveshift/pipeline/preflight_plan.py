"""Preflight execution plan.

Surfaces the orchestrator's intent to the operator BEFORE any hours of
encoding get spent. Addresses two operator-pain classes:

* multi-step pipelines must expose what they will do for a human audit
  pass.
* run every automatable check upfront, don't hand the operator failures
  one at a time.

The failure this guards against: a run silently re-encodes days of footage
when master files have been purged and ``per_day_completed`` is empty, with
nothing telling the operator. A preflight plan renders::

    EXECUTION PLAN
      Phase A:        SKIP (cached drive_scores.npz at <path>)
      Day 2024-01-01: SKIP (per_day_completed; master on disk)
      Day 2024-01-02: SKIP (per_day_completed; finals present, master purged — finals-only reconstruction)
      Day 2024-01-03: RUN (master partial 1.5GB / 15GB expected — re-encode)
      Day 2024-01-04: RUN (fresh)
      Phase D:        RUN
      Estimated wall-clock: 4h 30m  (breakdown: master 3h 50m, LFs 20m, shorts 15m, phase_d 5m)

      Proceed? (--dry-run to exit; add --force to skip this prompt)

Wall-clock estimate
-------------------

The estimate is a heuristic, not a measurement. Numbers come from run
timings observed on a reference machine:

* day master encode (color + audio loudnorm + OST): calibrated against
  a reference run — a ~2.7 h 4K HEVC source,
  hevc_nvenc preset p7, multipass=2 → ~3.9 h (0.7x realtime). A flat
  1.8h/day estimate would be ~2.2x too optimistic for the production
  preset. See
  ``estimate_master_encode_seconds`` for the model.
* per-LF re-encode (master slice → 4K HEVC NVENC LF + OST): calibrated
  against a reference run: a ~10 min source encodes in ~10 min wall-clock and
  an ~18 min source in ~20 min (~1.10x realtime). A flat 5 min/LF constant
  under-predicts longer LFs by ~4x. See ``estimate_lf_encode_seconds`` for the model.
* per-short encode (4K master slice → 1080×1920 9:16 NVENC + OST via
  ``shorts_extractor.render_short_ffmpeg``): the day-first 9:16
  path is too new for direct empirical data. Calibrated by scaling
  from a reference 16:9 trim path
  (~1.3x realtime) plus a 1.5x
  crop+scale+9:16-OST overhead → ~2.0x realtime. See
  ``estimate_short_encode_seconds`` for the model. Refine when 9:16
  wall-clock data exists.
* Phase A ingest + scoring: ~5 min on cached pyav; ~20 min fresh over
  20 episodes — estimate uses 10 min when fresh.
* Phase D (metadata, schedule, CSV, optional upload): ~5 min when
  ``skip_upload`` is True, ~12 min with uploads enabled.

The estimate is rounded to the nearest 5 minutes and labelled as an
``Estimated`` number in the rendered output. It is NOT load-bearing —
the caller (CLI) treats it as operator guidance only.

Public API
----------

* :func:`build_plan(state, stages_dir, probe_duration_fn=None, calibration=<saved>)` →
  :class:`PreflightPlan`. Pure read-only — inspects state + the stages
  directory, makes no mutations.
* :func:`print_plan(state, stages_dir, out=None, probe_duration_fn=None)`
  — render a ``PreflightPlan`` to ``out`` (defaults to ``sys.stdout``).
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Sequence

if TYPE_CHECKING:
    from .calibration import CalibrationRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wall-clock heuristic constants
# ---------------------------------------------------------------------------
# Conservative numbers derived from reference-machine run timings.
# See module docstring for provenance. All in seconds.
#
# Master-encode model (calibrated against a reference run):
#
#     time_seconds = source_duration_s
#                  * _RES_FACTOR[resolution]
#                  * _PRESET_FACTOR[preset]
#                  * _MULTIPASS_FACTOR[multipass]
#
# Empirical calibration point — 4K HEVC source, hevc_nvenc, preset p7,
# multipass=2, a ~2.7 h source → ffmpeg's own progress reports 0.7x realtime
# (21fps output for 30fps source) → ETA ~3.9 h.
#
#     check: 9800 * 1.00 (4K) * 1.10 (p7) * 1.30 (multipass=2) ≈ 14000s ✓
#
# A flat constant (1.8h per day, regardless of preset / multipass /
# resolution) would under-predict this case by ~2.2x.
_RES_FACTOR: dict[str, float] = {
    "1080p": 0.30,
    "1440p": 0.50,
    "4k": 1.00,
    "8k": 2.00,
}
_PRESET_FACTOR: dict[str, float] = {
    # NVENC presets p1 (fastest) → p7 (slowest, GMS production default).
    "p1": 0.40,
    "p2": 0.45,
    "p3": 0.55,
    "p4": 0.65,
    "p5": 0.80,
    "p6": 0.95,
    "p7": 1.10,
}
_MULTIPASS_FACTOR: dict[int, float] = {
    1: 1.00,
    2: 1.30,
}
# Defaults when a per-day case can't be classified — must NOT be the
# slowest combination (we don't want to scare the operator) but must
# match the GMS production default (4K / p7 / multipass=2 per
# encoding_constants.py and the preset selection in single_pass.py).
_DEFAULT_RESOLUTION = "4k"
_DEFAULT_PRESET = "p7"
_DEFAULT_MULTIPASS = 2

# Per-day source duration when the ingest plan hasn't been computed yet
# (1 hour — matches the GMS day cadence).
_DEFAULT_DAY_SOURCE_DURATION_SEC = 3600.0


def estimate_master_encode_seconds(
    source_duration_s: float,
    resolution: str,
    preset: str,
    multipass: int,
    calibration: "CalibrationRecord | None" = None,
) -> float:
    """Heuristic wall-clock estimate for one master encode.

    When ``calibration`` (a measurement of the real chain on THIS machine,
    see :mod:`.calibration`) is given it wins outright:
    ``source_duration_s * calibration.realtime_factor``. The measurement
    already includes the configured preset / multipass / resolution, so the
    fixed factors below are ignored.

    Calibrated against a reference run: a ~2.7 h 4K HEVC source,
    hevc_nvenc preset p7, multipass=2 → empirical ETA ~3.9 h
    (0.7x realtime). See ``_RES_FACTOR`` / ``_PRESET_FACTOR`` /
    ``_MULTIPASS_FACTOR`` block above for the model.

    Unknown ``preset`` / ``resolution`` / ``multipass`` values fall back
    to the documented defaults (``p4`` / ``4k`` / ``1``) and emit a
    WARNING — they MUST NOT raise, because preflight runs before any
    encoding work and a crash here would block the operator from
    seeing the rest of the plan.
    """
    if calibration is not None:
        return float(source_duration_s) * calibration.realtime_factor
    res_key = (resolution or "").lower()
    if res_key not in _RES_FACTOR:
        logger.warning(
            "estimate_master_encode_seconds: unknown resolution %r; falling back to %r",
            resolution,
            _DEFAULT_RESOLUTION,
        )
        res_key = _DEFAULT_RESOLUTION
    preset_key = (preset or "").lower()
    if preset_key not in _PRESET_FACTOR:
        logger.warning(
            "estimate_master_encode_seconds: unknown preset %r; falling back to %r",
            preset,
            "p4",
        )
        preset_key = "p4"
    if multipass not in _MULTIPASS_FACTOR:
        logger.warning(
            "estimate_master_encode_seconds: unknown multipass %r; falling back to 1",
            multipass,
        )
        multipass = 1
    return float(source_duration_s) * (
        _RES_FACTOR[res_key] * _PRESET_FACTOR[preset_key] * _MULTIPASS_FACTOR[multipass]
    )


# ---------------------------------------------------------------------------
# LF-encode model (calibrated against a reference day):
#
#     lf_seconds = max(lf_duration_s * _LF_REALTIME_FACTOR, _LF_FLOOR_SEC)
#
# Empirical calibration points (output mtimes vs source-content durations
# for one reference day):
#   - a ~10 min source → ~10 min wall-clock (~1.05x realtime)
#   - an ~18 min source → ~20 min wall-clock (~1.13x realtime)
# Pick 1.10 — midpoint that errs slightly conservative for longer LFs
# (per-LF NVENC re-encode + OST + montage overlay is ~realtime with a
# small overlay-compositing tax).
#
# A flat 5 min/LF constant would under-predict a typical 15-min
# LF by ~3x. Per-LF duration is already in state via
# ``regular_lfs[i].duration_s`` from the day-first plan.
_LF_REALTIME_FACTOR = 1.10
# Floor: keep ETA non-zero for trivially-short inputs. Even a 1-second LF
# pays the ffmpeg startup tax (probe + filter graph build + NVENC init).
_LF_FLOOR_SEC = 30
# Per-LF default when no per-LF duration is available (the legacy flat
# constant for backwards compat with callers that pass only counts).
_DEFAULT_LF_DURATION_SEC = 12 * 60  # ~12 min per LF


# ---------------------------------------------------------------------------
# Short-encode model (day-first 1080x1920 9:16 path via
# ``shorts_extractor.render_short_ffmpeg``):
#
#     short_seconds = max(
#         short_duration_s * _SHORT_REALTIME_FACTOR,
#         _SHORT_FLOOR_SEC,
#     )
#
# Calibration: the earlier 16:9 trim+OST path (now superseded)
# ran at ~1.3x realtime on a reference run (~39s per ~30s slice). The new
# 9:16 path adds a center-crop, scale-to-1080×1920, and 9:16 OST overlay;
# conservatively assume a 1.5x overhead vs the trim path → ~2.0x realtime.
# NVENC startup + filter-graph build dominate very-short slices, hence
# the floor.
#
# This is the LEAST-calibrated of the three encode estimates: no measured
# 9:16 wall-clock data exists yet. Refine once it does.
_SHORT_REALTIME_FACTOR = 2.0
_SHORT_FLOOR_SEC = 30
_DEFAULT_SHORT_DURATION_SEC = 30.0


def estimate_lf_encode_seconds(lf_duration_s: float) -> float:
    """Heuristic wall-clock for one regular-LF re-encode + OST + montage.

    Calibrated against a reference day:
    a ~10 min source → ~10 min wall-clock; an ~18 min source → ~20 min
    wall-clock. Model: ``lf_duration_s * 1.10`` with a 30s floor
    so trivially-short LFs still account for ffmpeg startup cost.

    Negative or non-finite ``lf_duration_s`` falls back to the documented
    default (12 min) and emits a WARNING — must NOT raise (preflight runs
    before any encoding work, and a crash here would block operator
    visibility into the rest of the plan).
    """
    try:
        d = float(lf_duration_s)
    except (TypeError, ValueError):
        d = -1.0
    if not (d > 0):
        logger.warning(
            "estimate_lf_encode_seconds: invalid duration %r; falling back to %ss",
            lf_duration_s,
            _DEFAULT_LF_DURATION_SEC,
        )
        d = float(_DEFAULT_LF_DURATION_SEC)
    return max(d * _LF_REALTIME_FACTOR, float(_LF_FLOOR_SEC))


def estimate_short_encode_seconds(short_duration_s: float) -> float:
    """Heuristic wall-clock for one day-first 9:16 short encode.

    Conservatively models the ``render_short_ffmpeg`` path (4K master
    slice → 1080×1920 9:16 NVENC + OST overlay) as ``2.0x`` realtime
    with a 30s floor for ffmpeg startup. Calibrated by scaling from the
    legacy 16:9 trim+OST path (~1.3x realtime on a reference day)
    plus a 1.5x overhead for the added crop / scale / 9:16 OST.

    Negative or non-finite ``short_duration_s`` falls back to the
    documented default (30s) and emits a WARNING — must NOT raise.
    """
    try:
        d = float(short_duration_s)
    except (TypeError, ValueError):
        d = -1.0
    if not (d > 0):
        logger.warning(
            "estimate_short_encode_seconds: invalid duration %r; falling back to %ss",
            short_duration_s,
            _DEFAULT_SHORT_DURATION_SEC,
        )
        d = float(_DEFAULT_SHORT_DURATION_SEC)
    return max(d * _SHORT_REALTIME_FACTOR, float(_SHORT_FLOOR_SEC))


# Legacy flat per-unit constants — retained as the fallback when neither
# per-LF / per-short duration data nor the calibrated estimators apply.
# The default ``_estimate_wall_clock`` path now sums the calibrated
# estimators; these constants are kept for callers that pass only counts.
_LF_PER_UNIT_SEC = 5 * 60  # per regular LF post-encode (legacy fallback)
_SHORT_PER_UNIT_SEC = 3 * 60  # per short (legacy fallback)
_PHASE_A_RUN_SEC = 10 * 60  # fresh Phase A ingest + scoring
_PHASE_D_SKIP_UPLOAD_SEC = 5 * 60
_PHASE_D_WITH_UPLOAD_SEC = 12 * 60

# LFs / shorts per day when we have no better estimate (guess: ~3 LFs/day,
# ~14 shorts/LF for the GMS Wed cadence = ~42 shorts/day).
_DEFAULT_LFS_PER_DAY = 3
_DEFAULT_SHORTS_PER_DAY = 42

# Partial-master threshold: raw concat duration below this fraction of the
# expected day duration is a "re-encode" signal. Matches
# day_master_encoder's own 0.99 fraction (see the ``expected_duration * 0.99`` checks in day_master_encoder.py).
_PARTIAL_RAW_FRACTION = 0.99


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PlanRow:
    """One row of the preflight plan.

    Rows represent phases or days. Status is "RUN" or "SKIP"; the reason
    is a short operator-readable string that explains WHY (e.g.
    "cached drive_scores.npz", "finals-only reconstruction").
    """

    label: str
    status: str  # "RUN" | "SKIP"
    reason: str


@dataclass
class PreflightPlan:
    """Read-only snapshot of what the orchestrator will do this run.

    Rendered by :func:`print_plan`; inspected directly by tests.
    """

    rows: list[PlanRow] = field(default_factory=list)
    wall_clock_estimate_seconds: int = 0
    wall_clock_breakdown: dict[str, int] = field(default_factory=dict)
    fast_skipped: list[str] = field(default_factory=list)
    calibration: "CalibrationRecord | None" = None


# ---------------------------------------------------------------------------
# Plan builder
# ---------------------------------------------------------------------------


def _format_duration(total_sec: int) -> str:
    """``9300`` → ``"2h 35m"``; short durations collapse to ``"5m"``."""
    if total_sec <= 0:
        return "0m"
    hours, rem = divmod(int(total_sec), 3600)
    minutes = rem // 60
    if hours == 0:
        return f"{minutes}m"
    return f"{hours}h {minutes}m"


def _collect_day_dates(state: Any) -> list[str]:
    """Extract the ordered list of YYYY-MM-DD day keys the pipeline will
    walk. Prefer state.per_day_completed order when present (authoritative
    checkpoint); fall back to episode_dates then ingest_plan enumeration.
    """
    # Prefer a recorded day_batches-style path from the ingest_plan so the
    # user sees every planned day (including ones with no checkpoint yet).
    ingest_plan = getattr(state, "ingest_plan", None) if state is not None else None
    episodes = list(getattr(ingest_plan, "episodes", None) or []) if ingest_plan else []
    if episodes:
        dates: list[str] = []
        seen: set[str] = set()
        for ep in episodes:
            dk = _episode_date_key(ep)
            if dk and dk not in seen:
                seen.add(dk)
                dates.append(dk)
        if dates:
            return sorted(dates)
    # Fallback: episode_dates list (already normalised in state).
    ep_dates = getattr(state, "episode_dates", None) if state is not None else None
    if isinstance(ep_dates, (list, tuple)) and ep_dates:
        return sorted({d for d in ep_dates if isinstance(d, str) and d})
    # Last resort: per_day_completed keys (dates for days already done).
    pdc = getattr(state, "per_day_completed", None) if state is not None else None
    if isinstance(pdc, dict) and pdc:
        return sorted(pdc.keys())
    return []


def _episode_date_key(episode: Any) -> Optional[str]:
    """Mirror ``day_grouper._episode_date_key`` — prefer creation_time,
    fall back to the string ``date`` attribute. Kept local to avoid
    importing day_grouper (which pulls extra deps)."""
    from datetime import datetime

    ct = getattr(episode, "creation_time", None)
    if isinstance(ct, datetime):
        return ct.strftime("%Y-%m-%d")
    date_str = getattr(episode, "date", None)
    if isinstance(date_str, str) and date_str.strip():
        return date_str.strip()
    return None


def _phase_a_row(state: Any, stages_dir: Path) -> PlanRow:
    """Phase A SKIPs when ``drive_scores.npz`` is on disk AND the
    checkpoint recorded phase_a as complete. The orchestrator's own
    resume path uses the same ``PHASE_ARTIFACTS`` check."""
    scores_path = stages_dir / "drive_scores.npz"
    phases_completed = getattr(state, "phases_completed", None) if state is not None else None
    phases_completed = set(phases_completed or [])
    if scores_path.exists() and "phase_a" in phases_completed:
        return PlanRow(
            label="Phase A",
            status="SKIP",
            reason=f"cached drive_scores.npz at {scores_path}",
        )
    if scores_path.exists():
        # Artifact exists but checkpoint hasn't recorded it — re-run.
        return PlanRow(
            label="Phase A",
            status="RUN",
            reason="drive_scores.npz present but phase_a not in phases_completed",
        )
    return PlanRow(label="Phase A", status="RUN", reason="fresh")


def _day_dir_has_finals(day_dir: Path) -> bool:
    """Mirror ``phase_bc_day_first._day_dir_has_finals``."""
    lf_dir = day_dir / "regular_lfs"
    shorts_dir = day_dir / "shorts"
    has_lf = lf_dir.is_dir() and any(
        p.is_file() and p.stat().st_size > 0 for p in lf_dir.glob("lf_[0-9][0-9].mp4")
    )
    has_short = shorts_dir.is_dir() and any(
        p.is_file() and p.stat().st_size > 0 for p in shorts_dir.glob("short_[0-9][0-9].mp4")
    )
    return has_lf and has_short


def _collect_lf_durations(state: Any) -> list[float]:
    """Return per-LF durations (seconds) from ``state["drive_plan"].regular_lfs``.

    Each ``regular_lfs`` entry is expected to expose a ``duration_s`` /
    ``duration_sec`` / ``total_duration`` attribute (the day-first plan
    populates one of these). Returns an empty list when ``drive_plan``
    is missing or no per-LF duration can be read — callers must fall
    back to the legacy flat estimate. Never raises.
    """
    if state is None:
        return []
    drive_plan = getattr(state, "drive_plan", None)
    lfs = list(getattr(drive_plan, "regular_lfs", None) or [])
    if not lfs:
        return []
    out: list[float] = []
    for lf in lfs:
        dur = (
            getattr(lf, "duration_s", None)
            or getattr(lf, "duration_sec", None)
            or getattr(lf, "total_duration", None)
            or 0.0
        )
        try:
            d = float(dur)
        except (TypeError, ValueError):
            d = 0.0
        if d > 0:
            out.append(d)
    return out


def _collect_short_durations(state: Any) -> list[float]:
    """Return per-short durations (seconds) from
    ``state["short_durations_per_lf"]`` or, failing that, derive a
    default-duration entry per known short path in
    ``state["shorts_per_lf"]``.

    The day-first plan populates ``shorts_per_lf: dict[int, list[Path]]``
    via the shorts extractor; per-short duration is not yet a first-class state field,
    so when only counts are known we conservatively assign each short the
    documented default duration (30s). Returns an empty list when no
    shorts data is available.
    """
    if state is None:
        return []
    explicit = getattr(state, "short_durations_per_lf", None)
    out: list[float] = []
    if isinstance(explicit, dict):
        for durs in explicit.values():
            for d in durs or []:
                try:
                    val = float(d)
                except (TypeError, ValueError):
                    val = 0.0
                if val > 0:
                    out.append(val)
    if out:
        return out
    shorts_per_lf = getattr(state, "shorts_per_lf", None)
    if isinstance(shorts_per_lf, dict):
        for paths in shorts_per_lf.values():
            for _ in paths or []:
                out.append(_DEFAULT_SHORT_DURATION_SEC)
    return out


def _expected_day_duration(state: Any, date: str) -> float:
    """Sum the durations of every episode whose date-key matches."""
    ingest_plan = getattr(state, "ingest_plan", None) if state is not None else None
    episodes = list(getattr(ingest_plan, "episodes", None) or []) if ingest_plan else []
    total = 0.0
    for ep in episodes:
        dk = _episode_date_key(ep)
        if dk != date:
            continue
        dur = float(getattr(ep, "total_duration", 0.0) or 0.0)
        total += dur
    return total


def _day_row(
    state: Any,
    stages_dir: Path,
    date: str,
    probe_duration_fn: Optional[Callable[[Path], Optional[float]]] = None,
) -> PlanRow:
    """Decide RUN vs SKIP for a single day, mirroring
    ``phase_bc_day_first``'s per-day bookkeeping logic."""
    recorded = getattr(state, "per_day_completed", None) if state is not None else None
    entry = recorded.get(date) if isinstance(recorded, dict) else None
    master_path_str = (entry or {}).get("master_path") if entry else None
    day_dir = stages_dir / f"day_{date}"
    label = f"Day {date}"

    if master_path_str and Path(master_path_str).exists():
        return PlanRow(
            label=label,
            status="SKIP",
            reason="per_day_completed; master on disk",
        )
    if entry and _day_dir_has_finals(day_dir):
        return PlanRow(
            label=label,
            status="SKIP",
            reason=(
                "per_day_completed; finals present, master purged — finals-only reconstruction"
            ),
        )
    # No completion record. Check for a partial raw concat so we can
    # explain why a fresh day is going to re-encode.
    raw_path = day_dir / f"day_{date}_raw.mp4"
    if raw_path.exists() and probe_duration_fn is not None:
        expected = _expected_day_duration(state, date)
        actual = 0.0
        try:
            probed = probe_duration_fn(raw_path)
            if probed is not None:
                actual = float(probed)
        except Exception as exc:  # noqa: BLE001 — best-effort probe
            logger.debug("preflight_plan probe failed for %s: %s", raw_path, exc)
        if expected > 0 and actual > 0 and actual < expected * _PARTIAL_RAW_FRACTION:
            return PlanRow(
                label=label,
                status="RUN",
                reason=(
                    f"master partial {_format_duration(int(actual))} / "
                    f"{_format_duration(int(expected))} expected — re-encode"
                ),
            )
    return PlanRow(label=label, status="RUN", reason="fresh")


def _phase_d_row(state: Any, stages_dir: Path) -> PlanRow:
    """Phase D SKIPs when drive_schedule.json is on disk AND the
    checkpoint recorded phase_d complete."""
    schedule_path = stages_dir / "drive_schedule.json"
    phases_completed = getattr(state, "phases_completed", None) if state is not None else None
    phases_completed = set(phases_completed or [])
    if schedule_path.exists() and "phase_d" in phases_completed:
        return PlanRow(
            label="Phase D",
            status="SKIP",
            reason=f"cached drive_schedule.json at {schedule_path}",
        )
    return PlanRow(label="Phase D", status="RUN", reason="fresh")


def _estimate_wall_clock(
    rows: Iterable[PlanRow],
    state: Any,
    calibration: "CalibrationRecord | None" = None,
) -> tuple[int, dict[str, int]]:
    """Heuristic wall-clock estimate.

    Returns ``(total_seconds, breakdown_dict)``. Breakdown keys:
    ``phase_a``, ``master``, ``lfs``, ``shorts``, ``superlong``, ``phase_d``.
    Only RUN rows contribute. SKIP rows contribute 0s.
    """
    phase_a_sec = 0
    master_sec = 0
    lfs_sec = 0
    shorts_sec = 0
    superlong_sec = 0
    phase_d_sec = 0

    # ``--superlong`` (opt-in per-day sl_NN.mp4 parts, additive to the
    # regular LF/shorts encodes — see day_pipeline._produce_superlong_parts).
    # Read from ``state`` rather than ``config.drive.superlong_enabled``:
    # by the time the orchestrator reaches preflight, ``run()`` has already
    # resolved this run's intent onto ``state`` — via
    # ``_build_initial_state``'s ``apply_cli_to_state`` on a fresh run, or
    # ``_try_resume``'s ``merge_on_resume`` on a resume — both of which
    # execute before ``run_preflight_and_prompt``. Same sentinel-read
    # pattern already used below for ``skip_upload``.
    superlong_enabled = (
        bool(getattr(state, "superlong_enabled", None)) if state is not None else False
    )

    days_to_run = 0
    for row in rows:
        if row.status != "RUN":
            continue
        if row.label == "Phase A":
            phase_a_sec = _PHASE_A_RUN_SEC
        elif row.label == "Phase D":
            skip_upload = bool(getattr(state, "skip_upload", None)) if state is not None else False
            phase_d_sec = _PHASE_D_SKIP_UPLOAD_SEC if skip_upload else _PHASE_D_WITH_UPLOAD_SEC
        elif row.label.startswith("Day "):
            days_to_run += 1
            # Sum the per-day master cost using the calibrated estimator
            # so the operator-facing wall-clock number actually reflects
            # preset + multipass + resolution choices. See module
            # docstring + ``estimate_master_encode_seconds``.
            date = row.label[len("Day ") :].strip()
            day_source_sec = _expected_day_duration(state, date)
            if day_source_sec <= 0:
                day_source_sec = _DEFAULT_DAY_SOURCE_DURATION_SEC
            master_sec += int(
                estimate_master_encode_seconds(
                    source_duration_s=day_source_sec,
                    resolution=_DEFAULT_RESOLUTION,
                    preset=_DEFAULT_PRESET,
                    multipass=_DEFAULT_MULTIPASS,
                    calibration=calibration,
                )
            )
            if superlong_enabled:
                # Superlong parts re-encode the same stop-removed day
                # timeline (split into <=4h parts by
                # superlong_planner.plan_superlong_parts), so the master
                # estimator is the right calibrated model — scaled by the
                # same planned per-day footage used for ``master_sec``.
                superlong_sec += int(
                    estimate_master_encode_seconds(
                        source_duration_s=day_source_sec,
                        resolution=_DEFAULT_RESOLUTION,
                        preset=_DEFAULT_PRESET,
                        multipass=_DEFAULT_MULTIPASS,
                        calibration=calibration,
                    )
                )

    # Calibrated LF + shorts cost: prefer per-LF / per-short duration data
    # from state when available (drive_plan.regular_lfs + per-day artifact
    # short_paths/short durations); otherwise fall back to the legacy flat
    # constants scaled by ``days_to_run`` so call-sites with no per-LF
    # plan still get a non-zero estimate. The calibrated path mirrors the
    # master-encode calibration.
    lf_durations = _collect_lf_durations(state) if days_to_run else []
    short_durations = _collect_short_durations(state) if days_to_run else []
    if lf_durations:
        lfs_sec = int(sum(estimate_lf_encode_seconds(d) for d in lf_durations))
    else:
        lfs_sec = (
            days_to_run
            * _DEFAULT_LFS_PER_DAY
            * int(estimate_lf_encode_seconds(_DEFAULT_LF_DURATION_SEC))
        )
    if short_durations:
        shorts_sec = int(sum(estimate_short_encode_seconds(d) for d in short_durations))
    else:
        shorts_sec = (
            days_to_run
            * _DEFAULT_SHORTS_PER_DAY
            * int(estimate_short_encode_seconds(_DEFAULT_SHORT_DURATION_SEC))
        )

    breakdown = {
        "phase_a": phase_a_sec,
        "master": master_sec,
        "lfs": lfs_sec,
        "shorts": shorts_sec,
        "superlong": superlong_sec,
        "phase_d": phase_d_sec,
    }
    total = sum(breakdown.values())
    return total, breakdown


_LOAD_CALIBRATION: Any = object()  # sentinel: "kwarg omitted -> load the saved record"


def build_plan(
    state: Any,
    stages_dir: Path,
    probe_duration_fn: Optional[Callable[[Path], Optional[float]]] = None,
    fast_skipped: Sequence[str] = (),
    calibration: Any = _LOAD_CALIBRATION,
) -> PreflightPlan:
    """Inspect state + the stages directory and return a plan snapshot.

    Pure read-only: mutates nothing. ``probe_duration_fn`` is injectable
    for tests; it defaults to lazy-imported ``_probe_duration`` from
    ``phase_bc_day_first`` at call time so the module stays
    import-lightweight. ``calibration`` omitted loads the saved record
    (``load_calibration``); pass ``None`` for "no record".
    """
    if calibration is _LOAD_CALIBRATION:
        from .calibration import load_calibration

        calibration = load_calibration()
    stages_dir = Path(stages_dir)
    rows: list[PlanRow] = []
    rows.append(_phase_a_row(state, stages_dir))
    for date in _collect_day_dates(state):
        rows.append(_day_row(state, stages_dir, date, probe_duration_fn))
    rows.append(_phase_d_row(state, stages_dir))
    total, breakdown = _estimate_wall_clock(rows, state, calibration)
    return PreflightPlan(
        rows=rows,
        wall_clock_estimate_seconds=total,
        wall_clock_breakdown=breakdown,
        fast_skipped=list(fast_skipped),
        calibration=calibration,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _eta_basis_line(cal: "CalibrationRecord | None") -> str:
    if cal is None:
        return (
            "  ETA basis: built-in reference model (not yet measured on this machine; "
            "the first run calibrates, or run: gomoveshift doctor --calibrate <clip>)"
        )
    return (
        f"  ETA basis: measured on this machine \u2014 {cal.chain_fps:.1f} fps "
        f"({cal.encoder}, {cal.width}x{cal.height}, blur {'on' if cal.blur else 'off'}, "
        f"{cal.measured_at[:10]})"
    )


def _render_plan_lines(plan: PreflightPlan) -> list[str]:
    """Build the list of rendered lines; kept separate for testability."""
    lines = ["EXECUTION PLAN"]
    # Compute label column width.
    label_width = max((len(row.label) for row in plan.rows), default=10)
    for row in plan.rows:
        # "  <label>:<padding> <status> (<reason>)"
        label_col = f"{row.label}:".ljust(label_width + 2)
        status_col = row.status.ljust(4)
        lines.append(f"  {label_col}{status_col} ({row.reason})")
    if plan.fast_skipped:
        lines.append(f"  fast mode: skipped {', '.join(plan.fast_skipped)}")
    bd = plan.wall_clock_breakdown
    bd_parts = []
    # Order: master, lfs, shorts, superlong, phase_a, phase_d — heavy work first.
    for key in ("master", "lfs", "shorts", "superlong", "phase_a", "phase_d"):
        val = bd.get(key, 0)
        if val > 0:
            bd_parts.append(f"{key} {_format_duration(val)}")
    bd_str = ", ".join(bd_parts) if bd_parts else "(nothing to run)"
    lines.append(_eta_basis_line(plan.calibration))
    lines.append(
        f"  Estimated wall-clock: {_format_duration(plan.wall_clock_estimate_seconds)} "
        f" (breakdown: {bd_str})"
    )
    lines.append("")
    lines.append("  Proceed? (--dry-run to exit; add --force to skip this prompt)")
    return lines


def print_plan(
    state: Any,
    stages_dir: Path,
    out=None,
    probe_duration_fn: Optional[Callable[[Path], Optional[float]]] = None,
    fast_skipped: Sequence[str] = (),
    calibration: Any = _LOAD_CALIBRATION,
) -> PreflightPlan:
    """Render the plan for ``state`` to ``out`` (defaults to ``sys.stdout``).

    Returns the underlying :class:`PreflightPlan` so callers that also
    need to act on the rows (e.g. tests, the orchestrator's prompt
    logic) don't have to re-build it.
    """
    if out is None:
        out = sys.stdout
    plan = build_plan(
        state,
        stages_dir,
        probe_duration_fn=probe_duration_fn,
        fast_skipped=fast_skipped,
        calibration=calibration,
    )
    for line in _render_plan_lines(plan):
        print(line, file=out)
    return plan
