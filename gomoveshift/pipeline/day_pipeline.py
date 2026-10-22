"""Day-first pipeline orchestration for the day-first pipeline.

Wires together every day-scoped module into a single `run_day_pipeline`
function, and provides a `run_all_days` loop that iterates over a list
of `DayBatch` objects producing one set of artifacts per day.

Per-day flow (executed by `run_day_pipeline`):

    1. `encode_day_master`         — concat raw episodes + single color/audio
                                     pass on the day's footage → day master
    2. `plan_regular_lfs`          — pick contiguous best ranges from the
                                     day-local slice of the drive scores
    3. `extract_and_concat_segments` — cut each scored range out of the day
                                     master and join them, no re-encode. Every
                                     cut start is snapped FORWARD onto a probed
                                     keyframe (and re-encoded when it cannot be),
                                     so a stream copy never replays the prior
                                     segment's tail.
    4. `apply_ost`                 — post-assembly OST burn-in on every regular
                                     LF (layout="regular_lf"). The day master is
                                     NOT OST'd (skip_ost=True).
    5. `plan_shorts`               — rank-ordered short picks from the same
                                     day-local score slice
    6. `render_short_ffmpeg`       — one ffmpeg pass per short: center-crop to
                                     9:16, scale to 1080×1920, and burn the
                                     9:16-positioned OST (`build_ost_filter`).
                                     Shorts do NOT go through apply_ost.

There is deliberately no bare stream-copy slice helper here. A `-ss/-to -c copy`
seek with no keyframe probe snaps its start BACKWARD and replays up to one GOP of
the preceding footage — cut with `extract_and_concat_segments` instead.

The module is deliberately thin — every heavy step lives in a sibling
module. `run_day_pipeline` is just the choreography. `drive_orchestrator`
drives it per day batch.

`run_all_days` slices the drive-wide score array by `file_boundaries`
so each day's planner sees a rebased 0-based timeline matching the day
master's own 0-based timeline. A failed day (returning None) logs and
moves on — it does NOT abort the whole run, so the rest of the trip
still produces output.

OST burn-in text priority for regular LFs (single source of
truth for OST, title and thumbnail):

    1. ``per_lf_ost_text[idx]``   ← canonical ``"{start} to {end} via
                                    {road}, {state}"``
    2. ``per_lf_locations[idx]``  ← county-level fallback
    3. ``location``               ← drive-wide default

Level 1 is the YouTube title text (sans dedup ``(Part N of M)``
suffix) and the thumbnail text — all three surfaces must match.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from keycut import (
    assign_clips_by_source_window,
    boundaries_from_durations,
    extract_and_concat_ranges,
    segments_to_master_ranges,
)

from ..config import resolve_lf_mode
from . import trail_opening, volume_source
from .blur_gate import BlurGateError
from .cleanup import purge_day_intermediates, purge_lf_intermediates
from .day_grouper import DayBatch
from .day_master_encoder import build_master_record, encode_day_master
from .ffmpeg_filters import crop_upscaled_from, get_video_dimensions, shorten_ost_for_shorts
from .gps_quality import trim_gps_to_clip_timeline, usable_fix_times, window_median_speed_mps
from .inserts import (
    InsertError,
    insert_label,
    insert_records,
    plan_drive_inserts,
    probe_media,
    shift_photo_inserts,
    splice_day_lf_inserts,
)
from .montage_builder import (
    GOP_DURATION_SEC,
    lf_reencode_args,
    prepend_montage,
    to_cut_segment,
)
from .ost_applier import apply_ost
from .ost_audit_sidecar import (
    OstAuditPendingError,
    OstPostBurnError,
    compute_ost_audit_hash,
    confirmation_is_stale,
    format_ost_audit,
    lf_gate_or_raise,
    postburn_ost_gate_or_raise,
    read_confirmed_hash,
    shorts_gate_or_raise,
    stamp_confirmed,
    superlong_gate_or_raise,
    system_review_passes,
    write_ost_audit,
    write_shorts_ost_audit,
    write_system_review_copy,
)
from .ost_system_review import record_surface_reviews
from .ost_validator import validate_ost
from .preflight import check_lf_work_space, check_superlong_output_space
from .regular_lf_planner import (
    DEFAULT_MIN_DURATION_SEC as DEFAULT_REGULAR_LF_MIN_DURATION_SEC,
)
from .regular_lf_planner import (
    plan_full_drive_parts,
    plan_regular_lfs,
)
from .route_claims import RouteClaimRefusedError
from .schedule_constants import required_shorts_per_lf
from .short_planner import (
    DEFAULT_MAX_DURATION_SEC,
    DEFAULT_MIN_DURATION_SEC,
    DEFAULT_MIN_GAP_SEC,
    DEFAULT_TARGET_DURATION_SEC,
    plan_shorts,
)
from .shorts_extractor import render_short_ffmpeg as _render_short_ffmpeg
from .social_generator import assert_day_title_stamps
from .stop_mask import (
    build_county_change_mask,
    build_gps_gap_mask,
    build_stop_mask,
    mask_summary,
    stop_mask_kwargs,
)
from .title_kit import TitleKit, require_title_kit
from .validation.content_gates import check_short_location_invariant, check_stop_free_windows

logger = logging.getLogger(__name__)

# Score-array column indices (mirror unified_scorer)
COL_TIMESTAMP = 0


def compute_day_file_boundaries(day_batch) -> list[tuple[float, float]]:
    """Return `[(start, end), ...]` master-local ranges for each episode in
    the day batch, accumulated from their durations.

    The day master is a stream-copy concat of the episodes in order, so each
    episode's offset in master time equals the sum of the prior episodes'
    durations.
    """
    return boundaries_from_durations(
        float(getattr(ep, "total_duration", 0.0) or 0.0)
        for ep in (getattr(day_batch, "episodes", None) or [])
    )


def _window_gps_centroid(
    gps_points: Optional[list[tuple[float, float, float]]], w0: float, w1: float
) -> tuple[float, float]:
    """Mean ``(lat, lon)`` of ``gps_points`` (``[(t, lat, lon), ...]``) with
    ``t`` in ``[w0, w1]``, else ``(0.0, 0.0)`` — the synthetic-segment GPS
    centroid for full_drive/superlong LFSegments. No drive-wide fallback: a
    window with zero fixes inside it stays the sentinel (the caller/consumer
    treats ``(0.0, 0.0)`` as "no GPS")."""
    if not gps_points:
        return (0.0, 0.0)
    lats = [lat for (t, lat, lon) in gps_points if w0 <= t <= w1]
    lons = [lon for (t, lat, lon) in gps_points if w0 <= t <= w1]
    if not lats:
        return (0.0, 0.0)
    return (sum(lats) / len(lats), sum(lons) / len(lons))


def assign_shorts_to_lfs_by_source_window(short_ranges, lf_plans, file_boundaries) -> dict:
    """Map each short to the LF whose source segments contain its midpoint.

    Each short is rendered from a `[start_sec, end_sec)` window of the day
    master; each LF is assembled from ``LFSegment``s that translate to
    master-time intervals via ``file_boundaries``. A short belongs to LF X
    when its midpoint falls inside any of LF X's intervals; a short that no
    LF claims (content the percentile cut excluded from every LF body but
    which still scored into the shorts list) attaches to the NEAREST LF.
    Never dropped — the short was rendered, it has to land somewhere.

    This replaced the prior equal-split assignment,
    which produced the correct *count* per LF but arbitrary attribution.

    Returns ``dict[int, list[int]]`` keyed by ``lf.index``, values being
    indices into ``short_ranges`` in their original emission order.
    """
    groups = {
        int(getattr(lf, "index", 0)): [
            to_cut_segment(seg) for seg in (getattr(lf, "segments", None) or [])
        ]
        for lf in lf_plans
    }
    clips = [
        (
            float(getattr(sr, "start_sec", 0.0) or 0.0),
            float(getattr(sr, "end_sec", 0.0) or 0.0),
        )
        for sr in short_ranges
    ]
    return assign_clips_by_source_window(clips, groups, file_boundaries)


def cap_shorts_to_per_lf_capacity(
    short_ranges: list,
    lf_plans: list,
    file_boundaries: list,
    per_lf_capacity: Optional[int],
) -> tuple[list, list]:
    """Drop shorts an LF's publish window physically cannot schedule.

    The planner is DAY-WIDE (``plan_shorts`` picks the best-scoring windows
    across the whole day, with no per-LF awareness) but the scheduler is
    PER-LF: an LF's shorts window runs ``lf_date → next_lf_date`` × the
    region's publish times, so it holds exactly ``required_shorts_per_lf``
    slots and not one more. Attribution is emergent, so a day can plan the
    right TOTAL and still hand one LF more shorts than it can ever post —
    e.g. a day planned for four LFs can hand one LF more shorts than its
    window holds; the drive-wide count gate sees no shortfall and the CSV
    export later refuses the slotless rows.

    Capping here is the robust-by-construction fix: a short that cannot be
    placed is never rendered, so the downstream gate has nothing to catch.
    Only OVER-full LFs are trimmed — an under-full LF keeps everything it
    has, because levelling every LF to the same count would throw away
    footage that does have a slot.

    Returns ``(kept, dropped)``, both in the input's order (which is rank
    order from ``short_planner``, best first). Within an LF the surviving
    shorts are its best-ranked; ``rank`` is read explicitly rather than
    inferred from position. A ``None``/non-positive capacity, no LFs, or an
    unmappable attribution is a no-op — deleting content on a failed mapping
    would be worse than the slotless short. ``_render_day_shorts`` re-raises
    an ambiguous multi-LF mapping, so that guard stays the single decider.
    """
    if not per_lf_capacity or per_lf_capacity <= 0 or not short_ranges or not lf_plans:
        return list(short_ranges), []
    try:
        buckets = assign_shorts_to_lfs_by_source_window(
            short_ranges=list(short_ranges),
            lf_plans=list(lf_plans),
            file_boundaries=file_boundaries,
        )
    except Exception as exc:  # noqa: BLE001 — fail-safe, see docstring
        logger.warning(
            "Shorts per-LF capacity cap skipped — short→LF mapping failed (%s); "
            "keeping all %d planned shorts",
            exc,
            len(short_ranges),
        )
        return list(short_ranges), []

    def _rank(i: int) -> tuple:
        return (float(getattr(short_ranges[i], "rank", i + 1) or (i + 1)), i)

    drop: set[int] = set()
    for lf_index, indices in (buckets or {}).items():
        valid = [i for i in indices if 0 <= i < len(short_ranges)]
        if len(valid) <= per_lf_capacity:
            continue
        surplus = sorted(valid, key=_rank)[per_lf_capacity:]
        drop.update(surplus)
        logger.warning(
            "LF %s drew %d shorts but its publish window holds %d — dropping the "
            "%d worst-ranked before render (they could never be scheduled)",
            lf_index,
            len(valid),
            per_lf_capacity,
            len(surplus),
        )
    if not drop:
        return list(short_ranges), []
    kept = [sr for i, sr in enumerate(short_ranges) if i not in drop]
    dropped = [sr for i, sr in enumerate(short_ranges) if i in drop]
    return kept, dropped


def extract_and_concat_segments(
    master_path: Path,
    segments: list,
    file_boundaries: list[tuple[float, float]],
    output_path: Path,
    work_dir: Path,
    stop_mask=None,
) -> Optional[Path]:
    """Cut each scored segment out of the day master and concat the pieces
    in drive_plan order, without re-encoding.

    Contiguous/overlapping ranges (and sub-GOP gaps, which a stream copy
    cannot excise cleanly) are merged into single extractions, and each
    remaining cut start is snapped FORWARD onto a probed keyframe so
    `-ss … -c copy` never replays the prior segment's tail or resurrects a
    dropped span.

    ``stop_mask`` (per-second bool sequence / callable in master seconds,
    ``True`` == parked) makes the cut layer STOP-aware: two kept ranges are
    never merged across a gap overlapping a masked second, and a range whose
    `-c copy` start would snap back across a masked second is RE-ENCODED with
    an accurate seek instead. When the keyframe probe is unavailable at all,
    EVERY range is re-encoded. ``None`` preserves legacy behaviour.

    ``source_file_index`` on each segment is DAY-LOCAL — it indexes into
    ``file_boundaries`` directly. Callers holding drive-global indices must
    remap first (``phase_bc_day_first._partition_lf_plans_by_date``).

    Returns the concatenated file, or None on any ffmpeg failure or missing
    input.
    """
    master_path = Path(master_path)
    output_path = Path(output_path)

    if not master_path.exists():
        logger.warning(
            "⚠ segment extract: preferred path (extract+concat from day master) "
            "unavailable — master file missing (%s); falling back to no-op "
            "(returning None; no LF will be assembled from this batch)",
            master_path,
        )
        return None
    if not segments:
        logger.warning(
            "⚠ segment extract: no segments to extract (master=%s, output=%s); "
            "falling back to no-op (returning None; no LF will be assembled)",
            master_path,
            output_path,
        )
        return None

    try:
        ranges = segments_to_master_ranges(
            (to_cut_segment(seg) for seg in segments), file_boundaries
        )
    except IndexError as exc:
        logger.warning(
            "⚠ segment extract: segment translation failed (%s); falling back "
            "to no-op (returning None; no LF will be assembled from this batch "
            "of %d segments)",
            exc,
            len(segments),
        )
        return None

    result = extract_and_concat_ranges(
        master_path,
        ranges,
        output_path,
        work_dir,
        exclude_mask=stop_mask,
        # Pin the pipeline's fixed GOP constant — keycut would otherwise
        # MEASURE the master's real GOP, changing the merge on an irregular one.
        max_gap_sec=GOP_DURATION_SEC,
        # Pin the shared long-form encoder args + AAC 256k audio — keycut
        # would otherwise derive an encoder from the master's codec at CRF 18
        # and stream-copy the audio.
        reencode_args=lf_reencode_args(),
        # This concat is an INTERMEDIATE: apply_ost re-encodes it into the
        # final artifact (which writes its own faststart) and cleanup deletes
        # it. keycut's default would rewrite the moov atom over the whole
        # concat for nothing — on a superlong part that is the entire ~118 GB
        # day, and it exceeded keycut's 1800s stream-copy timeout, which this
        # caller cannot raise (extract_and_concat_ranges takes no timeout).
        # Not doing the work beats waiting longer for it.
        faststart=False,
    )
    if result is None:
        logger.warning(
            "⚠ segment extract: stream-copy/concat failed (master=%s, "
            "output=%s, %d range(s)); falling back to no-op (returning None; "
            "no LF will be assembled from this batch)",
            master_path,
            output_path,
            len(ranges),
        )
    return result


@dataclass
class DayArtifacts:
    """Everything a single day produced under the day-first pipeline.

    * `master_path` — final OST-applied day master (super LF). ``None`` when
      the master has been purged post-run-day-pipeline and the day is being
      reconstructed from its final ``lf_XX.mp4`` / ``short_XX.mp4`` files
      alone. Downstream phases (Phase D) read only ``regular_lf_paths``
      and ``short_paths``, so the missing master is a strict superset of
      what's actually consumed.
    * `regular_lf_paths` — OST-applied regular LFs in temporal order.
    * `short_paths` — OST-applied shorts in RANK order (best first) at the
      day level. Downstream, ``phase_bc_day_first`` re-buckets these into
      per-LF lists sorted by source-time start so each LF's shorts publish
      chronologically.
    * `short_ranges` — parallel to ``short_paths``: the ShortRange for each
      emitted short carrying ``start_sec`` / ``end_sec`` in day-master
      timeline. Consumed by phase_bc_day_first to assign each short to
      the LF whose source segments contain its midpoint (shorts↔LF mapping must be source-window-based,
      not positional).
    """

    date: str
    master_path: Optional[Path]
    regular_lf_paths: list = field(default_factory=list)
    # Parallel to ``regular_lf_paths``: each rendered path's authoritative LF
    # index (``lf_plan.index``). A mid-day render skip leaves a gap in
    # ``regular_lf_paths``; this list carries the surviving LFs' identities so
    # downstream pairing keys off the index, never list position — otherwise a
    # later LF's file is silently attributed to an earlier LF's plan/label.
    regular_lf_indices: list = field(default_factory=list)
    short_paths: list = field(default_factory=list)
    short_ranges: list = field(default_factory=list)
    # encode-time integrity metadata (``expected_duration`` /
    # ``master_duration`` / ``master_fingerprint``), merged into
    # ``per_day_completed[date]`` so a later resume can validate the cached
    # master after the raw episodes are purged. ``None`` when no master landed.
    master_record: Optional[dict] = None
    # SHORTS-CONTRACT: the required shorts count this day was planned
    # against (``target_count`` in run_day_pipeline). Threaded to the
    # master-deletion gate in ``phase_bc_day_first`` so "count-matched"
    # compares produced-vs-REQUIRED, not produced-vs-produced. ``None``
    # when no target was in force (or on disk-reconstructed artifacts).
    planned_shorts_count: Optional[int] = None
    # The explicit per-LF target (``_effective_per_lf``) this day
    # actually ran under — set alongside ``planned_shorts_count`` (same
    # ``None``-ness) so a resume's ``_enforce_reconstructed_shortfall`` can
    # recognize a correctly per-LF-capacity-capped day instead of re-deriving
    # the pre-cap total from the CURRENT target and reading it as short.
    shorts_target_in_force: Optional[int] = None
    # The inserts each rendered LF carries, ``{lf_index: [record, ...]}``
    # (``inserts.insert_records``). Phase D reads the count for the description's
    # ``📸`` line and the chapter markers; empty when the run asked for none.
    inserts: dict = field(default_factory=dict)


def _enforce_shorts_contract(
    produced: int,
    required: Optional[int],
    date: str,
    allow_short_count: bool,
    stage: str = "planned",
) -> None:
    """Hard-fail when a day's shorts fall below the required count.

    Fires BEFORE any shorts encode/CSV/upload (``stage="planned"``) and
    again after the render loop (``stage="rendered"``) so render-time
    losses can't erode a passing plan. ``--allow-short-count`` is the
    operator escape (shortfall explicitly accepted).
    """
    if required is None or allow_short_count or produced >= required:
        return
    raise RuntimeError(
        f"SHORTS-CONTRACT: {stage} {produced} < required {required} for day "
        f"{date} — the plan_shorts relaxation ladder is already exhausted "
        f"(timeline physically can't hold {required} windows) or shorts were "
        f"lost at render; pass --allow-short-count to accept the shortfall"
    )


def _cap_shorts_target_by_moving_budget(
    target_count: Optional[int],
    stop_mask,
    min_gap_sec: Optional[float],
) -> Optional[int]:
    """Lower ``target_count`` to what the day's MOVING footage can physically hold.

    Consumes the per-day ``stop_mask`` (``True`` == parked second to exclude, the
    same mask the shorts planner honors). A short needs its full duration inside
    moving footage plus a ``min_gap`` of spacing, so the moving budget can hold at
    most ``moving_seconds / (target_duration + min_gap)`` shorts. Returns the
    lesser of that and ``target_count`` — only ever LOWERS it. A no-op (returns
    ``target_count`` unchanged) when there is no target, no mask, or the mask is
    empty, so GPS-less / mask-off days are unaffected.
    """
    if target_count is None or stop_mask is None or len(stop_mask) == 0:
        return target_count
    moving_seconds = int(len(stop_mask) - int(np.count_nonzero(stop_mask)))
    per_short = DEFAULT_TARGET_DURATION_SEC + (
        min_gap_sec if min_gap_sec is not None else DEFAULT_MIN_GAP_SEC
    )
    if per_short <= 0:
        return target_count
    supportable = int(moving_seconds // per_short)
    return min(target_count, supportable)


def _shorts_planning_mask(
    stop_mask,
    day_scores,
    day_gps_points,
    date: str,
    geocoder=None,
    half_window_s: float = DEFAULT_TARGET_DURATION_SEC,
):
    """The shorts planner's veto: the stop mask plus, on a day with GPS, every
    second a short could not vote its own place over (``build_gps_gap_mask``)
    and every second within one window of a county change
    (``build_county_change_mask``) — a short never spans a county line.

    Planning-only — the persisted ``stop_mask_<date>.npz`` stays the pure stop
    mask the downstream gates read. A GPS-less day, or a trace with no gap,
    plans under the stop mask unchanged; with no ``geocoder`` (legacy / GPS-less
    day) the county-line veto is skipped, exactly the no-GPS behaviour.
    """
    if not isinstance(day_scores, np.ndarray) or day_scores.size == 0:
        return stop_mask
    n = len(day_scores)
    veto = build_gps_gap_mask(n, day_gps_points)
    if veto.any():
        logger.info(
            "Day %s: GPS-gap veto — %d/%d s have no fix a short could name its place from",
            date,
            int(np.count_nonzero(veto)),
            int(veto.size),
        )
    county = build_county_change_mask(n, day_gps_points, geocoder, half_window_s=half_window_s)
    if county.any():
        logger.info(
            "Day %s: county-line veto — %d/%d s sit within %ds of a county change",
            date,
            int(np.count_nonzero(county)),
            int(county.size),
            int(half_window_s),
        )
        veto = veto | county
    if not veto.any():
        return stop_mask
    return veto if stop_mask is None else (np.asarray(stop_mask, dtype=bool) | veto)


def slice_scores_for_day(
    scores: np.ndarray,
    day_start_sec: float,
    day_end_sec: float,
) -> np.ndarray:
    """Return the slice of the drive-wide score array for a single day,
    with timestamps rebased to start at 0.

    The planners (`plan_regular_lfs`, `plan_shorts`) are timeline-agnostic
    — they work in the score array's own coordinate system. The day
    master's on-disk timeline is also 0-based. By rebasing here we make
    the planner's output directly usable as `-ss`/`-to` arguments.

    `day_end_sec` is EXCLUSIVE — this is a half-open interval so
    adjacent days can tile without overlap.
    """
    if scores is None or scores.size == 0:
        return np.zeros((0, 9), dtype=np.float64)

    timestamps = scores[:, COL_TIMESTAMP]
    mask = (timestamps >= day_start_sec) & (timestamps < day_end_sec)
    day_slice = scores[mask].copy()
    if day_slice.size == 0:
        return day_slice
    day_slice[:, COL_TIMESTAMP] -= day_start_sec
    return day_slice


_ARC_LABEL_RE = re.compile(r"\s+to\s+|\s+via\s+", re.IGNORECASE)


def _is_arc_label(text: Optional[str]) -> bool:
    """True when ``text`` is an ``"A to B [via {road}]"`` drive arc rather than a
    single place. A short is one ~38s clip; an arc naming two endpoints can never
    be that clip's own label, so an inherited arc that reaches the burn under an
    empty clip vote is an unverifiable far-corner desync."""
    return bool(text) and bool(_ARC_LABEL_RE.search(text))


def _resolve_short_ost_text(
    candidate: Optional[str],
    parent_ost_text: Optional[str],
    rank: int,
    date: str,
    drive_location: Optional[str] = None,
) -> tuple[str, int]:
    """:func:`_gate_short_ost_text` plus the provenance of what it picked.

    Returns ``(text, source_index)`` — the index into the cascade below that the
    burnable title actually came from (0 ``candidate`` / 1 ``parent_ost_text`` /
    2 ``drive_location``). The shorts loop needs the provenance, not just the
    string: the short-title region gate treats clip-DERIVED text (index 0, off this
    clip's own GPS vote) as corroborated by construction, while inherited text
    stays strictly checked against the clip's sampled counties.

    Every short MUST carry a title:
    a title-less short is never silently shipped. Try each source in cascade
    order — ``candidate`` (GPS-derived / per-LF), the parent LF's canonical OST
    (``parent_ost_text``), then the drive-wide ``location`` — shortening and
    validating each. A real place name that only OVERFLOWS the 2-line mobile
    frame is hard-fit to the frame rather than dropped (:func:`_fit_shorts_title`);
    only genuine garbage (``"None, None"`` / coordinate / HOA / placeholder) is
    skipped. When NO source yields a burnable title the day hard-fails (mirrors
    the long-form ``OST-EMPTY-LOCATION`` contract) so the operator supplies
    ``--per-lf-location`` — never a silently-blank ship. Each rejection logs a
    greppable ``SHORTS-OST-GATE`` WARN.
    """
    for which, source in enumerate((candidate, parent_ost_text, drive_location)):
        if not source or not source.strip():
            continue
        shortened = shorten_ost_for_shorts(source)
        ok, reason = validate_ost(shortened, "shorts")
        if ok:
            return shortened, which
        # A real label that merely overflows the 2-line frame is NOT garbage —
        # hard-fit it so the short still carries a title. Garbage patterns
        # (None/coord/HOA/placeholder) don't report "exceeds"/"overflows" and
        # fall through to the next source instead of being fitted.
        if "exceeds" in reason or "overflows" in reason:
            fitted = _fit_shorts_title(shortened)
            ok_fit, _ = validate_ost(fitted, "shorts")
            if ok_fit:
                logger.warning(
                    "SHORTS-OST-GATE: rank %d (day %s) hard-fit overlong %r -> %r to frame",
                    rank,
                    date,
                    source,
                    fitted,
                )
                return fitted, which
        logger.warning(
            "SHORTS-OST-GATE: rank %d (day %s) rejected %r->%r (%s)",
            rank,
            date,
            source,
            shortened,
            reason,
        )
    raise RuntimeError(
        f"SHORTS-OST-UNRESOLVED: rank {rank} (day {date}) — no burnable short "
        "title from candidate/parent/drive-location (all empty or garbage). A "
        "short must carry a title; supply --per-lf-location for the parent LF."
    )


def _gate_short_ost_text(
    candidate: Optional[str],
    parent_ost_text: Optional[str],
    rank: int,
    date: str,
    drive_location: Optional[str] = None,
) -> str:
    """Shorten + validate a short's OST text before the irreversible 9:16 burn —
    the burnable title alone. See :func:`_resolve_short_ost_text` for the cascade
    itself and for the caller that also needs to know which source won."""
    return _resolve_short_ost_text(
        candidate, parent_ost_text, rank, date, drive_location=drive_location
    )[0]


def _fit_shorts_title(text: str) -> str:
    """Truncate ``text`` until it fits the shorts overlay (<=2 lines, <=char
    budget), so a real place name that overflows the mobile frame still yields
    a burnable title instead of being dropped. Drops whole trailing comma
    clauses first (", County Kerry" — keeps the distinguishing town), then
    trailing words, then hard-cuts characters when a single word is itself too
    wide. Returns non-empty for any non-empty input.
    """
    from .ffmpeg_filters import SHORTS_MAX_TEXT_LEN, strip_trailing_connectors
    from .ost_validator import text_fits_overlay

    def _fits(s: str) -> bool:
        return len(s) <= SHORTS_MAX_TEXT_LEN and text_fits_overlay(s, "shorts")

    # Region-preserving: drop whole trailing ", County X" clauses before any
    # word/char chop so a survivor keeps the town intact ("Kenmare, County
    # Kerry" -> "Kenmare", never "Kenmare, County").
    parts = [p.strip() for p in text.split(",") if p.strip()]
    while len(parts) > 1:
        joined = ", ".join(parts)
        if _fits(joined):
            return joined
        parts = parts[:-1]
    text = ", ".join(parts) if parts else text.strip()

    words = text.split()
    while len(words) > 1:
        # Strip any orphaned trailing connector ("Killarney to") BEFORE the
        # fit check, so a chop landing right after a joiner token never
        # returns a dangling arc.
        joined = strip_trailing_connectors(" ".join(words))
        if _fits(joined):
            return joined
        words = words[:-1]
    single = words[0] if words else text.strip()
    while len(single) > 1 and not _fits(single):
        single = single[:-1]
    return single


def _is_publish_run(state) -> bool:
    """True only when ``state`` says this run publishes (``youtube_upload`` is True).

    Phase D's own publish test (``bool(state.youtube_upload)``), read off a
    ``PipelineState`` or a plain dict. ``cmd_drive`` writes it as a concrete
    bool on every invocation; the identity check keeps a mock or a missing
    key from reading as publish intent.
    """
    if state is None:
        return False
    if isinstance(state, dict):
        return state.get("youtube_upload") is True
    return getattr(state, "youtube_upload", None) is True


def _short_upscaled_from(master_path) -> Optional[list]:
    """``[crop_w, crop_h]`` when the master is too small for a native 1080x1920 crop, else None."""
    try:
        return crop_upscaled_from(*get_video_dimensions(master_path))
    except Exception:  # noqa: BLE001 — recording only; an unreadable master is gated elsewhere
        return None


def _render_day_shorts(
    *,
    day_dir: Path,
    date: str,
    day_batch: DayBatch,
    short_ranges: list,
    allow_short_count: bool,
    gps_points_by_date: Optional[dict],
    lf_plans: list,
    per_lf_ost_text: Optional[dict],
    per_lf_locations: Optional[dict],
    per_lf_overrides: Optional[dict] = None,
    location: str,
    short_ost_text_fn: Optional[Callable],
    master_source_for_extract: Path,
    short_regions_fn: Optional[Callable] = None,
    state=None,
    stop_mask=None,
    short_route_fn: Optional[Callable] = None,
) -> tuple:
    """Extract + OST-burn every planned short for one day (Step 5).

    Derives every planned short's final burned text first, then runs the
    pre-burn OST review gate (``shorts_gate_or_raise``) — the shorts analog of
    the per-LF gate — BEFORE any short is rendered, so a wrong shorts burn is
    caught before the irreversible pixels land. The SHORTS-CONTRACT post-render
    gate is unchanged. Returns
    ``(short_paths, short_ranges_emitted, short_ost_audit_entries)``.
    """
    # ── Step 5: extract + OST each short (rank-ordered) ───────────
    shorts_dir = day_dir / "shorts"
    shorts_dir.mkdir(parents=True, exist_ok=True)
    short_paths: list = []
    short_ranges_emitted: list = []  # parallel to short_paths
    short_ost_audit_entries: list = []  # per-short burned-text audit coverage
    # (short_range, duration, ost_text) specs collected in the derive pass;
    # the actual burns run in a second pass AFTER the pre-burn review gate.
    _render_specs: list = []
    # Per-rank route-claim outcomes for the Shorts system review record.
    _short_outcomes: dict = {}
    _short_route_records: dict = {}  # rank -> RouteEvidence.record() for the place gate
    # Every Short whose retained GPS cannot support its text, reported together.
    _route_refusals: list = []
    # Degraded-derivation visibility: per-day counts feeding ONE summary
    # WARN — how many GPS-bearing shorts shipped inherited OST text and how
    # many skipped the region gate (gate failing open).
    _gps_bearing_total = 0
    _ost_degrade_count = 0
    _region_skip_count = 0
    # GPS points for this day (if supplied). Every short — even
    # single-segment — routes its OST composition through
    # ``trim_gps_to_clip_timeline`` so the code path is uniform (no
    # one-offs) and the surface describes what's on-screen, not the
    # raw source footage.
    day_gps_points = None
    if gps_points_by_date is not None:
        day_gps_points = gps_points_by_date.get(date)
    # A day with usable fixes plans its shorts under the GPS-gap veto, so every
    # short on it must hold fixes of its own (checked per short below).
    _day_has_fixes = bool(usable_fix_times(day_gps_points))
    # Precompute short → parent-LF mapping
    # so the short OST burn can fall back to the parent LF's
    # per_lf_ost_text (single source of truth for
    # OST / title / thumbnail) BEFORE the drive-wide
    # ``location``. Without this, GPS-less drives with per-LF location
    # overrides shipped shorts whose OST showed only the drive-wide
    # ``--location`` (e.g. "Dingle Peninsula, County Kerry") while the
    # parent LF's OST correctly showed the per-LF override (e.g.
    # "Dingle Peninsula toward Gortmore, County Kerry") — three-surface
    # divergence is a content-integrity violation.
    _short_idx_to_lf_idx: dict[int, int] = {}
    if short_ranges and (lf_plans or []):
        try:
            _file_boundaries = compute_day_file_boundaries(day_batch)
            _lf_to_shorts = assign_shorts_to_lfs_by_source_window(
                short_ranges=list(short_ranges),
                lf_plans=list(lf_plans or []),
                file_boundaries=_file_boundaries,
            )
            for _lf_idx, _short_indices in (_lf_to_shorts or {}).items():
                for _i in _short_indices:
                    _short_idx_to_lf_idx[int(_i)] = int(_lf_idx)
        except Exception as exc:  # noqa: BLE001 — see the policy below
            # A swallowed mapping failure left ``_short_idx_to_lf_idx``
            # empty, so every short's parent LF became None — GPS-less shorts
            # skipped the anchor cascade and burned the drive-wide ``location``,
            # silently swapping an AUTHORITATIVE per-LF override for it, and the
            # per-short contract assertion + region gate + degrade WARN were all
            # guarded off. Mirror ``phase_bc_day_first``'s equal-split hard-fail:
            if len(lf_plans or []) > 1:
                # >1 LF: attribution is AMBIGUOUS — never guess. Re-raise.
                raise RuntimeError(
                    f"Day {date}: short→LF mapping failed ({exc}) with "
                    f"{len(lf_plans or [])} LFs — ambiguous short attribution; "
                    "cannot fall back to the drive-wide OST label. Re-run the "
                    "day so short_ranges/lf_plans persist, or repair the day "
                    "artifact."
                ) from exc
            # Exactly one LF: attribution is unambiguous by construction — map
            # every short to that sole LF's anchors (never the drive-wide fallback).
            _sole_lf_idx = int(getattr(lf_plans[0], "index", 0))
            for _i in range(len(short_ranges)):
                _short_idx_to_lf_idx[int(_i)] = _sole_lf_idx
            logger.warning(
                "Day %s: short→LF mapping failed (%s) — single LF, mapping all "
                "shorts to LF %d (unambiguous)",
                date,
                exc,
                _sole_lf_idx,
            )

    # Shorts go through ``render_short_ffmpeg`` (single
    # ffmpeg invocation: center-crop to 9:16 + scale to 1080×1920 +
    # 9:16-positioned OST). The previous loop used ``extract_slice``
    # (16:9 stream-copy trim) + ``apply_ost(layout="short")`` which
    # produced 3840×2160 16:9 outputs Metricool / YouTube Shorts /
    # TikTok / Reels all rejected.
    for _short_list_idx, short_range in enumerate(short_ranges):
        rank = short_range.rank
        short_duration = short_range.end_sec - short_range.start_sec
        if short_duration < 5.0:
            logger.warning(
                "Day %s: short rank %d is degenerate (%.1fs), skipping",
                date,
                rank,
                short_duration,
            )
            continue
        # OST text priority for shorts (single source of truth):
        #   1. short_ost_text_fn(trimmed_gps, short_range) — GPS-derived
        #      route text describing exactly what's on-screen (GPS-bearing).
        # When the clip has NO GPS on a GPS-less day, there is no on-screen
        # route to derive, so the short must NOT inherit the parent LF's
        # GPS-derived ``"A to B via Road"`` title — that title names two
        # places this short's footage may be neither. The GPS-less
        # anchor is, in order: an operator ``--per-lf-location`` override
        # (authoritative, verbatim), else the parent county, else the
        # drive-wide ``location`` (legacy last resort) — never the destination
        # POI, which is where the drive went, not where the clip was filmed.
        # On a day WITH GPS a clip with no fix of its own is never planned
        # (GPS-gap veto) and hard-fails here if one slips through.
        _trimmed_points = trim_gps_to_clip_timeline(
            day_gps_points,
            [(short_range.start_sec, short_range.end_sec)],
        )
        _parent_lf_idx = _short_idx_to_lf_idx.get(_short_list_idx)
        _gps_less = not _trimmed_points
        if _gps_less and _day_has_fixes:
            raise RuntimeError(
                f"Short place FAILED (day {date}, rank {rank}, parent LF "
                f"{_parent_lf_idx}): the day has GPS but the clip "
                f"{short_range.start_sec:.0f}-{short_range.end_sec:.0f}s has no GPS "
                "fix of its own, so its place cannot be verified and no LF label, "
                "destination or county may stand in. Do NOT encode/upload this short."
            )
        if _gps_less and _is_publish_run(state):
            # A publish run may name only the place a clip's OWN GPS verified.
            # A GPS-less short can only be labeled by the operator override,
            # the county or the drive-wide location — none of them its own —
            # so refuse it here, before any encode (Phase D would refuse the
            # sidecar anyway, after the render was paid for).
            raise RuntimeError(
                f"Short place FAILED (day {date}, rank {rank}, parent LF "
                f"{_parent_lf_idx}): this is a publish run and the clip has no GPS "
                "fix of its own, so its label would come from the operator "
                "override, the county or the drive-wide location, not its own "
                "clip. Do NOT encode/upload this short (re-run as an audit pass "
                "or drop the clip)."
            )
        if not _gps_less:
            _gps_bearing_total += 1
        # True once the burn text is this clip's OWN dominant_location vote (set
        # below, confirmed after the shorten/validate cascade) — the region gate
        # reads it to skip a check that is vacuous against that vote.
        _ost_from_clip_vote = False
        # Set only when derivation was actually ATTEMPTED (GPS present)
        # and came back empty — a genuine geocoder degrade, never "no GPS to
        # vote". The gate below reads this to hard-fail that one case.
        _gps_fallback_unverified = False
        short_ost_text = location
        # ``_parent_fallback`` is the anchor the OST gate falls back to when
        # the shorten/validate step rejects ``short_ost_text``; for a GPS-less
        # short it must be the GPS-less anchor (never the suppressed A→B title).
        _parent_fallback: Optional[str] = (
            per_lf_ost_text.get(_parent_lf_idx)
            if (per_lf_ost_text and _parent_lf_idx is not None)
            else None
        )
        if _gps_less and _parent_lf_idx is not None:
            _parent_override = (per_lf_overrides or {}).get(_parent_lf_idx)
            _parent_ost = (per_lf_ost_text or {}).get(_parent_lf_idx)
            _county = (per_lf_locations or {}).get(_parent_lf_idx)
            if isinstance(_parent_override, str) and _parent_override.strip():
                # Operator override is AUTHORITATIVE — burn it verbatim.
                short_ost_text = _parent_override
                _parent_fallback = _parent_override
            elif (_parent_ost or "").strip():
                # There IS a GPS-derived A→B title to SUPPRESS — anchor on the
                # county; never the A→B endpoints nor the destination POI.
                _anchor = _county if (_county or "").strip() else None
                short_ost_text = _anchor
                _parent_fallback = _anchor
            elif (_county or "").strip():
                # No A→B title — legacy county-level fallback.
                short_ost_text = _county
                _parent_fallback = _county
            # else: keep drive-wide ``location`` (legacy last resort).
        elif _parent_lf_idx is not None:
            # GPS-bearing short: prime from the parent LF's canonical text; the
            # ``short_ost_text_fn`` vote over this clip's own GPS replaces it
            # below, or the short hard-fails. An operator ``--per-lf-location``
            # label names the whole LF, never this clip, so it is not read here.
            if (
                per_lf_ost_text
                and _parent_lf_idx in per_lf_ost_text
                and per_lf_ost_text[_parent_lf_idx]
            ):
                short_ost_text = per_lf_ost_text[_parent_lf_idx]
            elif (
                per_lf_locations
                and _parent_lf_idx in per_lf_locations
                and per_lf_locations[_parent_lf_idx]
            ):
                short_ost_text = per_lf_locations[_parent_lf_idx]
        if short_ost_text_fn is not None and _trimmed_points:
            _degrade_reason = "derivation returned no label (geocode quorum miss / no county)"
            try:
                derived = short_ost_text_fn(_trimmed_points, short_range)
            except Exception as exc:  # noqa: BLE001 — never fail a short on OST text derivation
                _degrade_reason = f"short_ost_text_fn raised: {exc}"
                derived = None
            if derived:
                short_ost_text = derived
                _ost_from_clip_vote = True
            else:
                # Never SILENTLY ship inherited parent-LF/drive-wide
                # text on a GPS-bearing short whose own derivation degraded —
                # loud, upstream of the irreversible encode. A WARN alone
                # lets an inherited caption ship, so this also
                # flags the hard gate a few lines below.
                _ost_degrade_count += 1
                _gps_fallback_unverified = True
                logger.warning(
                    "SHORT-OST-DEGRADED: day %s short rank %d (parent LF %s) — %s; "
                    "shipping inherited text %r",
                    date,
                    rank,
                    _parent_lf_idx,
                    _degrade_reason,
                    short_ost_text,
                )

        # Contract assertion: if a GPS-BEARING short whose parent LF has
        # per-LF OST text still ended up with the drive-wide ``location``
        # verbatim, construction logic regressed (mirror of the LF OST-burn
        # assertion at run_day_pipeline's LF loop). GPS-aware: a GPS-less
        # short legitimately anchors on the override/county/drive-wide
        # ``location`` (the A→B title is deliberately suppressed), so this
        # check applies only to the GPS-bearing path.
        if (
            not _gps_less
            and _parent_lf_idx is not None
            and per_lf_ost_text
            and _parent_lf_idx in per_lf_ost_text
            and (per_lf_ost_text[_parent_lf_idx] or "").strip()
            and (location or "").strip()
            and (short_ost_text or "").strip() == (location or "").strip()
            and (per_lf_ost_text[_parent_lf_idx] or "").strip() != (location or "").strip()
        ):
            raise RuntimeError(
                f"Short OST contract violation: short rank {rank} "
                f"(parent LF {_parent_lf_idx}) burned drive-wide --location "
                f"{location!r} despite per_lf_ost_text[{_parent_lf_idx}]="
                f"{per_lf_ost_text[_parent_lf_idx]!r}. Construction logic "
                "regressed."
            )
        # Shorten for the 9:16 mobile frame then validate, falling
        # back to ``_parent_fallback`` — never burn raw
        # "None, None"/coord/HOA garbage into the 14 shorts/LF. For a
        # GPS-less short ``_parent_fallback`` is the GPS-less anchor
        # (override/county), NOT the suppressed A→B title; when
        # that anchor is None the gate routes to the empty-overlay path
        # (an "omit" — never a silently-blank ship), aligning with how the
        # burn handles a missing short OST text.
        short_ost_text, _ost_source = _resolve_short_ost_text(
            short_ost_text, _parent_fallback, rank, date, drive_location=location
        )
        # The cascade can reject the clip-derived candidate as garbage/overlong
        # and swap in the parent-LF or drive-wide fallback; that text was never
        # checked against this clip's GPS, so the short hard-fails upstream of
        # the encode instead of naming a place it may not be at.
        if _ost_from_clip_vote and _ost_source != 0:
            raise RuntimeError(
                f"Short place FAILED (day {date}, rank {rank}, parent LF "
                f"{_parent_lf_idx}): the 9:16 text cascade rejected this clip's own "
                f"GPS place and fell back to {short_ost_text!r}, which is not "
                "verified against this clip's GPS. Do NOT encode/upload this short."
            )
        # The long-form route-claim rule at Short scale, over this clip's own
        # retained window (one sample per GAP_S, no place floor): a gap, an
        # unsettled sample or an omitted place refuses or broadens the text.
        if short_route_fn is not None and _trimmed_points:
            _requested = short_ost_text
            try:
                _route_label, _route_note = short_route_fn(
                    day_gps_points, short_range, short_ost_text
                )
            except RouteClaimRefusedError as exc:
                _route_refusals.append(f"rank {rank}: {exc}")
                continue
            if _route_note:
                logger.warning("Short rank %d (day %s): %s", rank, date, _route_note)
                short_ost_text = _route_label
            _record = getattr(short_route_fn, "last_record", None)
            _short_route_records[rank] = _record if isinstance(_record, dict) else None
            # What this clip's own retained GPS was measured to support.
            _short_outcomes[rank] = {
                "refused": False,
                "reason": _route_note or "",
                "label_requested": _requested,
                "ost_text": short_ost_text,
            }
        # Cross-surface location gate for shorts: hard-fail UPSTREAM of
        # the irreversible 9:16 encode when the burned county is in ZERO of the
        # clip's own sampled GPS regions — the shorts analog of the LF location
        # invariant. ``short_regions_fn`` reuses ``dominant_location`` over the
        # SAME ``_trimmed_points`` the OST text derives from. No-op on a
        # GPS-less / empty clip (mirrors the LF invariant's legacy skip).
        # Clip-DERIVED text always reaches this gate with regions to check:
        # ``short_ost_text_fn`` only labels a quorum-passing clip (destination
        # included), and ``short_regions_fn`` passes on the same quorum — an
        # empty-regions skip here therefore only ever covers INHERITED
        # (parent-LF/drive-wide) text, which the LF-level gates already vetted.
        _clip_regions: Optional[list] = None
        _regions_error = False
        if short_regions_fn is not None and _trimmed_points:
            try:
                _clip_regions = short_regions_fn(_trimmed_points, short_range)
            except Exception as exc:  # noqa: BLE001
                # A genuine region-derivation ERROR is NOT the same as a
                # genuinely GPS-less clip. Swallowing it here skipped the
                # pre-encode gate AND left sampled_regions empty (vacuous CSV
                # gate). Distinguish the two: an exception hard-fails below
                # regardless of label form; a clean empty vote keeps the
                # existing arc-hard-fail / county-fail-open behavior.
                logger.error(
                    "Day %s: short_regions_fn raised for short rank %d: %s",
                    date,
                    rank,
                    exc,
                )
                _clip_regions = None
                _regions_error = True
            if _regions_error:
                raise RuntimeError(
                    f"Short region gate ERROR (day {date}, rank {rank}, parent LF "
                    f"{_parent_lf_idx}): deriving the clip's sampled regions failed. "
                    "This is a genuine error, not a GPS-less clip — do NOT "
                    "encode/upload with an unverifiable location."
                )
            if _clip_regions:
                _loc_gate = check_short_location_invariant(
                    short_ost_text,
                    _clip_regions,
                    clip_derived=_ost_from_clip_vote,
                    sampled_vote=getattr(_clip_regions, "vote", None),
                )
                if not _loc_gate.passed:
                    raise RuntimeError(
                        f"Short location invariant FAILED (day {date}, rank {rank}, "
                        f"parent LF {_parent_lf_idx}): {_loc_gate.message}. Do NOT "
                        "encode/upload this short — its burned location desynced from "
                        "its own clip GPS."
                    )
            else:
                # Decouple the region-vote skip from the geocode-quorum
                # miss that ALSO triggered parent-text inheritance above. An
                # inherited "A to B [via road]" arc is never a single clip's own
                # label; shipping it under an EMPTY clip vote (nothing to verify
                # it against) is exactly the far-corner desync this gate exists to
                # stop — hard-block upstream of the 9:16 encode rather than fail
                # open. A clean inherited county (degraded-but-correct, common)
                # carries no arc and still fails open below.
                if _is_arc_label(short_ost_text):
                    raise RuntimeError(
                        f"Short location invariant FAILED (day {date}, rank {rank}, "
                        f"parent LF {_parent_lf_idx}): burned text {short_ost_text!r} "
                        "is an inherited A-to-B drive arc but the clip's own GPS "
                        "vote is EMPTY (geocode quorum miss) — nothing verifies the "
                        "arc against this clip. Do NOT encode/upload; investigate "
                        "the geocoder (--per-lf-location labels the LF only, never "
                        "a GPS-bearing short)."
                    )
                # Empty/exception regions on a GPS-BEARING short means the
                # gate fails OPEN — count it into the per-day degrade summary.
                # (A GPS-less clip never reaches here: guarded on _trimmed_points.)
                _region_skip_count += 1
        # Stop-free cross-check: hard-fail UPSTREAM of the irreversible 9:16
        # encode when this short's own retained window is MAJORITY parked — the
        # failure class the per-day stop mask exists to prevent, verified here
        # per-second against GPS speed (mask as the per-second fallback). No-op
        # on a GPS-less day (mirrors the location invariant's legacy skip).
        _stopfree = check_stop_free_windows(
            [(float(short_range.start_sec), float(short_range.end_sec))],
            day_gps_points,
            stop_mask=stop_mask,
        )
        if not _stopfree.passed:
            raise RuntimeError(
                f"Short stop-free gate FAILED (day {date}, rank {rank}, parent LF "
                f"{_parent_lf_idx}): {_stopfree.message}. Do NOT encode/upload this "
                "short — its retained window is majority parked."
            )
        # Audit enrichment: the clip's GPS ground speed (a parked window reads
        # ~0). No loudness counterpart: shorts ship muted (``-an``), so nothing
        # measures a per-short LUFS and the audit must not imply one.
        _median_speed = window_median_speed_mps(
            day_gps_points, float(short_range.start_sec), float(short_range.end_sec)
        )
        # hard-fail UPSTREAM of the irreversible 9:16 encode when this
        # short's own clip GPS was present but its place derivation degraded
        # to the inherited parent-LF/drive-wide label — the fallback fired
        # for a reason OTHER than "no GPS" (geocoder quorum miss/exception).
        # A genuinely GPS-less short (nothing of its own to vote) is excluded
        # above — only an unverified degrade reaches here.
        if _gps_fallback_unverified:
            raise RuntimeError(
                f"Short place fallback FAILED (day {date}, rank {rank}, parent "
                f"LF {_parent_lf_idx}): this short's own clip GPS was present "
                f"but its place derivation degraded, so it fell back to the "
                f"inherited label {short_ost_text!r}. Do NOT encode/upload this "
                "short with an unverified fallback place — investigate the "
                "geocoder (--per-lf-location labels the LF only, never a "
                "GPS-bearing short)."
            )
        # Record the final burned text so the shorts surface gets the
        # audit coverage the per-LF sidecar (written pre-shorts) can't.
        # Persist this short's OWN clip-GPS vote regions — an independent
        # source from the label — so the per-short CSV region invariant checks
        # the shipped surfaces against the clip's footage, not the label itself.
        # Persist WHICH source won the burned text — ``_ost_from_clip_vote``
        # was already tracked above but discarded, so a reviewer/CSV consumer
        # couldn't tell a genuine clip-derived place from an inherited
        # parent-LF/drive-wide fallback.
        short_ost_audit_entries.append(
            {
                "rank": rank,
                "parent_lf_index": _parent_lf_idx,
                "ost_text": short_ost_text,
                "start_sec": float(short_range.start_sec),
                "end_sec": float(short_range.end_sec),
                "regions": list(_clip_regions or []),
                "roads": list(getattr(_clip_regions, "roads", None) or []),
                "route_record": _short_route_records.get(rank),
                "median_speed_mps": _median_speed,
                "place_source": "clip_gps" if _ost_from_clip_vote else "lf_label",
            }
        )
        # Defer the irreversible burn — collect the render spec so the
        # pre-burn review gate can run over ALL shorts' text before any pixels.
        _render_specs.append((short_range, short_duration, short_ost_text))

    # One per-day summary of degraded shorts derivation (loud, not a
    # hard fail — the pre-burn review gate below is the operator catch).
    if _ost_degrade_count or _region_skip_count:
        logger.warning(
            "SHORTS-DEGRADE-SUMMARY: day %s — %d of %d GPS-bearing shorts degraded to "
            "inherited (parent-LF/drive-wide) OST text; region gate skipped for %d of %d "
            "(degraded region derivation)",
            date,
            _ost_degrade_count,
            _gps_bearing_total,
            _region_skip_count,
            _gps_bearing_total,
        )

    # ── Pre-burn OST review gate for shorts ────────────────────
    # Mirror of the per-LF gate: run AFTER every short's text is derived but
    # BEFORE any pixels burn, so a wrong shorts burn is caught upstream of the
    # irreversible artefact instead of merely recorded after it.
    # Every entry needs a stamp and no two share a title prefix -- refuse
    # here, before any pixels burn, not at enqueue.
    if _route_refusals:
        raise RouteClaimRefusedError(
            f"Short route claims REFUSED (day {date}); nothing rendered:\n"
            + "\n".join(_route_refusals)
        )
    assert_day_title_stamps(date, short_ost_audit_entries)
    if state is not None:
        record_surface_reviews(state, "shorts", date, short_ost_audit_entries, _short_outcomes)
    shorts_gate_or_raise(day_dir.parent, date, short_ost_audit_entries, state)

    # ── Step 5b: burn every gated short (rank-ordered) ────────────
    # render_short_ffmpeg crops + scales to 1080×1920 9:16 AND composites the
    # short-layout OST in one pass (no separate extract_slice / apply_ost).
    for short_range, short_duration, short_ost_text in _render_specs:
        rank = short_range.rank
        final_short = shorts_dir / f"short_{rank:02d}.mp4"
        rendered_ok = _render_short_ffmpeg(
            source_video=master_source_for_extract,
            output_path=final_short,
            start_time=float(short_range.start_sec),
            duration=float(short_duration),
            location=short_ost_text,
        )
        if not rendered_ok:
            logger.warning(
                "Day %s: render_short_ffmpeg failed for short rank %d — skipping",
                date,
                rank,
            )
            continue
        short_paths.append(final_short)
        short_ranges_emitted.append(short_range)
        # Bind this rank's audit entry to the artefact it describes. The entry
        # was built pre-burn (nothing was rendered yet), so without this the
        # post-burn check has no file to open and the audit stays pure attestation.
        for _entry in short_ost_audit_entries:
            if _entry.get("rank") == rank:
                _entry["path"] = str(final_short)
                break

    # SHORTS-CONTRACT post-render check: degenerate-skip / render-failure
    # `continue`s above silently shrink the output — a render-time loss
    # must not erode a plan that passed the pre-encode gate.
    _enforce_shorts_contract(
        len(short_paths), len(short_ranges), date, allow_short_count, stage="rendered"
    )
    return short_paths, short_ranges_emitted, short_ost_audit_entries


def _lf_ost_fingerprint(location, montage_duration, crossfade_sec, segments=None) -> str:
    """Stable hash of the material render inputs for one regular LF.

    Persisted next to ``lf_NN.mp4`` as ``lf_NN.ost.sha256`` after a burn.
    On resume the per-LF loop reuses a final whose stored fingerprint still
    matches — unchanged inputs skip the ~10-15min NVENC re-encode (the
    documented shorts-OST-gate → confirm → resume flow). A materially changed
    LF re-burns.

    The hash covers the burned text/montage/crossfade AND:

    * the LF's SEGMENT WINDOWS (each ``(source_file_index, start, end)``) — a
      re-plan (e.g. the per-day stop mask now cutting a parked gap out of a
      segment) moves the windows, so the cached ``lf_NN.mp4`` (assembled from
      the OLD windows, possibly including parked footage) is NOT reused; and
    * the attenuate-only audio-filter policy (``AUDIO_FILTER_ALGO_VERSION`` +
      the ceiling/true-peak constants) — a run whose audio rule changed (the
      loudnorm=-20 boost → attenuate-only fix) re-derives the LF instead of
      shipping the old (audible) audio.
    """
    from . import encoding_constants as ec

    seg_sig = ";".join(
        f"{int(getattr(s, 'source_file_index', 0))}:"
        f"{float(getattr(s, 'start_time', 0.0)):.4f}:"
        f"{float(getattr(s, 'end_time', 0.0)):.4f}"
        for s in (segments or [])
    )
    audio_sig = (
        f"v{int(ec.AUDIO_FILTER_ALGO_VERSION)}:"
        f"{float(ec.AUDIO_SILENCE_CEILING_LUFS):.2f}:"
        f"{float(ec.AUDIO_TRUE_PEAK_CAP):.2f}"
    )
    raw = (
        f"{location or ''}\x00{float(montage_duration):.4f}\x00{float(crossfade_sec):.4f}"
        f"\x00{seg_sig}\x00{audio_sig}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _superlong_ost_fingerprint(
    location,
    ranges,
    crossfade_sec,
    lower_thirds=None,
    lower_third_duration_sec: float = 0.0,
    lower_third_min_gap_sec: float = 0.0,
) -> str:
    """Stable hash of the material render inputs for one superlong part.

    Persisted next to ``sl_NN.mp4`` as ``sl_NN.ost.sha256`` after a burn —
    the freshness mechanism, mirrored from ``_lf_ost_fingerprint`` for
    superlong's own ``apply_ost`` inputs (a part has no montage prepend,
    but DOES burn chapter-derived ``lower_thirds`` an LF never has).

    Covers ``ranges`` (the part's SEGMENT WINDOWS, i.e. ``part.ranges``) —
    a re-plan that shifts a part's boundaries while its part INDEX (and
    therefore ``sl_NN.mp4``'s filename) stays the same must NOT reuse the
    old-boundary bytes, same reasoning as the LF fingerprint's documented
    re-plan case — plus the burned title (``location``), the crossfade,
    and the lower-thirds text/timing, and the attenuate-only audio-filter
    policy (``apply_ost`` applies it identically to both surfaces).
    """
    from . import encoding_constants as ec

    range_sig = ";".join(f"{float(s):.4f}:{float(e):.4f}" for s, e in (ranges or []))
    lt_sig = ";".join(f"{lbl}:{float(t):.4f}" for lbl, t in (lower_thirds or []))
    audio_sig = (
        f"v{int(ec.AUDIO_FILTER_ALGO_VERSION)}:"
        f"{float(ec.AUDIO_SILENCE_CEILING_LUFS):.2f}:"
        f"{float(ec.AUDIO_TRUE_PEAK_CAP):.2f}"
    )
    raw = (
        f"{location or ''}\x00{float(crossfade_sec):.4f}\x00{range_sig}"
        f"\x00{lt_sig}\x00{float(lower_third_duration_sec):.4f}"
        f"\x00{float(lower_third_min_gap_sec):.4f}\x00{audio_sig}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _build_superlong_part_metadata(
    *,
    part,
    date: str,
    location: str,
    gpx_path: Optional[Path],
    resolved_location,
    config,
    stages_dir: Path,
    day_journey=None,
    kit: Optional[TitleKit] = None,
) -> dict:
    """Build one superlong part's title / chapters / description.

    ``location`` is the SAME string ``apply_ost`` burns onto the part's
    pixels (single source of truth — the title is never a different label
    than the burn) and ``resolved_location`` is the day's already-resolved
    ``ResolvedLocation`` (reused, never re-derived from raw GPX — see the
    per-LF chapters call this mirrors in ``phase_d_metadata``). ``day_journey``
    (built once per day by ``_produce_superlong_parts`` via
    ``chapters.generator.build_day_master_journey``) is the preferred GPS
    source — it spans every episode of the day, windowed onto THIS part's
    own ``part.ranges`` (day-master seconds) instead of one linear scale of
    a single episode's GPX. ``None`` (no ``gps_points_by_date`` entry for
    this day) falls back to the legacy ``gpx_path`` behaviour. GPS-less
    days (neither available) skip chapters entirely — ``has_gps=False`` is
    the same robust-by-construction no-op
    ``validation.content_gates.check_metadata_quality`` already honours
    for GPS-less regular LFs.
    """
    kit = require_title_kit(kit)
    title = kit.superlong_title(location, part.part_index, part.part_count)
    has_gps = day_journey is not None or gpx_path is not None
    chapters: list = []
    chapter_text = ""
    if has_gps:
        try:
            generator = kit.chapter_generator(config)
            chapters = generator.generate(
                gpx_path=gpx_path,
                day_journey=day_journey,
                video_duration=part.kept_duration_sec,
                start_location=location,
                max_chapters=kit.superlong_max_chapters(part.kept_duration_sec),
                resolved_location=resolved_location,
                stages_dir=stages_dir,
                included_segments=list(part.ranges) if day_journey is not None else None,
            )
            chapter_text = generator.format_chapters(chapters)
        except Exception as exc:  # noqa: BLE001 — never fail a part on chapter derivation
            logger.error(
                "Day %s: superlong part %d chapter generation failed: %s",
                date,
                part.part_index,
                exc,
            )
            chapters, chapter_text = [], ""

    description = (
        f"{title}\n\n"
        f"The whole day's drive from {location}, part {part.part_index} of "
        f"{part.part_count} — stops removed, nothing else cut."
    )
    if len(chapters) >= kit.min_chapters:
        description = f"{description}\n\n\U0001f4cd CHAPTERS\n{chapter_text}"

    return {
        "title": title,
        "description": description,
        "chapters": chapters,
        "video_duration": part.kept_duration_sec,
        "has_gps": has_gps,
        "location": location,
    }


def _produce_superlong_parts(
    *,
    date: str,
    stop_mask,
    episodes: list,
    file_boundaries: list[tuple[float, float]],
    master_path: Path,
    day_dir: Path,
    config,
    location: str,
    ost_crossfade_sec: float,
    lf_plans: Optional[list] = None,
    state=None,
    day_gps_points: Optional[list[tuple[float, float, float]]] = None,
    media_profile=None,
    kit: Optional[TitleKit] = None,
    superlong_route_fn: Optional[Callable] = None,
) -> dict:
    """Cut + OST-burn opt-in superlong parts (``sl_NN.mp4``) from the day
    master. Never turned into ``RegularLFPlan``s, so they can't reach
    ``lf_plans``/``num_lfs``. Resume-idempotent: an existing non-empty final
    is skipped. Every record also carries real metadata (title/chapters/
    description — see ``_build_superlong_part_metadata``) so an uploader
    can consume it without re-deriving. Returns ``{"part_count": int,
    "records": [...]}``.

    Each part's recurring LOCATION lower-thirds (``apply_ost``'s
    ``lower_thirds``) come from ``chapter_town_events`` on its OWN
    ``record["chapters"]`` — the SAME canonical, ``ResolvedLocation``-
    reconciled chapter pipeline every town-name surface uses, never raw GPX.
    Kept events persist onto the part's record ``lower_thirds`` key.

    ``day_gps_points``: optional ``[(t_master, lat, lon), ...]`` — the day's
    GPS points already projected onto the day master's own 0-based timeline
    (``gps_enrichment._build_gps_points_by_date`` via the caller's
    ``gps_points_by_date``), spanning every episode of the day. Builds the
    per-part chapter journey (see ``_build_superlong_part_metadata``);
    ``None`` falls back to the legacy single-episode-GPX chapters.

    ``superlong_route_fn`` (``route_claims.make_superlong_route_fn``): each
    part's title text and lower-thirds are checked against that part's OWN
    retained ranges before any part extracts or burns; an unsupported title is
    broadened or refused, an off-route lower-third refused, and every refusal is
    raised together. A GPS-bearing day with no route check refuses.
    """
    from ..models import LFSegment
    from .route_claims import lower_third_issues, route_claim_outcome
    from .superlong_planner import plan_superlong_parts

    _drive = getattr(config, "drive", None) if config is not None else None
    plans = plan_superlong_parts(
        stop_mask,
        part_max_duration_sec=int(getattr(_drive, "superlong_part_max_duration_sec", 14400)),
        split_grace=float(getattr(_drive, "superlong_split_grace", 0.08)),
    )
    # Output-space gate, scaled to THIS day's planned part footage from the
    # master already on disk — upstream of any part concat/OST re-encode
    # (checked before the irreversible step, not after). Runs here, not run-level preflight,
    # because the master and part plans only exist at this point.
    check_superlong_output_space(master_path, day_dir, sum(p.kept_duration_sec for p in plans))
    sl_dir = day_dir / "superlong"
    sl_dir.mkdir(parents=True, exist_ok=True)
    brand_logo_path = getattr(_drive, "brand_logo_path", None) if _drive is not None else None
    lt_duration_sec = float(getattr(_drive, "superlong_lower_third_duration_sec", 8.0))
    lt_min_gap_sec = float(getattr(_drive, "superlong_lower_third_min_gap_sec", 45.0))
    # Metadata inputs, computed once and reused for every part of the day
    # (never re-derived per part): the day's first regular LF's already-
    # resolved ``ResolvedLocation``, the whole-day GPS journey (preferred —
    # spans every episode, windowed per part below), and the first
    # GPS-bearing episode's GPX (fallback when no day journey is available).
    _resolved_location = None
    if lf_plans and state is not None:
        _first_idx = int(getattr(lf_plans[0], "index", 0))
        _resolved_location = (getattr(state, "per_lf_resolved_location", None) or {}).get(
            _first_idx
        )
    kit = require_title_kit(kit)
    _day_journey = kit.day_master_journey(day_gps_points)
    _gpx_path: Optional[Path] = None
    for _ep in episodes:
        _p = getattr(_ep, "gpx_path", None)
        if _p:
            _gpx_path = Path(_p)
            break
    # ── Pass 1: build every part's record, metadata and burned text — no
    # encode yet — so the superlong gate below audits it before pixels burn.
    # Same ``part_transitions`` derivation the old single-pass loop used.
    records: list[dict] = []
    _burn_work: list[dict] = []
    _route_refusals: list[str] = []
    if day_gps_points and superlong_route_fn is None:
        raise RouteClaimRefusedError(
            f"Day {date}: superlong text {location!r} has GPS but no route check to "
            "measure it against each part's retained footage; next: run through "
            "phase_bc_day_first, which wires make_superlong_route_fn"
        )
    for part in plans:
        final_sl = sl_dir / f"sl_{part.part_index:02d}.mp4"
        part_text, _route_ev, _route_note = location, None, ""
        if day_gps_points:
            try:
                part_text, _route_note, _route_ev = superlong_route_fn(
                    part.part_index, part.ranges, day_gps_points, location
                )
            except RouteClaimRefusedError as exc:
                _route_refusals.append(f"part {part.part_index}: {exc}")
                continue
            if _route_note:
                logger.warning(
                    "ROUTE-CLAIM: day %s superlong part %d %s", date, part.part_index, _route_note
                )
        record = {
            "path": str(final_sl),
            "part_index": part.part_index,
            "part_count": part.part_count,
            "duration_sec": part.kept_duration_sec,
        }
        record.update(
            _build_superlong_part_metadata(
                part=part,
                date=date,
                location=part_text,
                gpx_path=_gpx_path,
                resolved_location=_resolved_location,
                config=config,
                stages_dir=sl_dir,
                day_journey=_day_journey,
                kit=kit,
            )
        )
        part_transitions = kit.town_events(record.get("chapters"))
        if _route_ev is not None:
            _lt_issues = lower_third_issues(part_transitions, _route_ev)
            if _lt_issues:
                _route_refusals.append(
                    f"part {part.part_index}: " + "\n  - ".join(["lower-thirds:", *_lt_issues])
                )
                continue
            record["route_claim"] = {
                "part_index": part.part_index,
                **route_claim_outcome(
                    part.part_index, location, _route_ev, chosen=part_text, reason=_route_note
                ),
            }
            record["route_claim"].pop("lf_index", None)
        records.append(record)
        _burn_work.append(
            {
                "part": part,
                "record": record,
                "final_sl": final_sl,
                "part_transitions": part_transitions,
                "part_index": part.part_index,
                "part_count": part.part_count,
                # Burned title text: the day-level ``location`` from
                # ``run_day_pipeline``, as this part's route check kept it.
                "ost_text": part_text,
                "lower_thirds": [
                    {"label": lbl, "time_sec": t} for lbl, t in (part_transitions or [])
                ],
            }
        )

    if _route_refusals:
        for r in _route_refusals:
            logger.error("ROUTE-CLAIM SL: day %s %s", date, r.splitlines()[0])
        raise RouteClaimRefusedError(
            f"Day {date}: {len(_route_refusals)} superlong part(s) claim more than their "
            "retained GPS proves (nothing extracted or burned):\n" + "\n".join(_route_refusals)
        )

    # ── Superlong pre-burn OST review gate (third burned-text surface) ──
    # Audits every part's title text + lower-thirds before any part's
    # extract/apply_ost — mirrors the shorts pre-burn gate. The burn loop consumes
    # the gate's RETURNED work-list, so a part can only reach
    # ``apply_ost`` by having been in what this gate audited.
    if state is not None:
        record_surface_reviews(
            state,
            "superlong",
            date,
            _burn_work,
            {
                w["part_index"]: {**w["record"]["route_claim"], "ost_text": w["ost_text"]}
                for w in _burn_work
                if "route_claim" in w["record"]
            },
        )
    _gated_work = superlong_gate_or_raise(day_dir.parent, date, _burn_work, state)

    # ── Pass 2: extract + burn every gated part ──
    for entry in _gated_work:
        part = entry["part"]
        record = entry["record"]
        final_sl = entry["final_sl"]
        part_transitions = entry["part_transitions"]
        # Freshness (superlong): skip re-burning a final that already
        # exists AND whose stored fingerprint still matches this plan's
        # segment windows (``part.ranges``) + burned text/timing. A
        # re-plan that shifts the windows (or a pre-fingerprint render
        # with no sidecar) is NOT adopted on file size alone — see
        # ``_superlong_ost_fingerprint``.
        sl_fingerprint = _superlong_ost_fingerprint(
            entry["ost_text"],
            part.ranges,
            ost_crossfade_sec,
            part_transitions,
            lt_duration_sec,
            lt_min_gap_sec,
        )
        fp_path = sl_dir / f"sl_{part.part_index:02d}.ost.sha256"
        if (
            final_sl.is_file()
            and final_sl.stat().st_size > 0
            and fp_path.is_file()
            and fp_path.read_text().strip() == sl_fingerprint
        ):
            logger.info(
                "Day %s: superlong part %d final is fresh (OST unchanged) — skipping re-burn",
                date,
                part.part_index,
            )
            continue
        if final_sl.is_file() and final_sl.stat().st_size > 0:
            logger.info(
                "Day %s: superlong part %d final exists but its OST fingerprint "
                "is missing or stale (re-plan moved its segment windows, or a "
                "pre-fingerprint render) — re-burning",
                date,
                part.part_index,
            )
        segs = []
        for start, end in part.ranges:
            for ep_i, (eb_start, eb_end) in enumerate(file_boundaries):
                ov_start, ov_end = max(start, eb_start), min(end, eb_end)
                if ov_end <= ov_start:
                    continue
                _sl_lat, _sl_lon = _window_gps_centroid(day_gps_points, ov_start, ov_end)
                segs.append(
                    LFSegment(
                        source_file_index=ep_i,
                        source_path=Path(getattr(episodes[ep_i], "concat_path", "") or ""),
                        start_time=ov_start - eb_start,
                        end_time=ov_end - eb_start,
                        original_start_time=ov_start,
                        original_end_time=ov_end,
                        composite_score=0.0,
                        gps_center_lat=_sl_lat,
                        gps_center_lon=_sl_lon,
                        terrain_type="superlong",
                        contiguous_with_previous=bool(segs),
                    )
                )
        if not segs:
            logger.warning("Day %s: superlong part %d has no segments", date, part.part_index)
            continue
        work_dir = sl_dir / f".sl_{part.part_index:02d}_work"
        raw_concat = sl_dir / f"sl_{part.part_index:02d}_segments.mp4"
        concatenated = extract_and_concat_segments(
            master_path=master_path,
            segments=segs,
            file_boundaries=file_boundaries,
            output_path=raw_concat,
            work_dir=work_dir,
            stop_mask=stop_mask,
        )
        if concatenated is None:
            # HARD fail, never a skip. A warning here let the day report
            # lfs/shorts complete and walk into Phase D with no part on disk
            # — the same silent-superlong-loss the resume skip caused.
            # The upstream warning names the ffmpeg cause.
            raise RuntimeError(
                f"Day {date}: superlong part {part.part_index} concat FAILED "
                f"(extract+concat from {master_path} returned no output; see the "
                "'segment extract' warning above for the ffmpeg cause — a "
                "stream-copy timeout on a very long part is the known one). "
                "Refusing to continue: the day would otherwise complete with a "
                "missing superlong part."
            )
        ost_result = apply_ost(
            input_path=concatenated,
            output_path=final_sl,
            location=entry["ost_text"],
            layout="regular_lf",
            brand_logo_path=brand_logo_path,
            prefer_nvenc=True,
            crossfade_sec=ost_crossfade_sec,
            lower_thirds=part_transitions,
            lower_third_duration_sec=lt_duration_sec,
            lower_third_min_gap_sec=lt_min_gap_sec,
            media_profile=media_profile,
        )
        if ost_result is None:
            raise RuntimeError(f"Superlong OST failed: day {date} part {part.part_index}")
        # Persist the burned-OST fingerprint so a later resume can detect this
        # final is fresh and a re-plan that moved its windows is not.
        fp_path.write_text(sl_fingerprint)
        record["path"] = str(ost_result)
        # Reviewable ("Audit before publish" --skip-upload): the SAME burned text.
        if part_transitions:
            record["lower_thirds"] = [{"label": lbl, "time_sec": t} for lbl, t in part_transitions]
    return {"part_count": len(plans), "records": records}


def _gate_master_picture(day_batch, master_path: Path, date: str) -> None:
    """Hard-fail a day whose master came out softer than, or ringing above, its sources.

    Runs only when the day's colour cache (beside the master) records a step that
    changed the picture. The day's episode concats are measured with the same
    sampler and frame count as the master, so the comparison is like for like.
    A source that cannot be measured (concat gone) defers to Phase D's gate.
    """
    from . import quality
    from .validation.longform_validator import picture_quality_gate

    cache = master_path.parent / f"day_{date}_color_cache.json"
    steps = cache.with_name(f"day_{date}_picture_steps.json")
    # Neither record on disk: no step ran. Phase D's gate warns about it.
    if not (cache.is_file() or steps.is_file()) or not quality.picture_changed_for_day(cache):
        return
    episodes = list(getattr(day_batch, "episodes", None) or [])
    source = quality.sample_source_picture(
        [getattr(ep, "concat_path", None) or "" for ep in episodes],
        [float(getattr(ep, "total_duration", 0.0) or 0.0) for ep in episodes],
    )
    if source is None:
        logger.warning(
            "Day %s: day-master picture gate skipped -- sources unmeasurable; Phase D gates it",
            date,
        )
        return
    gate = picture_quality_gate(
        master_path,
        float(getattr(day_batch, "total_duration", 0.0) or 0.0),
        source_sharpness=source[0],
        changed=True,
        source_ringing=source[1],
    )
    logger.info("Day %s: day-master picture gate: %s", date, gate.message)
    if not gate.passed:
        raise RuntimeError(
            f"Day {date}: day master {master_path.name} failed the picture gate: {gate.message}"
        )


def run_day_pipeline(
    day_batch: DayBatch,
    day_scores: np.ndarray,
    output_dir: Path,
    location: str,
    config=None,
    *,
    lf_plans: list,
    per_lf_locations: Optional[dict] = None,
    per_lf_ost_text: Optional[dict] = None,
    gps_points_by_date: Optional[dict] = None,
    short_ost_text_fn: Optional[Callable] = None,
    short_regions_fn: Optional[Callable] = None,
    short_route_fn: Optional[Callable] = None,
    superlong_route_fn: Optional[Callable] = None,
    state=None,
    inserts_by_lf: Optional[dict] = None,
    kit: Optional[TitleKit] = None,
) -> Optional[DayArtifacts]:
    """Run the full day-first pipeline for a single day.

    Args:
        day_batch: The day's episodes (from `day_grouper.group_episodes_by_day`).
        day_scores: The day-local slice of the drive score array, with
            timestamps already rebased to 0 (see `slice_scores_for_day`).
        output_dir: Per-day output directory. Will be created if missing.
            The day master lands here, along with `regular_lfs/` and
            `shorts/` subdirectories for the derived outputs.
        location: OST text for this day. Every output (master, regular
            LFs, shorts) gets this same string — per-output endpoint-
            derived locations are a possible future enhancement.
        config: Optional `gomoveshift.config.Config` passed through to
            `encode_day_master`. When None, that module uses its default.
        per_lf_locations: Optional ``{lf_index: county_state_label}`` map
            used as the OST burn-in fallback when no canonical text is
            available for an LF.
        per_lf_ost_text: Optional ``{lf_index: canonical_route_text}``
            map. When an entry exists for an LF, it WINS over
            ``per_lf_locations`` for that LF's OST burn-in — keeps OST,
            YouTube title, and thumbnail aligned to one source. Missing entries
            fall back to ``per_lf_locations[idx]`` then ``location``.
        gps_points_by_date: Optional ``{date: gps_points}`` map supplying
            the day's GPS trace to the shorts OST composition path. Every
            short routes through ``trim_gps_to_clip_timeline`` keyed on
            its single ``(start_sec, end_sec)`` range — even when the
            map is None the helper is still invoked (uniform code path,
            no one-offs).
        short_ost_text_fn: Optional callable
            ``(trimmed_points, short_range) -> Optional[str]`` that
            derives a per-short OST text from the trim output. When
            supplied AND trimmed points are non-empty AND the callable
            returns a truthy string, that string wins over the drive-
            wide ``location`` fallback for the short's OST burn-in.
            The trim output is consumed here; otherwise the
            drive-wide string would always be burned.
        state: Optional pipeline state (``PipelineState`` or dict). When
            supplied, ``shorts_target`` / ``shorts_gap`` override the
            equivalents on ``config.drive`` for the plan_shorts call.

    Read-priority for the shorts target / ``shorts_min_gap_sec``:
        The per-LF count is Waypost's decision, read via
        ``volume_source.shorts_per_lf()``; ``state.get("shorts_target")`` (the
        ``--shorts-target`` flag) can only lower it (``volume_source.apply_flag``).
        ``state.get("shorts_gap")`` wins over ``shorts_min_gap_sec`` when set.
        State is authoritative because it persists across resume; config is
        mutated in-process and is wiped to YAML defaults on every fresh invocation. A mid-run edit
        to state (e.g. operator manually bumps shorts_target in
        pipeline_state.json) therefore overrides a stale CLI flag that
        was re-passed by habit.

    Returns:
        A `DayArtifacts` record on success, or None if the day is empty
        or the master encode fails. Partial progress is kept (if the
        master exists but some LF extractions fail, those are skipped
        and the surviving ones are returned).
    """
    output_dir = Path(output_dir)
    date = day_batch.date

    episodes = list(getattr(day_batch, "episodes", None) or [])
    if not episodes:
        logger.info("run_day_pipeline: day %s has no episodes, skipping", date)
        return None

    _raw_indices = [int(getattr(ep, "index", 0)) for ep in episodes]
    if not all(i >= 1 for i in _raw_indices):
        raise ValueError(
            f"Episode indices must be 1-based (got {_raw_indices}); caller "
            "violated the ingest convention. See execute_ingest / "
            "episode_grouper / _synthesize_ingest_plan_from_state."
        )

    day_dir = output_dir / f"day_{date}"
    day_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: encode day master (color + audio + OST in one pass) ─
    # OST is burned during the single-pass encode using the day master
    # layout (1s-9s). This eliminates a full re-encode that apply_ost
    # would otherwise require on the longest file in the pipeline.
    logger.info("Day %s: encoding master (single-pass with OST)...", date)
    # Thread this day's per_day_completed record so encode_day_master can
    # validate a cached master against recorded metadata when the raw episodes
    # are purged. Both PipelineState and a plain dict support ``.get``.
    _per_day_record = None
    if state is not None:
        _pdc = state.get("per_day_completed")
        if isinstance(_pdc, dict):
            _per_day_record = _pdc.get(date)
    # The tenant's MediaProfile, loaded once per run on the orchestration
    # side and handed to the engine (encode_day_master, apply_ost).
    kit = require_title_kit(kit)
    media_profile = kit.media_profile()
    master_path = encode_day_master(
        day_batch,
        day_dir,
        config=config,
        location=location,
        per_day_record=_per_day_record,
        profile=media_profile,
    )
    if master_path is None:
        logger.error("Day %s: master encode failed, aborting day", date)
        return None
    # Judge the picture where a rendition step changed it, before
    # any LF or short is cut from it (Phase D's final gate still runs as well).
    _gate_master_picture(day_batch, master_path, date)

    # The master already has OST burned in. It also serves as the
    # source for stream-copy extraction of regular LFs and shorts
    # (their OST is burned separately with different timing).
    master_source_for_extract = master_path
    master_final_path = master_path

    # ── Build the per-day STOP mask ──────────────────────────────
    # A published short shipped extended parked/roadside footage. Build one
    # per-second veto (True == stopped) from GPS speed — falling back to
    # optical motion where speed is unmeasurable — and thread it into the
    # shorts planner so a masked window can never be selected. Persist it as
    # the single source of truth downstream gates/thumbnails/DoD read.
    # Default on; disabled via config.drive.stop_mask_enabled or when the
    # day has no scores.
    stop_mask = None
    _drive_cfg = getattr(config, "drive", None) if config is not None else None
    _stop_mask_enabled = getattr(_drive_cfg, "stop_mask_enabled", True) if _drive_cfg else True
    if _stop_mask_enabled and isinstance(day_scores, np.ndarray) and day_scores.size > 0:
        day_gps_points = gps_points_by_date.get(date) if gps_points_by_date else None
        # Knobs come from the TOP-LEVEL ``config.stop_detection`` via the shared
        # reader (this used to look on ``config.drive`` and always fell back to
        # the hard-coded defaults). Same reader as the day-first drive plan's
        # full_drive mask, so the two masks agree.
        stop_mask = build_stop_mask(day_scores, day_gps_points, **stop_mask_kwargs(config))
        _summary = mask_summary(stop_mask)
        logger.info(
            "Day %s: stop mask — %d/%d s masked (%.1f%%)",
            date,
            _summary["masked_seconds"],
            _summary["total_seconds"],
            _summary["masked_fraction"] * 100,
        )
        # Persist as the single source of truth: <stages_base or
        # output_base/_stages>/stop_mask_<date>.npz (best-effort — a persist
        # failure must not abort the day).
        try:
            _stages_base = state.get("stages_base") if state is not None else None
            _output_base = state.get("output_base") if state is not None else None
            if _stages_base:
                _mask_base = Path(_stages_base)
            elif _output_base:
                _mask_base = Path(_output_base) / "_stages"
            else:
                _mask_base = output_dir / "_stages"
            _mask_base.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(_mask_base / f"stop_mask_{date}.npz", mask=stop_mask)
        except Exception as exc:  # noqa: BLE001 — persistence is best-effort
            logger.warning("Day %s: stop mask persist failed: %s", date, exc)

    # ── Step 2: assemble regular LFs ──────────────────────────────
    lf_dir = day_dir / "regular_lfs"
    lf_dir.mkdir(parents=True, exist_ok=True)
    # --keep-intermediates: read once, used by the per-LF purge in the loop
    # below and by the day-end sweep.
    keep_intermediates = bool(state.get("keep_intermediates")) if state is not None else False
    regular_lf_paths: list = []
    # Parallel to ``regular_lf_paths``: the authoritative LF index for each
    # rendered path, so a mid-day render skip leaves an index-keyed gap the
    # downstream pairing can honour instead of shifting later LFs up a slot.
    regular_lf_indices: list = []

    # Segment-driven path (correct per drive_plan). Each LF is:
    #   concat(scored segments from master) → prepend 30s montage
    #   → apply_ost(layout=regular_lf, OST at t=2s-10s + logo + persistent URL)
    file_boundaries = compute_day_file_boundaries(day_batch)

    # Segments arrive with drive-global ``source_file_index`` (remapped
    # in ``LongFormAssembler._assemble_with_adaptive_threshold``), but
    # ``file_boundaries`` is day-local — so we must remap indices back
    # to day-local before ``extract_and_concat_segments`` /
    # ``prepend_montage`` run. Without this step, any day whose episodes
    # don't happen to sit at drive-global indices 0..N-1 silently drops
    # its LFs.
    #
    # INDEX SPACE: segments carry 0-based drive-global positions (from
    # ``enumerate(episode_dates)`` in longform_assembler).
    # ``Episode.index`` is 1-based (``enumerate(..., 1)`` in pipeline/ingest/__init__.py),
    # so we subtract 1 to align spaces. A 1-based vs 0-based
    # mismatch surfaces as "segment source_file_index 0 not in this
    # day's episode set [1]".
    day_episode_indices = [i - 1 for i in _raw_indices]
    # Brand logo watermark — read the field from the nested DriveConfig
    # (`config.drive.brand_logo_path`); a bare `getattr(config, ...)` returns
    # None silently, which silently ships a missing logo. The path is threaded
    # into apply_ost below, but the only long-form layout (regular_lf) sets
    # show_logo=False — the brand logo is intentionally disabled on the YouTube
    # surface, where it would collide with YouTube's own channel watermark (see
    # ost_applier.LAYOUTS). A missing brand_logo_path is therefore EXPECTED, not
    # a misconfiguration, so we do NOT warn. The path
    # stays threaded so any future show_logo=True layout works unchanged.
    _drive = getattr(config, "drive", None) if config is not None else None
    brand_logo_path = getattr(_drive, "brand_logo_path", None) if _drive is not None else None
    # Thread montage duration + OST title crossfade into ``apply_ost``
    # so the LOCATION title fills the full montage and crossfades out as
    # the main video begins. BRAND text shares the same montage
    # gate + crossfade (was persistent full-video; that decision burned
    # center-screen brand text on every video frame — reverted).
    # Source of truth is config; fall back to the previous hardcoded 15s
    # default when config is unavailable (e.g. test harness without config).
    montage_duration = (
        getattr(_drive, "montage_max_duration_sec", 15.0) if _drive is not None else 15.0
    )
    ost_crossfade_sec = (
        getattr(_drive, "ost_title_crossfade_sec", 0.5) if _drive is not None else 0.5
    )
    # Insta360 / full-drive-LF: resolve LF selection mode from
    # config.drive.lf_selection_mode. When "full_drive" AND the caller
    # didn't supply pre-built ``lf_plans``, synthesize a single-LF plan
    # spanning the full post-stop-removal timeline by calling
    # ``plan_regular_lfs(day_scores, mode="full_drive", ...)`` — the
    # planner is the robust-by-construction single source of truth for
    # what shape the LF list takes. ``best_moments`` keeps the legacy
    # external-caller contract (lf_plans built by LongFormAssembler).
    lf_selection_mode = resolve_lf_mode(_drive) if _drive is not None else "best_moments"
    full_drive_max_duration_sec = (
        int(getattr(_drive, "full_drive_max_duration_sec", 7200)) if _drive is not None else 7200
    )
    if lf_selection_mode == "full_drive" and not (lf_plans or []):
        from ..models import LFSegment, RegularLFPlan

        if day_scores is None or day_scores.size == 0:
            raise ValueError(
                f"Day {date}: lf_selection_mode='full_drive' requires a non-empty "
                "day_scores array (post-stop-removal scored timeline). Caller "
                "supplied empty or None."
            )
        part_min = getattr(_drive, "full_drive_part_min_sec", None) if _drive else None
        part_max = getattr(_drive, "full_drive_part_max_sec", None) if _drive else None
        split_parts = part_min is not None and part_max is not None
        if split_parts:
            # Balanced K-part split; min/ceiling enforced per part.
            part_ranges = plan_full_drive_parts(
                day_scores,
                min_duration_sec=float(
                    getattr(_drive, "min_lf_duration", DEFAULT_REGULAR_LF_MIN_DURATION_SEC)
                ),
                full_drive_max_duration_sec=full_drive_max_duration_sec,
                stop_mask=stop_mask,
                part_min_sec=part_min,
                part_max_sec=part_max,
                day=date,
            )
        else:
            part_ranges = [
                plan_regular_lfs(
                    day_scores,
                    mode="full_drive",
                    full_drive_max_duration_sec=full_drive_max_duration_sec,
                    stop_mask=stop_mask,
                )
            ]
        # plan_regular_lfs(mode="full_drive") raises on ceiling violation
        # or sub-min kept duration, so reaching this point means ≥1 range.
        # Each part's ranges are the maximal contiguous UNMASKED spans (parked
        # spans cut out) in day-master seconds; with no mask and no split it is
        # one whole-day span. Build one LFSegment per (range ∩ episode)
        # so the assembled LF PHYSICALLY excludes the masked stop spans — the
        # trim_ratio then reflects the stripped stops instead of falsely
        # reading 1.0. ``file_boundaries`` (day-local, computed above) tiles
        # the episodes in master time.
        region = getattr(_drive, "region", "") if _drive is not None else ""
        lf_plans = []
        for part_no, full_drive_ranges in enumerate(part_ranges, 1):
            kept_duration = sum(r.duration_sec for r in full_drive_ranges)
            avg_score = sum(r.integrated_score for r in full_drive_ranges) / max(kept_duration, 1.0)
            synth_segments: list = []
            for rng in full_drive_ranges:
                for ep_i, (eb_start, eb_end) in enumerate(file_boundaries):
                    ov_start = max(rng.start_sec, eb_start)
                    ov_end = min(rng.end_sec, eb_end)
                    if ov_end <= ov_start:
                        continue
                    ep = episodes[ep_i]
                    _seg_lat, _seg_lon = _window_gps_centroid(day_gps_points, ov_start, ov_end)
                    synth_segments.append(
                        LFSegment(
                            source_file_index=ep_i,
                            source_path=Path(getattr(ep, "concat_path", "") or ""),
                            start_time=ov_start - eb_start,
                            end_time=ov_end - eb_start,
                            original_start_time=ov_start,
                            original_end_time=ov_end,
                            composite_score=avg_score,
                            gps_center_lat=_seg_lat,
                            gps_center_lon=_seg_lon,
                            terrain_type="full_drive",
                            contiguous_with_previous=len(synth_segments) > 0,
                        )
                    )
            if not synth_segments:
                raise ValueError(
                    f"Day {date}: lf_selection_mode='full_drive' but no episodes "
                    "with positive total_duration (or the whole day is masked) — "
                    "cannot synthesize an LF plan."
                )
            lf_plans.append(
                RegularLFPlan(
                    index=part_no,
                    segments=synth_segments,
                    montage_peaks=[],
                    shorts_target=0,
                    region=region,
                    total_duration=kept_duration,
                    part_index=part_no if len(part_ranges) > 1 else None,
                    part_count=len(part_ranges) if len(part_ranges) > 1 else None,
                )
            )
            logger.info(
                "Day %s: full_drive mode — synthesized LF plan %d/%d "
                "(%.1fs kept across %d segment(s); %d unmasked span(s))",
                date,
                part_no,
                len(part_ranges),
                kept_duration,
                len(synth_segments),
                len(full_drive_ranges),
            )
    elif lf_selection_mode == "full_drive" and (
        getattr(_drive, "full_drive_part_min_sec", None) is not None
        and getattr(_drive, "full_drive_part_max_sec", None) is not None
    ):
        # Balanced parts: K supplied plans are fine when each is a distinct
        # LF (unique index → unique lf_NN.mp4 / sidecars) within the ceiling.
        _idx = [int(getattr(p, "index", 0)) for p in lf_plans]
        if len(set(_idx)) != len(_idx):
            raise ValueError(
                f"Day {date}: lf_selection_mode='full_drive' parts need unique "
                f"LF indices, got {_idx}."
            )
        for _p in lf_plans:
            _dur = float(getattr(_p, "total_duration", 0.0) or 0.0)
            if _dur > full_drive_max_duration_sec:
                raise ValueError(
                    f"Day {date}: full_drive part {getattr(_p, 'index', '?')} is "
                    f"{_dur:.1f}s, which exceeds the {full_drive_max_duration_sec}s ceiling."
                )
    elif lf_selection_mode == "full_drive" and len(lf_plans or []) > 1:
        raise ValueError(
            f"Day {date}: lf_selection_mode='full_drive' requires exactly one "
            f"LF plan, got {len(lf_plans)}. Caller must reconcile the planner "
            "contract with the supplied lf_plans list."
        )

    logger.info(
        "Day %s: %d LF plan(s) supplied — assembling from segments",
        date,
        len(lf_plans or []),
    )

    # ── OST audit gate ───────────────────────────────────
    # Burning OST pixels onto an LF without an operator-visible audit
    # artefact is unrepresentable (robust by construction). Write
    # ``<output_dir>/ost_audit_<date>.json`` listing every LF's
    # canonical OST text, YouTube title, thumbnail-text source proxy,
    # and recording date BEFORE the per-LF apply_ost loop runs.
    #
    # ``run_all_days`` runs this SAME gate (``lf_gate_or_raise``)
    # for every day UPFRONT before the first master encode, so on a
    # front-loaded run this per-day check fast-passes on the persisted
    # marker instead of stopping the run post-encode. The per-day gate is
    # retained as defense-in-depth for callers that bypass run_all_days.
    #
    # Operator confirms via:
    #   1. ``<output_dir>/ost_audit_<date>.confirmed`` file, or
    #   2. ``state["ost_audit_passed"][date] = True`` (resume marker), or
    #   3. ``--ost-audit-confirmed`` CLI flag → state["ost_audit_confirmed"].
    #
    # Hash mismatch (operator edits per-LF labels and re-runs phase_bc)
    # re-prompts: the stored hash no longer matches the just-computed
    # one, so OstAuditPendingError fires again.
    #
    _gate_lf_indices = [int(getattr(plan, "index", i + 1)) for i, plan in enumerate(lf_plans or [])]
    # Burn-loop-consumes-gate-return (fix): the burn loop below reads its
    # per-LF text from THIS dict — populated from the audit payload the gate
    # just wrote/confirmed — rather than re-resolving its own answer from
    # ``per_lf_ost_text``/``per_lf_locations`` alone. One resolved value,
    # one source, so the pixels can never diverge from the sidecar an
    # operator reviewed.
    _gate_lf_text: dict = {}
    if _gate_lf_indices:
        # PipelineState supports dict-protocol sugar (state.get / state["x"])
        # — both shapes (dataclass + raw test dict) flow through this gate.
        if state is None:
            _ost_audit_state = {}
            _ost_audit_state.setdefault("per_lf_ost_text", per_lf_ost_text or {})
            _ost_audit_state.setdefault("per_lf_locations", per_lf_locations or {})
            _ost_audit_state.setdefault("per_lf_titles", {})
        else:
            _ost_audit_state = {
                "per_lf_ost_text": state.get("per_lf_ost_text") or per_lf_ost_text or {},
                "per_lf_titles": state.get("per_lf_titles") or {},
                "per_lf_recording_date": state.get("per_lf_recording_date") or {},
                "per_lf_locations": state.get("per_lf_locations") or per_lf_locations or {},
            }
            _ost_audit_state.setdefault("per_lf_recording_date", {})
        # Pass drive-wide ``location`` so the audited text equals the
        # per-LF burn loop's resolved ``lf_location`` — review-text
        # == burn-text even for a GPS-less drive whose only label is
        # ``--location``.
        payload = format_ost_audit(
            date,
            _ost_audit_state,
            _gate_lf_indices,
            location=location,
            inserts=insert_records(inserts_by_lf or {}),
        )
        current_hash = compute_ost_audit_hash(payload)
        _gate_lf_text = {int(e["lf_index"]): e["ost_text"] for e in payload.get("lfs", [])}

        # PipelineState supports dict-protocol sugar — works on both dataclass + raw test dict.
        if state is None:
            _passed_map: dict = {}
            _hash_map: dict = {}
            _cli_confirmed = False
        else:
            _passed_map = state.get("ost_audit_passed") or {}
            _hash_map = state.get("ost_audit_hash") or {}
            # Only a THIS-run confirmation bypasses without a stored hash; a
            # persisted flag no longer auto-passes a changed burn after a kill.
            _cli_confirmed = bool(state.get("ost_audit_confirmed_this_run"))
        confirmed_file = Path(output_dir) / f"ost_audit_{date}.confirmed"

        # A stale confirmation — ``.confirmed`` on disk OR
        # ``--ost-audit-confirmed`` — is INSUFFICIENT when the resolved
        # label changed materially since the prior confirmed run. The
        # prior confirmed hash is ``_hash_map[date]``; ``confirmation_is_stale``
        # is False on a first-ever confirmation (no prior hash) so the
        # non-interactive bypass still works on a clean run. The
        # state-marker path already required a matching hash; this extends
        # the same binding to the two bypass signals (tightens, never
        # weakens, the three confirmation signals).
        # Bind the ``.confirmed`` marker to the audit hash it confirmed.
        # ``--fresh`` wipes ``_hash_map`` (and purges the marker), but on any
        # path where the state hash is absent a stamped marker still supplies
        # the prior hash, so ``exists()`` alone can't pass a materially changed,
        # never-reviewed burn text.
        _prior_hash = _hash_map.get(date) or read_confirmed_hash(confirmed_file)
        _stale = confirmation_is_stale(_prior_hash, current_hash)

        gate_passes = False
        reason = ""
        if _stale:
            gate_passes = False  # explicit: a material label change reopens the gate
        elif _cli_confirmed:
            gate_passes = True
            reason = "--ost-audit-confirmed"
        elif confirmed_file.exists():
            gate_passes = True
            reason = f"{confirmed_file.name} on disk"
        elif _passed_map.get(date) is True and _hash_map.get(date) == current_hash:
            gate_passes = True
            reason = "state['ost_audit_passed'] + matching hash"
        else:
            # The opt-in system review (``ost_system_review``).
            gate_passes, reason = system_review_passes(state, date, _gate_lf_text)

        if not gate_passes:
            sidecar_path = write_ost_audit(Path(output_dir), date, payload)
            logger.info(
                "OST audit sidecar written: %s. Review and create %s "
                "(or pass --ost-audit-confirmed) to proceed.",
                sidecar_path,
                confirmed_file,
            )
            raise OstAuditPendingError(
                f"OST audit pending for day {date}: review {sidecar_path} "
                f"and create {confirmed_file} (or re-run with "
                f"--ost-audit-confirmed) before OST burn-in." + (f" {reason}." if reason else "")
            )
        if reason.startswith("system review record"):
            write_system_review_copy(Path(output_dir), date, state)

        # Persist the pass marker and hash so later invocations skip the
        # file/CLI check via the state path; both are ``PipelineState``
        # fields (``ost_audit_passed`` / ``ost_audit_hash``).
        if state is not None:
            passed_map = state.get("ost_audit_passed")
            if passed_map is None:
                passed_map = {}
                state["ost_audit_passed"] = passed_map
            passed_map[date] = True
            hash_map = state.get("ost_audit_hash")
            if hash_map is None:
                hash_map = {}
                state["ost_audit_hash"] = hash_map
            hash_map[date] = current_hash
        # Bind the marker to this audit hash so a later run whose burn
        # text changed reads a stale binding and reopens the gate.
        if confirmed_file.exists():
            stamp_confirmed(confirmed_file, current_hash)
        logger.info("OST audit gate passed for day %s (%s).", date, reason)

    for lf_plan in lf_plans or []:
        idx = int(getattr(lf_plan, "index", len(regular_lf_paths) + 1))
        segments = list(getattr(lf_plan, "segments", []) or [])
        peaks = list(getattr(lf_plan, "montage_peaks", []) or [])
        if not segments:
            logger.warning(
                "Day %s: LF %d has no segments, skipping",
                date,
                idx,
            )
            continue
        # CONTRACT: segments + peaks must already be day-local when they
        # reach this function. The authoritative remap lives in
        # phase_bc_day_first._partition_lf_plans_by_date, which mutates
        # both segments and montage_peaks in-place before handing them
        # off via run_all_days → run_day_pipeline. A second remap here
        # (previous behaviour) silently dropped every LF on any day at
        # non-zero drive offset because it treated already-day-local
        # values as drive-global. This broke any day whose
        # single episode sits at a non-zero drive index.
        _max_day_local = len(day_episode_indices) - 1
        for seg in list(segments) + list(peaks):
            gi = int(getattr(seg, "source_file_index", 0))
            if gi < 0 or gi > _max_day_local:
                raise ValueError(
                    f"Day {date}: LF {idx} segment/peak source_file_index "
                    f"{gi} outside day-local range [0, {_max_day_local}]; "
                    "caller must remap drive-global → day-local before "
                    "run_day_pipeline. See _partition_lf_plans_by_date."
                )
        # OST burn-in text priority (single source of truth):
        #   1. per_lf_ost_text[idx]     ← canonical "{start} to {end}
        #                                  via {road}, {state}"
        #   2. per_lf_locations[idx]    ← county-level fallback
        #   3. location                 ← drive-wide default
        # (1) is the single
        # source of truth for OST burn / YouTube title / thumbnail.
        # Resolved BEFORE extract/montage so a resume can skip re-burning
        # an already-final LF whose OST text is unchanged.
        # Burn EXACTLY the text the audit gate resolved and wrote to the
        # sidecar (``_gate_lf_text``, state-preferring) — never re-derive an
        # independent answer from the raw kwargs here, or the burned pixels
        # can diverge from what the operator reviewed.
        lf_location = _gate_lf_text.get(idx, location)
        per_lf_supplied = bool(
            (per_lf_locations and idx in per_lf_locations)
            or (per_lf_ost_text and idx in per_lf_ost_text and per_lf_ost_text[idx])
        )

        # Contract assertion at OST-burn entry. NOT a
        # quality gate: this fires only if the upstream construction
        # logic regresses (drive-wide label leaked into a per-LF slot
        # via the per_lf_locations / per_lf_ost_text path). Only checked
        # when per-LF text was supplied — fixture-driven legacy tests
        # that pass only ``location`` are unaffected. The robust-by-
        # construction CLI gate in phase_bc_day_first already refused
        # to enter encoding when an LF lacks GPS + an override, so this
        # is cheap defense-in-depth.
        if (
            per_lf_supplied
            and (location or "").strip()
            and (lf_location or "").strip() == (location or "").strip()
        ):
            raise RuntimeError(
                f"OST-burn contract violation: LF {idx} per-LF label "
                f"{lf_location!r} equals drive-wide --location "
                f"{location!r} verbatim. Construction logic regressed "
                "— the drive-wide fallback path in phase_bc was removed."
            )

        final_lf = lf_dir / f"lf_{idx:02d}.mp4"
        # Per-LF freshness: skip re-burning a final that already exists
        # and whose burned-OST fingerprint is unchanged. The shorts OST-gate
        # stops the run AFTER every LF is final; a plain gate→confirm→resume
        # would otherwise re-NVENC (~10-15min each) every unchanged LF.
        ost_fingerprint = _lf_ost_fingerprint(
            lf_location, montage_duration, ost_crossfade_sec, segments
        )
        _lf_inserts = list((inserts_by_lf or {}).get(idx) or [])
        if _lf_inserts:  # a changed insert list re-burns, so a stale final never ships
            ost_fingerprint = hashlib.sha256(
                (ost_fingerprint + repr(insert_records({idx: _lf_inserts}))).encode()
            ).hexdigest()
        fp_path = lf_dir / f"lf_{idx:02d}.ost.sha256"
        if (
            final_lf.is_file()
            and final_lf.stat().st_size > 0
            and fp_path.is_file()
            # a swapped-in trail opening is folded into the stored fingerprint
            and fp_path.read_text().strip()
            == trail_opening.expected_fingerprint(lf_dir, idx, ost_fingerprint)
        ):
            logger.info(
                "Day %s: LF %d final is fresh (OST unchanged) — skipping re-burn",
                date,
                idx,
            )
            regular_lf_paths.append(final_lf)
            regular_lf_indices.append(idx)
            continue

        # Upstream of the stream copy + montage prepend + OST re-encode: this
        # LF's three coexisting artifacts must fit on the cache volume now.
        # The day's length is the end of its last file boundary.
        check_lf_work_space(
            master_source_for_extract,
            lf_dir,
            sum(
                float(getattr(s, "end_time", 0.0)) - float(getattr(s, "start_time", 0.0))
                for s in segments
            ),
            float(file_boundaries[-1][1]) if file_boundaries else 0.0,
            lf_index=idx,
        )
        trail_opening.reset(lf_dir, idx)  # this burn restores the montage lead
        work_dir = lf_dir / f".lf_{idx:02d}_work"
        raw_concat = lf_dir / f"lf_{idx:02d}_segments.mp4"
        concatenated = extract_and_concat_segments(
            master_path=master_source_for_extract,
            segments=segments,
            file_boundaries=file_boundaries,
            output_path=raw_concat,
            work_dir=work_dir,
            # STOP-aware cut layer: the per-day mask (master seconds) keeps the
            # extractor from merging across a parked gap or snapping a `-c copy`
            # start back into parked footage. None on GPS-less/legacy/
            # disabled days → legacy behavior.
            stop_mask=stop_mask,
        )
        if concatenated is None:
            logger.warning("Day %s: LF %d concat failed, skipping", date, idx)
            continue

        with_montage = lf_dir / f"lf_{idx:02d}_with_montage.mp4"
        prepended = prepend_montage(
            raw_lf_path=concatenated,
            peaks=peaks,
            output_path=with_montage,
            work_dir=work_dir,
            master_path=master_source_for_extract,
            file_boundaries=file_boundaries,
        )
        if prepended is None:
            logger.warning(
                "Day %s: LF %d montage prepend failed, skipping",
                date,
                idx,
            )
            continue

        # prefer_nvenc=True: the day master just finished on NVENC, so the
        # LF OST re-encode should too. The helper retries the NVENC probe
        # once to ride out post-encode session release (otherwise a busy
        # NVENC session falls back to libx265, which can exceed the 1h timeout).
        ost_result = apply_ost(
            input_path=prepended,
            output_path=final_lf,
            location=lf_location,
            layout="regular_lf",
            brand_logo_path=brand_logo_path,
            prefer_nvenc=True,
            montage_duration=montage_duration,
            crossfade_sec=ost_crossfade_sec,
            media_profile=media_profile,
        )
        if ost_result is None:
            raise RuntimeError(
                f"OST application failed for LF {idx} ({lf_location}). "
                f"Input: {prepended}, Output: {final_lf}"
            )
        if _lf_inserts:
            splice_day_lf_inserts(
                ost_result,
                _lf_inserts,
                config=config,
                label=insert_label(per_lf_locations, idx, lf_location),
                profile=media_profile,
                work_dir=work_dir,
                montage_planned_s=montage_duration,
                prepended=prepended,
                concatenated=concatenated,
            )
        # Keep the lead and the OST recipe so the trail film, rendered after this
        # LF, can replace the montage at 0 s once it exists (the intermediates are purged).
        try:
            _lead_s = float(probe_media(prepended).duration) - float(
                probe_media(concatenated).duration
            )
            trail_opening.record_lead(
                lf_dir,
                idx,
                lead_s=_lead_s,
                location=lf_location,
                crossfade_sec=ost_crossfade_sec,
                brand_logo_path=brand_logo_path,
                inserts=shift_photo_inserts(_lf_inserts, _lead_s - montage_duration),
                planned_s=montage_duration,
            )
        except Exception as exc:  # noqa: BLE001 -- the LF is fine; only its trail opening is lost
            logger.warning("Day %s: LF %d lead not recorded (%s)", date, idx, exc)
        # Persist the burned-OST fingerprint so a later resume can detect this
        # final is fresh and skip the re-burn.
        fp_path.write_text(ost_fingerprint)
        regular_lf_paths.append(ost_result)
        regular_lf_indices.append(idx)
        # Release this LF's work files now, not at day end: a multi-part day
        # otherwise carries every finished part's segments + with_montage
        # while the next part burns.
        if not keep_intermediates:
            try:
                purge_lf_intermediates(day_dir, idx)
            except Exception as exc:  # noqa: BLE001 — never block the pipeline on cleanup
                logger.warning("Day %s: LF %d work-file purge failed: %s", date, idx, exc)

    # ── Step 4: plan shorts from the same day scores ──────────────
    # Shorts budget is per-LF: plan `<decided per-LF count> * len(lf_plans)`
    # shorts from the day master. Phase D partitions them across each LF's
    # publish window by datetime.
    #
    # Source: the per-LF count is Waypost's decision, read through
    # ``volume_source.shorts_per_lf()`` -- no config override, no derived
    # fallback. ``state["shorts_target"]`` (operator CLI, resume-persisted) can
    # only LOWER it (``volume_source.apply_flag``). The count derived from the
    # publish plan (``required_shorts_per_lf``) remains only as the slot
    # capacity check below. An unreadable store raises and plans nothing.
    # region=None on the derivation: the drive handler already resolved the
    # region/--shorts-times precedence into shorts_publish_times.
    # config=None (test/legacy path with no publish plan) keeps the
    # planner's own default behavior.
    _shorts_kw: dict = {}
    _decided_per_lf = None
    _config_gap = None
    _derived_per_lf = None
    if config is not None:
        _drive = getattr(config, "drive", None)
        if _drive is not None:
            _decided_per_lf = volume_source.shorts_per_lf().value
            _config_gap = getattr(_drive, "shorts_min_gap_sec", 60.0)
            # Configurable short duration bounds. Falls
            # back to the short_planner module defaults when config.drive
            # doesn't carry these fields (legacy/test stubs).
            _shorts_kw["min_duration_sec"] = getattr(
                _drive, "shorts_min_duration_sec", DEFAULT_MIN_DURATION_SEC
            )
            _shorts_kw["max_duration_sec"] = getattr(
                _drive, "shorts_max_duration_sec", DEFAULT_MAX_DURATION_SEC
            )
            _shorts_kw["target_duration_sec"] = getattr(
                _drive, "shorts_target_duration_sec", DEFAULT_TARGET_DURATION_SEC
            )
            # A drive config without shorts_publish_times (legacy/test
            # stubs) has no publish plan to derive from — skip derivation.
            if getattr(_drive, "shorts_publish_times", None):
                # --shorts-per-day N truncates the scheduler to N windows
                # (Phase D), so derive the planner target from the SAME N — else
                # the planner demands full-window shorts the scheduler has no
                # slots for and every surplus ships slotless.
                _spd = state.get("shorts_per_day") if state is not None else None
                _derived_per_lf = required_shorts_per_lf(config, shorts_per_day=_spd)
    _state_target = None
    _state_gap = None
    if state is not None:
        _state_target = state.get("shorts_target")
        _state_gap = state.get("shorts_gap")
    # ``--shorts-target N`` (which lands as ``state["shorts_target"]``)
    # is operator-intent-aligned to mean "N shorts per LF", not a per-day
    # total cap. Multiply by ``num_lfs`` so a 14/LF target on a 3-LF day
    # plans 42 shorts.
    num_lfs = len(lf_plans or [])
    # The per-LF number behind ``target_count``, kept so the render step can cap
    # each LF at what its publish window can actually schedule (see
    # ``cap_shorts_to_per_lf_capacity``). Derived from the SAME precedence as
    # the day total so the two cannot drift.
    _effective_per_lf: Optional[int] = None
    if _decided_per_lf is not None:
        _effective_per_lf = volume_source.apply_flag(_decided_per_lf, _state_target)
        target_count = _effective_per_lf * num_lfs
    elif _state_target is not None:
        # No drive config (legacy/test path): there is no decision to lower.
        target_count = _state_target * num_lfs
        _effective_per_lf = _state_target
    else:
        target_count = None
    _rung = (
        "cli"
        if _state_target is not None
        else "decision"
        if _decided_per_lf is not None
        else "none"
    )
    # A brand's policy.per_location_cap (brand_policy.py) further
    # trims the per-LF target -- never raises it, and never applies to an
    # explicit --shorts-target (operator intent always wins). A thin drive
    # still yields fewer shorts via the moving-budget cap below; this is an
    # additional, brand-scoped ceiling on top of it. The SHORTS-CONTRACT
    # gate below checks THIS cap-derived target, never the uncapped
    # config/derived value or the unrelated min_shorts_per_lf floor.
    if _state_target is None and target_count is not None:
        _policy_cap = kit.per_location_cap()
        if _policy_cap is not None and _policy_cap < _effective_per_lf:
            _effective_per_lf = _policy_cap
            target_count = _effective_per_lf * num_lfs
            _rung = "brand-policy-cap"
    logger.info(
        "Day %s: shorts target %s (per LF %s × %d LFs, rung=%s)",
        date,
        target_count,
        _effective_per_lf,
        num_lfs,
        _rung,
    )
    min_gap_sec = _state_gap if _state_gap is not None else _config_gap
    # A short names only the place its own clip's GPS votes: on a day with GPS
    # the planner honours the GPS-gap veto on top of the stop mask, so it cannot
    # pick a window without the fixes that vote needs. Planning-only — the
    # persisted stop mask above stays the pure stop mask.
    _plan_mask = _shorts_planning_mask(
        stop_mask,
        day_scores,
        gps_points_by_date.get(date) if gps_points_by_date else None,
        date,
        # the shared cached geocoder the shorts OST closure was built with
        geocoder=getattr(short_ost_text_fn, "geocoder", None),
        half_window_s=_shorts_kw.get("target_duration_sec", DEFAULT_TARGET_DURATION_SEC),
    )
    # Cap the required target by the day's MOVING-seconds budget so a
    # parked-heavy (or GPS-gap-heavy) day doesn't pressure routine
    # --allow-short-count use.
    _capped = _cap_shorts_target_by_moving_budget(target_count, _plan_mask, min_gap_sec)
    if _capped is not None and target_count is not None and _capped < target_count:
        logger.warning(
            "Day %s: capping shorts target %d -> %d — the moving-seconds budget "
            "after the stop mask and the GPS-gap veto can't hold more shorts "
            "(parked-heavy or GPS-gap-heavy day)",
            date,
            target_count,
            _capped,
        )
    target_count = _capped
    if target_count is not None:
        _shorts_kw["target_count"] = target_count
    if min_gap_sec is not None:
        _shorts_kw["min_gap_sec"] = min_gap_sec
    short_ranges = (
        plan_shorts(day_scores, stop_mask=_plan_mask, **_shorts_kw)
        if day_scores is not None
        else []
    )
    logger.info("Day %s: %d short ranges planned", date, len(short_ranges))
    # SHORTS-CONTRACT gate: the
    # required count and the planned count are both in scope right here —
    # compare them BEFORE any shorts encode/CSV/upload instead of logging
    # a shortfall and continuing.
    allow_short_count = bool(state.get("allow_short_count")) if state is not None else False
    if day_scores is not None:
        _enforce_shorts_contract(len(short_ranges), target_count, date, allow_short_count)

    # PER-LF CAPACITY: the contract above is drive-wide and
    # the planner is day-wide, but slots are per-LF — so the right TOTAL can
    # still leave one LF holding more shorts than its publish window can post.
    # Trim the surplus BEFORE any render, so no short exists without a slot and
    # the Metricool CSV gate has nothing to refuse. ``_derived_per_lf`` is the
    # scheduler's own ceiling; an explicit target above it is unschedulable no
    # matter what the operator asked for, so it binds too.
    _per_lf_capacity = _effective_per_lf
    if _derived_per_lf is not None and _per_lf_capacity is not None:
        _per_lf_capacity = min(_per_lf_capacity, _derived_per_lf)
    if day_scores is not None and short_ranges:
        short_ranges, _capacity_dropped = cap_shorts_to_per_lf_capacity(
            short_ranges,
            lf_plans or [],
            compute_day_file_boundaries(day_batch),
            _per_lf_capacity,
        )
        if _capacity_dropped:
            logger.warning(
                "Day %s: dropped %d of %d planned shorts with no schedulable slot "
                "(per-LF capacity %d) — %d will be rendered",
                date,
                len(_capacity_dropped),
                len(_capacity_dropped) + len(short_ranges),
                _per_lf_capacity,
                len(short_ranges),
            )

    # ── Step 5: extract + OST each short (rank-ordered) ───────────
    # Location anchor for a short on a GPS-less day: the operator
    # ``--per-lf-location`` overrides (authoritative, verbatim), sourced from
    # the canonical pipeline state. Never the destination POI — where the
    # drive went, not where the clip was filmed.
    per_lf_overrides = (
        dict(getattr(state, "per_lf_overrides", None) or {}) if state is not None else {}
    )
    short_paths, short_ranges_emitted, short_ost_audit_entries = _render_day_shorts(
        day_dir=day_dir,
        date=date,
        day_batch=day_batch,
        short_ranges=short_ranges,
        allow_short_count=allow_short_count,
        gps_points_by_date=gps_points_by_date,
        lf_plans=lf_plans,
        per_lf_ost_text=per_lf_ost_text,
        per_lf_locations=per_lf_locations,
        per_lf_overrides=per_lf_overrides,
        location=location,
        short_ost_text_fn=short_ost_text_fn,
        master_source_for_extract=master_source_for_extract,
        short_regions_fn=short_regions_fn,
        short_route_fn=short_route_fn,
        state=state,
        stop_mask=stop_mask,
    )

    logger.info(
        "Day %s complete: master=%s, regular_lfs=%d, shorts=%d",
        date,
        master_final_path.name,
        len(regular_lf_paths),
        len(short_paths),
    )

    # Refresh the operator-visible audit of every short's burned text for
    # the downstream phases (phase_d per-short labels). The pre-burn gate
    # already wrote this file for review before the burn; this re-emits the
    # final confirmed copy after a passing run.
    if short_ost_audit_entries:
        try:
            write_shorts_ost_audit(Path(output_dir), date, short_ost_audit_entries)
        except Exception as exc:  # noqa: BLE001 — audit write must never fail a day
            logger.warning("Day %s: shorts OST audit write failed: %s", date, exc)
        # And then PROVE it. Everything above attests what the burn was ASKED
        # to composite; this reads the rendered pixels and hard-fails a short whose
        # title band is provably bare — the last point upstream of phase_d's upload.
        # Deliberately OUTSIDE the swallow above: a lost audit file is a lost
        # report, a title-less short is an irreversible publish.
        postburn_ost_gate_or_raise(date, short_ost_audit_entries)

    # ── Superlong parts (opt-in, additive; NEVER folded into lf_plans) ──
    # Dispatched here — AFTER the day's LFs and shorts are fully burned and
    # proven (postburn gate above), BEFORE the intermediate purge below —
    # instead of before the shorts work, so the publishable day-first
    # output lands ~2.5h sooner instead of queuing behind a superlong part
    # that runs to a 4h/part cap. Every input read below (stop_mask,
    # episodes, file_boundaries, master_source_for_extract, day_dir,
    # lf_plans) is set once earlier in this function and untouched by the
    # shorts work, so it's still valid here; `purge_day_intermediates`
    # itself already gates superlong's own segment/work-dir cleanup on
    # `superlong/sl_XX.mp4` finals existing, so running before it (not
    # after) is required, not incidental.
    if _drive is not None and bool(getattr(_drive, "superlong_enabled", False)):
        if stop_mask is None:
            # Both reasons a mask can be legitimately absent (detection
            # disabled by config, or the day has no scores) are real
            # operator situations — but --superlong is an explicit opt-in
            # that REQUIRES one (mirrors superlong_planner.plan_superlong_
            # parts's own ValueError). Hard-fail naming which case this is
            # rather than silently skipping and reporting a green run with
            # zero superlong output.
            if not _stop_mask_enabled:
                _reason = "config.drive.stop_mask_enabled is False for this run"
            else:
                _reason = f"day {date} has no drive scores (day_scores missing/empty)"
            raise ValueError(
                f"Day {date}: --superlong requires a stop mask, but {_reason}, "
                "so none was built. Enable config.drive.stop_mask_enabled and "
                "ensure day scores exist, or omit --superlong for this day."
            )
        else:
            # Recomputed here (not reused from the stop-mask branch above,
            # which is conditional on day_scores and doesn't always run) so
            # every part's chapters chart the WHOLE day's GPS, not just the
            # first GPS-bearing episode's GPX linearly smeared across parts.
            _day_gps_points = gps_points_by_date.get(date) if gps_points_by_date else None
            _sl_plan = _produce_superlong_parts(
                date=date,
                stop_mask=stop_mask,
                episodes=episodes,
                file_boundaries=file_boundaries,
                master_path=master_source_for_extract,
                day_dir=day_dir,
                config=config,
                location=location,
                ost_crossfade_sec=ost_crossfade_sec,
                lf_plans=lf_plans,
                state=state,
                day_gps_points=_day_gps_points,
                media_profile=media_profile,
                kit=kit,
                superlong_route_fn=superlong_route_fn,
            )
            if state is not None:
                state.setdefault("superlong_parts", {})[date] = _sl_plan

    # Inline intermediate purge — bounds the cache at ~one day's working set
    # by dropping per-LF/short work files (segments, with_montage, .lf_XX_work,
    # short_XX_raw) once their final OST-applied outputs exist. Gate is
    # per-LF/short so a partial day is safe. The master + finals are
    # never touched here — those are Phase D's responsibility.
    # (LF work files already went per LF above; this sweeps shorts, superlong
    # and any LF whose per-LF purge failed.)
    if not keep_intermediates:
        try:
            purge_day_intermediates(day_dir)
        except Exception as exc:  # noqa: BLE001 — never block the pipeline on cleanup
            logger.warning("Day %s: intermediate purge failed: %s", date, exc)

    # Capture the master's integrity metadata (duration + fingerprint) ONCE
    # here so per_day_completed can carry it for cached-master validation on
    # resume after the raw episodes are purged.
    master_record = build_master_record(day_batch, master_final_path)
    # A master narrower than a 9:16 short's output makes every short an upscale.
    master_record["short_upscaled_from"] = _short_upscaled_from(master_final_path)
    return DayArtifacts(
        inserts=insert_records(
            {i: v for i, v in (inserts_by_lf or {}).items() if i in regular_lf_indices}
        ),
        date=date,
        master_path=master_final_path,
        regular_lf_paths=regular_lf_paths,
        regular_lf_indices=regular_lf_indices,
        short_paths=short_paths,
        short_ranges=short_ranges_emitted,
        master_record=master_record,
        # POST-CAP expectation: the master-deletion gate count-matches against
        # what this day is actually meant to produce. Using the uncapped
        # ``target_count`` would make a correctly-capped day look short and
        # pin its ~118 GB master forever.
        planned_shorts_count=(
            (None if target_count is None else len(short_ranges))
            if day_scores is not None
            else None
        ),
        shorts_target_in_force=(
            (None if target_count is None else _effective_per_lf)
            if day_scores is not None
            else None
        ),
    )


def _derive_day_score_bounds(
    day_batch: DayBatch,
    file_boundaries: list,
    episode_index_by_path: dict,
) -> Optional[tuple]:
    """Given a day batch and the drive-wide file_boundaries list, find
    the [global_start_sec, global_end_sec) bounds of this day's scored
    footage.

    Each episode in the day batch maps to an entry in `file_boundaries`
    via `episode_index_by_path` (the concat_path → drive-episode-index
    map built by the caller). The day's bounds are the min start and
    max end across its episodes.

    Returns None if the day has no resolvable bounds (e.g. none of its
    episodes appear in the boundaries map).
    """
    starts: list = []
    ends: list = []
    for ep in day_batch.episodes:
        concat_path = getattr(ep, "concat_path", None)
        if concat_path is None:
            continue
        idx = episode_index_by_path.get(str(concat_path))
        if idx is None or idx >= len(file_boundaries):
            continue
        start, end = file_boundaries[idx]
        starts.append(float(start))
        ends.append(float(end))
    if not starts:
        return None
    return (min(starts), max(ends))


def run_all_days(
    day_batches: list,
    drive_scores: np.ndarray,
    file_boundaries: list,
    output_base: Path,
    location: str,
    config=None,
    episode_paths: Optional[list] = None,
    per_day_locations: Optional[dict] = None,
    lf_plans_by_date: Optional[dict] = None,
    per_lf_locations_by_date: Optional[dict] = None,
    per_lf_ost_text_by_date: Optional[dict] = None,
    gps_points_by_date: Optional[dict] = None,
    short_ost_text_fn: Optional[Callable] = None,
    short_regions_fn: Optional[Callable] = None,
    short_route_fn: Optional[Callable] = None,
    superlong_route_fn: Optional[Callable] = None,
    state=None,
    on_day_complete: Optional[Callable] = None,
    state_save_fn: Optional[Callable] = None,
    all_day_batches: Optional[list] = None,
    all_gps_points_by_date: Optional[dict] = None,
    kit: Optional[TitleKit] = None,
) -> list:
    """Run the day-first pipeline for every day in `day_batches`.

    Slices `drive_scores` per day using `file_boundaries`, calls
    `run_day_pipeline` for each day, and returns the list of successful
    `DayArtifacts`. Failed days (where `run_day_pipeline` returns None)
    are logged and skipped so the rest of the trip still produces
    output.

    Args:
        day_batches: List of `DayBatch` in chronological order.
        drive_scores: The drive-wide score array from
            `UnifiedSegmentScorer.merge_file_scores`.
        file_boundaries: List of `(start_sec, end_sec)` tuples, one per
            episode file in the drive, in the same order the scorer saw
            them. Together with the episode list this lets us map a day
            batch to its global score bounds.
        output_base: Parent directory for all per-day output. Each day
            lands in `output_base/day_YYYY-MM-DD/`.
        location: Drive-wide OST fallback used for any day that doesn't
            have an entry in `per_day_locations`.
        config: Optional `gomoveshift.config.Config` passed through.
        episode_paths: Ordered list of concat paths the scorer saw. When
            provided, used to build the path → index map for slicing.
            When None, falls back to a positional walk through the day
            batches (works when days contain all episodes in order).
        per_day_locations: Optional mapping of `{date: ost_text}`.
            When set, each day receives its own OST string so multi-day
            drives don't bleed Day 1's location onto Day 2's outputs
            (multi-day drives).
        per_lf_locations_by_date: Optional
            `{date: {lf_index: county_state_label}}` map threaded through
            to `run_day_pipeline`'s per-LF OST burn-in.
        per_lf_ost_text_by_date: Optional
            `{date: {lf_index: canonical_route_text}}` map. When
            an entry exists it WINS over `per_lf_locations_by_date` for
            that LF's OST burn-in, keeping the burned pixels in sync
            with the YouTube title and thumbnail text.
        gps_points_by_date: Optional `{date: gps_points}` map forwarded
            to each `run_day_pipeline` call so the shorts OST loop can
            trim GPS to each short's clip-timeline range.
            Without this forwarding the kwarg would always be None
            inside the per-day call, making the trim a no-op.
        short_ost_text_fn: Optional callable
            `(trimmed_points, short_range) -> Optional[str]` forwarded
            to each `run_day_pipeline` call. Without
            a consumer the trim output was computed and discarded, so
            every short burned the drive-wide `location` fallback.
            The callable turns the trimmed points (which describe
            what's on-screen, not the source footage) into a short-
            specific OST string.
        all_day_batches / all_gps_points_by_date: On a resume, EVERY day of
            the drive (finished ones too) and their GPS. Inserts are placed
            against all of them (a photo of a finished day is still placed,
            the clip stays on the drive's first day) and gated only for the
            days in `day_batches`. Omitted: `day_batches` is the whole drive.

    Returns:
        List of `DayArtifacts`, one per successful day.
    """
    if not day_batches:
        return []

    # Build a path → drive-episode-index map. Preferred source is the
    # explicit `episode_paths` list from the orchestrator (matches the
    # scorer's own ordering). When unavailable, walk the day batches in
    # order and assume the days cover every scored episode positionally.
    episode_index_by_path: dict = {}
    if episode_paths is not None:
        for i, path in enumerate(episode_paths):
            episode_index_by_path[str(path)] = i
    else:
        pos = 0
        for batch in day_batches:
            for ep in batch.episodes:
                cp = getattr(ep, "concat_path", None)
                if cp is not None:
                    episode_index_by_path[str(cp)] = pos
                    pos += 1

    per_day_map = per_day_locations or {}

    # Place AND gate every requested insert BEFORE the first encode (a photo that fits
    # no day, a county outside the LF label, an unblurred clip all refuse here), so the
    # audit below covers them and nothing is burned for a run that cannot finish.
    _inserts_by_date = plan_drive_inserts(
        config,
        all_day_batches or day_batches,
        lf_plans_by_date=lf_plans_by_date,
        gps_points_by_date=all_gps_points_by_date or gps_points_by_date,
        render_dates={b.date for b in day_batches},
        label_of=lambda d, i: insert_label(
            (per_lf_locations_by_date or {}).get(d), i, per_day_map.get(d, location)
        ),
    )

    # ── Front-load every day's LF OST audit before the first encode ──
    # Every day's per-LF burn text is already resolved (``per_lf_ost_text_by_date``
    # / pre-seeded state) BEFORE any master encode, so run the LF OST-audit gate
    # for ALL days HERE — writing each day's sidecar and raising ONCE upfront —
    # instead of the per-day gate inside ``run_day_pipeline`` raising AFTER that
    # day's multi-hour master encode (one stop/resume cycle per day). Same
    # output_base + per-day ``location`` + per-LF maps as the encode loop below,
    # so the hashes match and the per-day gate then fast-passes. (Shorts keep
    # their per-day gate — their text needs per-day short planning.)
    _pending_ost_days: list = []
    for _fl_batch in day_batches:
        _fl_plans = (lf_plans_by_date or {}).get(_fl_batch.date) or []
        _fl_indices = [int(getattr(p, "index", i + 1)) for i, p in enumerate(_fl_plans)]
        if not _fl_indices:
            continue
        try:
            lf_gate_or_raise(
                Path(output_base),
                _fl_batch.date,
                _fl_indices,
                state,
                location=per_day_map.get(_fl_batch.date, location),
                per_lf_ost_text=(per_lf_ost_text_by_date or {}).get(_fl_batch.date),
                per_lf_locations=(per_lf_locations_by_date or {}).get(_fl_batch.date),
                inserts=insert_records(_inserts_by_date.get(_fl_batch.date) or {}),
            )
        except OstAuditPendingError:
            _pending_ost_days.append(_fl_batch.date)
    if _pending_ost_days:
        raise OstAuditPendingError(
            "OST audit pending for day(s) "
            f"{sorted(str(d) for d in _pending_ost_days)}: review each "
            "ost_audit_<date>.json sidecar and create the matching .confirmed "
            "file (or re-run with --ost-audit-confirmed) before the first "
            "master encode (front-load)."
        )

    # Persist the OST audit hash markers the front-load gate just
    # set BEFORE the first multi-hour master encode. Previously they only landed
    # at day-complete, so a SIGKILL mid-encode lost them — and a label edit
    # before resume then auto-passed a materially changed, never-reviewed burn.
    # With the markers durable here, the resume sees the real prior hash and the
    # stale-reopen logic fires on any change.
    if state is not None and state_save_fn is not None:
        try:
            state_save_fn(state)
        except Exception as _persist_err:  # non-fatal: encode still guarded per-day
            logger.warning(
                "run_all_days: front-load OST hash persist failed (%s); "
                "day-complete save remains the backstop.",
                _persist_err,
            )

    results: list = []
    for day_batch in day_batches:
        bounds = _derive_day_score_bounds(
            day_batch,
            file_boundaries,
            episode_index_by_path,
        )
        if bounds is None:
            logger.warning(
                "run_all_days: day %s has no score bounds, skipping",
                day_batch.date,
            )
            continue
        day_start, day_end = bounds
        day_scores = slice_scores_for_day(drive_scores, day_start, day_end)

        # Per-day location override (fixes multi-day location leakage —
        # every output of this day gets its own OST text, not the
        # drive-wide fallback).
        day_location = per_day_map.get(day_batch.date, location)

        # Wrap ``run_day_pipeline``
        # in try/except so one day's crash (or soft-None return) does
        # NOT abort the outer loop. The prior behaviour propagated the
        # exception up through ``PhaseBCDayFirst.run`` and the resume
        # marker for the crashed day was never written — even when
        # finals or a raw concat had actually landed on disk before the
        # crash. The finals-only reconstruction fast-path
        # (``_day_dir_has_finals`` / ``_reconstruct_day_artifacts_from_disk``
        # in ``phase_bc_day_first.py``) wants a per_day_completed
        # entry; without it, the next resume re-encodes from scratch
        # (8+ hours wasted).
        from .media_api import RunSpec, render_day

        day_dir = Path(output_base) / f"day_{day_batch.date}"
        try:
            artifacts = render_day(
                RunSpec(
                    day_batch=day_batch,
                    day_scores=day_scores,
                    output_dir=output_base,
                    location=day_location,
                    config=config,
                    lf_plans=(lf_plans_by_date or {}).get(day_batch.date),
                    per_lf_locations=(per_lf_locations_by_date or {}).get(day_batch.date),
                    # Canonical OST text wins over the county-level label
                    # inside run_day_pipeline. See run_day_pipeline docstring for
                    # the 3-level priority cascade.
                    per_lf_ost_text=(per_lf_ost_text_by_date or {}).get(day_batch.date),
                    # Forward per-date GPS + short OST text callback so
                    # the shorts OST composition inside run_day_pipeline can
                    # derive its text from ``trim_gps_to_clip_timeline``'s
                    # output instead of always falling back to the drive-wide
                    # ``location`` string. Without this wiring the trim
                    # helper would be invoked but its output discarded.
                    gps_points_by_date=gps_points_by_date,
                    short_ost_text_fn=short_ost_text_fn,
                    short_regions_fn=short_regions_fn,
                    short_route_fn=short_route_fn,
                    superlong_route_fn=superlong_route_fn,
                    state=state,
                    inserts_by_lf=_inserts_by_date.get(day_batch.date) or None,
                    kit=kit,
                )
            )
        except OstAuditPendingError:
            # The OST audit gate is a pipeline-control signal, not a
            # crash. Re-raise unwrapped so
            # ``DrivePipelineOrchestrator.run``'s
            # ``except OstAuditPendingError`` clause
            # (in drive_orchestrator.py) can exit with
            # ``success=True`` and partial ``stages_completed``
            # instead of the generic-exception wrap collapsing it
            # into ``RuntimeError``. Only this specific pipeline-
            # control exception is exempt from the crash-recovery
            # wrap — every other exception keeps the existing
            # "log + skip the day + continue" behaviour below.
            raise
        except (BlurGateError, OstPostBurnError, InsertError):
            # Privacy hard-fail (blur required but the master isn't provably blurred),
            # or a short whose burned title is provably absent from its pixels:
            # like OstAuditPendingError, these are pipeline-control signals, NOT a
            # recoverable per-day crash. Re-raise so the run aborts LOUDLY upstream of
            # any upload instead of silently skipping the day (and shipping the rest of
            # the week thinner) — a broken OST composite breaks EVERY day, so a per-day
            # skip would just repeat the failure while reporting progress. Operator
            # re-runs --fresh or with --skip-blur.
            raise
        except Exception as exc:  # noqa: BLE001 — crash recovery is the whole point
            logger.error(
                "run_all_days: day %s run_day_pipeline raised — %s. "
                "Attempting on-disk recovery checkpoint before continuing.",
                day_batch.date,
                exc,
            )
            _maybe_fire_recovery_callback(day_batch.date, day_dir, state, on_day_complete)
            continue
        if artifacts is None:
            logger.warning(
                "run_all_days: day %s produced no artifacts, skipping",
                day_batch.date,
            )
            # Soft failure. Same recovery logic as the exception path —
            # if finals or a raw concat survived, mark the day so the
            # next resume picks it up via the finals-only fast-path.
            _maybe_fire_recovery_callback(day_batch.date, day_dir, state, on_day_complete)
            continue
        results.append(artifacts)

        # Per-day write-through: record completion + persist state BEFORE
        # starting the next day. Without this, a mid-phase crash (e.g.
        # day N master encode fails after days 0..N-1 landed cleanly)
        # leaves `per_day_completed` empty on disk and the next resume
        # can't short-circuit the already-done days. Callback is
        # best-effort — failures are logged, not propagated, because
        # losing disk-side bookkeeping is recoverable but losing the
        # remaining days' encoded work is not.
        if on_day_complete is not None:
            try:
                on_day_complete(artifacts, state)
            except Exception as exc:  # noqa: BLE001 — bookkeeping never blocks
                logger.warning(
                    "run_all_days: on_day_complete callback failed for day %s: %s — "
                    "continuing with remaining days",
                    day_batch.date,
                    exc,
                )

    return results


def _day_dir_has_finals_or_raw(day_dir: Path) -> bool:
    """Return True when ``day_dir`` has a final LF + short OR a raw concat
    on disk — the two conditions under which the next resume can still
    make progress without re-encoding. Mirrors
    ``phase_bc_day_first._day_dir_has_finals`` plus an additional check
    for ``day_<date>_raw.mp4`` (partial encode that landed before the
    crash).
    """
    if not day_dir.is_dir():
        return False
    lf_dir = day_dir / "regular_lfs"
    shorts_dir = day_dir / "shorts"
    has_lf = lf_dir.is_dir() and any(
        p.is_file() and p.stat().st_size > 0 for p in lf_dir.glob("lf_[0-9][0-9].mp4")
    )
    has_short = shorts_dir.is_dir() and any(
        p.is_file() and p.stat().st_size > 0 for p in shorts_dir.glob("short_[0-9][0-9].mp4")
    )
    if has_lf and has_short:
        return True
    # Raw concat (pre-master-encode intermediate) is enough for the next
    # resume to skip re-concatenating — the finals-only reconstruction
    # fast-path is LF/shorts-driven so a bare raw-only day still produces
    # a "re-encode from raw" code path downstream. A non-empty raw file
    # (size > 0 to guard against an aborted write) is the trigger.
    raw_candidates = list(day_dir.glob("day_*_raw.mp4"))
    for raw in raw_candidates:
        if raw.is_file() and raw.stat().st_size > 0:
            return True
    return False


def _maybe_fire_recovery_callback(
    date: str,
    day_dir: Path,
    state,
    on_day_complete: Optional[Callable],
) -> None:
    """If the day directory holds finals or a raw concat, fire
    ``on_day_complete`` with a DayArtifacts(master_path=None) so the
    caller can mark the day as recorded for the finals-only fast-path.

    Called from the exception / None-return branches of ``run_all_days``.
    A no-op when nothing on disk suggests
    progress — we do NOT want a bogus entry that would block the next
    resume from re-encoding.
    """
    if on_day_complete is None:
        return
    if not _day_dir_has_finals_or_raw(day_dir):
        logger.info(
            "run_all_days: day %s has nothing on disk yet — no recovery "
            "checkpoint written (next resume will re-encode from scratch)",
            date,
        )
        return

    # A crash MID-render (mid-LF / mid-shorts) leaves the day master on
    # disk alongside a *partial* finals set (e.g. a 9-of-14 shorts day).
    # Writing a finals-only "complete" marker here would let the next resume
    # take the reconstruction fast-path and SHIP THAT THIN DAY — no
    # count-vs-target shortfall gate fires on a reconstructed (skipped) day.
    # Because the expensive master survived, the day must instead RE-RUN on
    # resume: ``encode_day_master``'s own resume shortcut reuses the cached
    # master (no ~8h re-encode) while ``run_day_pipeline`` re-plans and
    # re-renders the FULL shorts set, and the existing SHORTS-CONTRACT gate
    # guarantees the target count. Skip the marker so partition routes this
    # day to a re-run — robust-by-construction, no late gate needed.
    master_on_disk = day_dir / f"day_{date}_master.mp4"
    if master_on_disk.is_file() and master_on_disk.stat().st_size > 0:
        logger.warning(
            "run_all_days: day %s crashed mid-render but its master survived "
            "on disk (%s) — NOT writing a finals-only recovery marker; the "
            "next resume re-runs this day (reusing the cached master) so the "
            "full shorts set is produced instead of shipping a thin day.",
            date,
            master_on_disk,
        )
        return

    lf_dir = day_dir / "regular_lfs"
    shorts_dir = day_dir / "shorts"
    # Pair recovered LFs to plans by the render-fingerprint sidecar
    # (``lf_NN.ost.sha256``), NOT a bare glob: a current-code burn writes the
    # sidecar right after the encode, so a globbed ``lf_NN.mp4`` WITHOUT one is
    # a pre-fingerprint / stale render that must not be recorded as this day's
    # artifact (the next resume would ship it un-re-derived). Excluding it routes
    # that LF back through a re-render.
    regular_lf_paths = (
        sorted(
            p
            for p in lf_dir.glob("lf_[0-9][0-9].mp4")
            if p.is_file()
            and (fp := p.with_suffix(".ost.sha256")).is_file()
            and fp.stat().st_size > 0
        )
        if lf_dir.is_dir()
        else []
    )
    short_paths = (
        sorted(p for p in shorts_dir.glob("short_[0-9][0-9].mp4") if p.is_file())
        if shorts_dir.is_dir()
        else []
    )
    attempted = DayArtifacts(
        date=date,
        master_path=None,  # master never landed — recovery marker
        regular_lf_paths=regular_lf_paths,
        # Carry each final's drive-global LF index parsed from its
        # ``lf_NN`` stem (named from ``lf_plan.index`` above; the
        # ``lf_[0-9][0-9]`` glob guarantees the stem parses) — same as
        # ``_reconstruct_day_artifacts_from_disk``, so the next resume
        # pairs files to plans by index, never by list position.
        regular_lf_indices=[int(p.stem.split("_")[1]) for p in regular_lf_paths],
        short_paths=short_paths,
    )
    logger.warning(
        "run_all_days: day %s — firing recovery on_day_complete with "
        "master_path=None (finals/raw on disk: %d LFs, %d shorts). "
        "The next resume can pick up via finals-only reconstruction.",
        date,
        len(regular_lf_paths),
        len(short_paths),
    )
    try:
        on_day_complete(attempted, state)
    except Exception as exc:  # noqa: BLE001 — bookkeeping never blocks
        logger.warning(
            "run_all_days: recovery on_day_complete callback failed for "
            "day %s: %s — continuing with remaining days",
            date,
            exc,
        )
