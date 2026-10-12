"""Post-encoding per-LF invariants + computed metadata maps.

Extracted from ``drive_phases.phase_bc_day_first``. After the
day-pipeline finishes encoding LFs, Phase BC populates four per-LF
maps that downstream surfaces consume:

- ``per_lf_trim_ratio`` — title-verb / first-comment band selector.
- ``per_lf_distance_km`` — ``🛣️ Distance:`` description line.
- ``per_lf_elevation_gain_m`` — ``⛰️ Elevation gain:`` description line.
- ``per_lf_regions`` — ``🗺️ Through:`` description line.

Each map has a paired invariant that fires at the BC→D boundary so a
missing entry surfaces here rather than mid-Phase D with a silently
empty description line. This module owns both the compute helpers
and their invariants.

No behaviour change vs the inlined originals; the only difference is
that ``_derive_per_lf_trim_ratio_from_paths`` now reaches back into
``phase_bc_day_first._probe_duration`` via a lazy import (the helper
itself is a small wrapper around ``ffprobe`` that lives next to the
phase entry point).
"""

import logging
from pathlib import Path
from types import SimpleNamespace

from .exceptions import StateInvariantError
from .state_invariants import STATE_INVARIANT_ERROR_PREFIX

logger = logging.getLogger(__name__)


def _day_local_source_durations(
    per_lf_recording_date: dict, episode_dates: list, file_durations: list
) -> dict[int, list]:
    """Map ``{lf_index: [day-local source durations]}``.

    ``file_durations`` is DRIVE-WIDE — a flat list indexed by the GLOBAL
    episode index. But ``_partition_lf_plans_by_date`` remaps every LF
    segment's ``source_file_index`` from GLOBAL to DAY-LOCAL. Indexing the
    drive-wide list with a day-local index therefore reads a DIFFERENT day's
    episode on day 2+ of a multi-day drive (single-day drives are
    coincidentally correct because day-local == global). This returns, per
    LF, a durations slice indexed in the SAME day-local space as the LF's
    segments: the durations of that LF's calendar day's episodes, in
    day-local (chronological) order.

    Built from ``episode_dates`` (per-global-episode date — the same source
    the partition keys its day-local map on) bucketed in ascending global
    order, which is chronological, so the per-day order matches the
    partition's day-local index. ``per_lf_recording_date`` (already the
    authoritative LF→date map, derived from the partition output) selects
    each LF's day. Returns an empty map — caller falls back to the drive-wide
    list — when any input is missing (e.g. a single-day drive, or a legacy
    resume state without ``episode_dates``/``per_lf_recording_date``).
    """
    if not per_lf_recording_date or not episode_dates or not file_durations:
        return {}
    durations_by_date: dict[str, list] = {}
    for gi, date in enumerate(episode_dates):
        dur = float(file_durations[gi] or 0.0) if gi < len(file_durations) else 0.0
        durations_by_date.setdefault(date, []).append(dur)
    out: dict[int, list] = {}
    for lf_idx_raw, date in per_lf_recording_date.items():
        try:
            lf_idx = int(lf_idx_raw)
        except (TypeError, ValueError):
            continue
        day_durations = durations_by_date.get(date)
        if day_durations is not None:
            out[lf_idx] = day_durations
    return out


def _compute_per_lf_trim_ratio(
    skeleton_lfs: list, file_durations: list, per_lf_source_durations: dict | None = None
) -> dict:
    """Compute ``{lf.index: trim_ratio}`` from planner output.

    ``trim_ratio = lf.total_duration / sum(unique source-file durations
    that fed the LF)``. The unique source files are the distinct
    ``segment.source_file_index`` values referenced by the LF's
    segments. ``file_durations[i]`` is the full source clip duration.

    ``source_file_index`` is DAY-LOCAL after ``_partition_lf_plans_by_date``,
    while the drive-wide ``file_durations`` is GLOBAL-indexed. Pass
    ``per_lf_source_durations`` (``{lf_index: [day-local durations]}``) to
    index each LF's segments in their own day's space; an LF absent from that
    map falls back to the drive-wide ``file_durations`` (correct for
    single-day drives, where day-local == global).

    The trim ratio is the single source of truth for the trim
    band consumed by the metadata generator (title verb + description
    hook) and the shorts CSV first-comment lead. Values are clamped to
    1.0 (a planner that over-counts kept time vs source must not push
    the band into "Driving" via a >1.0 ratio).

    Returns an empty mapping when ``skeleton_lfs`` is empty. LFs whose
    source-file durations are unresolvable (e.g. ``source_file_index``
    out of range, or all referenced durations zero) are OMITTED from
    the result so the caller can fall back to the path-derivation
    backstop (see ``_derive_per_lf_trim_ratio_from_paths``); silently
    emitting 0.0 would mis-band every legacy resume into "Highlights".
    """
    if not skeleton_lfs:
        return {}
    out: dict[int, float] = {}
    for lf in skeleton_lfs:
        lf_idx = getattr(lf, "index", None)
        if lf_idx is None:
            continue
        durations = (per_lf_source_durations or {}).get(int(lf_idx), file_durations)
        final_duration = float(getattr(lf, "total_duration", 0.0) or 0.0)
        seen_indices: set[int] = set()
        source_total = 0.0
        for seg in getattr(lf, "segments", None) or []:
            sfi = int(getattr(seg, "source_file_index", -1))
            if sfi < 0 or sfi in seen_indices:
                continue
            seen_indices.add(sfi)
            if sfi < len(durations):
                source_total += float(durations[sfi] or 0.0)
        if source_total <= 0 or final_duration <= 0:
            continue
        ratio = final_duration / source_total
        if ratio > 1.0:
            ratio = 1.0
        out[int(lf_idx)] = ratio
    return out


def _derive_per_lf_trim_ratio_from_paths(
    skeleton_lfs: list,
    regular_lf_paths: list,
    file_durations: list,
    probe_duration_fn=None,
    per_lf_source_durations: dict | None = None,
) -> dict:
    """Resume-path backstop for ``per_lf_trim_ratio``.

    When ``state["per_lf_trim_ratio"]`` is empty (state file written
    under an older revision) but the LFs already exist on disk,
    derive each LF's final duration via ffprobe and combine with the
    same source-file durations used by the fresh-encode path. Returns
    a mapping keyed by LF index; LFs whose final-duration probe
    returns 0 or whose source duration can't be resolved are OMITTED
    (caller decides whether to error or proceed degraded — see the
    Phase BC invariant).

    ``per_lf_source_durations`` (``{lf_index: [day-local durations]}``)
    indexes each LF's DAY-LOCAL ``source_file_index`` in its own day's
    space; an LF absent from that map falls back to the drive-wide
    ``file_durations``. Same aliasing fix as the fresh-encode path.

    ``probe_duration_fn(path) -> float`` defaults to the module-level
    ``_probe_duration`` helper; tests inject a fake.
    """
    if not skeleton_lfs or not regular_lf_paths:
        return {}
    if probe_duration_fn is None:
        # Lazy import — _probe_duration lives in the phase entry-point
        # module to keep the ffprobe shim co-located with its other
        # callers. Importing at call time avoids a circular module
        # import (phase_bc_day_first imports from this module).
        from .drive_phases.phase_bc_day_first import _probe_duration

        probe_duration_fn = _probe_duration
    probe = probe_duration_fn
    out: dict[int, float] = {}
    for lf, lf_path in zip(skeleton_lfs, regular_lf_paths):
        lf_idx = getattr(lf, "index", None)
        if lf_idx is None:
            continue
        path_obj = Path(str(lf_path))
        if not path_obj.exists():
            continue
        try:
            final_duration = float(probe(path_obj) or 0.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "  per_lf_trim_ratio resume: ffprobe failed for LF %d at %s: %s",
                lf_idx,
                path_obj,
                exc,
            )
            continue
        if final_duration <= 0:
            continue
        durations = (per_lf_source_durations or {}).get(int(lf_idx), file_durations)
        seen_indices: set[int] = set()
        source_total = 0.0
        for seg in getattr(lf, "segments", None) or []:
            sfi = int(getattr(seg, "source_file_index", -1))
            if sfi < 0 or sfi in seen_indices:
                continue
            seen_indices.add(sfi)
            if sfi < len(durations):
                source_total += float(durations[sfi] or 0.0)
        if source_total <= 0:
            continue
        ratio = final_duration / source_total
        if ratio > 1.0:
            ratio = 1.0
        out[int(lf_idx)] = ratio
    return out


# ── One cleaned trace + physical-plausibility ceilings ──────────────────
# Every per-LF numeric fact (distance, elevation gain/profile, speed, bearing)
# is derived from the SAME quality-filtered, teleport-stripped trace via
# ``_cleaned_waypoints``, then bounded by a physical ceiling so a single bad GPS
# fix can't fabricate an inflated number (the cold-start class).

# Whole-LF average-speed ceiling for the DISTANCE gate (mirrors the per-STEP
# ``_SPEED_GLITCH_CEILING_KMH`` below): distance/duration above this is a
# teleport the cleaning missed — no consumer drive sustains >200 km/h average.
_DISTANCE_AVG_SPEED_CEILING_KMH = 200.0

# Elevation-gain ceiling as cumulative climb PER KILOMETRE. Even the steepest
# sustained roads stay well under 150 m/route-km averaged over a drive; above
# this it is summed GPS altitude noise, not terrain, so the gain is omitted.
_ELEVATION_GAIN_PER_KM_CEILING_M = 150.0


def _cleaned_waypoints(journey) -> list:
    """Return a journey's waypoints quality-filtered + teleport-stripped.

    The single canonical cleaning step every per-LF numeric fact routes through.
    Reuses ``gps_quality.filter_high_quality_points`` (null-island +
    spatial-outlier + shared km/SEC temporal-teleport strip — no new
    haversine/teleport logic). That filter reads lat/lon/time via tuple-aware
    accessors keyed on ``lat``/``lon``/``time`` — NOT the
    ``latitude``/``longitude``/``timestamp`` the ``Waypoint`` model uses — so a
    raw waypoint would no-op it; we feed positional ``(ts, lat, lon, idx)``
    tuples (lat@1, lon@2, ts@0; trailing ``idx`` ignored by every accessor) and
    map survivors back to the ORIGINAL objects by index, leaving the downstream
    summation unchanged. ``min_gap_sec=0`` disables the sub-second dedup (the
    compute fns never deduped; the teleport strip, not dedup, kills the bug).
    Returns the input unchanged for < 3 points or if cleaning drops everything.
    """
    waypoints = list(getattr(journey, "waypoints", None) or [])
    if len(waypoints) < 3:
        return waypoints
    from .gps_quality import filter_high_quality_points

    tuples: list = []
    for idx, wp in enumerate(waypoints):
        ts = getattr(wp, "timestamp", None)
        ts_val: float | None
        if ts is None:
            ts_val = None
        elif hasattr(ts, "timestamp"):
            try:
                ts_val = float(ts.timestamp())
            except (TypeError, ValueError):
                ts_val = None  # not a real datetime (e.g. a bare MagicMock)
        else:
            try:
                ts_val = float(ts)
            except (TypeError, ValueError):
                ts_val = None
        tuples.append((ts_val, getattr(wp, "latitude", None), getattr(wp, "longitude", None), idx))
    survivors = filter_high_quality_points(tuples, min_gap_sec=0.0)
    kept_idx = [t[3] for t in survivors if isinstance(t, (tuple, list)) and len(t) >= 4]
    if not kept_idx:
        return waypoints
    return [waypoints[i] for i in kept_idx]


def _trace_duration_hours(waypoints: list) -> float | None:
    """First→last ``.timestamp`` span in hours (datetime or epoch float), or None
    if unknown — bounds the distance gate's implied average speed."""
    if len(waypoints) < 2:
        return None
    t0 = getattr(waypoints[0], "timestamp", None)
    t1 = getattr(waypoints[-1], "timestamp", None)
    if t0 is None or t1 is None:
        return None
    try:
        if hasattr(t0, "timestamp") and hasattr(t1, "timestamp"):
            sec = t1.timestamp() - t0.timestamp()
        else:
            sec = float(t1) - float(t0)
    except (TypeError, ValueError):
        return None
    return sec / 3600.0 if sec > 0 else None


def _window_journey_to_included_segments(journey, included_segments):
    """Restrict ``journey`` to the LF's INCLUDED source-timeline segments.

    Every windowed per-LF description fact (distance / elevation / speed /
    bearing) parses the representative episode's whole GPX, keyed only
    on ``per_lf_gpx_path``. A best-moments LF (trim ~0.3) keeps ~30% of that
    episode, so without windowing it advertised the FULL episode's distance
    (an inflated-but-plausible number the speed ceiling can't catch) and listed
    counties whose footage was cut. Trimming the parsed trace to the segments
    the LF is actually assembled from makes that impossible by construction.

    Reuses ``gps_quality.trim_gps_to_clip_timeline`` with the same
    datetime→source-second projection the chapter generator uses, anchored at
    the first *valid* fix via the shared ``effective_offset`` helper —
    a leading GoPro cold-start anomaly (null-island ``(0,0)`` or stale-lock
    run) would otherwise pull every window onto the garbage prefix and
    exclude the real footage. ``included_segments`` None/empty is
    identity — an LF whose segment window is unknown keeps the whole trace
    (prior behaviour); a trace with < 2 points or a non-datetime origin is also
    returned unchanged. Returns a journey-like object exposing only the windowed
    ``.waypoints`` (every compute reads the trace through that attribute).
    """
    if not included_segments:
        return journey
    waypoints = list(getattr(journey, "waypoints", None) or [])
    if len(waypoints) < 2:
        return journey
    t0 = getattr(waypoints[0], "timestamp", None)
    if t0 is None or not hasattr(t0, "timestamp"):
        return journey
    from ..analysis.shorts_models import effective_offset
    from .gps_quality import trim_gps_to_clip_timeline

    try:
        # Project to relative seconds, then let the shared helper (dict-
        # trackpoint form) return the anomalous-lead seconds to skip — 0.0 for
        # a clean trace, i.e. prior behaviour exactly.
        rel = [
            {
                "lat": float(getattr(wp, "latitude", 0.0) or 0.0),
                "lon": float(getattr(wp, "longitude", 0.0) or 0.0),
                "timestamp": (wp.timestamp - t0).total_seconds(),
            }
            for wp in waypoints
        ]
        skip = effective_offset({"trackpoints": rel}, 0.0)
        trimmed = trim_gps_to_clip_timeline(
            waypoints,
            included_segments,
            time_key=lambda wp: (wp.timestamp - t0).total_seconds() - skip,
        )
    except (TypeError, ValueError, AttributeError):
        return journey
    return SimpleNamespace(waypoints=trimmed)


def _segment_isolated_traces(journey, ranges):
    """Window ``journey`` to EACH range in ``ranges`` separately.

    ``ranges`` (an LF's included source-timeline segments) is typically
    several NON-CONTIGUOUS windows — a best-moments LF keeps several
    separate clips out of one long episode. Windowing the whole list at once
    (``_window_journey_to_included_segments(journey, ranges)``) concatenates
    every kept clip into ONE flat trace; summing consecutive-point
    distance/speed over that trace folds the straight-line gap BETWEEN two
    kept clips — footage that was CUT — into the reported number, so the
    reported distance can far exceed what the kept footage actually covers.

    Returns one cleaned waypoint list per range (``ranges`` falsy/None →
    a single-item list windowing the whole trace, i.e. prior behaviour),
    so a caller summing per-trace never bridges a gap between two ranges.
    """
    if not ranges:
        waypoints = _cleaned_waypoints(_window_journey_to_included_segments(journey, None))
        # An empty trace is "no data", never a zero-length trace: callers omit
        # the LF (invariant fires) rather than recording 0.0 for it.
        return [waypoints] if waypoints else []
    traces = []
    for rng in ranges:
        windowed = _window_journey_to_included_segments(journey, [rng])
        waypoints = _cleaned_waypoints(windowed)
        if waypoints:
            traces.append(waypoints)
    return traces


def _cleaned_traces_for_pairs(pairs, parse_fn) -> tuple[list, bool]:
    """Parse + window + clean each ``(gpx_path, ranges)`` pair an LF spans;
    return ``(cleaned_waypoint_lists, parse_failed)``. Summing a per-LF metric
    across the traces fixes the multi-episode understatement: a
    single-episode LF yields one trace (prior behaviour exactly), a
    multi-episode LF one per episode. ``ranges`` is further split per segment
    (``_segment_isolated_traces``) so two non-contiguous kept clips
    never get bridged into one summed distance/speed. Empty cleaned traces
    are dropped; the FIRST parse failure returns ``parse_failed=True`` so the
    caller leaves the LF un-tolerated (invariant fires, as for a corrupt
    single GPX).
    """
    traces: list = []
    for gpx_path, ranges in pairs:
        if not gpx_path:
            continue
        try:
            journey = parse_fn(gpx_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("  per_lf GPX parse failed at %s: %s", gpx_path, exc)
            return traces, True
        traces.extend(_segment_isolated_traces(journey, ranges))
    return traces, False


def _lf_episode_pairs(lf_idx, gpx_path, per_lf_included_segments, per_lf_gpx_episode_paths):
    """The ``(gpx_path, ranges)`` pairs to sum a per-LF metric over.

    Multi-episode map present for this LF → one pair per episode it spans.
    Absent (legacy / resume / GPS-less) → a single pair on the
    representative GPX windowed by its included segments, i.e. the prior path.
    """
    episode_map = (per_lf_gpx_episode_paths or {}).get(lf_idx)
    if episode_map:
        return list(episode_map.items())
    return [(gpx_path, (per_lf_included_segments or {}).get(lf_idx))]


def _compute_per_lf_distance(
    per_lf_gpx_path: dict,
    parse_fn=None,
    plausibility_rejected: set | None = None,
    per_lf_included_segments: dict | None = None,
    per_lf_gpx_episode_paths: dict | None = None,
) -> dict[int, float]:
    """Compute ``{lf.index: distance_km}`` from each LF's GPX trajectory.

    Distance is SUMMED across every source episode an LF spans
    (``per_lf_gpx_episode_paths``) — a multi-episode LF otherwise
    shipped only its representative episode's distance. Episode map absent
    (legacy/resume/GPS-less) degenerates to the prior single-rep behaviour.

    LFs whose GPX fails to parse, or whose cleaned trajectory sums to zero
    distance, are OMITTED from the result — the description line is then simply
    omitted rather than surfacing a misleading "0 km". The Phase BC invariant
    (``_assert_per_lf_distance_complete``) fires when this gap is unexpected.

    ``parse_fn(path) -> Journey`` defaults to ``GPXParser().parse``; tests
    inject a fake. State-resume tolerance: keys may arrive as strings
    after a ``pipeline_state.json`` round-trip — they are coerced to int.
    """
    if not per_lf_gpx_path:
        return {}
    if parse_fn is None:
        from gps_journey.core.gpx_parser import GPXParser

        _parser = GPXParser()
        parse_fn = _parser.parse

    from gps_journey.analysis.route_analyzer import calculate_distance_km

    out: dict[int, float] = {}
    for lf_idx_raw, gpx_path in per_lf_gpx_path.items():
        try:
            lf_idx = int(lf_idx_raw)
        except (TypeError, ValueError):
            continue
        pairs = _lf_episode_pairs(
            lf_idx, gpx_path, per_lf_included_segments, per_lf_gpx_episode_paths
        )
        if not any(pair[0] for pair in pairs):
            # GPS-less LF (no usable GPX in any spanned episode) — skip
            # silently; it is not an expected key for the completeness invariant.
            continue
        traces, parse_failed = _cleaned_traces_for_pairs(pairs, parse_fn)
        if parse_failed:
            # A genuine parse failure stays un-tolerated: the completeness
            # invariant should fire exactly as it did for a corrupt single GPX.
            logger.warning("  per_lf_distance: GPX parse failed for LF %d", lf_idx)
            continue
        distance_km = sum(calculate_distance_km(wp) for wp in traces)
        if not traces or distance_km <= 0:
            # Cleaned trace has no usable distance — every surviving point
            # collapsed to one coordinate (a mostly-stationary / parked-with-
            # jitter LF whose spatial-outlier median step is ~0, so the real
            # movement is filtered out), or cleaning emptied the trace. This is
            # a LEGITIMATE omission of the optional ``🛣️ Distance:`` line (we do
            # not fabricate a "0 km" or wrong number), NOT a parse failure — so
            # it is TOLERATED by the completeness invariant exactly like the
            # plausibility rejection below. Without this an entire multi-hour
            # run aborts on one stationary LF.
            logger.warning(
                "  PER-LF-DISTANCE-OMIT: LF %d has no usable cleaned distance "
                "(stationary/over-filtered trace); line omitted, tolerated",
                lf_idx,
            )
            if plausibility_rejected is not None:
                plausibility_rejected.add(lf_idx)
            continue
        # Plausibility gate: omit a distance whose implied sustained average
        # speed is physically impossible — a teleport the cleaning missed. Prefer
        # omitting the optional ``🛣️ Distance:`` line over publishing an impossible figure.
        # Duration summed over the same traces; any None → unknown → skip gate.
        _durations = [_trace_duration_hours(wp) for wp in traces]
        dur_h = None if any(d is None for d in _durations) else sum(_durations)
        if dur_h and (distance_km / dur_h) > _DISTANCE_AVG_SPEED_CEILING_KMH:
            logger.warning(
                "  PLAUSIBILITY-GATE: per_lf_distance LF %d rejected — "
                "%.1f km over %.2f h implies %.0f km/h avg (> %.0f ceiling); "
                "line omitted",
                lf_idx,
                distance_km,
                dur_h,
                distance_km / dur_h,
                _DISTANCE_AVG_SPEED_CEILING_KMH,
            )
            if plausibility_rejected is not None:
                plausibility_rejected.add(lf_idx)
            continue
        out[lf_idx] = float(distance_km)
    return out


def _assert_mapping_complete(
    expected_keys: set[int],
    actual: dict,
    *,
    field_name: str,
    expected_source_desc: str = "",
    example_fix: str = "",
) -> None:
    """Phase BC invariant primitive — every expected LF index is in ``actual``.

    Coerces ``actual``'s keys to int before comparison so a state file
    that round-tripped through JSON (where ints became strings) doesn't
    spuriously fail. Raises ``RuntimeError`` listing the sorted missing
    indices when the contract fails; no-ops when ``expected_keys`` is
    empty. Used by the per-field wrappers (trim_ratio, distance,
    elevation, regions) below — each adds field-specific framing.
    """
    if not expected_keys:
        return
    have: set[int] = set()
    for k in actual or {}:
        try:
            have.add(int(k))
        except (TypeError, ValueError):
            continue
    missing = sorted(expected_keys - have)
    if not missing:
        return
    source_clause = f" from {expected_source_desc}" if expected_source_desc else ""
    fix_clause = f" {example_fix}" if example_fix else ""
    raise RuntimeError(
        f"phase_bc_day_first: {field_name} map is missing entries "
        f"for LF indices {missing} (expected {len(expected_keys)}{source_clause}, "
        f"got {len(have)}).{fix_clause}"
    )


def _gpx_path_expected_keys(per_lf_gpx_path: dict) -> set[int]:
    """Coerce ``per_lf_gpx_path`` keys to int for invariant comparison.

    LFs whose value is ``None`` (GPS-less LFs under the robust-by-
    construction contract — the map carries an explicit entry per LF with
    ``None`` for GPS-less ones) are excluded — they have no GPX path to
    parse, so the distance/elevation/regions invariants must not demand
    entries for them.
    """
    expected: set[int] = set()
    for lf_idx_raw, gpx_path in (per_lf_gpx_path or {}).items():
        if not gpx_path:
            continue
        try:
            expected.add(int(lf_idx_raw))
        except (TypeError, ValueError):
            continue
    return expected


def _assert_post_encoding_invariants(
    *,
    state,
    skeleton_lfs,
    regular_lf_paths,
) -> None:
    """Compute, validate, and persist the four post-encoding per-LF maps.

    Each map drives a downstream Phase D surface and has a Phase BC
    invariant — fail at the boundary, not silently mid-Phase D where the
    symptom is a missing description line (or, for trim_ratio, the wrong
    title verb) across the published batch.

    Maps populated on ``state``:

    - ``per_lf_trim_ratio`` — title verb +
      first-comment lead band selector.
    - ``per_lf_distance_km`` — ``🛣️ Distance:`` description line.
    - ``per_lf_elevation_gain_m`` — ``⛰️ Elevation gain:`` line.
    - ``per_lf_regions`` — ``🗺️ Through:`` line.

    GPS-less drives (empty ``state["per_lf_gpx_path"]``) skip the three
    GPX-keyed invariants by design — legacy runs and any pre-GPX run
    produce no map, no description line, no failure. The trim_ratio
    invariant fires only when LFs carry segments and ``file_durations``
    is populated; synthetic / test-fixture paths degrade to legacy
    framing rather than aborting.
    """
    # ── trim_ratio ──────────────────────────────────────────────────
    # Fresh-encode path computes from skeleton_lfs + state["file_durations"];
    # resume path falls back to ffprobing finals on disk + the same
    # source-file durations.
    _file_durations = state.file_durations or []
    # source_file_index is DAY-LOCAL post-partition, but _file_durations
    # is GLOBAL-indexed. Build each LF's day-local durations slice so a
    # multi-day drive's day-2+ LFs divide by their OWN day's source clips.
    _per_lf_source_durations = _day_local_source_durations(
        state.per_lf_recording_date or {}, state.episode_dates or [], _file_durations
    )
    per_lf_trim_ratio = _compute_per_lf_trim_ratio(
        skeleton_lfs, _file_durations, per_lf_source_durations=_per_lf_source_durations
    )
    if len(per_lf_trim_ratio) < len(skeleton_lfs):
        path_trim = _derive_per_lf_trim_ratio_from_paths(
            skeleton_lfs=skeleton_lfs,
            regular_lf_paths=regular_lf_paths,
            file_durations=_file_durations,
            per_lf_source_durations=_per_lf_source_durations,
        )
        # Record which indices the fresh-compute path missed, so the resume-time
        # ffprobe backstop is observable. Record the ones the path-derivation
        # recovered (or attempted to). Only
        # record indices that actually filled a gap, not full coverage.
        _recovered_trim: list[int] = []
        for lf_idx, ratio in path_trim.items():
            if lf_idx not in per_lf_trim_ratio:
                _recovered_trim.append(int(lf_idx))
            per_lf_trim_ratio.setdefault(lf_idx, ratio)
        if _recovered_trim:
            from .state_invariants import record_backstop_recovery

            record_backstop_recovery(
                state,
                "per_lf_trim_ratio",
                _recovered_trim,
                "fresh-compute path returned incomplete map; ffprobed LF "
                "finals to fill the gap (resume into an older state, or "
                "file_durations missing for the LFs' source_file_indices)",
            )
    _has_segments = any(getattr(lf, "segments", None) for lf in skeleton_lfs or [])
    if _has_segments and _file_durations:
        _assert_per_lf_trim_ratio_complete(skeleton_lfs, per_lf_trim_ratio)
    elif _has_segments and skeleton_lfs:
        # Real LFs have segments but no file_durations → Phase A
        # ffprobe step corrupted/partial. Refuse to ship the wrong title
        # verb silently. Production always populates file_durations.
        raise StateInvariantError(
            f"{STATE_INVARIANT_ERROR_PREFIX} per_lf_trim_ratio: "
            f"skeleton LFs carry segments but state['file_durations'] "
            f"is empty — Phase A ffprobe step did not populate source "
            f"durations. Re-run from phase_a to recover, or wipe state "
            f"for a clean start."
        )
    elif skeleton_lfs:
        logger.warning(
            "  per_lf_trim_ratio: skipping coverage invariant — "
            "skeleton LFs have no segments (%d) and state['file_durations'] "
            "is empty. Per-LF trim band defaults to 'Driving' "
            "for any LF without an entry.",
            sum(1 for lf in skeleton_lfs if not getattr(lf, "segments", None)),
        )
    state.per_lf_trim_ratio = dict(per_lf_trim_ratio)

    # ── distance / elevation / regions — all keyed on per_lf_gpx_path ──
    # Distance + elevation_gain route through one cleaned/teleport-stripped
    # trace then a physical-plausibility gate; an LF the gate omits is a
    # legitimate omission, so its index is passed to the completeness invariant
    # as tolerated (else the invariant would false-fire like a parse failure).
    _per_lf_gpx_path = state.per_lf_gpx_path or {}
    # The per-episode span map lets distance/elevation SUM across every
    # episode an LF spans (not just its representative). Absent → the computes
    # fall back to the single-representative windowed path (prior behaviour).
    _per_lf_gpx_episode_paths = state.per_lf_gpx_episode_paths or {}

    # Window every per-LF description stat to the segments the LF is
    # actually assembled from — a best-moments LF (trim ~0.3) must not advertise
    # the whole representative episode's distance/counties. Reuses phase_d's
    # included-segments builder (episode-local segment ranges for the LF's
    # representative GPX — the same GPX per_lf_gpx_path resolves to, so the
    # ranges share this trace's time base); lazy-imported to avoid a module-load
    # cycle. An LF with no resolvable segment window yields ``[]`` → the whole
    # trace (prior behaviour), so GPS-less / synthetic LFs degrade gracefully.
    from .drive_phases.phase_d import _build_included_segments_for_lf

    _per_lf_included_segments = {
        int(getattr(lf, "index")): _build_included_segments_for_lf(lf)
        for lf in (skeleton_lfs or [])
        if getattr(lf, "index", None) is not None
    }

    _dist_rejected: set[int] = set()
    per_lf_distance_km = _compute_per_lf_distance(
        _per_lf_gpx_path,
        plausibility_rejected=_dist_rejected,
        per_lf_included_segments=_per_lf_included_segments,
        per_lf_gpx_episode_paths=_per_lf_gpx_episode_paths,
    )
    if _per_lf_gpx_path:
        _assert_per_lf_distance_complete(
            _per_lf_gpx_path, per_lf_distance_km, tolerated_indices=_dist_rejected
        )
    state.per_lf_distance_km = dict(per_lf_distance_km)

    _elev_rejected: set[int] = set()
    per_lf_elevation_gain_m = _compute_per_lf_elevation_gain(
        _per_lf_gpx_path,
        plausibility_rejected=_elev_rejected,
        per_lf_included_segments=_per_lf_included_segments,
        per_lf_gpx_episode_paths=_per_lf_gpx_episode_paths,
    )
    if _per_lf_gpx_path:
        _assert_per_lf_elevation_gain_complete(
            _per_lf_gpx_path, per_lf_elevation_gain_m, tolerated_indices=_elev_rejected
        )
    state.per_lf_elevation_gain_m = dict(per_lf_elevation_gain_m)

    # Persist the LEGITIMATELY-omitted GPX-keyed indices so the BC-exit AND
    # D-entry boundary contracts (state_invariants) exclude them from the
    # GPX-keyed coverage check — an LF the plausibility/zero-distance gate
    # dropped is not "missing data". Without this, the aggregate contract
    # re-fails on the same LF the per-field invariant above already tolerated
    # (a stationary LF yields an empty distance).
    state.per_lf_gpx_omitted = {
        "per_lf_distance_km": sorted(_dist_rejected),
        "per_lf_elevation_gain_m": sorted(_elev_rejected),
    }

    # Label-substrate hardening: elevation PROFILE (range + net
    # climb/descent), the shape complement to the scalar gain above. No
    # invariant: a GPX may legitimately carry no elevation data, so an LF
    # absent from the profile map is tolerated (the description line is just
    # omitted) — unlike gain, where the GPX is expected to yield a number.
    per_lf_elevation_profile = _compute_per_lf_elevation_profile(
        _per_lf_gpx_path, per_lf_included_segments=_per_lf_included_segments
    )
    state.per_lf_elevation_profile = dict(per_lf_elevation_profile)

    # Label-substrate hardening: per-LF speed/character (avg/max km/h +
    # std), surfacing the ``🚗 Pace:`` description line. No invariant — a GPX
    # without timestamps / too sparse yields no entry (line omitted), same
    # tolerance as the elevation profile above.
    per_lf_speed = _compute_per_lf_speed(
        _per_lf_gpx_path, per_lf_included_segments=_per_lf_included_segments
    )
    state.per_lf_speed = dict(per_lf_speed)

    # Label-substrate hardening: net travel BEARING (8-point compass,
    # first→last waypoint), surfacing the ``🧭 Heading:`` description line. No
    # invariant — a loop drive (tiny net displacement) or GPS-less LF yields no
    # entry (line omitted).
    per_lf_bearing = _compute_per_lf_bearing(
        _per_lf_gpx_path, per_lf_included_segments=_per_lf_included_segments
    )
    state.per_lf_bearing = dict(per_lf_bearing)

    per_lf_regions = _compute_per_lf_regions(
        _per_lf_gpx_path,
        per_lf_resolved=state.per_lf_resolved_location or {},
    )
    if _per_lf_gpx_path and per_lf_regions:
        _assert_per_lf_regions_complete(_per_lf_gpx_path, per_lf_regions)
    state.per_lf_regions = dict(per_lf_regions)


def _assert_per_lf_distance_complete(
    per_lf_gpx_path: dict, mapping: dict, tolerated_indices: set | None = None
) -> None:
    """Phase BC invariant: every LF with a GPX path has a distance entry.

    GPS-less drives (empty ``per_lf_gpx_path``) are tolerated by design.
    ``tolerated_indices`` names LFs whose distance was OMITTED by the
    physical-plausibility gate, not a parse failure — a legitimate omission
    (the line is simply dropped), so those indices are excluded from the
    "expected" set and never false-fire the completeness invariant.
    """
    expected = _gpx_path_expected_keys(per_lf_gpx_path)
    if tolerated_indices:
        expected = expected - {int(i) for i in tolerated_indices}
    _assert_mapping_complete(
        expected,
        mapping or {},
        field_name="per_lf_distance_km",
        expected_source_desc="per_lf_gpx_path",
        example_fix=(
            "The compute path parses each LF's GPX and sums Haversine "
            "waypoints via gps_journey.calculate_distance_km. If an LF's "
            "GPX failed to parse or returned no waypoints, the operator "
            "should re-run phase_a to recover the trajectory or accept the "
            "missing line by clearing the LF's per_lf_gpx_path entry."
        ),
    )


def _compute_per_lf_elevation_gain(
    per_lf_gpx_path: dict,
    parse_fn=None,
    plausibility_rejected: set | None = None,
    per_lf_included_segments: dict | None = None,
    per_lf_gpx_episode_paths: dict | None = None,
) -> dict[int, float]:
    """Compute ``{lf.index: elevation_gain_m}`` from each LF's GPX trajectory.

    Gain is SUMMED across every source episode an LF spans
    (``per_lf_gpx_episode_paths``), mirroring per-LF distance. Positive
    elevation deltas come from ``route_analyzer.calculate_elevation_gain_m``.

    Unlike per-LF distance (where a zero-distance LF is omitted because
    surfacing ``"0 km"`` would mislead), elevation gain of ``0.0`` is an
    honest measurement of a flat or pure-descent drive and is recorded.
    The 50 m suppression heuristic — flat/coastal drives don't need an
    elevation line — lives at the description formatter, not here. LFs
    whose GPX file fails to parse or returns no waypoints are still
    omitted from the result; the Phase BC invariant fires on that gap.

    ``parse_fn(path) -> Journey`` defaults to ``GPXParser().parse``; tests
    inject a fake. State-resume tolerance: keys may arrive as strings
    after a ``pipeline_state.json`` round-trip — they are coerced to int.
    """
    if not per_lf_gpx_path:
        return {}
    if parse_fn is None:
        from gps_journey.core.gpx_parser import GPXParser

        _parser = GPXParser()
        parse_fn = _parser.parse

    from gps_journey.analysis.route_analyzer import (
        calculate_distance_km,
        calculate_elevation_gain_m,
    )

    out: dict[int, float] = {}
    for lf_idx_raw, gpx_path in per_lf_gpx_path.items():
        try:
            lf_idx = int(lf_idx_raw)
        except (TypeError, ValueError):
            continue
        pairs = _lf_episode_pairs(
            lf_idx, gpx_path, per_lf_included_segments, per_lf_gpx_episode_paths
        )
        if not any(pair[0] for pair in pairs):
            # GPS-less LF (no usable GPX in any spanned episode) — skip silently.
            continue
        traces, parse_failed = _cleaned_traces_for_pairs(pairs, parse_fn)
        if parse_failed:
            # A genuine parse failure stays un-tolerated (invariant fires).
            logger.warning("  per_lf_elevation_gain: GPX parse failed for LF %d", lf_idx)
            continue
        if not traces:
            # Cleaning emptied every trace (all teleport/null-island) — a
            # legitimate omission, NOT a parse failure; tolerate it so the
            # completeness invariant does not abort the run (mirrors the
            # per_lf_distance omission).
            logger.warning(
                "  PER-LF-ELEVATION-OMIT: LF %d cleaned trace empty; gain omitted, tolerated",
                lf_idx,
            )
            if plausibility_rejected is not None:
                plausibility_rejected.add(lf_idx)
            continue
        # Gain (and the distance used for the plausibility ratio) are SUMMED
        # across every episode the LF spans.
        gain = sum(calculate_elevation_gain_m(wp) for wp in traces)
        # Plausibility gate: omit a gain whose climb-per-km exceeds what
        # real terrain sustains — that is summed GPS altitude noise, not a hill.
        dist_km = sum(calculate_distance_km(wp) for wp in traces)
        if dist_km > 0 and (gain / dist_km) > _ELEVATION_GAIN_PER_KM_CEILING_M:
            logger.warning(
                "  PLAUSIBILITY-GATE: per_lf_elevation_gain LF %d rejected — "
                "%.0f m over %.1f km is %.0f m/km (> %.0f ceiling); line omitted",
                lf_idx,
                gain,
                dist_km,
                gain / dist_km,
                _ELEVATION_GAIN_PER_KM_CEILING_M,
            )
            if plausibility_rejected is not None:
                plausibility_rejected.add(lf_idx)
            continue
        out[lf_idx] = float(gain)
    return out


def _compute_per_lf_elevation_profile(
    per_lf_gpx_path: dict, parse_fn=None, per_lf_included_segments: dict | None = None
) -> dict[int, dict]:
    """Compute ``{lf.index: {min_m, max_m, net_m, descent_m}}`` per LF.

    The elevation *shape* complement to the scalar
    ``_compute_per_lf_elevation_gain`` (cumulative positive climb only).
    Reads ``waypoint.elevation`` directly: ``min_m``/``max_m`` = altitude range,
    ``net_m`` = signed end−start change, ``descent_m`` = total negative-delta
    magnitude. No new gps-journey helper (no min/max primitive exists there).
    LFs whose GPX fails to parse / has no elevation-bearing waypoints are
    omitted — NO completeness invariant (a GPX may legitimately lack elevation;
    the ``📈 Elevation`` line is then just omitted). ``parse_fn`` defaults to
    ``GPXParser().parse``; str keys (resume) coerced to int.
    """
    if not per_lf_gpx_path:
        return {}
    if parse_fn is None:
        from gps_journey.core.gpx_parser import GPXParser

        parse_fn = GPXParser().parse

    out: dict[int, dict] = {}
    for lf_idx_raw, gpx_path in per_lf_gpx_path.items():
        try:
            lf_idx = int(lf_idx_raw)
        except (TypeError, ValueError):
            continue
        if not gpx_path:
            continue
        try:
            journey = parse_fn(gpx_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "  per_lf_elevation_profile: GPX parse failed for LF %d at %s: %s",
                lf_idx,
                gpx_path,
                exc,
            )
            continue
        journey = _window_journey_to_included_segments(
            journey, (per_lf_included_segments or {}).get(lf_idx)
        )
        waypoints = _cleaned_waypoints(journey)
        elevations = [
            float(wp.elevation) for wp in waypoints if getattr(wp, "elevation", None) is not None
        ]
        if not elevations:
            continue
        descent = 0.0
        for e0, e1 in zip(elevations, elevations[1:]):
            delta = e1 - e0
            if delta < 0:
                descent += -delta
        out[lf_idx] = {
            "min_m": min(elevations),
            "max_m": max(elevations),
            "net_m": elevations[-1] - elevations[0],
            "descent_m": descent,
        }
    return out


# Per-step speeds above this are GPS glitches (cold-start teleports / stale
# locks), excluded from the avg/max/std so one bad fix can't fabricate a
# "180 km/h" pace. Matches the spirit of the teleport ceiling.
_SPEED_GLITCH_CEILING_KMH = 200.0


def _compute_per_lf_speed(
    per_lf_gpx_path: dict, parse_fn=None, per_lf_included_segments: dict | None = None
) -> dict[int, dict]:
    """Compute ``{lf.index: {avg_kmh, max_kmh, std_kmh}}`` per LF.

    Per-step speeds are derived from consecutive waypoint pairs WITHIN one
    included segment — ``_segment_isolated_traces`` prevents a step
    from bridging two non-contiguous kept clips, which would fold the CUT
    footage in between into the reported pace. Distance via the public
    ``calculate_distance_km`` (reused, not re-implemented) over each pair,
    divided by the timestamp delta. ``avg_kmh`` is total-distance /
    total-time over non-glitch steps (time-weighted, not a mean-of-means).
    Steps with non-positive ``dt`` or an implied speed above
    ``_SPEED_GLITCH_CEILING_KMH`` are excluded. LFs with < 2 timestamped
    waypoints / no valid step are omitted — NO completeness invariant (the
    ``🚗 Pace:`` line is then just omitted). ``parse_fn`` defaults to
    ``GPXParser().parse``; str keys (resume) coerce to int.
    """
    if not per_lf_gpx_path:
        return {}
    if parse_fn is None:
        from gps_journey.core.gpx_parser import GPXParser

        parse_fn = GPXParser().parse

    from gps_journey.analysis.route_analyzer import calculate_distance_km

    out: dict[int, dict] = {}
    for lf_idx_raw, gpx_path in per_lf_gpx_path.items():
        try:
            lf_idx = int(lf_idx_raw)
        except (TypeError, ValueError):
            continue
        if not gpx_path:
            continue
        try:
            journey = parse_fn(gpx_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "  per_lf_speed: GPX parse failed for LF %d at %s: %s",
                lf_idx,
                gpx_path,
                exc,
            )
            continue
        traces = _segment_isolated_traces(journey, (per_lf_included_segments or {}).get(lf_idx))
        speeds: list[float] = []
        total_km = 0.0
        total_h = 0.0
        for waypoints in traces:
            for a, b in zip(waypoints, waypoints[1:]):
                ta, tb = getattr(a, "timestamp", None), getattr(b, "timestamp", None)
                if ta is None or tb is None:
                    continue
                dt_s = (tb - ta).total_seconds()
                if dt_s <= 0:
                    continue
                dist_km = calculate_distance_km([a, b])
                kmh = dist_km / (dt_s / 3600.0)
                if kmh > _SPEED_GLITCH_CEILING_KMH:
                    continue
                speeds.append(kmh)
                total_km += dist_km
                total_h += dt_s / 3600.0
        if not speeds or total_h <= 0:
            continue
        import numpy as np

        out[lf_idx] = {
            "avg_kmh": total_km / total_h,
            "max_kmh": max(speeds),
            "std_kmh": float(np.std(speeds)),
        }
    return out


# Net straight-line displacement (first→last waypoint) below this is a loop /
# round-trip drive with no meaningful single heading — the bearing line is
# suppressed (omitted) rather than reporting a near-arbitrary direction.
_BEARING_MIN_NET_KM = 2.0


def _compute_per_lf_bearing(
    per_lf_gpx_path: dict, parse_fn=None, per_lf_included_segments: dict | None = None
) -> dict[int, str]:
    """Compute ``{lf.index: cardinal}`` net travel bearing per LF.

    WIRES the existing ``_drive_helpers._compass_from_waypoints`` (8-point
    compass, no new haversine-bearing code) on each LF's first→last waypoint.
    LFs whose net straight-line displacement is below ``_BEARING_MIN_NET_KM``
    (loop / round-trip — no meaningful heading), have < 2 waypoints, or fail to
    parse are omitted — NO completeness invariant (the ``🧭 Heading:`` line is
    then just omitted). ``parse_fn`` defaults to ``GPXParser().parse``; str keys
    (resume) coerce to int.
    """
    if not per_lf_gpx_path:
        return {}
    if parse_fn is None:
        from gps_journey.core.gpx_parser import GPXParser

        parse_fn = GPXParser().parse

    from gps_journey.analysis.route_analyzer import calculate_distance_km

    from ._drive_helpers import _compass_from_waypoints

    out: dict[int, str] = {}
    for lf_idx_raw, gpx_path in per_lf_gpx_path.items():
        try:
            lf_idx = int(lf_idx_raw)
        except (TypeError, ValueError):
            continue
        if not gpx_path:
            continue
        try:
            journey = parse_fn(gpx_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "  per_lf_bearing: GPX parse failed for LF %d at %s: %s",
                lf_idx,
                gpx_path,
                exc,
            )
            continue
        journey = _window_journey_to_included_segments(
            journey, (per_lf_included_segments or {}).get(lf_idx)
        )
        waypoints = _cleaned_waypoints(journey)
        if len(waypoints) < 2:
            continue
        first, last = waypoints[0], waypoints[-1]
        if calculate_distance_km([first, last]) < _BEARING_MIN_NET_KM:
            continue
        out[lf_idx] = _compass_from_waypoints(
            first.latitude, first.longitude, last.latitude, last.longitude
        )
    return out


def _assert_per_lf_elevation_gain_complete(
    per_lf_gpx_path: dict, mapping: dict, tolerated_indices: set | None = None
) -> None:
    """Phase BC invariant: every LF with a GPX path has an elevation entry.

    GPS-less drives (empty ``per_lf_gpx_path``) are tolerated by design.
    ``tolerated_indices`` excludes LFs whose gain the plausibility gate
    omitted (noise-inflated climb), a legitimate omission — see
    ``_assert_per_lf_distance_complete``.
    """
    expected = _gpx_path_expected_keys(per_lf_gpx_path)
    if tolerated_indices:
        expected = expected - {int(i) for i in tolerated_indices}
    _assert_mapping_complete(
        expected,
        mapping or {},
        field_name="per_lf_elevation_gain_m",
        expected_source_desc="per_lf_gpx_path",
        example_fix=(
            "The compute path parses each LF's GPX and sums positive "
            "elevation deltas via gps_journey.calculate_elevation_gain_m. "
            "If an LF's GPX failed to parse or returned no waypoints, the "
            "operator should re-run phase_a to recover the trajectory or "
            "accept the missing line by clearing the LF's per_lf_gpx_path "
            "entry."
        ),
    )


def _compute_per_lf_regions(
    per_lf_gpx_path: dict,
    per_lf_resolved: dict | None = None,
) -> dict[int, list[dict]]:
    """Build ``{lf.index: [{"name": <region>}, ...]}`` from each LF's canonical
    ``ResolvedLocation.regions``.

    ``resolved.regions`` is the SINGLE source of truth the title/OST and the
    location-claim gates already consume. This replaces an independent FOURTH
    geocode pass over the raw GPX (``RouteAnalyzer.get_regions_passed_through``)
    that could name a county no other surface named and that no gate
    cross-checked — sourcing the ``🗺️ Through:`` line from the resolved model
    makes that divergence structurally impossible.

    Keyed off ``per_lf_gpx_path`` so the Phase BC completeness invariant still
    holds (every GPS LF gets an entry); a GPS LF whose resolved model carries no
    regions gets an honest ``[]``. GPS-less drives (empty ``per_lf_gpx_path``)
    yield ``{}``. Keys may arrive as ``str`` after a ``pipeline_state.json``
    round-trip; both int and str probes are honoured. A resolved value that
    rehydrated as a plain dict (legacy state) is supported alongside the
    dataclass.
    """
    if not per_lf_gpx_path:
        return {}
    resolved_map = per_lf_resolved or {}
    out: dict[int, list[dict]] = {}
    for lf_idx_raw in per_lf_gpx_path:
        try:
            lf_idx = int(lf_idx_raw)
        except (TypeError, ValueError):
            continue
        resolved = resolved_map.get(lf_idx) or resolved_map.get(str(lf_idx))
        regions = getattr(resolved, "regions", None)
        if regions is None and isinstance(resolved, dict):
            regions = resolved.get("regions")
        out[lf_idx] = [{"name": r} for r in (regions or []) if isinstance(r, str) and r.strip()]
    return out


def _assert_per_lf_regions_complete(per_lf_gpx_path: dict, mapping: dict) -> None:
    """Phase BC invariant: every LF with a GPX path has a regions entry.

    GPS-less drives (empty ``per_lf_gpx_path``) are tolerated by design, and
    a legacy drive with no resolved models yields an entirely empty
    ``mapping`` — both are graceful.
    """
    if not mapping:
        return
    _assert_mapping_complete(
        _gpx_path_expected_keys(per_lf_gpx_path),
        mapping,
        field_name="per_lf_regions",
        expected_source_desc="per_lf_gpx_path",
        example_fix=(
            "per_lf_regions is now sourced from each LF's canonical "
            "ResolvedLocation.regions. A missing index means the LF has "
            "a GPX path but no resolved model — re-run phase_a to re-resolve "
            "the LF, or clear its per_lf_gpx_path entry to accept the omitted "
            "🗺️ Through: line."
        ),
    )


def _assert_per_lf_trim_ratio_complete(skeleton_lfs: list, mapping: dict) -> None:
    """Phase BC invariant: every skeleton-LF index with segments has a non-None entry.

    LFs constructed via the defensive fallback path (no segments) are
    excluded — their source clip metadata is unrecoverable; downstream
    falls back to the legacy ``Driving`` framing. ``None`` values are
    treated as missing (the compute helper only inserts resolved
    ratios, but a stale state file could carry ``None``).
    """
    expected: set[int] = set()
    for lf in skeleton_lfs or []:
        lf_idx = getattr(lf, "index", None)
        if lf_idx is None or not getattr(lf, "segments", None):
            continue
        expected.add(int(lf_idx))
    actual_nonnull = {k: v for k, v in (mapping or {}).items() if v is not None}
    _assert_mapping_complete(
        expected,
        actual_nonnull,
        field_name="per_lf_trim_ratio",
        expected_source_desc=f"rendered {len(skeleton_lfs)} LFs",
        example_fix=(
            "The fresh-encode path computes from skeleton_lfs + "
            "state['file_durations']; the resume path uses "
            "_derive_per_lf_trim_ratio_from_paths to ffprobe finals on "
            "disk. If both paths leave an LF unmapped, the source clip "
            "metadata is unrecoverable for that LF — surface a clear "
            "error rather than silently degrading."
        ),
    )
