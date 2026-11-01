"""Phase D metadata helpers extracted from ``phase_d.py``.

Pure cut-and-move:
the function below was lifted verbatim from ``phase_d.py``; no semantic
changes. ``phase_d.py`` re-exports the symbol so the public import path
(``gomoveshift.pipeline.drive_phases.phase_d._build_metadata_per_lf``) is
preserved for tests and the inline orchestrator call site.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ...config import Config
from .._atomic_write import atomic_write_json
from ..chapters.generator import YOUTUBE_MIN_CHAPTERS
from ..drive_text_adapter import attach_drive_text, reattach_with_chapters
from ..final_text_record import build_final_text_review
from ..location_evidence import refuse_unsupported_title
from ..region_resolver import derive_region, is_us_region
from ..route_profile import RouteProfile, build_lf_profile
from ..title_builder import ensure_unique_titles

# Use the phase_d logger name so existing log-capture tests + grep-trails
# stay valid across the module split (tests assert on
# "gomoveshift.pipeline.drive_phases.phase_d" log records).
logger = logging.getLogger("gomoveshift.pipeline.drive_phases.phase_d")


def _lf_inserts(state) -> dict[int, list[dict]]:
    """``{lf_index: [insert record, ...]}`` the drive run spliced (``Deliverables.inserts``)."""
    from ..media_api import deliverables_for_publish

    out: dict[int, list[dict]] = {}
    for d in deliverables_for_publish(state):
        # a trail opening that replaced the montage moved the stills by its lead change
        moved = {
            int(o["lf_index"]): float(o.get("insert_delta_s") or 0.0) for o in d.trail_openings
        }
        for i, recs in d.inserts.items():
            if not recs:
                continue
            delta = moved.get(i, 0.0)
            out[i] = [
                {**r, "at_s": round(max(0.0, float(r["at_s"]) + delta), 3)}
                if delta and r.get("kind") == "still"
                else r
                for r in recs
            ]
    return out


def _lf_leads(state) -> dict[int, float]:
    """``{lf_index: seconds of montage in front of the LF body}`` the file really carries.

    Read from the lead the burn recorded beside the LF (``lf_NN.lead.json``); a trail
    opening that replaced the montage supplies its own length. An LF with no record
    (rendered before the lead was kept, or no montage) has lead 0.
    """
    from ..media_api import deliverables_for_publish
    from ..trail_opening import _lead_path, _read_json

    out: dict[int, float] = {}
    for d in deliverables_for_publish(state):
        opened = {int(o["lf_index"]): float(o.get("opening_s") or 0.0) for o in d.trail_openings}
        for idx, path in zip(d.regular_lf_indices, d.regular_lf_paths):
            recipe = _read_json(_lead_path(Path(path).parent, idx))
            lead = float(recipe.get("lead_s") or 0.0) if recipe else 0.0
            out[idx] = max(0.0, opened.get(idx, lead))
    return out


def _rescope_arc_to_lf(profile: dict, resolved) -> None:
    """Window a day-aggregated route profile's journey arc + towns to ONE LF.

    ``_build_route_profiles_per_lf`` aggregates every episode recorded on the LF's
    calendar DAY, so up to four sibling LFs share one day-wide profile — each then
    renders "Started in X, ended in Y, passing through …" + town tags covering the
    WHOLE day, contradicting its own sub-arc title/OST. The per-LF
    ``ResolvedLocation`` is the single source of truth the title/OST already
    consumed, so overwrite the arc + town-list fields IN PLACE from its
    ``ordered_place_sequence`` (route-ordered, town-level, consecutively deduped).
    No-op when the LF has no resolved sequence — a GPS-less / legacy LF keeps the
    day aggregate. Features stay day-scoped; the road list is windowed separately
    by ``_rescope_roads_to_lf`` (``ResolvedLocation`` carries no per-LF road list).

    The "Started in X, ended in Y" endpoints come from the SAME record as the
    title — the burned ``ost_signature``'s own "A to B" endpoints — never the
    first/last geocoded town (a title naming one pair of towns while its
    description named another is a contradiction). A label that names no
    endpoints makes no endpoint claim.
    """
    from gomoveshift.pipeline.validation.content_gates import title_endpoints

    seq = list(getattr(resolved, "ordered_place_sequence", None) or []) if resolved else []
    if not seq:
        return
    profile["towns_traversed"] = seq
    start, end = title_endpoints(getattr(resolved, "ost_signature", None) or "") or (None, None)
    profile["start_town"] = start
    profile["end_town"] = end


def _rescope_roads_to_lf(profile: dict, own_profiles: list[RouteProfile], lf_idx: int) -> None:
    """Window the day-aggregated ``roads_traversed`` to ONE LF's own episodes
    (road tags + the "Roads traveled:" description line).

    The day aggregate spans every sibling LF's episodes, so on a multi-LF day
    LF 2's tags/description named roads only LF 1 drove. Re-aggregate the road
    list from the LF's OWN episode span (``state.per_lf_gpx_episode_paths``)
    through the same ``build_lf_profile`` merge the day aggregate used — the
    traversal ordering/dedup and the per-episode ``ROADS_TRAVERSED_MIN_KM``
    distance floor (applied when each episode profile was built) are reused,
    never re-derived. No-op when ``own_profiles`` is empty: a GPS-less / legacy
    state without an episode-span map keeps the day aggregate, and the caller
    gates single-LF days out (no sibling to leak into).
    """
    if not own_profiles:
        return
    profile["roads_traversed"] = build_lf_profile(
        lf_index=lf_idx,
        episode_profiles=own_profiles,
        gpx_files=[],
        episode_indices=[],
        geocoder=None,
        route_analyzer=None,
    ).roads_traversed


def _publish_time_bearing(state) -> dict[int, str]:
    """Per-LF ``🧭 Heading:`` re-derived from every episode each LF spans.

    Phase B+C used to store the heading of the LF's representative GPX alone,
    so an out-and-back whose return leg was the representative shipped the
    return leg's heading (Beaufort 2026-02-20: "Northeastbound"). Recompute
    from the persisted span map for each LF whose GPX files are all still on
    disk; any other LF keeps its stored heading.
    """
    stored = {int(k): v for k, v in (state.per_lf_bearing or {}).items()}
    episode_map = {int(k): v for k, v in (state.per_lf_gpx_episode_paths or {}).items()}
    readable = {
        idx for idx, paths in episode_map.items() if paths and all(Path(p).is_file() for p in paths)
    }
    if not readable:
        return stored
    from ..post_encoding_invariants import _compute_per_lf_bearing

    fresh = _compute_per_lf_bearing(
        {int(k): v for k, v in (state.per_lf_gpx_path or {}).items() if int(k) in readable},
        per_lf_gpx_episode_paths=episode_map,
    )
    out = {idx: v for idx, v in stored.items() if idx not in readable}
    out.update(fresh)
    return out


def _build_route_profiles_per_lf(state, drive_plan) -> dict[int, dict]:
    """Aggregate the persisted per-EPISODE route profiles into per-LF profiles.

    ``state.episode_profiles`` is 0-based GLOBAL-episode-keyed (parallel to
    ``gpx_files`` / ``episode_dates``); the metadata loop reads
    ``route_profiles.get(lf.index)`` (LF-index-keyed). A segment's
    ``source_file_index`` is DAY-LOCAL after ``_partition_lf_plans_by_date``, so
    indexing the drive-wide episode list with it would alias ANOTHER day's
    episodes. Map each LF to its calendar day's GLOBAL episode indices via
    ``per_lf_recording_date`` x ``episode_dates`` (the same day bucket the
    segment partition keys on), aggregate that LF's episode profiles with
    ``build_lf_profile``, and key the result by ``lf.index``. LFs with no GPS
    episodes are omitted (consumer sees ``None`` -> no enrichment).
    """
    episode_profiles = state.episode_profiles or []
    episode_dates = state.episode_dates or []
    per_lf_recording_date = state.per_lf_recording_date or {}
    per_lf_resolved = state.per_lf_resolved_location or {}
    regular_lfs = getattr(drive_plan, "regular_lfs", None) or []
    if not (episode_profiles and episode_dates and per_lf_recording_date and regular_lfs):
        return {}

    # Bucket global episode indices by recording date (ascending == chronological,
    # matching the partition's day-local order).
    episodes_by_date: dict[str, list[int]] = {}
    for gi, date in enumerate(episode_dates):
        episodes_by_date.setdefault(date, []).append(gi)

    # Road-list inputs: the LF's own episode span (GPX-path-keyed, survives
    # the day-local index remap) mapped back to GLOBAL indices, plus the number
    # of sibling LFs per recording date (roads are only windowed on multi-LF
    # days — a single-LF day keeps the day aggregate).
    per_lf_episode_paths = state.per_lf_gpx_episode_paths or {}
    gi_by_gpx = {str(p): gi for gi, p in enumerate(state.gpx_files or []) if p}
    date_lf_counts: dict[str, int] = {}
    for lf in regular_lfs:
        _idx = getattr(lf, "index", None)
        if _idx is None:
            continue
        _d = per_lf_recording_date.get(int(_idx)) or per_lf_recording_date.get(str(_idx))
        if _d:
            date_lf_counts[_d] = date_lf_counts.get(_d, 0) + 1

    out: dict[int, dict] = {}
    for lf in regular_lfs:
        lf_idx = getattr(lf, "index", None)
        if lf_idx is None:
            continue
        lf_idx = int(lf_idx)
        lf_date = per_lf_recording_date.get(lf_idx) or per_lf_recording_date.get(str(lf_idx))
        if not lf_date:
            continue
        day_gis = [
            gi
            for gi in episodes_by_date.get(lf_date, [])
            if gi < len(episode_profiles) and episode_profiles[gi]
        ]
        lf_profiles = [RouteProfile.from_dict(episode_profiles[gi]) for gi in day_gis]
        if not lf_profiles:
            continue
        # gpx_files/episode_indices/geocoder are only touched on the empty-profile
        # fallback, which the non-empty ``lf_profiles`` guarantees we never hit.
        aggregated = build_lf_profile(
            lf_index=lf_idx,
            episode_profiles=lf_profiles,
            gpx_files=[],
            episode_indices=[],
            geocoder=None,
            route_analyzer=None,
        )
        profile_dict = aggregated.to_dict()
        # The aggregate above spans the LF's whole recording DAY. Re-scope its
        # journey arc + towns to THIS LF via its own ``ResolvedLocation`` so sibling
        # LFs on a multi-LF day can't inherit each other's sub-arc.
        _rescope_arc_to_lf(profile_dict, per_lf_resolved.get(lf_idx))
        # On a multi-LF day, window the road list to THIS LF's own
        # episode span. Single-LF days and GPS-less/legacy state (no span-map
        # entry) keep the day aggregate — _rescope_roads_to_lf no-ops on [].
        own_profiles: list[RouteProfile] = []
        if date_lf_counts.get(lf_date, 0) >= 2:
            own_gis = {
                gi_by_gpx[p] for p in (per_lf_episode_paths.get(lf_idx) or {}) if p in gi_by_gpx
            }
            own_profiles = [
                RouteProfile.from_dict(episode_profiles[gi]) for gi in day_gis if gi in own_gis
            ]
        _rescope_roads_to_lf(profile_dict, own_profiles, lf_idx)
        out[lf_idx] = profile_dict
    return out


def _lf_full_drive_stats(lf, state) -> Optional[list]:
    """Which of THIS LF's distance/speed numbers (if present) were
    computed from the whole (pre-cut) GPX trace rather than its kept
    segments.

    Mirrors ``post_encoding_invariants._lf_episode_pairs`` exactly, since
    that is what actually produced ``state.per_lf_distance_km`` /
    ``state.per_lf_speed``: a ``per_lf_gpx_episode_paths`` entry for this LF
    means the number was SUMMED per source episode (windowed, never
    unwindowed); its absence falls through to
    ``phase_d._build_included_segments_for_lf`` — an EMPTY range list there
    (no segments, or no GPX-bearing representative episode) is the
    unwindowed whole-trace case those numbers were actually derived from.
    Returns ``None`` (never stamp ``full_drive_stats``) for the windowed
    (kept-segment) case, so that output stays unchanged.
    """
    episode_map = (state.per_lf_gpx_episode_paths or {}).get(lf.index)
    if episode_map:
        return None
    from . import phase_d as _pd

    ranges = _pd._build_included_segments_for_lf(lf)
    if ranges:
        return None
    return ["distance", "speed"]


def _build_metadata_per_lf(
    *,
    meta_gen,
    drive_plan,
    regular_lf_paths,
    state,
    location: str,
    publish_day_full: str,
    project_base: str,
    output_base: Path,
) -> dict:
    """Build the per-LF YouTube metadata dict and write each LF's
    ``<project>_youtube.json``; return ``{lf.index: metadata}``.

    Mirrors the inline metadata loop's contract verbatim:
      * Title-override priority: ``state["per_lf_ost_text"]`` →
        ``ensure_unique_titles`` (dedup-suffixed locally) → fallback to
        ``state["per_lf_titles"]`` for legacy resume.
      * Per-LF date / trim / distance / elevation / regions are read
        from state; missing entries degrade to "no line emitted" by
        the formatter (graceful for GPS-less drives / legacy state).
      * After each generate, ``check_description_redundancy`` raises
        ``RuntimeError`` so a redundant description can't reach disk
        (hard-fail gate).
    """
    from ..validation.content_gates import check_description_redundancy

    per_lf_locations_override = state.per_lf_locations or {}
    per_lf_resolved = state.per_lf_resolved_location or {}
    per_lf_ost_text_state = state.per_lf_ost_text or {}
    pre_render_decisions = getattr(state, "per_lf_pre_render_title_decisions", None) or {}
    if per_lf_ost_text_state:
        from ..longform_metadata import known_channel_titles

        # A split full_drive day's parts share a label by design and are told
        # apart by the deliberate ``— Part k of K`` suffix applied below; the
        # incidental ``(Part N of M)`` dedup must not stack a second one.
        _split_idx = {
            lf.index for lf in drive_plan.regular_lfs if (getattr(lf, "part_count", None) or 1) > 1
        }
        _title_map_input = {
            int(k): v
            for k, v in per_lf_ost_text_state.items()
            if int(k) not in _split_idx and int(k) not in pre_render_decisions
        }
        per_lf_titles_override = ensure_unique_titles(
            _title_map_input, published_titles=known_channel_titles()
        )
        per_lf_titles_override.update(
            {int(k): v for k, v in per_lf_ost_text_state.items() if int(k) in _split_idx}
        )
        per_lf_titles_override.update(
            {int(k): decision["input_label"] for k, decision in pre_render_decisions.items()}
        )
    else:
        per_lf_titles_override = state.per_lf_titles or {}
    route_profiles = state.route_profiles or {}
    if not route_profiles:
        # Wire the GPS-verified per-LF enrichment. ``state.route_profiles``
        # was read here but never written, so the enrichment was dead. Build it
        # from the persisted per-episode profiles (aggregating by recording day)
        # and persist so a resume reuses it.
        route_profiles = _build_route_profiles_per_lf(state, drive_plan)
        if route_profiles:
            state.route_profiles = route_profiles
    per_lf_trim_ratio = state.per_lf_trim_ratio or {}
    per_lf_recording_date = state.per_lf_recording_date or {}
    per_lf_distance_km = state.per_lf_distance_km or {}
    per_lf_elevation_gain_m = state.per_lf_elevation_gain_m or {}
    per_lf_elevation_profile = state.per_lf_elevation_profile or {}
    per_lf_speed = state.per_lf_speed or {}
    per_lf_bearing = _publish_time_bearing(state)
    per_lf_regions = state.per_lf_regions or {}
    per_lf_gpx_path_map = state.per_lf_gpx_path or {}

    lf_inserts = _lf_inserts(state)
    metadata_per_lf: dict[int, dict] = {}
    for lf, _lf_path in zip(drive_plan.regular_lfs, regular_lf_paths):
        project_name = f"{project_base}_lf{lf.index:02d}"
        # A populated per-LF map
        # missing THIS index is state drift — silently falling back to the
        # drive-wide label shipped wrong-region metadata. An entirely empty
        # map (GPS-less / legacy state) still degrades to the drive-wide
        # label by design.
        if per_lf_locations_override:
            if lf.index not in per_lf_locations_override:
                raise RuntimeError(
                    f"phase_d metadata: LF {lf.index} missing from "
                    f"per_lf_locations (have {sorted(per_lf_locations_override)}) "
                    f"— refusing drive-wide fallback {location!r}; state drift "
                    "would ship wrong-region metadata."
                )
            lf_location = per_lf_locations_override[lf.index]
        else:
            lf_location = location
        # Mirror the per_lf_locations hard-fail above — a populated
        # per_lf_ost_text map (source of ``per_lf_titles_override``) missing
        # THIS index is state drift. Silently dropping to no title_override
        # lets the SEO manager template the title, diverging the YouTube
        # title from the burned OST pixels. An entirely empty map (GPS-less
        # / legacy per_lf_titles resume) still degrades to the SEO title.
        if per_lf_ost_text_state and lf.index not in per_lf_titles_override:
            raise RuntimeError(
                f"phase_d metadata: LF {lf.index} missing from "
                f"per_lf_ost_text (have {sorted(per_lf_titles_override)}) "
                "— refusing SEO-template title fallback; state drift would "
                "diverge the YouTube title from the burned OST pixels."
            )
        lf_title_override = per_lf_titles_override.get(lf.index) or None
        preselected = pre_render_decisions.get(lf.index)
        route_record = (getattr(state, "per_lf_route_record", None) or {}).get(lf.index)
        if preselected and not route_record:
            from ..title_decision import PreRenderTitleError

            raise PreRenderTitleError(f"LF {lf.index} lost its retained route after title approval")
        if route_record:
            from ..title_decision import PreRenderTitleError, route_fingerprint

            if not preselected:
                raise PreRenderTitleError(
                    f"LF {lf.index} has retained-route evidence but no pre-render Waypost title decision"
                )
            if preselected.get("ost_text") != per_lf_ost_text_state.get(lf.index):
                raise PreRenderTitleError(f"LF {lf.index} OST changed after title approval")
            if preselected.get("route_fingerprint") != route_fingerprint(route_record):
                raise PreRenderTitleError(
                    f"LF {lf.index} retained route changed after title approval"
                )
        lf_profile = route_profiles.get(lf.index)

        # Units + playlist region come from the SAME canonical
        # ResolvedLocation the title used; WARN when the label parse disagrees.
        # ``per_lf_resolved_location`` is an ``_INT_KEY_DICT_FIELDS`` member, so
        # ``_coerce_int_key_dict`` normalizes its keys to ``int`` on every JSON
        # load — a ``str(lf.index)`` fallback would be dead and would mask a
        # deserialization regression that the canonical-key invariant should
        # surface instead.
        lf_resolved = per_lf_resolved.get(lf.index)
        if lf_resolved is not None and is_us_region(lf_location) != is_us_region(
            lf_location, resolved=lf_resolved
        ):
            logger.warning(
                "UNITS-DIVERGE: LF %d label %r is_us=%s but resolved model says "
                "is_us=%s; using the resolved model",
                lf.index,
                lf_location,
                is_us_region(lf_location),
                is_us_region(lf_location, resolved=lf_resolved),
            )

        lf_recording_date: Optional[str] = None
        date_val = per_lf_recording_date.get(lf.index)
        if isinstance(date_val, str) and date_val.strip():
            lf_recording_date = date_val.strip()
        elif hasattr(date_val, "strftime"):
            lf_recording_date = date_val.strftime("%Y-%m-%d")

        lf_trim: Optional[float] = None
        _trim_val = per_lf_trim_ratio.get(lf.index)
        if isinstance(_trim_val, (int, float)):
            lf_trim = float(_trim_val)
        lf_source_duration: Optional[float] = None
        if lf_trim is not None and lf_trim > 0:
            lf_source_duration = lf.total_duration / lf_trim
        elif lf_trim == 0:
            # A literal-zero ratio means "unmeasured" (no segments / ffprobe
            # gap), NOT a 100%-trimmed highlight. Threading 0 downstream made
            # meta_gen select the "Highlights from" verb for a possibly
            # full-length drive. Normalise to None at the source so the verb
            # stays "Driving" and no Final-cut disclosure is implied, and WARN
            # (was debug) so the regression is grep-traceable.
            logger.warning(
                "TRIM-RATIO-ZERO: trim_ratio=0 for LF %d — treating as unmeasured "
                "(verb stays 'Driving', no Final-cut disclosure)",
                lf.index,
            )
            lf_trim = None

        lf_distance_km: Optional[float] = None
        _dist_val = per_lf_distance_km.get(lf.index)
        if isinstance(_dist_val, (int, float)) and _dist_val > 0:
            lf_distance_km = float(_dist_val)

        lf_elevation_gain_m: Optional[float] = None
        _elev_val = per_lf_elevation_gain_m.get(lf.index)
        if isinstance(_elev_val, (int, float)) and _elev_val >= 0:
            lf_elevation_gain_m = float(_elev_val)

        # Elevation profile (range + net climb/descent). Tolerate
        # str-keyed resume state. Empty/missing → None (line omitted).
        lf_elevation_profile: Optional[dict] = None
        _prof_val = per_lf_elevation_profile.get(lf.index) or per_lf_elevation_profile.get(
            str(lf.index)
        )
        if isinstance(_prof_val, dict) and _prof_val:
            lf_elevation_profile = _prof_val

        # Speed/character ({avg_kmh, max_kmh, std_kmh}). Same str-key
        # resume tolerance; empty/missing → None (Pace line omitted).
        lf_speed: Optional[dict] = None
        _speed_val = per_lf_speed.get(lf.index) or per_lf_speed.get(str(lf.index))
        if isinstance(_speed_val, dict) and _speed_val:
            lf_speed = _speed_val

        # Net travel bearing (cardinal string). Same str-key tolerance.
        lf_bearing: Optional[str] = None
        _bearing_val = per_lf_bearing.get(lf.index) or per_lf_bearing.get(str(lf.index))
        if isinstance(_bearing_val, str) and _bearing_val:
            lf_bearing = _bearing_val

        lf_regions: Optional[list] = None
        _regions_val = per_lf_regions.get(lf.index)
        if isinstance(_regions_val, list):
            lf_regions = _regions_val

        # Drive the chapter-density skip from the same
        # state["per_lf_gpx_path"] that controls chapter generation.
        # tolerate str-keyed resume state by also probing the str key.
        lf_has_gps = (
            per_lf_gpx_path_map.get(lf.index) is not None
            or per_lf_gpx_path_map.get(str(lf.index)) is not None
        )

        # Distance/speed numbers computed from the whole (pre-cut)
        # trace, not this LF's kept segments, must flag the gap so the
        # generator renders "Full drive:" and the redundancy gate doesn't
        # mistake it for an unscoped kept-segment number.
        lf_full_drive_stats = _lf_full_drive_stats(lf, state)
        if lf_full_drive_stats and (getattr(lf, "part_count", None) or 1) > 1:
            # A split part must never carry whole-trace numbers (they would
            # describe the whole day, not this part): with no windowed source
            # of its own, drop distance/speed instead of flagging them.
            lf_distance_km = None
            lf_speed = None
            lf_full_drive_stats = None

        metadata = meta_gen.generate(
            location=lf_location,
            publish_day=publish_day_full,
            video_duration=lf.total_duration,
            project_name=project_name,
            route_profile=lf_profile,
            title_override=lf_title_override,
            recording_date=lf_recording_date,
            trim_ratio=lf_trim,
            source_duration=lf_source_duration,
            distance_km=lf_distance_km,
            elevation_gain_m=lf_elevation_gain_m,
            elevation_profile=lf_elevation_profile,
            speed=lf_speed,
            bearing=lf_bearing,
            regions=lf_regions,
            has_gps=lf_has_gps,
            resolved_location=lf_resolved,
            full_drive_stats=lf_full_drive_stats,
            insert_count=len(lf_inserts.get(lf.index, [])),
            # The evidence record behind the title's place claims; the
            # part suffix goes INTO Waypost's decision, never appended after.
            location_evidence=(state.per_lf_location_evidence or {}).get(lf.index),
            preselected_title_decision=preselected,
            title_part=(
                (lf.part_index, lf.part_count)
                if (getattr(lf, "part_count", None) or 1) > 1
                else None
            ),
        )
        if preselected and (
            metadata.get("title") != preselected.get("chosen_title")
            or metadata.get("title_decision") != preselected
        ):
            from ..title_decision import PreRenderTitleError

            raise PreRenderTitleError(
                f"LF {lf.index} metadata replaced the pre-render Waypost title decision"
            )
        # The shipped title's place claims must be backed by this LF's
        # evidence record before anything is written; no record is a refusal.
        refuse_unsupported_title(
            metadata.get("title", ""),
            (state.per_lf_location_evidence or {}).get(lf.index),
            lf_index=lf.index,
        )
        # Waypost's review of the final text against the retained route;
        # ``None`` for a GPS-less LF. Upload re-verifies it before insert.
        attach_drive_text(
            (state.per_lf_route_record or {}).get(lf.index),
            metadata,
            lf_location,
            final_seconds=lf.total_duration,
            source_seconds=lf_source_duration,
            trim_ratio=lf_trim,
            distance_km=lf_distance_km,
            bearing=lf_bearing,
            recording_date=lf_recording_date,
            insert_count=len(lf_inserts.get(lf.index, [])),
            us=is_us_region(lf_location, resolved=lf_resolved),
        )
        metadata["final_text_review"] = build_final_text_review(state, lf, metadata)
        metadata.setdefault("location", lf_location)
        # Stamp the resolved region (JSON-safe) so the playlist
        # consumes the SAME model the units used (resolve_playlist's region key).
        if lf_resolved is not None and (_rk := derive_region(lf_location, resolved=lf_resolved)):
            metadata.setdefault("region", _rk)
        # Thread the canonical vote + sampled-regions union
        # (JSON-safe) so the post-encode location-invariant gate can reconcile
        # the title county against the GPS trace and hard-fail the far-road
        # class. Absent for GPS-less / legacy LFs → the gate no-ops. The same
        # builder feeds the pre-encode gate, so both read identical fields.
        if lf_resolved is not None:
            from ..validation.content_gates import resolved_location_gate_dict

            metadata["resolved_location"] = resolved_location_gate_dict(lf_resolved)
        if lf_recording_date is not None:
            metadata.setdefault("recording_date", lf_recording_date)
            # Pass the day's native-resolution claim through for the
            # publish side (False = the day was upscaled; do not label it native).
            _day = (getattr(state, "per_day_completed", None) or {}).get(lf_recording_date)
            if isinstance(_day, dict) and "native_resolution_claim" in _day:
                metadata["native_resolution_claim"] = bool(_day["native_resolution_claim"])
        if lf_source_duration is not None:
            metadata.setdefault("source_duration", lf_source_duration)
        # Format-aware playlist routing (playlist_resolver's derived
        # ``format`` key): thread the same trim_ratio already passed to
        # meta_gen.generate above.
        if lf_trim is not None:
            metadata.setdefault("trim_ratio", lf_trim)
        metadata.setdefault("video_duration", lf.total_duration)
        metadata_per_lf[lf.index] = metadata

        _redundancy = check_description_redundancy(metadata)
        if not _redundancy.passed:
            raise RuntimeError(
                f"description redundancy gate failed for LF {lf.index}: {_redundancy.message}"
            )

        meta_path = output_base / f"{project_name}_youtube.json"
        atomic_write_json(meta_path, metadata)

        logger.info("  LF %d: %s", lf.index, metadata.get("title", "N/A"))

    return metadata_per_lf


def _below_chapter_floor(chapters: list, video_duration: float) -> bool:
    """True when ``chapters`` miss the metadata gate's middle-chapter floor."""
    from ..chapters.generator import intermediate_chapters, min_intermediate_chapters

    return len(intermediate_chapters(chapters)) < min_intermediate_chapters(video_duration)


def _lf_chapters(config: Config, lf, state, location: str, stages_dir: Path):
    """``(chapters, chapter_text)`` for one LF from its GPS, or ``None`` when
    the LF has no GPX. A generator error yields ``([], "")``, logged."""
    from . import phase_d as _pd

    gpx_files = state.gpx_files or []
    per_lf_gpx_path = state.per_lf_gpx_path or {}
    per_lf_locations_override = state.per_lf_locations or {}
    per_lf_resolved = state.per_lf_resolved_location or {}
    gpx_path_str = per_lf_gpx_path.get(lf.index)
    if gpx_path_str is None and gpx_files:
        first_seg_idx = lf.segments[0].source_file_index if lf.segments else None
        if first_seg_idx is not None and first_seg_idx < len(gpx_files):
            gpx_path_str = gpx_files[first_seg_idx]
            logger.warning(
                "  LF %d: per_lf_gpx_path missing in state — "
                "falling back to segment-index lookup "
                "(may be wrong on multi-day resumes from "
                "pre-fix state files)",
                lf.index,
            )
    if not gpx_path_str:
        logger.debug("  LF %d: no GPX file, skipping chapters", lf.index)
        return None
    gpx_path = Path(gpx_path_str)
    if not gpx_path.exists():
        logger.debug("  LF %d: GPX file %s not found, skipping", lf.index, gpx_path)
        return None

    lf_location = per_lf_locations_override.get(lf.index, location)
    # Int-keyed by construction (see the metadata-loop note above) — no
    # ``str(lf.index)`` fallback.
    lf_resolved = per_lf_resolved.get(lf.index)
    # Pass gpx_files so the window's representative episode is chosen
    # among GPX-bearing episodes — the same one per_lf_gpx_path/gpx_path
    # above resolves to, so chapters trim the actual charted GPX.
    lf_included_segments = _pd._build_included_segments_for_lf(lf, gpx_files=gpx_files)
    lf_day_journey = None
    multi = _build_lf_multi_episode_journey(
        lf, getattr(state, "gps_per_episode", None), gpx_files=gpx_files
    )
    if multi is not None:
        lf_day_journey, lf_included_segments = multi
        logger.info(
            "  LF %d: charting %d waypoints across every GPX-bearing episode "
            "(drive timeline, %d ranges)",
            lf.index,
            len(lf_day_journey.waypoints),
            len(lf_included_segments),
        )
    try:
        chapters, chapter_text = _pd._generate_chapters_for_lf(
            config=config,
            gpx_path=gpx_path,
            video_duration=lf.total_duration,
            location=lf_location,
            stages_dir=stages_dir,
            included_segments=lf_included_segments or None,
            resolved_location=lf_resolved,
            day_journey=lf_day_journey,
        )
    except Exception as e:
        # The single-video path returns success=False on chapter
        # failure (stages/chapters.py); the drive path used to swallow it at
        # WARN and embed an empty block. Fail loud with a greppable ERROR.
        logger.error("CHAPTER-GATE: LF %d chapter generation failed: %s", lf.index, e)
        chapters = []
        chapter_text = ""
    return chapters, chapter_text


def assert_chapter_floor_before_encode(
    config: Config, drive_plan, state, location: str, stages_dir: Path
) -> None:
    """Refuse to START an encode whose LF chapters cannot clear the
    metadata gate's middle-chapter floor.

    Runs the same per-LF chapter computation (and the same retry-once repair)
    as Phase D's D1.5, against the planned LF durations, so a trace too sparse
    to chapter fails in minutes instead of at Phase D after a multi-hour
    encode. An LF with no GPX is skipped, as the metadata gate skips it
    (``has_gps`` False). The geocodes it makes land in the shared geocoding
    cache, so D1.5 reuses them.
    """
    short: list[str] = []
    for lf in getattr(drive_plan, "regular_lfs", None) or []:
        computed = _lf_chapters(config, lf, state, location, stages_dir)
        if computed is None or not _below_chapter_floor(computed[0], lf.total_duration):
            continue
        retry = _lf_chapters(config, lf, state, location, stages_dir)
        if retry is not None and not _below_chapter_floor(retry[0], lf.total_duration):
            continue
        from ..chapters.generator import intermediate_chapters, min_intermediate_chapters

        got = len(intermediate_chapters((retry or computed)[0]))
        short.append(
            f"LF {lf.index}: {got} middle chapter(s) for "
            f"{lf.total_duration / 60:.0f}min (need {min_intermediate_chapters(lf.total_duration)}+)"
        )
    if short:
        raise RuntimeError(
            "CHAPTER-GATE: refusing to start the encode — the GPS trace cannot "
            "chapter these LFs: " + "; ".join(short)
        )


def _generate_chapters_all_lfs(
    *,
    config: Config,
    drive_plan,
    state,
    metadata_per_lf: dict,
    location: str,
    project_base: str,
    output_base: Path,
    stages_dir: Path,
) -> None:
    """Generate YouTube chapters from GPS for every LF that has a GPX.

    Mutates ``metadata_per_lf`` in place (sets ``chapters``,
    ``chapter_text``, embeds chapters into ``description``) and
    re-writes each LF's ``<project>_youtube.json`` with the embedded
    chapter text. Re-runs the description-redundancy gate after embed
    (the embed mutates the description text).

    GPX resolution order per LF:
      1. ``state["per_lf_gpx_path"][lf.index]`` — authoritative,
         captured before ``_partition_lf_plans_by_date`` mutates segment
         indices (post-fix).
      2. ``state["gpx_files"][segment.source_file_index]`` — legacy
         fallback; may be wrong on multi-day resumes from pre-fix
         state files (logs a warning when used).
    """
    # Late-bind to phase_d so test monkey-patches on phase_d._generate_chapters_for_lf,
    # phase_d._build_included_segments_for_lf, and phase_d._LFMeta._embed_chapters
    # still flow through after the cut-and-move.
    from ..validation.content_gates import (
        check_description_redundancy as _check_red,
    )
    from . import phase_d as _pd

    gpx_files = state.gpx_files or []
    per_lf_gpx_path = state.per_lf_gpx_path or {}
    lf_inserts = _lf_inserts(state)
    lf_leads = _lf_leads(state)
    if not (per_lf_gpx_path or gpx_files):
        logger.info("  D1.5: No GPX files — skipping chapter generation")
        return

    logger.info("  D1.5: Generating YouTube chapters from GPS...")
    for lf in drive_plan.regular_lfs:
        computed = _lf_chapters(config, lf, state, location, stages_dir)
        if computed is None:
            continue
        chapters, chapter_text = computed
        if _below_chapter_floor(chapters, lf.total_duration):
            # A shortfall the generator can fix (a geocoder blip that
            # emptied the first pass) is repaired here, once, instead of
            # failing the run at the metadata gate after the encode.
            logger.warning(
                "CHAPTER-GATE: LF %d first pass under the chapter floor — regenerating",
                lf.index,
            )
            chapters, chapter_text = _lf_chapters(config, lf, state, location, stages_dir) or (
                chapters,
                chapter_text,
            )

        meta = metadata_per_lf.get(lf.index, {})
        if chapters and lf_leads.get(lf.index):
            # chapters were timed against the body; the file opens with the montage
            from ..chapters.marker_merger import shift_chapters

            chapters = shift_chapters(chapters, lf_leads[lf.index])
            chapter_text = "\n".join(f"{c['timestamp']} {c['title']}" for c in chapters)
        if chapters and lf_inserts.get(lf.index):
            from ..chapters.marker_merger import add_insert_markers

            chapters = add_insert_markers(chapters, [r["at_s"] for r in lf_inserts[lf.index]])
            chapter_text = "\n".join(f"{c['timestamp']} {c['title']}" for c in chapters)

        meta["chapters"] = chapters
        meta["chapter_text"] = chapter_text

        # A description chapter block only renders on YouTube with
        # >= YOUTUBE_MIN_CHAPTERS timestamps. The pad runs inside generate(), so
        # < that here means the LF is too short to host a renderable block —
        # embedding it just clutters the description with a non-rendering stub.
        # Skip the embed and fail loud (greppable ERROR) instead.
        if "description" in meta:
            if len(chapters) >= YOUTUBE_MIN_CHAPTERS:
                meta["description"] = _pd._LFMeta._embed_chapters(
                    meta["description"],
                    chapter_text,
                )
            else:
                logger.error(
                    "CHAPTER-GATE: LF %d produced %d < %d chapters — not embedding a "
                    "non-rendering block into the description",
                    lf.index,
                    len(chapters),
                    YOUTUBE_MIN_CHAPTERS,
                )

        # The embed above rewrote the description, so the review is
        # built from the text that will be sent, after it.
        if len(chapters) >= YOUTUBE_MIN_CHAPTERS:
            reattach_with_chapters((state.per_lf_route_record or {}).get(lf.index), meta, chapters)
        meta["final_text_review"] = build_final_text_review(state, lf, meta)
        metadata_per_lf[lf.index] = meta

        _red2 = _check_red(meta)
        if not _red2.passed:
            raise RuntimeError(
                f"description redundancy gate failed for LF {lf.index} "
                f"after chapter embed: {_red2.message}"
            )

        project_name = f"{project_base}_lf{lf.index:02d}"
        meta_path = output_base / f"{project_name}_youtube.json"
        atomic_write_json(meta_path, meta)

        logger.info("  LF %d: %d chapters generated", lf.index, len(chapters))


def _build_lf_multi_episode_journey(lf, gps_per_episode, gpx_files=None):
    """One GPS journey + trim ranges on the LF's DRIVE timeline, when the LF
    spans more than one GPX-bearing episode.

    ``_pick_representative_episode`` charts an LF against ONE episode's GPX;
    a full-drive LF crossing episodes then gets that single trace linearly
    stretched over the whole LF (e.g. a 701 s episode charted
    across 41.5 min — chapters on the wrong roads, too few to clear the
    metadata gate). Here every GPX-bearing episode's trackpoints are
    projected onto drive-global seconds (``original_start_time -
    start_time`` is the episode's drive offset) and the segments' own
    drive-global windows become the trim ranges — the same day-journey path
    the superlong parts use (``build_day_master_journey``).

    Returns ``(journey, ranges)`` rebased so the journey's first waypoint is
    second 0 (the generator keys trims on it), or ``None`` when the LF
    touches at most one GPX-bearing episode (callers keep the per-episode
    GPX path).
    """
    from ...analysis.shorts_models import effective_offset as _effective_offset
    from ..chapters.generator import build_day_master_journey

    segments = list(getattr(lf, "segments", []) or [])
    gps_per_episode = gps_per_episode or []

    def _has_gpx(idx: int) -> bool:
        if gpx_files is not None and (idx >= len(gpx_files) or not gpx_files[idx]):
            return False
        return 0 <= idx < len(gps_per_episode) and bool(gps_per_episode[idx])

    ep_offsets: dict[int, float] = {}
    ep_windows: dict[int, list[tuple[float, float]]] = {}
    for seg in segments:
        idx = int(getattr(seg, "source_file_index", 0))
        if not _has_gpx(idx):
            continue
        g_start = getattr(seg, "original_start_time", None)
        g_end = getattr(seg, "original_end_time", None)
        if g_start is None or g_end is None:
            return None
        local = float(getattr(seg, "start_time", 0.0))
        ep_offsets.setdefault(idx, float(g_start) - local)
        if float(g_end) > float(g_start):
            ep_windows.setdefault(idx, []).append((float(g_start), float(g_end)))
    if len(ep_offsets) < 2:
        return None

    points: list[tuple[float, float, float]] = []
    for idx, ep_start in ep_offsets.items():
        entry = gps_per_episode[idx]
        gps_data = entry[0]
        gps_offset = float(entry[1]) if len(entry) > 1 and entry[1] is not None else 0.0
        if gps_data is None:
            continue
        gps_offset = _effective_offset(gps_data, gps_offset)
        trackpoints = getattr(gps_data, "trackpoints", None)
        if trackpoints is None and isinstance(gps_data, dict):
            trackpoints = gps_data.get("trackpoints") or []
        ep_end = max(e for _, e in ep_windows.get(idx, [(ep_start, ep_start)]))
        for tp in trackpoints or []:
            if isinstance(tp, dict):
                ts, lat, lon = tp.get("timestamp"), tp.get("lat"), tp.get("lon")
            else:
                ts = getattr(tp, "timestamp", None)
                lat = getattr(tp, "lat", None)
                lon = getattr(tp, "lon", None)
            if ts is None or lat is None or lon is None:
                continue
            t_drive = ep_start + float(ts) - gps_offset
            if t_drive < ep_start or t_drive > ep_end:
                continue
            points.append((t_drive, float(lat), float(lon)))
    if not points:
        return None
    t0 = min(p[0] for p in points)
    journey = build_day_master_journey([(t - t0, lat, lon) for t, lat, lon in points])
    if journey is None:
        return None
    ranges = sorted((s - t0, e - t0) for windows in ep_windows.values() for s, e in windows)
    return journey, ranges


def _generate_lf_thumbnails(
    *,
    drive_plan,
    regular_lf_paths,
    metadata_per_lf: dict,
    state,
    output_base: Path,
) -> None:
    """Generate the canonical ``thumbnails/lf_NN.jpg`` per LF.

    Runs UPSTREAM of every upload and HARD-FAILS: a generator exception, a
    ``None`` return and a failed canonical promotion all raise — all three
    used to warn, letting a long-form publish with no thumbnail while the
    run still reported success. Attach needs a video ID, so it lives
    post-upload in ``_attach_lf_thumbnails``.

    Generation is unconditional (DoD validator requires the canonical
    path). Title source: canonical ``state["per_lf_ost_text"]``
    (pre-dedup), with fallback to ``meta["title"]`` for legacy state
    files; if both are missing we raise rather than burn a
    ``"Drive LF N"`` placeholder onto a public thumbnail (no gate ever
    sees thumbnail text). Stale per-LF subdir + canonical file are
    removed before regen so a bug in generation can't be masked by
    stale output.
    """
    if not drive_plan.regular_lfs:
        return
    logger.info("  D-thumbs: Generating thumbnails per LF...")
    import shutil as _shutil

    from . import thumbnail_generator as tg

    thumbs_root = output_base / "thumbnails"
    thumbs_root.mkdir(parents=True, exist_ok=True)

    per_lf_ost_text_map = state.per_lf_ost_text or {}
    for lf, lf_path in zip(drive_plan.regular_lfs, regular_lf_paths):
        meta = metadata_per_lf.get(lf.index, {})
        title = (
            per_lf_ost_text_map.get(lf.index)
            or per_lf_ost_text_map.get(str(lf.index))
            or meta.get("title")
        )
        if not title:
            raise ValueError(
                f"LF {lf.index}: no thumbnail title — per_lf_ost_text and "
                "meta['title'] are both missing. Refusing to burn a "
                "'Drive LF N' placeholder onto a public thumbnail (no gate "
                "ever reviews thumbnail text). Populate OST text or gated "
                "metadata before thumbnail generation."
            )

        thumb_subdir = thumbs_root / f"lf_{lf.index:02d}"
        canonical_thumb = thumbs_root / f"lf_{lf.index:02d}.jpg"
        if thumb_subdir.exists():
            _shutil.rmtree(thumb_subdir, ignore_errors=True)
        if canonical_thumb.exists():
            try:
                canonical_thumb.unlink()
            except OSError as e:
                logger.warning(
                    "  LF %d: could not remove stale thumbnail %s: %s",
                    lf.index,
                    canonical_thumb,
                    e,
                )

        try:
            # No ``is_masked``/``unmasked_intervals`` threaded here. The LF
            # FINAL video is already structurally stop-free (the stop-cut layer
            # removes parked spans before this stage), so in FINAL-local time
            # every second is "kept" — the day-master STOP mask does not map
            # cheaply back to FINAL-local seconds at this call site. The
            # generator param exists for callers that DO have the FINAL-local
            # kept intervals; this belt-and-suspenders sampler stays legacy here.
            variant_path = tg.generate_lf_thumbnail(
                video_path=Path(lf_path),
                title=title,
                subtitle=None,
                output_dir=thumb_subdir,
                duration=lf.total_duration,
            )
        except Exception as e:
            raise RuntimeError(f"LF {lf.index}: thumbnail generation failed: {e}") from e

        if not variant_path:
            raise RuntimeError(f"LF {lf.index}: thumbnail generation produced nothing")

        try:
            _shutil.copyfile(str(variant_path), str(canonical_thumb))
        except OSError as e:
            raise RuntimeError(
                f"LF {lf.index}: failed to promote thumbnail to {canonical_thumb}: {e}"
            ) from e


def _attach_lf_thumbnails(
    *,
    drive_plan,
    output_base: Path,
    yt,
    youtube_uploads: dict,
    state=None,
) -> None:
    """Attach each canonical ``thumbnails/lf_NN.jpg`` to its uploaded video.

    Necessarily LATE — YouTube has no video ID before the upload — so a
    failure is recorded on the upload entry (``thumbnail_attach_failed``)
    rather than raised: the video is already public and aborting cannot
    un-publish it. The generation gate above prevents the bad ship.

    ``state`` (optional, default ``None`` for legacy callers) carries the
    cut-from-master identity cross-check — same best-effort,
    never-gating contract as the LF provenance sidecar. Absent ``state``
    degrades every entry to null-with-reason rather than raising.
    """
    if not drive_plan.regular_lfs or yt is None:
        return
    from ..lf_provenance_sidecar import attach_master_identity_to_record

    thumbs_root = output_base / "thumbnails"
    recording_dates = getattr(state, "per_lf_recording_date", None) or {}
    per_day_completed = getattr(state, "per_day_completed", None) or {}
    stages_base = Path(getattr(state, "stages_base", None) or (output_base / "_stages"))
    for lf in drive_plan.regular_lfs:
        entry = youtube_uploads.get(lf.index)
        canonical_thumb = thumbs_root / f"lf_{lf.index:02d}.jpg"
        if not entry or not canonical_thumb.exists():
            continue
        try:
            yt.set_thumbnail(entry["video_id"], str(canonical_thumb))
            entry["thumbnail_path"] = str(canonical_thumb)
        except Exception as e:
            logger.warning("  LF %d: set_thumbnail failed: %s", lf.index, e)
            entry["thumbnail_attach_failed"] = str(e)
        date = recording_dates.get(lf.index) or recording_dates.get(str(lf.index))
        attach_master_identity_to_record(
            entry, date=date, per_day_completed=per_day_completed, stages_base=stages_base
        )


def _generate_superlong_thumbnails(*, state, output_base: Path) -> None:
    """Generate one canonical thumbnail per superlong part.

    Reusing the day's LF thumbnail for every part was explicitly rejected:
    a 3-part day would show three IDENTICAL images in subscriptions, worse
    for telling parts apart than YouTube's auto-picked frame. Mirrors
    ``_generate_lf_thumbnails``'s placement (upstream of the irreversible
    upload) and hard-fail contract — a generator exception or an empty
    result raises, so a broken run costs a re-run, not a public part with
    no thumbnail.

    Filename ``thumbnails/sl_<date>_<NN>.jpg`` cannot collide with
    ``lf_NN.jpg`` (distinct prefix, plus a date component no LF filename
    carries). The part number is stamped as the SUBTITLE (``Part N of
    M``) so same-day parts are distinguishable at a glance; the TITLE is
    the part's own burned OST location text (``record["location"]``) —
    the same burn-source-of-truth the LF thumbnail follows, never the
    YouTube title (which carries the redundant ``Part N of M`` suffix).
    """
    import shutil as _shutil

    from .. import youtube_sidecar
    from . import thumbnail_generator as tg

    parts = youtube_sidecar.iter_superlong_parts(state)
    if not parts:
        return
    logger.info("  D-thumbs: Generating thumbnails per superlong part...")
    thumbs_root = output_base / "thumbnails"
    thumbs_root.mkdir(parents=True, exist_ok=True)

    for part_key, record in parts:
        title = record.get("location") or record.get("title")
        if not title:
            raise ValueError(
                f"superlong part {part_key}: no thumbnail title — record has "
                "neither 'location' nor 'title'. Refusing to burn a blank "
                "thumbnail onto a public video."
            )
        part_index = int(record.get("part_index") or 0)
        part_count = record.get("part_count")
        subtitle = f"Part {part_index} of {part_count}" if part_count else None
        date = part_key.split(":", 1)[0]
        thumb_subdir = thumbs_root / f"sl_{date}_{part_index:02d}"
        canonical_thumb = thumbs_root / f"sl_{date}_{part_index:02d}.jpg"
        if thumb_subdir.exists():
            _shutil.rmtree(thumb_subdir, ignore_errors=True)
        if canonical_thumb.exists():
            try:
                canonical_thumb.unlink()
            except OSError as e:
                logger.warning(
                    "  superlong part %s: could not remove stale thumbnail %s: %s",
                    part_key,
                    canonical_thumb,
                    e,
                )

        try:
            variant_path = tg.generate_lf_thumbnail(
                video_path=Path(record["path"]),
                title=title,
                subtitle=subtitle,
                output_dir=thumb_subdir,
                duration=float(record.get("duration_sec") or 0.0),
            )
        except Exception as e:
            raise RuntimeError(
                f"superlong part {part_key}: thumbnail generation failed: {e}"
            ) from e
        if not variant_path:
            raise RuntimeError(f"superlong part {part_key}: thumbnail generation produced nothing")

        try:
            _shutil.copyfile(str(variant_path), str(canonical_thumb))
        except OSError as e:
            raise RuntimeError(
                f"superlong part {part_key}: failed to promote thumbnail to {canonical_thumb}: {e}"
            ) from e


def _attach_superlong_thumbnails(*, state, output_base: Path, yt) -> None:
    """Attach each superlong part's canonical thumbnail to its uploaded video.

    Part analogue of ``_attach_lf_thumbnails``: necessarily LATE (needs a
    video ID) so a failure is recorded on the part's OWN
    ``superlong_uploads.json`` sidecar entry (``thumbnail_attach_failed``)
    rather than raised — the video is already public. A part with no
    generated thumbnail file, or no upload record for its key, is skipped
    quietly so the wrong image can never attach to the wrong part.
    """
    from .. import youtube_sidecar
    from ..lf_provenance_sidecar import attach_master_identity_to_record

    if yt is None:
        return
    parts = youtube_sidecar.iter_superlong_parts(state)
    if not parts:
        return
    sidecar_uploads = youtube_sidecar.load_superlong_sidecar(output_base)
    thumbs_root = output_base / "thumbnails"
    per_day_completed = getattr(state, "per_day_completed", None) or {}
    stages_base = Path(getattr(state, "stages_base", None) or (output_base / "_stages"))
    dirty = False
    for part_key, record in parts:
        entry = sidecar_uploads.get(part_key)
        date = part_key.split(":", 1)[0]
        part_index = int(record.get("part_index") or 0)
        canonical_thumb = thumbs_root / f"sl_{date}_{part_index:02d}.jpg"
        if not entry or not entry.get("video_id") or not canonical_thumb.exists():
            continue
        try:
            yt.set_thumbnail(entry["video_id"], str(canonical_thumb))
            entry["thumbnail_path"] = str(canonical_thumb)
        except Exception as e:
            logger.warning("  superlong part %s: set_thumbnail failed: %s", part_key, e)
            entry["thumbnail_attach_failed"] = str(e)
        # Same best-effort, never-gating identity cross-check as the
        # LF provenance sidecar — this part's date is already known (from
        # its own part_key), so no per-part recording-date lookup is needed.
        attach_master_identity_to_record(
            entry, date=date, per_day_completed=per_day_completed, stages_base=stages_base
        )
        dirty = True
    if dirty:
        youtube_sidecar.save_superlong_sidecar(output_base, sidecar_uploads, state=state)
