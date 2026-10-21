"""Pre-encoding per-LF GPS + location enrichment helpers.

Extracted from ``drive_phases.phase_bc_day_first`` so the
phase entry point stays focused on orchestration. No behaviour change
— every function below is the verbatim move of its phase_bc original.

Public-API expectations (re-exported by phase_bc_day_first for
backward compat):

- ``_enrich_per_lf_gps`` — single entry point called from ``PhaseBCDayFirst.run``.
- ``derive_per_lf_locations`` — used by the post-encoding fallback path
  in ``PhaseBCDayFirst.run``.
- ``_partition_lf_plans_by_date`` — date-partition helper called by
  ``_enrich_per_lf_gps`` (lives here to keep the LF-source-of-truth
  remap colocated with the GPS-slicing helpers that depend on it
  running first).

Tests import several of the private helpers directly via the legacy
``phase_bc_day_first`` import path; those re-exports are preserved.
"""

import logging
from collections import Counter
from types import SimpleNamespace
from typing import Any, Optional

from ..models import DrivePlan, ResolvedLocation
from .ost_validator import validate_ost
from .title_builder import (
    build_lf_title,
    differentiate_sibling_labels,
    dominant_location,
    is_us_state,
    ranked_lf_features,
)

logger = logging.getLogger(__name__)

# Geocode quorum floor: the winning county needs ≥ this many successfully
# reverse-geocoded samples, else the builder degrades (``dominant_*`` stay None)
# rather than letting a minority win the vote.
GEOCODE_QUORUM_MIN = 2

# per-segment A→B endpoint-loss floor (seconds). The journey's start (or
# end) is the first (or last) surviving trackpoint of the segment supplying it;
# when that segment's head (or tail) waypoints have dropped out, that survivor
# sits this many seconds or more inside the segment's requested window, so the
# derived origin/destination is SHIFTED. A gap wider than this is treated as a
# genuine dropout and the endpoint is withheld rather than burned as
# authoritative; below it is normal first-fix latency / sampling jitter (a
# leading null-island / stale-lock run is already corrected by effective_offset,
# so it never reaches this gate).
_ENDPOINT_LOSS_GAP_SEC = 5.0


def _resolve_lf_destination(lf, outcome: Optional[dict] = None) -> Optional[str]:
    """Best-effort destination POI for one LF's montage-peak footage.

    The WIRE half. Resolves once from the LF's featured (montage-peak)
    segments (the top-``composite_score`` clips ``longform_assembler`` picks for
    the highlight montage) via ``destination_resolver.resolve_destination``,
    which reuses the allowlisted chapters Overpass machinery. Failure-tolerant
    by construction: a resolver miss or any exception yields ``None`` so the
    caller's A→B title cascade runs — the destination never raises into the
    pipeline. Returns the destination name or ``None``.

    ``outcome`` (optional out-dict) receives ``"source"`` (``overpass`` / ``none``
    / ``failed``, see ``destination_resolver.resolve_destination``).
    """
    if outcome is not None:
        outcome["source"] = "none"
    peaks = list(getattr(lf, "montage_peaks", None) or [])
    if not peaks:
        return None
    try:
        from .destination_resolver import resolve_destination

        return resolve_destination(peaks, outcome=outcome)
    except Exception as exc:  # noqa: BLE001 — degrade to the A→B cascade
        logger.warning(
            "destination_resolver.resolve_destination failed for LF %d: %s",
            int(getattr(lf, "index", 0)),
            exc,
        )
        if outcome is not None:
            outcome["source"] = "failed"
        return None


def _resolve_short_destination(trimmed_points, *, poi_source=None) -> Optional[str]:
    """Best-effort destination POI for one short's own clip footage.

    The SHORT-side mirror of ``_resolve_lf_destination``. A short is a
    single ~38s clip, not a montage of scattered peaks, so there is no
    ``montage_peaks`` list to vote over. We synthesise ONE featured-segment PER
    trimmed on-screen GPS fix (null-island points dropped), each a unit-score
    sample, then route them through ``destination_resolver.resolve_destination``
    with ``require_sample_quorum=True``. A short shot *at* Donnelley WMA thus
    names the WMA instead of the nearest town.

    Voting the actual fixes (not the clip's MEAN point) closes the
    centroid-of-arc hole — a curved clip's centroid can sit inside a feature the
    footage never entered, so the mean-point containment named a destination the
    clip only bowed around. The quorum requires a minimum fraction of the clip's
    fixes to fall INSIDE the winning polygon, so a grazed feature is not named.

    Failure-tolerant by construction: an empty/degenerate clip, a resolver miss,
    or any exception yields ``None`` so the caller's ``dominant_location``
    cascade runs — the destination never raises into the shorts path.
    ``poi_source`` is injectable so tests stay offline (``None`` → the
    production chapters Overpass query).
    """
    from .title_builder import _point_latlon

    coords = [ll for ll in (_point_latlon(p) for p in (trimmed_points or [])) if ll]
    coords = [(lat, lon) for lat, lon in coords if not (lat == 0.0 and lon == 0.0)]
    if not coords:
        return None
    samples = [
        SimpleNamespace(gps_center_lat=lat, gps_center_lon=lon, composite_score=1.0)
        for lat, lon in coords
    ]
    try:
        from .destination_resolver import resolve_destination

        return resolve_destination(samples, poi_source=poi_source, require_sample_quorum=True)
    except Exception as exc:  # noqa: BLE001 — degrade to the dominant_location cascade
        logger.warning("destination_resolver.resolve_destination failed for short: %s", exc)
        return None


def _coerce_gps_points_to_dicts(points) -> list:
    """Convert bare ``(t, lat, lon)`` tuples → dicts BEFORE quality filtering.
    ``filter_high_quality_points`` reads ``_get(p, "lat")``
    which is ``None`` on a positional tuple, so the null-island/hdop/speed
    gates are a no-op on the tuple path (scorer + shorts-OST trace shape).
    Converting upstream arms the filter; dict/object points pass through. NOTE:
    does NOT edit ``gps_quality.py``.
    """
    out: list = []
    for p in points or []:
        if isinstance(p, (tuple, list)) and len(p) >= 3:
            out.append({"time": p[0], "lat": p[1], "lon": p[2]})
        else:
            out.append(p)
    return out


_UNRESOLVED_SLOTS = frozenset({"", "none", "null", "unknown"})


def region_label(county, state="") -> "str | None":
    """``"County, State"`` for one county vote, or ``None`` when the county slot
    is unresolved — a geocoder miss (e.g. an independent city) is never a
    ``"None, Virginia"`` region. An unresolved state collapses to the bare
    county (never ``"County, None"``)."""
    county = str(county if county is not None else "").strip()
    state = str(state if state is not None else "").strip()
    if county.lower() in _UNRESOLVED_SLOTS:
        return None
    if state.lower() in _UNRESOLVED_SLOTS:
        return county
    return f"{county}, {state}"


def build_resolved_location(points, *, geocoder, withhold_endpoints=None) -> ResolvedLocation:
    """Build ONE canonical :class:`ResolvedLocation` from an LF's GPS trace:
    tuple→dict coerce → quality filter → modal-county vote
    with a geocode quorum floor → ``vote`` + ``regions`` →
    ``is_us``/``country_code`` once → ``road`` (the winning county's road)
    → raw ``endpoints``. Empty when no geocoder or no usable point (callers
    fall back).

    ``withhold_endpoints`` — an iterable subset of ``{"start", "end"}``
    (from :func:`_journey_endpoint_loss_by_lf`) whose A→B endpoint was corrupted
    by a per-segment head/tail GPS dropout. Each named key is OMITTED from
    ``endpoints`` so a coordinate SHIFTED by the loss is never emitted as
    authoritative — the same withhold-by-omission discipline as the quorum gate
    below (a missing key, never a fabricated replacement). ``None`` (the default)
    withholds nothing, so a fully-covered trace behaves exactly as before.

    ``road`` is populated here from ``dom.road`` so callers (notably
    ``_populate_resolved_locations``) never re-run ``dominant_location`` just
    to recover it — a second geocode pass that doubled the quota on uncached
    runs.

    The quorum gate runs BEFORE any geocode-derived evidence is
    populated, so a quorum-miss return carries NO road / place sequence /
    regions / vote — only the trace-derived ``endpoints``. The pre-fix
    "regions/vote kept" contract leaked sub-floor evidence downstream: the
    degraded model was stored (``regions`` non-empty), the title fallback
    used the 1-sample road, and the far-road gate reconciled correct quorum-derived
    titles against a single-flake region set (post-encode hard fail).
    """
    rl = ResolvedLocation()
    filtered = _coerce_gps_points_to_dicts(points)
    if not filtered:
        return rl

    from .gps_quality import filter_high_quality_points
    from .title_builder import _point_latlon

    filtered = filter_high_quality_points(filtered) or filtered
    coords = [ll for ll in (_point_latlon(p) for p in filtered) if ll and ll != (0.0, 0.0)]
    if coords:
        # Withhold an endpoint whose supplying segment lost its head/tail
        # waypoints — the survivor is shifted, so omit the key rather than burn
        # a wrong origin/destination as authoritative.
        _withhold = set(withhold_endpoints or ())
        ep: dict = {}
        if "start" not in _withhold:
            ep["start"] = list(coords[0])
        if "end" not in _withhold:
            ep["end"] = list(coords[-1])
        rl.endpoints = ep
    if geocoder is None or not coords:
        return rl

    dom = dominant_location(filtered, geocoder)
    if dom is None:
        return rl

    # Gate on the COUNT of successfully-geocoded samples, NOT the footage-
    # weighted vote mass (``dom.votes`` weights each sample by its catchment
    # span, so on a long trace a single flaky geocode carries a mass above the
    # floor — the far-road class the quorum exists to block).
    # The gate runs BEFORE road/sequence/regions/vote are populated —
    # sub-floor evidence must not ship on the degraded model (see docstring).
    if dom.geocoded_sample_count < GEOCODE_QUORUM_MIN:
        logger.warning(
            "RESOLVED-LOCATION-QUORUM: winning county %r had only %d geocoded "
            "sample(s) (< %d) — degrading to an evidence-free model "
            "(endpoints only; road/sequence/regions/vote withheld)",
            f"{dom.county}, {dom.state}",
            dom.geocoded_sample_count,
            GEOCODE_QUORUM_MIN,
        )
        return rl

    # The winning county's road (callers must not re-geocode to
    # recover it — that doubled the geocode quota on uncached runs).
    rl.road = dom.road
    # Raw winning-county sample refs for the ladder's via-road support check.
    rl.road_refs = list(getattr(dom, "road_refs", None) or [])
    # The full winning-county road tally, for the provenance sidecar.
    rl.road_votes = dict(getattr(dom, "road_votes", None) or {})

    # route-ordered place sequence (town-level, consecutively deduped)
    # — consumed by the description ``🛤️ Route:`` line + chapter names.
    rl.ordered_place_sequence = list(dom.ordered_places)

    total = sum(dom.votes.values()) or 1  # vote/regions from dom.votes ("county|state")
    for key, n in dom.votes.items():
        county, _, state = key.partition("|")
        label = region_label(county, state)
        if label is None:
            continue  # unresolved county (independent city): no region claim
        rl.vote[label] = round(n / total, 4)
        if label not in rl.regions:
            rl.regions.append(label)

    # Which county each endpoint sits in: lets the location coverage gate map
    # an "A to B" title's towns to counties. A withheld / unresolved endpoint
    # leaves its key absent (never a fabricated label).
    for _end, _ll in (("start", coords[0]), ("end", coords[-1])):
        if _end not in rl.endpoints:
            continue
        try:
            _d = geocoder.reverse_geocode_detailed(_ll[0], _ll[1]) or {}
        except Exception as exc:  # noqa: BLE001 — a geocode miss leaves the key absent
            logger.debug("endpoint_regions: %s geocode failed: %s", _end, exc)
            continue
        _c, _s = _d.get("county"), _d.get("state")
        _s = "" if _s in (None, "None") else _s
        if _c:
            rl.endpoint_regions[_end] = f"{_c}, {_s}" if _s else _c

    rl.dominant_county = dom.county
    rl.dominant_state = dom.state
    if dom.state:
        rl.is_us = is_us_state(dom.state)
    # Carry the GEOCODED country INDEPENDENT of state, so every region
    # surface (playlist/tags) routes off the one ResolvedLocation instead of the
    # label parse (which returns the label's last component — a county/province,
    # never the country). A STATELESS county (Irish county, US independent
    # city) has no state to gate on but its country is still known — set it
    # whenever a country resolves, never drop it. ``is_us`` stays False without a
    # US state, so a degraded/absent geocoder (no country) still never flips
    # non-US → US.
    rl.country_code = "US" if rl.is_us else (dom.country or None)
    return rl


def validate_burned_ost_text(canonical: Optional[str], lf_location: str, lf_idx: int) -> str:
    """Gate a ``build_lf_title`` result before it becomes burned OST pixels.

    The per-LF OST text was the one label surface with NO validator between
    geocoder output and the irreversible encode — ``validate_ost`` ran only on
    the episode-OST log line and the drive fallback (``phase_a``), never on the
    title that actually burns. Run the same garbage checks here (broken
    ``None``/``null``, coordinate strings, pure digits, HOA/subdivision
    leakage, placeholder literals) under the 100-char ``lf_title`` length
    budget. On failure log a greppable ``OST-GATE`` WARN and fall back to the
    clean ``per_lf_location`` base rather than burning garbage. Robust-by-
    construction is the span guard in ``title_builder``; this is the alarm.
    """
    text = canonical or lf_location
    ok, reason = validate_ost(text, "lf_title")
    if not ok:
        logger.warning(
            "OST-GATE: invalid burned OST text for LF %d: %r (%s) — "
            "falling back to per_lf_location %r",
            lf_idx,
            text,
            reason,
            lf_location,
        )
        return lf_location
    return text


def _differentiate_and_revalidate_siblings(
    per_lf_ost_text_by_date: dict,
    gps_points_by_lf: dict,
    geocoder,
) -> None:
    """DETERMINISTIC sibling-LF differentiation, RE-VALIDATED.

    When ≥2 LFs collapse to the same burned label (the no-settlement ACE-Basin
    class), differentiate each by its modal named feature via a mechanical vote
    instead of the "(Part N of M)" filler.

    ``differentiate_sibling_labels`` inserts a feature into the label, but
    that mutated string never passed ``validate_burned_ost_text`` (it ran on the
    pre-diff label only) — a malformed feature could burn ungated. Re-validate
    each changed label against its already-validated original, so a bad feature
    degrades to the clean region label. Mutates ``per_lf_ost_text_by_date`` in
    place.
    """
    _flat: dict[int, str] = {}
    for _day in per_lf_ost_text_by_date.values():
        _flat.update(_day)
    _collisions = {lbl for lbl, n in Counter(_flat.values()).items() if n > 1 and lbl}
    if not _collisions:
        return
    _features = {
        idx: ranked_lf_features(gps_points_by_lf.get(idx), geocoder)
        for idx, lbl in _flat.items()
        if lbl in _collisions and gps_points_by_lf.get(idx)
    }
    _diff = differentiate_sibling_labels(_flat, _features)
    for _day in per_lf_ost_text_by_date.values():
        for idx in list(_day):
            if idx not in _diff or _diff[idx] == _day[idx]:
                continue
            # Fall back to the original (already-validated) label on failure.
            _revalidated = validate_burned_ost_text(_diff[idx], _day[idx], idx)
            if _revalidated != _day[idx]:
                logger.info(
                    "sibling-diff: LF %d %r -> %r",
                    idx,
                    _day[idx],
                    _revalidated,
                )
            _day[idx] = _revalidated


def canonical_lf_ost_text(
    lf: Any,
    *,
    lf_location: str,
    overrides: dict,
    geocoder: Any,
    gps_points: Any,
    destination: Optional[str],
) -> str:
    """One LF's canonical burn text — the string its OST, thumbnail and YouTube
    title baseline derive from. The render (``_enrich_per_lf_gps``) and
    ``--label-dry-run`` both call this, so the dry-run cannot drift from it.
    """
    lf_idx = int(getattr(lf, "index", 0))
    # Operator override is AUTHORITATIVE — burn it verbatim (the
    # documented ``derive_per_lf_locations`` contract: "overrides[idx]
    # wins over centroid-derived labels"). Without this, an explicit
    # ``--per-lf-location`` on a GPS-BEARING LF was silently reshaped by
    # build_lf_title's GPS cascade (e.g. "Saluda to Lake Toxaway" came
    # out "Quebec to Toxaway Falls"). A GPS-less LF already routed its
    # override through the cascade's county fallback; this makes the
    # behaviour uniform and predictable for both. The override also wins
    # over the destination anchor — the operator was explicit.
    _override = overrides.get(lf_idx)
    if isinstance(_override, str) and _override.strip():
        canonical = _override.strip()
    else:
        try:
            canonical = build_lf_title(
                list(getattr(lf, "segments", None) or []),
                geocoder=geocoder,
                per_lf_location=lf_location,
                gps_points=gps_points,
                resolved_destination=destination,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "title_builder.build_lf_title failed for LF %d "
                "(pre-run_all_days OST text derivation): %s — "
                "falling back to per_lf_location string",
                lf_idx,
                exc,
            )
            canonical = lf_location
    return validate_burned_ost_text(canonical, lf_location, lf_idx)


def _enrich_per_lf_gps(
    *,
    state,
    real_plan: Optional[DrivePlan],
    day_batches,
    episode_files,
    per_day_locations: dict,
    drive_wide_location: str,
    gps_cache_dir: Optional[str] = None,
) -> SimpleNamespace:
    """Pre-encoding per-LF GPS / location / OST setup.

    Extracts the pre-``run_all_days`` GPS-and-OST setup that used to
    live inline in ``PhaseBCDayFirst.run``. Behaviour
    is identical to the inlined sequence; only the surface changes.

    Steps (in order, matching the original inline flow):

    1. **Per-LF raw-GPS slicing:** slice
       ``state["gps_per_episode"]`` per LF BEFORE
       ``_partition_lf_plans_by_date`` remaps each segment's
       ``source_file_index`` from GLOBAL to DAY-LOCAL. The LF
       ``.index`` is not mutated by the partition, so a global
       ``{lf_index: trackpoints}`` dict is safe to thread into the
       derive calls below. Without this, the
       ``filter_high_quality_points`` branch in
       ``derive_per_lf_locations`` is dead in production.
    2. **Per-LF GPX path snapshot (chapter-generation dataflow fix):**
       Phase D's chapter generator needs each LF's GLOBAL source GPX
       index, but ``_partition_lf_plans_by_date`` remaps the index to
       day-local. Snapshotting ``{lf.index: gpx_path}`` here while
       indices are still global lets Phase D look up the right GPX
       per LF on multi-day drives. Sets ``state["per_lf_gpx_path"]``
       as a side effect (preserves the original semantics).
    3. **Day-partition LF plans:** delegate to
       ``_partition_lf_plans_by_date`` — empty dict when ``real_plan``
       is None.
    4. **Geocoder + overrides + validation gate:** one
       shared ``NominatimRoadsClient`` session is built here and
       reused by every downstream caller in this phase (derive,
       title builder fallback, shorts OST closure, regions). The
       post-discovery validation gate
       ``_assert_per_lf_locations_resolvable`` refuses to enter
       encoding when an LF has neither centroid-derivable GPS nor an
       operator override.
    5. **Per-LF location labels:** county-level labels via
       ``derive_per_lf_locations``, keyed by day.
    6. **Canonical per-LF OST text:** the
       ``"{start} to {end} via {road}, {state}"`` form via
       ``build_lf_title``, snapshotted BEFORE
       ``ensure_unique_titles`` — the ``"(Part N of M)"`` dedup
       suffix belongs on YouTube titles only, not in burned OST
       pixels.

    Returns a namespace with: ``gps_points_by_lf``,
    ``lf_plans_by_date``, ``geocoder``, ``per_lf_overrides``,
    ``per_lf_locations_by_date``, ``per_lf_ost_text_by_date``,
    ``destination_by_lf`` (the once-resolved destination POI per LF
    index, reused by ``_populate_resolved_locations``) — every field
    consumed by the encoding call and the post-encoding fallback derive
    in ``PhaseBCDayFirst.run``.
    """
    gps_points_by_lf = _build_gps_points_by_lf(
        real_plan,
        state.gps_per_episode,
    )
    # Flag any LF whose A→B start/end is shifted by a per-segment head/tail
    # GPS dropout, computed here (indices still GLOBAL, BEFORE the day-partition
    # remap — same constraint as _build_gps_points_by_lf) and threaded to
    # _populate_resolved_locations so the builder withholds the shifted endpoint.
    endpoint_loss_by_lf = _journey_endpoint_loss_by_lf(
        real_plan,
        state.gps_per_episode,
    )

    if real_plan is not None:
        per_lf_gpx_path = _build_per_lf_gpx_path_map(
            real_plan,
            state.gpx_files or [],
        )
        # ALWAYS assign — the map is total across both selection
        # modes (best_moments + full_drive) and GPS modes (with/without
        # gpx_files). Empty result only happens when drive_plan itself
        # has no regular_lfs (zero-LF drive), in which case the
        # phase_bc cross-field invariant is already in the empty-LF
        # legitimate-path branch.
        if per_lf_gpx_path:
            state.per_lf_gpx_path = per_lf_gpx_path
        # Additive per-episode span map, built here (indices still
        # GLOBAL) for the SAME reason as per_lf_gpx_path above. Lets the
        # post-encoding distance/elevation computes sum across every episode an
        # LF spans instead of only its single representative episode.
        per_lf_gpx_episode_paths = _build_per_lf_gpx_episode_map(
            real_plan,
            state.gpx_files or [],
        )
        if per_lf_gpx_episode_paths:
            state.per_lf_gpx_episode_paths = per_lf_gpx_episode_paths

    lf_plans_by_date = (
        _partition_lf_plans_by_date(
            real_plan,
            episode_files=episode_files,
            episode_dates=state.episode_dates or [],
            day_batches=day_batches,
        )
        if real_plan is not None
        else {}
    )

    geocoder = _build_shared_geocoder(cache_dir=gps_cache_dir)
    per_lf_overrides: dict[int, str] = dict(getattr(state, "per_lf_overrides", None) or {})
    _assert_per_lf_locations_resolvable(
        lf_plans_by_date or {},
        gps_points_by_lf=gps_points_by_lf or {},
        overrides=per_lf_overrides,
    )
    per_lf_locations_by_date: dict[str, dict[int, str]] = {}
    for date_key, day_real_lfs in (lf_plans_by_date or {}).items():
        per_lf_locations_by_date[date_key] = derive_per_lf_locations(
            day_real_lfs or [],
            geocoder=geocoder,
            gps_fallback=None,
            overrides=per_lf_overrides,
            gps_points_by_lf=gps_points_by_lf or None,
        )

    per_lf_ost_text_by_date: dict[str, dict[int, str]] = {}
    # Destination POI per LF, resolved ONCE from each LF's montage-peak
    # footage and reused by ``_populate_resolved_locations`` (computed-once
    # invariant). Threaded into ``build_lf_title`` below so the destination
    # anchors the burned OST (the single source of truth for title/thumbnail).
    destination_by_lf: dict[int, str] = {}
    for date_key, day_real_lfs in (lf_plans_by_date or {}).items():
        day_labels = per_lf_locations_by_date.get(date_key, {})
        day_ost_text: dict[int, str] = {}
        for lf in day_real_lfs or []:
            lf_idx = int(getattr(lf, "index", 0))
            lf_location = day_labels.get(
                lf_idx, per_day_locations.get(date_key, drive_wide_location)
            )
            destination = _resolve_lf_destination(lf)
            if destination:
                destination_by_lf[lf_idx] = destination
            day_ost_text[lf_idx] = canonical_lf_ost_text(
                lf,
                lf_location=lf_location,
                overrides=per_lf_overrides,
                geocoder=geocoder,
                gps_points=gps_points_by_lf.get(lf_idx),
                destination=destination,
            )
        per_lf_ost_text_by_date[date_key] = day_ost_text

    _differentiate_and_revalidate_siblings(per_lf_ost_text_by_date, gps_points_by_lf, geocoder)
    # The label may claim only what the RETAINED footage's own GPS
    # proves — checked after planning, before the burn (route_claims).
    from .boundary_proof import CensusBoundaryProver
    from .final_text_record import boundary_review_available
    from .ost_system_review import record_system_reviews
    from .route_claims import enforce_route_claims

    # Keep what the check decided (refusals included, written before
    # the raise) as the system record the opt-in OST gate signal reads.
    claim_outcomes: dict = {}
    try:
        route_claims_by_lf = enforce_route_claims(
            per_lf_ost_text_by_date,
            _build_route_windows_by_lf(real_plan, state.gps_per_episode),
            geocoder,
            overrides=per_lf_overrides,
            outcomes=claim_outcomes,
            boundary_prover=CensusBoundaryProver(),
            require_review_support=True,
            review_support_available=boundary_review_available(),
        )
    finally:
        record_system_reviews(state, per_lf_ost_text_by_date, claim_outcomes)
    # Persist the evidence the final-text review is built from (Phase D)
    # and re-verified against at upload.
    from .final_text_record import route_record

    state.per_lf_route_record = {
        idx: route_record(ev) for idx, ev in route_claims_by_lf.items()
    } or None

    return SimpleNamespace(
        gps_points_by_lf=gps_points_by_lf,
        lf_plans_by_date=lf_plans_by_date,
        geocoder=geocoder,
        per_lf_overrides=per_lf_overrides,
        per_lf_locations_by_date=per_lf_locations_by_date,
        per_lf_ost_text_by_date=per_lf_ost_text_by_date,
        destination_by_lf=destination_by_lf,
        endpoint_loss_by_lf=endpoint_loss_by_lf,
        route_claims_by_lf=route_claims_by_lf,
    )


def _build_shared_geocoder(cache_dir: Optional[str] = None):
    """Construct a NominatimRoadsClient for reverse-geocoding per-LF
    centroids. Returns None when the gps_journey dependency isn't
    importable — callers MUST handle None gracefully (by falling back
    to the day-wide label).

    ``cache_dir`` is the geocoding cache directory; when ``None`` (legacy
    callers / tests), it falls back to the ONE canonical namespace
    (``DriveConfig.resolved_geocode_cache_dir``) shared with Phase A and
    Phase D. ``~`` is expanded here at the read site so callers can pass
    the raw config string unchanged.
    """
    try:
        from pathlib import Path as _Path

        from gps_journey.geocoding.cache import GeocodingCache
        from gps_journey.geocoding.nominatim_roads import NominatimRoadsClient

        if cache_dir is None:
            # Lazy import to avoid a hard config dep when this helper is
            # called from a test that doesn't construct a Config.
            from gomoveshift.config import DriveConfig

            cache_dir = DriveConfig().resolved_geocode_cache_dir()
        cache_dir_path = _Path(cache_dir).expanduser()
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        cache = GeocodingCache(
            cache_path=cache_dir_path / "geocoding_cache.json",
            ttl_days=90,
        )
        return NominatimRoadsClient(
            user_agent="gomoveshift-video/0.1.0",
            cache=cache,
            rate_limit=1.0,
        )
    except Exception as exc:  # noqa: BLE001 — dependency optional at test time
        logger.debug("Shared geocoder unavailable: %s", exc)
        return None


def _centroid_for_segments(segments) -> tuple[float, float]:
    """Return the mean (lat, lon) across segments with non-zero GPS
    fields, or (0.0, 0.0) when no segment has GPS data."""
    lats: list[float] = []
    lons: list[float] = []
    for seg in segments or []:
        lat = float(getattr(seg, "gps_center_lat", 0.0) or 0.0)
        lon = float(getattr(seg, "gps_center_lon", 0.0) or 0.0)
        if lat == 0.0 and lon == 0.0:
            continue
        lats.append(lat)
        lons.append(lon)
    if not lats:
        return 0.0, 0.0
    return sum(lats) / len(lats), sum(lons) / len(lons)


def _segment_centroid_points(segments) -> list[tuple[float, float]]:
    """Return the list of non-zero ``(lat, lon)`` segment centroids.

    The resume centroid-fallback must VOTE the modal county across the
    individual segment centroids, NOT geocode their arithmetic MEAN. On a
    drive crossing a county/state line the mean lands on the wrong side
    (the far-road bug), whereas voting keeps the
    label on the county holding the plurality of segments. Each centroid is a
    bare ``(lat, lon)`` pair — ``dominant_location``/``_point_latlon`` accept
    that shape directly.
    """
    out: list[tuple[float, float]] = []
    for seg in segments or []:
        lat = float(getattr(seg, "gps_center_lat", 0.0) or 0.0)
        lon = float(getattr(seg, "gps_center_lon", 0.0) or 0.0)
        if lat == 0.0 and lon == 0.0:
            continue
        out.append((lat, lon))
    return out


def _build_gps_points_by_date(
    day_batches,
    episode_files,
    gps_per_episode,
) -> dict[str, list[tuple[float, float, float]]]:
    """Project raw GPS trackpoints into each day's master-video timeline.

    Returns ``{date: [(t_master, lat, lon), ...]}`` — tuple shape keeps
    the existing ``trim_gps_to_clip_timeline`` default time-accessor
    working (it reads ``point[0]``), and matches the expectation of
    ``short_ost_text_fn`` below which reads ``(t, lat, lon)``.

    The day master is a stream-copy concat of the day's episodes in
    order (see ``segment_extractor.compute_day_file_boundaries``), so
    an episode-local second ``t_local`` maps to master seconds
    ``ep_master_start + t_local``. Trackpoint ``.timestamp`` values are
    in GPX-absolute seconds; episode-local = ``timestamp - gps_offset``
    (matches the convention in ``longform_assembler``).

    Episodes without GPS (missing entry, ``GPSData is None``, or empty
    trackpoints) contribute nothing; days with no GPS on any episode
    yield an empty list and are omitted from the result (caller passes
    the dict as ``gps_points_by_date or None`` so the shorts loop sees
    ``None`` and skips the trim branch).

    Without this helper, no caller of
    ``run_all_days`` built the kwarg, so the GPS trim was a no-op.
    """
    if not day_batches or not gps_per_episode:
        return {}

    # Path → drive-episode-index map (same idiom as run_all_days).
    idx_by_path: dict[str, int] = {}
    for i, path in enumerate(episode_files or []):
        idx_by_path[str(path)] = i

    result: dict[str, list[tuple[float, float, float]]] = {}
    for batch in day_batches:
        date = getattr(batch, "date", None)
        if not date:
            continue
        episodes = list(getattr(batch, "episodes", None) or [])
        if not episodes:
            continue
        # Per-episode day-master offsets (same computation the day
        # pipeline itself performs). Running cursor, not an index into
        # file_boundaries, because file_boundaries is drive-wide.
        day_cursor = 0.0
        day_points: list[tuple[float, float, float]] = []
        for ep in episodes:
            ep_master_start = day_cursor
            duration = float(getattr(ep, "total_duration", 0.0) or 0.0)
            day_cursor += duration
            ep_idx = idx_by_path.get(str(getattr(ep, "concat_path", "")))
            if ep_idx is None or ep_idx < 0 or ep_idx >= len(gps_per_episode):
                continue
            entry = gps_per_episode[ep_idx]
            if not entry:
                continue
            gps_data = entry[0]
            gps_offset = float(entry[1]) if len(entry) > 1 and entry[1] is not None else 0.0
            if gps_data is None:
                continue
            # Correct a cold-start offset BEFORE the window
            # test below (raw offset lands every t_local out of range).
            from ..analysis.shorts_models import effective_offset as _effective_offset

            gps_offset = _effective_offset(gps_data, gps_offset)
            # JSON-roundtrip reconstitution (some resume paths store
            # GPSData as a dict — mirror longform_assembler's handling).
            trackpoints = getattr(gps_data, "trackpoints", None)
            if trackpoints is None and isinstance(gps_data, dict):
                trackpoints = gps_data.get("trackpoints") or []
            if not trackpoints:
                continue
            ep_points_before = len(day_points)
            for tp in trackpoints:
                ts = getattr(tp, "timestamp", None)
                lat = getattr(tp, "lat", None)
                lon = getattr(tp, "lon", None)
                if ts is None and isinstance(tp, dict):
                    ts = tp.get("timestamp")
                    lat = tp.get("lat")
                    lon = tp.get("lon")
                if ts is None or lat is None or lon is None:
                    continue
                t_local = float(ts) - gps_offset
                if t_local < 0 or t_local > duration:
                    # Trackpoint falls outside this episode's window —
                    # GPX covers a broader range than the clipped file,
                    # which is expected. Skip.
                    continue
                day_points.append((ep_master_start + t_local, float(lat), float(lon)))
            # An episode with trackpoints but ZERO in-window points (even
            # after the corrected offset) was silently dropped — WARN instead.
            if len(day_points) == ep_points_before:
                logger.warning(
                    "GPS-WINDOW-EMPTY: ep %d gave 0 in-window points to date %s "
                    "(offset=%.1f) — shorts-OST trace may be empty",
                    ep_idx,
                    date,
                    gps_offset,
                )
        if day_points:
            result[date] = day_points
    return result


def _build_drive_gps_points(
    file_boundaries,
    gps_per_episode,
) -> list[tuple[float, float, float]]:
    """Project every episode's GPS trackpoints into DRIVE-master seconds.

    ``[(t_drive, lat, lon), ...]`` — the drive-wide analog of
    ``_build_gps_points_by_date`` (which is day-master). Episode ``i`` occupies
    ``file_boundaries[i] = (start, end)`` on the scorer's drive timeline
    (``state.drive_scores``), so an episode-local second
    ``t_local = timestamp - effective_offset`` lands at ``start + t_local``.
    Out-of-window points are skipped (a sidecar GPX covers more than the
    clip). Empty when no episode carries GPS. Consumed by the day-first drive
    plan's full_drive stop mask (``phase_bc_day_first._build_drive_stop_mask``).
    """
    if not file_boundaries or not gps_per_episode:
        return []
    from ..analysis.shorts_models import effective_offset as _effective_offset

    points: list[tuple[float, float, float]] = []
    for i, (fb_start, fb_end) in enumerate(file_boundaries):
        if i >= len(gps_per_episode):
            break
        entry = gps_per_episode[i]
        if not entry or entry[0] is None:
            continue
        gps_data = entry[0]
        gps_offset = float(entry[1]) if len(entry) > 1 and entry[1] is not None else 0.0
        gps_offset = _effective_offset(gps_data, gps_offset)
        trackpoints = getattr(gps_data, "trackpoints", None)
        if trackpoints is None and isinstance(gps_data, dict):
            trackpoints = gps_data.get("trackpoints") or []
        duration = float(fb_end) - float(fb_start)
        for tp in trackpoints or []:
            ts = getattr(tp, "timestamp", None)
            lat = getattr(tp, "lat", None)
            lon = getattr(tp, "lon", None)
            if ts is None and isinstance(tp, dict):
                ts = tp.get("timestamp")
                lat = tp.get("lat")
                lon = tp.get("lon")
            if ts is None or lat is None or lon is None:
                continue
            t_local = float(ts) - gps_offset
            if t_local < 0 or t_local > duration:
                continue
            points.append((float(fb_start) + t_local, float(lat), float(lon)))
    return points


def _make_short_ost_text_fn(geocoder, *, poi_source=None, places_fn=None):
    """Build the ``short_ost_text_fn`` closure consumed by run_day_pipeline.

    Signature: ``(trimmed_points, short_range) -> Optional[str]``.
    ``trimmed_points`` are the tuples produced by
    ``_build_gps_points_by_date`` and filtered through
    ``trim_gps_to_clip_timeline`` — i.e. the GPS fixes on-screen during
    the short's day-master window.

    Destination-first: the clip's own footage is routed through
    ``_resolve_short_destination`` so a short shot *at* a named destination
    (a WMA / reserve / protected-area / viewpoint) names that destination
    instead of the nearest town — mirroring the LF surface. The destination
    is only consulted when the county vote met the geocode quorum (``dom``
    resolved): a quorum-miss clip must never burn a bare destination the
    short-title region gate cannot corroborate (``short_regions_fn`` returns
    ``[]`` off the same miss). When a destination resolves it is the
    preferred label (state appended from the voted county when available);
    otherwise the existing behaviour stands:
    the modal (county, state) is voted across the trimmed points via
    ``dominant_location`` rather than reverse-geocoding the single mean
    centroid — a short spanning a county/state line would otherwise inherit
    whichever county the mean lands in. Empty trimmed input or geocoder
    failure returns ``None`` so the per-day call falls back to the drive-wide
    ``location`` string. ``poi_source`` is injectable so tests stay offline.

    Trace rule: a voted settlement name (``dom.name``) is the reverse
    geocoder's address claim, which can name a hamlet kilometres off the
    road. With ``places_fn`` (forward settlement lookup ``query -> [(name,
    lat, lon)]``) the name burns only when a settlement of exactly that name
    lies within the LF surface's trace distance of one of the clip's own
    fixes; otherwise — or when the lookup fails — the short burns the voted
    county, which the clip's own GPS does corroborate. Without ``places_fn``
    the vote stands unchecked.
    """

    def _fn(trimmed_points, short_range):
        pts = list(trimmed_points or [])
        if not pts:
            return None
        # Coerce positional (t, lat, lon) tuples → dicts BEFORE the vote so
        # the GPS quality filter inside dominant_location reads lat/lon/time by
        # key instead of no-opping on the bare-tuple shape (mirrors
        # build_resolved_location, which coerces once at :141).
        pts = _coerce_gps_points_to_dicts(pts)
        try:
            dom = dominant_location(pts, geocoder, min_votes=GEOCODE_QUORUM_MIN)
        except Exception as exc:  # noqa: BLE001 — OST derivation never fails a short
            # An exception degrade must be LOUD — the caller silently
            # inherits parent-LF/drive-wide text when this returns None.
            logger.warning(
                "short_ost_text_fn: dominant_location failed for short rank %s: %s",
                getattr(short_range, "rank", "?"),
                exc,
            )
            dom = None
        # Prefer the clip's own destination POI over the bare town/county —
        # but ONLY when the county vote corroborates it (``dom`` carries the
        # geocode quorum). On a quorum miss / geocoder failure a bare
        # destination would burn with ZERO GPS corroboration, and the short-title
        # region gate downstream no-ops because ``short_regions_fn`` returns
        # ``[]`` off the SAME quorum miss — so degrade to the caller fallback
        # instead of shipping ungated text.
        destination = (
            _resolve_short_destination(pts, poi_source=poi_source) if dom is not None else None
        )
        if destination:
            return f"{destination}, {dom.state}" if dom.state else destination
        if dom is None:
            # quorum-miss/vote-less degrade — WARN with the rank instead
            # of silently handing the caller its inherited-text fallback.
            logger.warning(
                "short_ost_text_fn: short rank %s has no dominant location "
                "(geocode quorum miss or no county votes) — degrading to caller fallback",
                getattr(short_range, "rank", "?"),
            )
            return None
        primary = dom.name or dom.county
        if primary and dom.name and dom.county and places_fn is not None:
            if not _settlement_on_clip_trace(dom, pts, places_fn, short_range):
                primary = dom.county
        if not primary:
            return None
        return f"{primary}, {dom.state}" if dom.state else primary

    # Expose the shared cached geocoder so the shorts planner's county-line veto
    # reuses it instead of building another.
    _fn.geocoder = geocoder
    return _fn


def _settlement_on_clip_trace(dom, pts, places_fn, short_range) -> bool:
    """True when a settlement named exactly ``dom.name`` lies within
    ``_TRACE_VERIFIED_PLACE_KM`` of one of the short's own fixes.

    Same forward-lookup + trace rule the LF surface uses for an endpoint town
    (``content_gates.trace_verified_endpoint_towns``), scoped to the voted
    county. No timed fixes or a failing geocoder: unverified (fail closed).
    """
    from . import gps_quality as gq
    from .validation.content_gates import _TRACE_VERIFIED_PLACE_KM, _nearest_same_name_km

    rank = getattr(short_range, "rank", "?")
    scope = ", ".join(s for s in (dom.county, dom.state) if s)
    query = f"{dom.name}, {scope}" if scope else dom.name
    window = gq.trace_time_window(pts)
    nearest = None
    if window is not None:
        try:
            nearest = _nearest_same_name_km(dom.name, query, pts, window, places_fn)
        except Exception as exc:  # noqa: BLE001 — unverified, never a crash
            logger.warning(
                "SHORT-PLACE-UNVERIFIED: rank %s %r — geocoder failed (%s)", rank, query, exc
            )
    if nearest is not None and nearest <= _TRACE_VERIFIED_PLACE_KM:
        return True
    logger.warning(
        "SHORT-PLACE-UNVERIFIED: rank %s %r — no settlement of that name within %g km "
        "of the clip's GPS (nearest: %s); burning the county %r instead",
        rank,
        dom.name,
        _TRACE_VERIFIED_PLACE_KM,
        "none" if nearest is None else f"{nearest:.2f} km",
        dom.county,
    )
    return False


def _assert_per_lf_locations_resolvable(
    lf_plans_by_date: dict,
    gps_points_by_lf: Optional[dict] = None,
    overrides: Optional[dict] = None,
) -> None:
    """Post-discovery, pre-encode validation gate.

    Refuses to enter encoding when an LF has neither a centroid-
    derivable GPS source nor an operator override. Single chokepoint
    that replaces the deleted drive-wide fallback path in
    ``derive_per_lf_locations``.

    A "centroid-derivable" LF has at least one segment with non-zero
    ``gps_center_lat``/``gps_center_lon`` OR a non-empty entry in
    ``gps_points_by_lf``. Operator overrides cover any LF whose index is
    a key in ``overrides``.

    The empty-``lf_plans_by_date`` case (plan build failed → ``real_plan``
    None) is handled UPSTREAM by ``_assert_drive_plan_built`` so this
    gate iterating nothing can no longer silently pass a drive-wide-fallback
    encode.

    Raises:
        ValueError: With the missing LF indices and an example
            ``--per-lf-location`` invocation.
    """
    overrides = overrides or {}
    gps_points_by_lf = gps_points_by_lf or {}
    missing: list[int] = []
    for _date, day_lfs in (lf_plans_by_date or {}).items():
        for lf in day_lfs or []:
            idx = int(getattr(lf, "index", 0))
            if idx in overrides and (overrides[idx] or "").strip():
                continue
            # GPS check: any segment with a non-zero centroid OR raw
            # trackpoints recorded for this LF.
            has_gps = False
            if gps_points_by_lf.get(idx):
                has_gps = True
            else:
                for seg in getattr(lf, "segments", None) or []:
                    lat = float(getattr(seg, "gps_center_lat", 0.0) or 0.0)
                    lon = float(getattr(seg, "gps_center_lon", 0.0) or 0.0)
                    if lat != 0.0 or lon != 0.0:
                        has_gps = True
                        break
            if not has_gps:
                missing.append(idx)
    if missing:
        sample = " ".join(f"--per-lf-location '{i}=<route>'" for i in sorted(missing))
        raise ValueError(
            "phase_bc: cannot enter encoding — LF index(es) "
            f"{sorted(missing)} have no usable GPS and no "
            "--per-lf-location override. Re-invoke with "
            "--list-lfs to see all indices, then add overrides: "
            f"{sample}"
        )


def derive_per_lf_locations(
    regular_lfs: list,
    geocoder=None,
    gps_fallback=None,
    overrides: Optional[dict] = None,
    gps_points_by_lf: Optional[dict] = None,
) -> dict:
    """For each LF, compute a unique location label.

    This function is now TOTAL. Inputs are **per-LF GPS centroids**
    (computed inside) **OR** an operator-supplied ``overrides`` map.
    There is no drive-wide fallback — the prior ``day_fallback``
    parameter has been removed because the post-discovery validation
    gate at PhaseBC entry refuses to enter encoding when an LF has
    neither GPS nor an override.

    Args:
        regular_lfs: Sequence of LF-plan-like objects with `.segments`.
            Each segment must expose `gps_center_lat`/`gps_center_lon`.
        geocoder: A NominatimRoadsClient-like instance exposing
            `reverse_geocode_detailed(lat, lon)` returning a dict with
            county/state fields (dominant-county vote) and a `"road"`
            field for disambiguation.
        gps_fallback: Optional callable `(seg) -> (lat, lon)` used when
            a segment's own centroid is (0, 0). Not required.
        overrides: Optional ``{lf_index: label}`` operator overrides.
            When present, ``overrides[idx]`` wins over centroid-derived
            labels for that LF.

    Returns a dict `{lf_index: label}` with non-empty labels. Centroid-derived
    labels are made distinct; operator overrides are surfaced VERBATIM and may
    repeat across sibling LFs (the operator was explicit — title uniqueness is
    handled by ``ensure_unique_titles``' ``(Part N of M)``).

    Stateless county case: a stateless county's label (Irish counties,
    US independent cities — no ``state``) carries the geocoded ``country``
    as its tail (``"County Cork, Ireland"``), never the bare county alone.
    ``build_lf_title`` derives its settlement-anchor/via-road/A-to-B cascade
    gate (``region_label``) from the label's tail after the first comma; a
    bare county has no comma, so ``region_label`` came back empty and every
    cascade branch was skipped — collapsing straight to the raw-label fallback
    even when a recognizable place (``ordered_place_sequence``) and a
    unanimous road were both available.

    Raises:
        ValueError: When an LF has neither a usable GPS centroid nor an
            override entry, or its county vote fails the geocode quorum
            (no bare-centroid fallback) — the function is TOTAL and
            refuses to silently substitute a fallback label. The CLI gate
            upstream of this call should make the raise unreachable for
            legitimate runs; the raise is a defensive backstop.
    """
    # Filter noisy / stationary GPS before centroid reducer when
    # raw points are supplied per-LF. Keeps centroids from drifting into
    # neighboring towns.
    from .gps_quality import filter_high_quality_points
    from .title_builder import _centroid_from_gps_points

    results: dict[int, str] = {}
    used_labels: set[str] = set()
    override_idxs: set[int] = set()  # LFs whose label is a verbatim operator override
    overrides = overrides or {}
    missing_indices: list[int] = []

    for i, lf in enumerate(regular_lfs, start=1):
        idx = int(getattr(lf, "index", i))
        segments = list(getattr(lf, "segments", []) or [])

        # Operator override wins when present.
        override_label = overrides.get(idx)
        if isinstance(override_label, str):
            override_label = override_label.strip()
        if override_label:
            # Operator override is AUTHORITATIVE — surfaced verbatim.
            # Sibling LFs deliberately sharing one label (e.g. two parts of one
            # continuous GPS-less drive) keep that EXACT label on the location
            # surface; YouTube-title uniqueness is handled separately by
            # ensure_unique_titles' "(Part N of M)". Do NOT append a "(Part N)"
            # suffix here — it leaked into the description's "📍 Location:" line.
            label = override_label
            used_labels.add(label)
            results[idx] = label
            override_idxs.add(idx)
            continue

        lat: float = 0.0
        lon: float = 0.0
        if gps_points_by_lf and idx in gps_points_by_lf:
            filtered = filter_high_quality_points(gps_points_by_lf[idx] or [])
            centroid = _centroid_from_gps_points(filtered)
            if centroid is not None:
                lat, lon = centroid
        if lat == 0.0 and lon == 0.0:
            lat, lon = _centroid_for_segments(segments)
        if (lat == 0.0 and lon == 0.0) and gps_fallback is not None:
            try:
                lat, lon = gps_fallback(segments[len(segments) // 2]) if segments else (0.0, 0.0)
            except Exception as exc:  # noqa: BLE001 — log + fall through
                logger.debug("gps_fallback raised for LF %d: %s", idx, exc)
                lat, lon = 0.0, 0.0

        if (lat == 0.0 and lon == 0.0) or geocoder is None:
            # No drive-wide fallback. Collect the
            # missing-input LF for a single aggregated raise after the
            # loop completes — operators get every missing index in
            # one error, not a sequence of one-off failures.
            missing_indices.append(idx)
            continue
        # Cross-county-line robustness: vote the modal county across the
        # LF's sampled GPS points instead of trusting the single centroid
        # geocode. A boundary-crossing drive's centroid lands in the wrong
        # county (SC: Beaufort/Fripp vs Colleton/Edisto); the vote keeps
        # this county-level label on the county holding the plurality of
        # footage. When no per-LF point trace is available the vote runs
        # over the per-segment centroids instead.
        if gps_points_by_lf and gps_points_by_lf.get(idx):
            # Fresh path: vote over the rebuilt per-LF raw trace.
            vote_points = gps_points_by_lf[idx]
        else:
            # Resume path: ``gps_points_by_lf`` round-trips empty (the raw
            # trace lives in ``gps_per_episode`` and is rebuilt by
            # ``_build_gps_points_by_lf`` at Phase-BC entry; when that yields
            # nothing for this LF we still must NOT geocode the bare-mean
            # centroid). Vote the modal county across the individual segment
            # centroids — the same dominant-county reducer the fresh path uses —
            # so a county-line-crossing LF lands on the plurality county instead
            # of the wrong-side arithmetic mean.
            vote_points = _segment_centroid_points(segments)
        # Geocode quorum floor. The bare-centroid geocode
        # fallback is DELETED (a centroid lands on the wrong side of a
        # county line — the SC Colleton case): a quorum-degraded or
        # vote-less LF joins the aggregated ambiguous-input raise below.
        dom = (
            dominant_location(vote_points, geocoder, min_votes=GEOCODE_QUORUM_MIN)
            if vote_points
            else None
        )
        if dom is None or not dom.county:
            missing_indices.append(idx)
            continue
        # Stateless county case: a stateless county (Irish counties, US independent
        # cities) with NO tail after the county leaves build_lf_title's
        # ``region_label`` empty ("County Cork" has no comma to split on),
        # which gates OFF every settlement-anchor/via-road/A-to-B branch in
        # the cascade and collapses straight to the Level-5 raw-label
        # fallback — a recognizable place in ``ordered_place_sequence``
        # (e.g. "West Cork") and a unanimous road never get a chance to
        # anchor the title, even when both are available. Append the
        # geocoded country (already carried on the vote) as the tail so
        # a stateless label reads "County Cork, Ireland" — a real,
        # non-fabricated region token — instead of the bare county. No
        # country signal (legacy/degraded geocoder) keeps the old bare label.
        if dom.state:
            label = f"{dom.county}, {dom.state}"
        elif dom.country:
            label = f"{dom.county}, {dom.country}"
        else:
            label = dom.county

        # Disambiguate against labels already assigned. The "(road)" suffix
        # comes from the dominant-location vote's winning-county road
        # (the modal, quality-filtered road sampled INSIDE the voted county) —
        # never from a reverse geocode of the bare arithmetic-mean centroid,
        # which on a boundary-crossing LF sits off-route / in the wrong
        # county and named a road the footage plurality never drove.
        if label and label in used_labels:
            disambiguated = False
            road = dom.road
            if road and f"{label} ({road})" not in used_labels:
                label = f"{label} ({road})"
                disambiguated = True
            if not disambiguated:
                n = 2
                while f"{label} (Part {n})" in used_labels:
                    n += 1
                label = f"{label} (Part {n})"

        used_labels.add(label)
        results[idx] = label

    if missing_indices:
        # Aggregate raise. The CLI gate upstream of this call
        # makes this raise unreachable for legitimate runs; reaching
        # here means a code path bypassed the gate.
        sample = ", ".join(f"--per-lf-location '{i}=...'" for i in sorted(missing_indices))
        raise ValueError(
            "derive_per_lf_locations: ambiguous input for LF index(es) "
            f"{sorted(missing_indices)} — neither usable GPS county vote "
            "nor operator override supplied. Re-invoke with: "
            f"{sample}"
        )

    # Uniqueness postcondition applies to CENTROID-DERIVED labels only.
    # Operator overrides are authoritative and may intentionally repeat across
    # sibling LFs (e.g. two parts of one GPS-less drive); title uniqueness is
    # handled by ensure_unique_titles. Exclude overridden indices.
    derived = [v for idx, v in results.items() if v and idx not in override_idxs]
    if len(derived) != len(set(derived)):
        raise RuntimeError(f"Duplicate LF labels detected: {list(results.values())}")

    return results


def episodes_with_ingest_gps(state) -> set:
    """Episode (source-file) indices that carried GPS AT INGEST.

    Distinguishes a genuinely GPS-less episode (never had a trace) from
    an episode whose GPS was successfully derived at ingest. The persisted
    proof of the latter is a resolved model with real evidence
    (``dominant_county``/``regions``) — mapped back to its recording ``.gpx``
    by DATE via ``per_lf_recording_date`` (LF→date) and ``episode_dates``
    (parallel to ``gpx_files``). Neither ``drive_plan`` nor the raw trace is
    needed, so it works at the early resume point. An episode with NO resolved
    model attributing to it (GPS-less / legacy) is absent from the result, so
    a missing/None ``.gpx`` for it degrades quietly — never a hard fail.
    """
    resolved = getattr(state, "per_lf_resolved_location", None) or {}
    rec_date = getattr(state, "per_lf_recording_date", None) or {}
    episode_dates = getattr(state, "episode_dates", None) or []
    if not resolved or not episode_dates:
        return set()
    gps_dates = {
        rec_date.get(int(lf_idx))
        for lf_idx, rl in resolved.items()
        if getattr(rl, "dominant_county", None) or getattr(rl, "regions", None)
    }
    gps_dates.discard(None)
    return {i for i, d in enumerate(episode_dates) if d in gps_dates}


def rebuild_gps_per_episode(gpx_files: Optional[list], *, had_gps_indices=None) -> list:
    """Reconstruct ``state.gps_per_episode`` from on-disk ``.gpx`` files.

    ``gps_per_episode`` (the full per-episode trackpoint traces) is no
    longer persisted into ``pipeline_state.json`` — it is recovered
    on resume by re-parsing the ``.gpx`` paths kept in ``state.gpx_files``.
    Mirrors the parse loop in ``drive_phases.phase_a`` (minus the
    external-provenance correspondence gate, which only runs at fresh
    ingest): each entry is ``(GPSData | None, gps_offset)`` and stays
    index-parallel with ``gpx_files``. ``gps_offset`` is ``effective_offset``
    (the footage origin for a stamped GPX), identical to what fresh ingest
    stores. A ``None`` path, a vanished file, or a
    parse failure yields ``(None, 0.0)`` so ``source_file_index`` alignment is
    preserved.

    Pass ``had_gps_indices`` — the episode indices that
    carried GPS AT INGEST (from :func:`episodes_with_ingest_gps`). A re-parse
    failure for one of THOSE episodes is a derivation FAILURE, not a GPS-less
    episode: degrading it to ``(None, 0.0)`` would make a GPS trace look
    GPS-less on resume and silently SKIP the pre-encode far-road gate.
    Such a failure HARD FAILS the resume; restore the ``.gpx`` or pass an
    explicit ``--per-lf-location`` override. A failure for an episode with NO
    recorded ingest GPS (``had_gps_indices`` omitted / not containing it — a
    genuinely GPS-less or legacy episode) tolerates as before.
    """
    from ..analysis.gps_parser import parse_gpx_file
    from ..analysis.shorts_models import effective_offset

    had_gps = set(had_gps_indices or ())
    result: list = []
    failures: list[str] = []
    for i, gpx_path in enumerate(gpx_files or []):
        gps_data = None
        gps_offset = 0.0
        if gpx_path:
            try:
                gps_data = parse_gpx_file(gpx_path)
                gps_offset = effective_offset(gps_data, gps_data.start_time or 0.0)
            except Exception as e:  # noqa: BLE001
                if i in had_gps:
                    failures.append(f"  [{i}] {gpx_path}: {e}")
                    continue
                logger.warning(
                    "Resume: GPX re-parse failed for %s: %s (no recorded ingest "
                    "GPS for this episode — tolerated)",
                    gpx_path,
                    e,
                )
        result.append((gps_data, gps_offset))
    if failures:
        raise RuntimeError(
            "Resume GPX re-parse HARD FAIL — the .gpx for an episode that HAD "
            "GPS at ingest (a persisted resolved location attributes to it) no "
            "longer re-parses. Degrading it to a GPS-less episode would silently "
            "skip the pre-encode far-road gate on a trace that produced "
            "a label. Restore/fix the .gpx (or pass an explicit --per-lf-location "
            "override) and resume.\nAffected episodes:\n" + "\n".join(failures)
        )
    return result


def _segment_trackpoints(
    seg, gps_per_episode, lf_idx: int, *, with_episode: bool = False
) -> Optional[tuple]:
    """``(corrected_offset, trackpoints)`` for one segment's window, or ``None``.

    Looks up the source episode's GPSData via the segment's GLOBAL
    ``source_file_index`` and returns the fixes in ``[start_time + offset,
    end_time + offset]``; the offset maps segment media seconds to GPS time.
    ``with_episode`` appends the episode's full trackpoint list (the fixes
    either side of the window bound a dropout a cut falls inside).
    """
    file_idx = int(getattr(seg, "source_file_index", -1))
    if file_idx < 0 or file_idx >= len(gps_per_episode):
        return None
    entry = gps_per_episode[file_idx]
    if not entry:
        return None
    gps_data, gps_offset = entry[0], entry[1] if len(entry) > 1 else 0.0
    if gps_data is None:
        return None
    # JSON-roundtrip reconstitution (mirrors logic in
    # longform_assembler.segment_context.populate_gps).
    if isinstance(gps_data, dict) and "trackpoints" in gps_data:
        try:
            from ..analysis.shorts_models import GPSData, GPSTrackpoint

            gps_data = GPSData(trackpoints=[GPSTrackpoint(**tp) for tp in gps_data["trackpoints"]])
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "GPSData reconstitution failed for LF %d seg ep %d: %s", lf_idx, file_idx, exc
            )
            return None
    from ..analysis.shorts_models import effective_offset as _effective_offset

    corrected = _effective_offset(gps_data, gps_offset)
    start = float(getattr(seg, "start_time", 0.0) or 0.0) + corrected
    end = float(getattr(seg, "end_time", 0.0) or 0.0) + corrected
    try:
        in_range = gps_data.get_points_in_range(start, end)
    except Exception as exc:  # noqa: BLE001
        logger.debug("get_points_in_range failed for LF %d seg ep %d: %s", lf_idx, file_idx, exc)
        in_range = []
    if with_episode:
        return corrected, list(in_range or []), list(getattr(gps_data, "trackpoints", None) or [])
    return corrected, list(in_range or [])


def _nearest_outside_fixes(points, corrected: float, start_s: float, end_s: float) -> tuple:
    """``(prev_fix_s, next_fix_s)``: the nearest valid fixes of the episode
    before ``start_s`` and at/after ``end_s``, in media seconds (``None`` if none)."""
    from .route_claims import _valid

    prev_s = next_s = None
    for p in points:
        if not _valid(getattr(p, "lat", None), getattr(p, "lon", None)):
            continue
        t = float(p.timestamp) - corrected
        if t < start_s and (prev_s is None or t > prev_s):
            prev_s = t
        elif t >= end_s and (next_s is None or t < next_s):
            next_s = t
    return prev_s, next_s


def _nearest_outside_fix_points(points, corrected: float, start_s: float, end_s: float) -> tuple:
    """The same cut-boundary anchors with coordinates for bounded reachability."""
    from .route_claims import _valid

    before = after = None
    for p in points:
        if not _valid(getattr(p, "lat", None), getattr(p, "lon", None)):
            continue
        item = (float(p.timestamp) - corrected, float(p.lat), float(p.lon))
        if item[0] < start_s and (before is None or item[0] > before[0]):
            before = item
        elif item[0] >= end_s and (after is None or item[0] < after[0]):
            after = item
    return before, after


def _build_route_windows_by_lf(drive_plan: Optional[DrivePlan], gps_per_episode) -> dict:
    """``{lf_index: [RouteWindow]}`` — each retained segment with its own fixes
    in segment media seconds (no dedup: a gap must stay visible). Same GLOBAL
    index constraint as :func:`_build_gps_points_by_lf`."""
    from .route_claims import RouteWindow

    if drive_plan is None or not gps_per_episode:
        return {}
    out: dict[int, list] = {}
    for lf in getattr(drive_plan, "regular_lfs", None) or []:
        lf_idx = int(getattr(lf, "index", 0))
        windows = []
        for n, seg in enumerate(getattr(lf, "segments", None) or []):
            hit = _segment_trackpoints(seg, gps_per_episode, lf_idx, with_episode=True)
            if hit is None:
                continue
            corrected, pts, episode_pts = hit
            start_s = float(getattr(seg, "start_time", 0.0) or 0.0)
            end_s = float(getattr(seg, "end_time", 0.0) or 0.0)
            prev_s, next_s = _nearest_outside_fixes(episode_pts, corrected, start_s, end_s)
            prev_fix, next_fix = _nearest_outside_fix_points(episode_pts, corrected, start_s, end_s)
            windows.append(
                RouteWindow(
                    segment_id=f"lf{lf_idx:02d}-s{n}",
                    start_s=start_s,
                    end_s=end_s,
                    fixes=[(float(p.timestamp) - corrected, p.lat, p.lon) for p in pts],
                    source=f"ep{int(getattr(seg, 'source_file_index', -1))}",
                    prev_fix_s=prev_s,
                    next_fix_s=next_s,
                    prev_fix=prev_fix,
                    next_fix=next_fix,
                )
            )
        if windows:
            out[lf_idx] = windows
    return out


def _build_gps_points_by_lf(
    drive_plan: Optional[DrivePlan],
    gps_per_episode: Optional[list],
) -> dict:
    """Slice raw GPS trackpoints per LF from ``state["gps_per_episode"]``.

    Walks each LF's segments, looks up the source-episode's GPSData
    via the segment's GLOBAL ``source_file_index``, and collects
    trackpoints falling in the segment's ``[start_time + gps_offset,
    end_time + gps_offset]`` range. LFs with no resolvable GPS yield
    an empty list (omitted from the result so callers see a missing
    key and fall back to the segment-centroid path).

    Returns ``{lf_index: [GPSTrackpoint, ...]}``.

    IMPORTANT: must be called BEFORE ``_partition_lf_plans_by_date``
    mutates segment ``source_file_index`` to day-local indices. After
    the partition the episode lookup below would point at the wrong
    episode (or out-of-range) and silently drop all points.

    Without this helper, the two production callers
    of ``derive_per_lf_locations`` did not pass ``gps_points_by_lf``,
    so ``filter_high_quality_points`` was never exercised and per-LF
    centroids were computed from unfiltered segment centroids.
    """
    if drive_plan is None or not getattr(drive_plan, "regular_lfs", None):
        return {}
    if not gps_per_episode:
        return {}

    result: dict[int, list] = {}
    for lf in drive_plan.regular_lfs:
        lf_idx = int(getattr(lf, "index", 0))
        points: list = []
        _last_coord: Optional[tuple] = None  # deduplicate consecutive stuck-GPS coordinates
        for seg in getattr(lf, "segments", None) or []:
            hit = _segment_trackpoints(seg, gps_per_episode, lf_idx)
            in_range = hit[1] if hit else []
            if in_range:
                # Collapse consecutive identical (lat, lon) pairs so a stuck GPS fix
                # (e.g. GoPro reporting the same Bahamas coordinate for hundreds of
                # seconds) doesn't numerically dominate the spatial outlier filter in
                # filter_high_quality_points and evict the correct minority of points.
                for _pt in in_range:
                    _coord = (getattr(_pt, "lat", None), getattr(_pt, "lon", None))
                    if _coord != _last_coord:
                        points.append(_pt)
                        _last_coord = _coord
        if points:
            result[lf_idx] = points
    return result


def _journey_endpoint_loss_by_lf(
    drive_plan: Optional[DrivePlan],
    gps_per_episode: Optional[list],
) -> dict:
    """Detect per-LF A→B endpoint loss from a head/tail GPS dropout.

    ``build_resolved_location`` derives the journey's ``start`` from the FIRST
    surviving trackpoint of the trace and its ``end`` from the LAST. Those are
    supplied by the first / last segment that yields in-range points here. When
    that boundary segment has lost its head (or tail) waypoints, its first (or
    last) surviving fix sits well inside the segment's requested window, so the
    derived origin (or destination) is SHIFTED with no signal it is wrong.

    Mirrors :func:`_build_gps_points_by_lf`'s per-segment window resolution
    (same episode lookup + ``effective_offset`` correction) but only inspects
    the boundary gaps. Returns ``{lf_index: {"start"?, "end"?}}`` for LFs with a
    lost endpoint (LFs / endpoints that are clean are omitted), and emits a
    ``logger.warning`` naming each affected endpoint so the shift is never
    silent. Like ``_build_gps_points_by_lf`` this MUST run BEFORE
    ``_partition_lf_plans_by_date`` remaps ``source_file_index`` to day-local.
    """
    if drive_plan is None or not getattr(drive_plan, "regular_lfs", None):
        return {}
    if not gps_per_episode:
        return {}

    from ..analysis.shorts_models import GPSData, GPSTrackpoint
    from ..analysis.shorts_models import effective_offset as _effective_offset

    result: dict[int, set] = {}
    for lf in drive_plan.regular_lfs:
        lf_idx = int(getattr(lf, "index", 0))
        first_bound: Optional[tuple] = None  # (requested_start, first_survivor_ts)
        last_bound: Optional[tuple] = None  # (requested_end, last_survivor_ts)
        for seg in getattr(lf, "segments", None) or []:
            file_idx = int(getattr(seg, "source_file_index", -1))
            if file_idx < 0 or file_idx >= len(gps_per_episode):
                continue
            entry = gps_per_episode[file_idx]
            if not entry:
                continue
            gps_data, gps_offset = entry[0], entry[1] if len(entry) > 1 else 0.0
            if gps_data is None:
                continue
            if isinstance(gps_data, dict) and "trackpoints" in gps_data:
                try:
                    gps_data = GPSData(
                        trackpoints=[GPSTrackpoint(**tp) for tp in gps_data["trackpoints"]]
                    )
                except Exception:  # noqa: BLE001 — mirror _build_gps_points_by_lf
                    continue
            corrected = _effective_offset(gps_data, gps_offset)
            start = float(getattr(seg, "start_time", 0.0) or 0.0) + corrected
            end = float(getattr(seg, "end_time", 0.0) or 0.0) + corrected
            try:
                in_range = gps_data.get_points_in_range(start, end)
            except Exception:  # noqa: BLE001
                continue
            if not in_range:
                continue
            if first_bound is None:
                first_bound = (start, float(in_range[0].timestamp))
            last_bound = (end, float(in_range[-1].timestamp))

        lost: set = set()
        if first_bound is not None and first_bound[1] - first_bound[0] > _ENDPOINT_LOSS_GAP_SEC:
            lost.add("start")
        if last_bound is not None and last_bound[0] - last_bound[1] > _ENDPOINT_LOSS_GAP_SEC:
            lost.add("end")
        if not lost:
            continue
        for which, edge in (("start", "head"), ("end", "tail")):
            if which in lost:
                logger.warning(
                    "A2B-ENDPOINT-LOSS: LF %d %s endpoint withheld — the segment "
                    "supplying it lost its %s GPS waypoints (boundary gap > %.0fs); "
                    "the shifted coordinate is NOT emitted as authoritative.",
                    lf_idx,
                    which,
                    edge,
                    _ENDPOINT_LOSS_GAP_SEC,
                )
        result[lf_idx] = lost
    return result


def _build_per_lf_gpx_path_map(drive_plan, gpx_files) -> dict:
    """Build ``{lf.index: gpx_path_str_or_None}`` while segment indices
    are still GLOBAL (i.e. parallel to ``state["gpx_files"]`` /
    ``state["episode_files"]``).

    IMPORTANT: must be called BEFORE
    ``_partition_lf_plans_by_date``. That helper mutates every
    segment's ``source_file_index`` from global to DAY-LOCAL; after
    the mutation, indexing the drive-wide ``gpx_files`` list with
    a day-local index returns the WRONG GPX file on day 2+ of any
    multi-day drive (day-local 0 → global 0 → day 1's GPX). Phase D
    chapter generation depends on this map to avoid that aliasing.

    Robust-by-construction contract: every LF in
    ``drive_plan.regular_lfs`` gets an entry — either the discovered
    GPX path string or explicit ``None`` for GPS-less LFs. The map's
    KEYS are the universe of LF indices that completed phase_bc;
    ``None`` values mean "no GPX available" (legitimate for GPS-less
    full_drive). The downstream
    ``_gpx_path_expected_keys`` filter excludes ``None`` values so
    GPS-derived invariants (distance / elevation / regions) skip
    those LFs by construction. The resume-time
    ``state_invariants._check_per_lf_completeness`` gate then accepts
    the state because every LF index is present.

    Multi-episode span attribution: an LF is
    attributed across the UNION of every episode its segments touch —
    NOT ``segments[0]`` alone. Keying off only the first segment's
    ``source_file_index`` aliased two distinct LFs that START in the
    same episode but SPAN different later episodes to the SAME GPX
    file, producing byte-identical per-LF GPS stats (the published
    lf_04/lf_05 distance/elevation defect). The downstream consumers
    (``_compute_per_lf_distance`` / ``_compute_per_lf_elevation_gain``
    in ``post_encoding_invariants.py``) index a SINGLE gpx path per
    LF, so the contract stays ``{lf.index: str_or_None}`` — but the
    chosen representative is the GPX of the episode contributing the
    MOST segments to this LF (ties broken by lowest global episode
    index, deterministic), so two LFs over different episode sets
    resolve to GPX files matching THEIR span. Two LFs that genuinely
    span the identical episode set legitimately share one GPX.

    Returns ``{}`` only when ``drive_plan`` itself has no regular_lfs.
    """
    if drive_plan is None or not getattr(drive_plan, "regular_lfs", None):
        return {}
    # Shared representative-episode selection: the SAME function the
    # segment-window picker uses, so the resolved GPX and the trim window
    # can never come from different episodes.
    from .drive_phases.phase_d import _pick_representative_episode

    result: dict[int, Optional[str]] = {}
    for lf in drive_plan.regular_lfs:
        lf_idx = int(getattr(lf, "index", 0))
        segments = getattr(lf, "segments", None) or []
        if not segments or not gpx_files:
            result[lf_idx] = None
            continue
        # Representative = episode contributing the most GPX-bearing segments
        # (ties → lowest global index), so it reflects the LF's actual span,
        # not segments[0].
        rep_gi = _pick_representative_episode(segments, gpx_files)
        if rep_gi is None:
            result[lf_idx] = None
            continue
        result[lf_idx] = str(gpx_files[rep_gi])
    return result


def _build_per_lf_gpx_episode_map(drive_plan, gpx_files) -> dict:
    """Build ``{lf.index: {gpx_path_str: [episode_local_ranges]}}`` for every
    distinct episode an LF spans that has a usable GPX.

    ADDITIVE companion to :func:`_build_per_lf_gpx_path_map` (whose single-rep
    contract stays unchanged — chapters/regions/"Through:" consume it): lets
    ``post_encoding_invariants`` SUM per-LF distance/elevation across all the
    LF's episodes, not just the representative one. Windows key on
    the GPX PATH (not an index) so the map survives the later
    ``source_file_index`` remap and a JSON round-trip. Like the sibling map this
    MUST be built BEFORE ``_partition_lf_plans_by_date``. Episodes with a
    None/out-of-range ``gpx_files`` entry are skipped; ``{}`` if no LF has GPX.
    """
    if drive_plan is None or not getattr(drive_plan, "regular_lfs", None) or not gpx_files:
        return {}
    from .drive_phases.phase_d import _build_included_segments_by_episode_for_lf

    result: dict[int, dict[str, list[tuple[float, float]]]] = {}
    for lf in drive_plan.regular_lfs:
        lf_idx = int(getattr(lf, "index", 0))
        by_episode = _build_included_segments_by_episode_for_lf(lf)
        episode_gpx: dict[str, list[tuple[float, float]]] = {}
        for gi, ranges in by_episode.items():
            if gi < 0 or gi >= len(gpx_files) or gpx_files[gi] is None or not ranges:
                continue
            episode_gpx[str(gpx_files[gi])] = ranges
        if episode_gpx:
            result[lf_idx] = episode_gpx
    return result


def _partition_lf_plans_by_date(
    drive_plan: DrivePlan,
    episode_files: list,
    episode_dates: list,
    day_batches: list,
) -> dict:
    """Partition the drive plan's regular_lfs by day using each LF's
    segment episode indices. Remaps `source_file_index` on every
    segment (and montage peak) from global episode-index to the
    position of that episode within its DayBatch, so the day-local
    `compute_day_file_boundaries` indices match.
    """
    if not drive_plan or not drive_plan.regular_lfs:
        return {}
    # Global episode index → date
    idx_to_date: dict = {}
    for i, date in enumerate(episode_dates or []):
        if i < len(episode_files):
            idx_to_date[i] = date

    # For each day, build global-index → day-local-index map based on
    # the day batch episode order.
    day_local_map: dict = {}
    for batch in day_batches:
        mapping: dict = {}
        for local_i, ep in enumerate(batch.episodes):
            concat_path = getattr(ep, "concat_path", None)
            if concat_path is None:
                continue
            for gi, path in enumerate(episode_files):
                if str(path) == str(concat_path):
                    mapping[gi] = local_i
                    break
        day_local_map[batch.date] = mapping

    result: dict = {}
    for lf in drive_plan.regular_lfs:
        # Determine LF's day from its first segment's episode date
        if not lf.segments:
            continue
        first_gi = int(getattr(lf.segments[0], "source_file_index", 0))
        date = idx_to_date.get(first_gi)
        if date is None:
            continue
        mapping = day_local_map.get(date, {})
        if not mapping:
            continue
        # Remap every segment + montage peak to day-local index
        for seg in list(lf.segments) + list(lf.montage_peaks):
            gi = int(getattr(seg, "source_file_index", 0))
            if gi in mapping:
                seg.source_file_index = mapping[gi]
        result.setdefault(date, []).append(lf)
    return result
