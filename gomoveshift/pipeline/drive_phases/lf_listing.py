"""LF index listing helpers for ``--list-lfs`` and ``--label-dry-run``.

Integrity gates run upstream of the OST burn: when the
operator passes ``--list-lfs``, the drive subcommand runs Phase A
(clip discovery), builds the drive plan from the discovery outputs,
prints a table of LF indices + suggested labels (centroid-derived from
GPS when available; full_drive LFs with no segment centroid resolve via the
encode path's ``derive_per_lf_locations`` over the GPX fixes in their span; ``"<no GPS>"`` otherwise), then exits 0 — without
encoding. The operator uses the table to author ``--per-lf-location``
overrides for a follow-up invocation that actually encodes.

``--label-dry-run`` (``print_label_dry_run``) goes one step further: it
resolves each LF's *corrected* label through the real dominant-county
reducer with the live geocoder and prints the label + county vote
breakdown — the verification gate for confirming labels (e.g. after a
mislabel report) before committing to an expensive re-encode.

When ``config.drive.sign_verify`` is on, the dry-run additionally prints
the road-sign evidence for each LF right beside the vote it is evidence
about — the only pre-burn place an operator sees an *independent* check
on the GPS→geocoder chain. The evidence is advisory here: a conflict
prints, WARNs with a greppable ``SIGN-VERIFY-CONFLICT`` tag, and the
gate still exits normally. Escalating a conflict to a hard failure is
the separate opt-in ``sign_verify_strict`` gate in ``phase_bc_day_first``.

``print_lf_listing`` is deliberately import-light (no network); the
dry-run path is heavier (one geocode per sample) but still Phase-A bound.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


def print_lf_listing(config: Any, state: Any) -> None:
    """Print the LF index → suggested-label table to stdout.

    Args:
        config: Service ``Config``. Forwarded into
            ``_build_drive_plan_from_state``.
        state: Orchestrator state dict / PipelineState after Phase A.
    """
    # Local import to avoid circulars at module load.
    from .phase_bc_day_first import _build_drive_plan_from_state

    drive_plan = _build_drive_plan_from_state(config, state)
    print()
    print("=" * 60)
    print("  LF index → suggested label (use --per-lf-location to override)")
    print("=" * 60)
    if drive_plan is None or not getattr(drive_plan, "regular_lfs", None):
        print("  (no LFs detected in discovery — nothing to list)")
        print("=" * 60)
        return
    span_labels = _labels_from_span_gps(drive_plan, state)
    for lf in drive_plan.regular_lfs:
        idx = int(getattr(lf, "index", 0))
        label = _suggest_label(lf)
        if label == _NO_GPS_LABEL and idx in span_labels:
            label = span_labels[idx]
        print(f"  {idx:>3}  {label}")
    print("=" * 60)
    print("  Re-invoke with --per-lf-location 'INDEX=LABEL' for any LF marked <no GPS>.")
    print()


def print_label_dry_run(config: Any, state: Any) -> None:
    """Print each LF's *corrected* label without encoding (``--label-dry-run``).

    Unlike ``print_lf_listing`` (which only prints raw centroid coords to
    keep the listing network-free), this runs the real dominant-county
    reducer with the live geocoder against each LF's sampled GPS trace —
    the same code the encode path uses — and prints the resolved label plus
    the county vote breakdown. Lets an operator confirm the labels are right
    before committing to an expensive re-encode. Phase-A only; exits via the
    orchestrator short-circuit afterward.
    """
    from ..gps_enrichment import _build_gps_points_by_lf, _build_shared_geocoder
    from ..title_builder import _road_ref_members, dominant_location
    from .phase_bc_day_first import _build_drive_plan_from_state

    drive_plan = _build_drive_plan_from_state(config, state)
    print()
    print("=" * 64)
    print("  LF index → corrected label (dominant-county vote — NO encode)")
    print("=" * 64)
    if drive_plan is None or not getattr(drive_plan, "regular_lfs", None):
        print("  (no LFs detected in discovery — nothing to list)")
        print("=" * 64)
        return
    gps_points_by_lf = _build_gps_points_by_lf(drive_plan, getattr(state, "gps_per_episode", None))
    geocoder = _build_shared_geocoder()
    # Sign evidence is bucketed ONCE for the whole (unpartitioned) plan — the
    # per-LF loop below only compares. Any failure here degrades to "unavailable"
    # per LF; it never costs the operator their labels.
    sign_on = _sign_verify_enabled(config, state)
    sign_hits_by_lf: dict = {}
    sign_error = ""
    if sign_on:
        try:
            sign_hits_by_lf = _bucket_sign_hits(state, drive_plan)
        except Exception as exc:  # noqa: BLE001 — advisory evidence never breaks the gate
            sign_error = str(exc)
            logger.warning("  sign evidence: bucketing failed (%s) — evidence unavailable", exc)
    for lf in drive_plan.regular_lfs:
        idx = int(getattr(lf, "index", 0))
        pts = gps_points_by_lf.get(idx) if gps_points_by_lf else None
        dom = dominant_location(pts, geocoder) if pts else None
        if dom is None or not dom.county:
            print(f"  {idx:>3}  <no GPS — --per-lf-location override required>")
            continue
        label = f"{dom.county}, {dom.state}" if dom.state else dom.county
        if dom.name:
            label = f"{dom.name}, {label}"
        print(f"  {idx:>3}  {label}")
        votes = ", ".join(f"{k}={v}" for k, v in sorted(dom.votes.items(), key=lambda kv: -kv[1]))
        print(f"       county vote: {votes}")
        # Surface the resolved via-road + its per-member vote tally so the
        # ground-truth harness can verify via-road behaviour (not just county).
        # The tally is reconstructed from the winning county's raw road refs the
        # same way ``dominant_location`` votes them (each ';'-concurrency member
        # credited once per sample), so it mirrors the encode path exactly.
        print(f"       via road: {dom.road or '<none>'}")
        road_votes: Counter[str] = Counter()
        for ref in dom.road_refs:
            for member in _road_ref_members(ref):
                road_votes[member] += 1
        if road_votes:
            rv = ", ".join(
                f"{k}={v}" for k, v in sorted(road_votes.items(), key=lambda kv: (-kv[1], kv[0]))
            )
            print(f"       road vote: {rv}")
        if sign_on:
            try:
                if sign_error:
                    raise RuntimeError(sign_error)
                _print_lf_sign_evidence(config, idx, dom, sign_hits_by_lf.get(idx) or [])
            except Exception as exc:  # noqa: BLE001 — advisory evidence never breaks the gate
                logger.warning("  sign evidence: LF %s comparison failed: %s", idx, exc)
                print(f"       sign evidence: unavailable ({exc})")
    print("=" * 64)
    print("  Labels reflect the dominant-county fix. No encode performed.")
    print()


def _sign_verify_enabled(config: Any, state: Any) -> bool:
    """True when the feature is on AND Phase A actually recorded sign hits.

    Read defensively (``getattr``) so a legacy config / state object — or a
    resume from a state file written before the feature existed — is a plain
    no-op rather than an ``AttributeError`` at the label gate.
    """
    drive_cfg = getattr(config, "drive", None)
    if not bool(getattr(drive_cfg, "sign_verify", False)):
        return False
    return bool(getattr(state, "sign_hits_by_date", None))


def _bucket_sign_hits(state: Any, drive_plan: Any) -> dict:
    """``state.sign_hits_by_date`` → ``{lf_index: [SignHit, ...]}``.

    ``drive_plan`` must be the UNPARTITIONED plan (segment
    ``source_file_index`` still in GLOBAL episode-index space), which is what
    the dry-run path builds — bucketing matches a hit's ``episode_index`` /
    ``t_local`` against segment spans in that same space.

    Imported lazily so the listing module does not pull in the sign-analysis
    modules unless sign hits are bucketed.
    """
    from ...analysis.sign_evidence import hits_by_date_from_state
    from ...analysis.sign_verify import bucket_hits_by_lf

    hits_by_date = hits_by_date_from_state(getattr(state, "sign_hits_by_date", None))
    return bucket_hits_by_lf(hits_by_date, drive_plan) or {}


def _print_lf_sign_evidence(config: Any, idx: int, dom: Any, hits: list) -> None:
    """Print one LF's sign evidence beneath its county/road vote lines.

    No hits is printed EXPLICITLY as no-signal rather than skipped: an operator
    scanning the gate must never read silence as agreement (a failed OCR pass
    and a corroborated label would otherwise look identical).
    """
    if not hits:
        print("       sign evidence: no-signal — 0 sign hits (absence of evidence, NOT agreement)")
        return

    from ...analysis.sign_evidence import STATUS_CONFLICTS
    from ...analysis.sign_verify import resolved_from_dominant, verify_lf

    # The repo's own sanitised + allowlisted forward geocoder. Advisory surface,
    # so a live lookup is acceptable here (the strict gate deliberately passes
    # None instead). Imported at call time so the test/hermetic stub applies.
    from ..chapters.nominatim_source import _nominatim_search

    # The dry-run has a dominant-county reduction, not a persisted
    # ``ResolvedLocation`` — build the comparison target from it.
    resolved = resolved_from_dominant(dom)
    drive_cfg = getattr(config, "drive", None)
    min_hits = int(getattr(drive_cfg, "sign_verify_min_corroborating_hits", 2) or 2)
    verdict = verify_lf(
        idx,
        hits,
        resolved,
        min_corroborating_hits=min_hits,
        geocode_fn=_nominatim_search,
    )
    hit_count = int(getattr(verdict, "hit_count", 0) or len(hits))
    print(f"       sign evidence: {getattr(verdict, 'status', '?')} ({hit_count} hit(s))")
    for reason in getattr(verdict, "reasons", None) or []:
        print(f"         - {reason}")
    if getattr(verdict, "status", "") == STATUS_CONFLICTS:
        claimed = ", ".join(str(c) for c in (getattr(verdict, "conflicting_counties", None) or []))
        regions = ", ".join(str(r) for r in (getattr(resolved, "regions", None) or []))
        logger.warning(
            "SIGN-VERIFY-CONFLICT LF %s: road signs claim %s, but the label resolves to "
            "%s / %s and the sampled regions are [%s]. Advisory only at --label-dry-run — "
            "re-check the label (or --per-lf-location) before encoding.",
            idx,
            claimed or "<unnamed county>",
            getattr(resolved, "dominant_county", None),
            getattr(resolved, "dominant_state", None),
            regions or "<none>",
        )


_NO_GPS_LABEL = "<no GPS — override required>"


def _labels_from_span_gps(drive_plan: Any, state: Any) -> dict[int, str]:
    """Labels for LFs whose segments carry no ``gps_center`` but whose span
    has GPX fixes (full_drive ``--list-lfs`` skips the A3 scoring pass).

    Reuses the encode path's resolution — ``_build_gps_points_by_lf`` then
    ``derive_per_lf_locations`` — never a second location deriver. An LF the
    resolver cannot label is simply absent, so the caller keeps the
    ``<no GPS>`` text. Any failure degrades to ``{}`` (advisory listing).
    """
    from ..gps_enrichment import (
        _build_gps_points_by_lf,
        _build_shared_geocoder,
        derive_per_lf_locations,
    )

    try:
        points = _build_gps_points_by_lf(drive_plan, getattr(state, "gps_per_episode", None))
        if not points:
            return {}
        geocoder = _build_shared_geocoder()
    except Exception as exc:  # noqa: BLE001 — listing stays advisory
        logger.warning("list-lfs: span GPS label resolution unavailable (%s)", exc)
        return {}
    labels: dict[int, str] = {}
    for lf in drive_plan.regular_lfs:
        idx = int(getattr(lf, "index", 0))
        if not points.get(idx):
            continue
        try:
            labels.update(
                derive_per_lf_locations(
                    [lf], geocoder=geocoder, gps_points_by_lf={idx: points[idx]}
                )
            )
        except ValueError:
            continue  # no usable county vote — keep the override text
        except Exception as exc:  # noqa: BLE001
            logger.warning("list-lfs: LF %s label resolution failed (%s)", idx, exc)
    return labels


def _suggest_label(lf: Any) -> str:
    """Best-effort suggestion: centroid lat/lon when GPS is available,
    else ``"<no GPS — override required>"``.

    Reverse-geocoding is intentionally NOT done here — it costs a
    network round-trip per LF and the operator only needs a hint to
    decide whether the LF needs an override. The ``derive_per_lf_locations``
    pass during phase_bc does the full lookup.
    """
    segments = list(getattr(lf, "segments", None) or [])
    coords: list[tuple[float, float]] = []
    for seg in segments:
        lat = float(getattr(seg, "gps_center_lat", 0.0) or 0.0)
        lon = float(getattr(seg, "gps_center_lon", 0.0) or 0.0)
        if lat != 0.0 or lon != 0.0:
            coords.append((lat, lon))
    if not coords:
        return _NO_GPS_LABEL
    avg_lat = sum(c[0] for c in coords) / len(coords)
    avg_lon = sum(c[1] for c in coords) / len(coords)
    return f"GPS centroid ({avg_lat:.4f}, {avg_lon:.4f})"
