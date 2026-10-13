"""Phase B+C combined under the day-first architecture.

Per-LF OST text flow.

Per-LF labels: ``derive_per_lf_locations`` runs BEFORE ``run_all_days`` so
each LF's OST burn-in inside ``day_pipeline.run_day_pipeline`` reads the
LF-specific label via the
``per_lf_locations_by_date`` kwarg. Prior behaviour derived the labels
AFTER ``run_all_days`` returned — which meant multi-LF days always fell
back to the drive-wide ``location`` string and produced identical
burned OST text for every LF on the day.

Canonical text: layered on top of the per-LF labels, compute the CANONICAL
``"{start} to {end} via {road}, {state}"`` text per LF (the same format
``title_builder.build_lf_title`` produces for YouTube titles) BEFORE
``run_all_days``, store it as ``state["per_lf_ost_text"]``, and thread
``per_lf_ost_text_by_date`` through ``run_all_days`` →
``run_day_pipeline`` so OST burn-in uses the canonical text — not the
coarser county-level label that ``derive_per_lf_locations`` produces.
Fall back to the county label when the canonical path can't resolve
(no road name, or geocoder returns None for start/end towns).
Per-LF titles stored in ``state["per_lf_titles"]`` go through
``ensure_unique_titles`` (adding ``"(Part N of M)"`` suffixes on
collisions), but the OST text itself does NOT — dedup suffixes belong
on YouTube titles, not in burned video text.

Quality contract: OST, YouTube title and thumbnail text must all derive from
the same per-LF source (``state["per_lf_ost_text"]``).
"""

import json
import logging
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np

from ...analysis.unified_scorer import COL_TIMESTAMP
from ...config import Config
from ...models import DrivePlan, RegularLFPlan
from .. import volume_source
from ..day_grouper import group_episodes_by_day
from ..day_pipeline import DayArtifacts, _enforce_shorts_contract, run_all_days
from ..gps_enrichment import (
    _assert_per_lf_locations_resolvable,
    _build_drive_gps_points,
    _build_gps_points_by_date,
    _build_gps_points_by_lf,
    _build_per_lf_gpx_path_map,
    _build_shared_geocoder,
    _centroid_for_segments,
    _enrich_per_lf_gps,
    _make_short_ost_text_fn,
    _partition_lf_plans_by_date,
    build_resolved_location,
    derive_per_lf_locations,
    validate_burned_ost_text,
)
from ..longform_assembler import LongFormAssembler
from ..media_api import Deliverables, deliverables_to_state, lift_artifacts
from ..post_encoding_invariants import (
    _assert_mapping_complete,
    _assert_per_lf_distance_complete,
    _assert_per_lf_elevation_gain_complete,
    _assert_per_lf_regions_complete,
    _assert_per_lf_trim_ratio_complete,
    _assert_post_encoding_invariants,
    _compute_per_lf_bearing,
    _compute_per_lf_distance,
    _compute_per_lf_elevation_gain,
    _compute_per_lf_elevation_profile,
    _compute_per_lf_regions,
    _compute_per_lf_speed,
    _compute_per_lf_trim_ratio,
    _derive_per_lf_trim_ratio_from_paths,
    _gpx_path_expected_keys,
)
from ..short_planner import ShortRange
from ..state_invariants import (
    assert_phase_bc_per_lf_complete,
)
from ..stop_mask import build_stop_mask, mask_summary, stop_mask_kwargs
from ..title_builder import build_lf_title, ensure_unique_titles
from .trail_film import attach_trail_film

# Re-export the moved helpers for backward compatibility — multiple
# tests + downstream modules import these names from this module.
__all__ = [
    "PhaseBCDayFirst",
    "_assert_mapping_complete",
    "_assert_per_lf_distance_complete",
    "_assert_per_lf_elevation_gain_complete",
    "_assert_per_lf_locations_resolvable",
    "_assert_per_lf_regions_complete",
    "_assert_per_lf_trim_ratio_complete",
    "_assert_post_encoding_invariants",
    "_assert_drive_plan_built",
    "_assert_skeleton_lf_has_location",
    "_build_drive_plan_from_state",
    "_build_gps_points_by_date",
    "_build_gps_points_by_lf",
    "_build_per_lf_gpx_path_map",
    "_build_per_lf_recording_date_map",
    "_build_shared_geocoder",
    "_centroid_for_segments",
    "_compute_per_lf_bearing",
    "_compute_per_lf_distance",
    "_compute_per_lf_elevation_gain",
    "_compute_per_lf_elevation_profile",
    "_compute_per_lf_regions",
    "_compute_per_lf_speed",
    "_compute_per_lf_trim_ratio",
    "_day_dir_has_finals",
    "_derive_per_lf_recording_date_from_paths",
    "_derive_per_lf_trim_ratio_from_paths",
    "_enrich_per_lf_gps",
    "_gpx_path_expected_keys",
    "_make_short_ost_text_fn",
    "_partition_lf_plans_by_date",
    "_probe_duration",
    "_probe_master_integrity",
    "_record_recording_date_recovery",
    "_resolve_ingest_plan",
    "_synthesize_ingest_plan_from_state",
    "derive_per_lf_locations",
]

logger = logging.getLogger(__name__)


def _probe_master_integrity(
    master_path: Path,
    expected_duration: float,
    batch_date,
    ffprobe_timeout_sec: int = 30,
) -> bool:
    """Integrity gate for a recorded day-master.

    Returns ``True`` when the master at ``master_path`` may be trusted
    by the resume fast-path (i.e. its ffprobed duration is ≥ 99 % of
    ``expected_duration``, or no expected duration is available and we
    fall back to existence-only behaviour). Returns ``False`` when the
    file fails the duration probe and must be re-encoded.

    A killed-mid-encode master leaves valid bytes on disk with no
    trailing ``moov`` atom; the existence-only gate would skip
    re-encoding and reconstruct ``DayArtifacts`` from the corrupt
    master, breaking downstream consumers. Mirrors the probe pattern
    in ``day_master_encoder.encode_day_master``.

    When ``expected_duration <= 0`` (synthesised batches with no
    episode durations attached), we cannot compute the threshold —
    fall back to existence-only behaviour and log a WARNING so the
    gap is visible in the run log.
    """
    if expected_duration >= 1.0:
        # Lazy import keeps the module-level dependency surface
        # unchanged and matches ``day_master_encoder``'s pattern.
        import subprocess as _subprocess

        from ..validation.ffprobe_client import FFprobeClient

        probe_error: str | None = None
        try:
            actual_duration = FFprobeClient(
                _subprocess, timeout=ffprobe_timeout_sec
            ).probe_duration(master_path)
        except Exception as exc:  # noqa: BLE001 — defensive; probe is best-effort
            # Name the exception class so "probe misbehaved"
            # is distinguishable from "probe returned a low value"
            # in the WARN line below.
            actual_duration = 0.0
            probe_error = f"{type(exc).__name__}: {exc}"
        if actual_duration >= expected_duration * 0.99:
            return True
        logger.warning(
            "  day-first: day %s recorded master %s failed "
            "duration probe (%.1fs < %.1fs expected × 0.99, "
            "probe_error=%s) — treating as incomplete "
            "(integrity gate)",
            batch_date,
            master_path,
            actual_duration,
            expected_duration,
            probe_error or "none",
        )
        return False
    # No expected duration available — preserve legacy existence-only
    # behaviour but flag the gap so the operator notices in logs.
    logger.warning(
        "  day-first: day %s has no expected duration "
        "(batch.total_duration=%.2f) — skipped master "
        "integrity validation, trusting existence only",
        batch_date,
        expected_duration,
    )
    return True


def _expected_shorts_for_master_drop(art, *, allow_short_count: bool) -> Optional[int]:
    """Expected shorts count for ``drop_day_master``'s integrity gate.

    SHORTS-CONTRACT: the count-match must
    compare against the PLANNED target, not ``len(art.short_paths)`` —
    produced-vs-produced let a 9-of-14 day pass "count-matched" by
    construction, then delete the only source for the missing 5. With
    ``--allow-short-count`` the operator accepted the shortfall → fall
    back to the actual produced count (legacy behaviour).
    """
    planned = getattr(art, "planned_shorts_count", None)
    if planned is not None and not allow_short_count:
        return planned
    art_shorts = getattr(art, "short_paths", None) or []
    return len(art_shorts) if art_shorts else None


def _maybe_drop_day_master(artifacts, state, stages_dir: Path) -> None:
    """Drop this day's master right after its own bookkeeping lands
    (~59 GB/day). Recovery callers pass ``master_path=None`` — gate on that
    so a resume's shorts re-render is never forced into an ~8h re-encode."""
    if getattr(artifacts, "master_path", None) is None or getattr(state, "keep_masters", None):
        return
    date = getattr(artifacts, "date", None)
    if date is None:
        return
    from ..cleanup import drop_day_master

    day_dir = stages_dir / f"day_{date}"
    expected_shorts = _expected_shorts_for_master_drop(
        artifacts, allow_short_count=bool(getattr(state, "allow_short_count", None))
    )
    sl_record = (getattr(state, "superlong_parts", None) or {}).get(date)
    expected_superlong = sl_record.get("part_count") if isinstance(sl_record, dict) else None
    try:
        drop_day_master(
            day_dir,
            date=date,
            expected_shorts_count=expected_shorts,
            expected_superlong_parts=expected_superlong,
        )
    except Exception as exc:  # noqa: BLE001 — cleanup never blocks
        logger.warning("  day-first: master drop failed for %s: %s", date, exc)


# Code-rule version for Phase B+C (LF assembly, master encode, shorts render).
# Bumped whenever a rule change means a checkpointed phase_bc's on-disk renders
# (masters/LFs/shorts) can no longer be trusted on resume — e.g. the
# per-day stop mask (parked-footage exclusion), attenuate-only audio, and
# privacy blur changes. The orchestrator's resume path re-runs phase_bc when the
# persisted version differs from this constant — mirrors ``PHASE_D_CODE_VERSION``.
PHASE_BC_CODE_VERSION = 1


class PhaseBCDayFirst:
    """Phase B+C combined — the only LF assembly path (day-first by design).

    Delegates encoding, regular-LF extraction, and shorts selection to
    `day_pipeline.run_all_days`, then populates `state` with the
    structures Phase D expects. The legacy non-day-first split
    (phase_b + phase_c) was removed; LFs are always single-day.
    """

    name = "phase_bc_day_first"

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self, state, probe_duration_fn=None, state_save_fn=None) -> None:
        _probe = probe_duration_fn or _probe_duration

        logger.info("Phase B+C (day-first): encoding per day, extracting LFs and shorts")

        ingest_plan = _resolve_ingest_plan(state)

        if ingest_plan is None or not getattr(ingest_plan, "episodes", None):
            logger.warning("day-first: no ingest plan and no usable state, nothing to encode")
            if state.regular_lf_paths is None:
                state.regular_lf_paths = []
            if state.shorts_per_lf is None:
                state.shorts_per_lf = {}
            if state.drive_plan is None:
                state.drive_plan = None
            if state.day_artifacts is None:
                state.day_artifacts = []
            if state.per_day_locations is None:
                state.per_day_locations = {}
            if state.per_lf_locations is None:
                state.per_lf_locations = {}
            if state.per_lf_ost_text is None:
                state.per_lf_ost_text = {}
            if state.per_lf_titles is None:
                state.per_lf_titles = {}
            return

        day_batches = group_episodes_by_day(ingest_plan)
        logger.info("  day-first: grouped into %d day batch(es)", len(day_batches))

        # Guard: episodes exist but grouping yielded nothing → refuse
        # to mark the phase complete on zero work.
        if not day_batches:
            raise RuntimeError(
                "phase_bc_day_first: %d episode(s) provided but produced 0 day "
                "batches — refusing to record phase as complete on zero work. "
                "Likely cause: episodes lack a usable .date or .creation_time "
                "attribute (resume-time shallow rehydrate)." % len(ingest_plan.episodes)
            )

        output_base = Path(state.output_base)
        stages_dir = Path(state.stages_base or (output_base / "_stages"))
        stages_dir.mkdir(parents=True, exist_ok=True)

        # -- Per-day location derivation
        drive_wide_location = state.location or ""
        episode_locations = state.episode_locations or []
        episode_files = state.episode_files or []
        path_to_episode_idx = {str(p): i for i, p in enumerate(episode_files)}
        per_day_locations: dict = {}
        for batch in day_batches:
            labels: list[str] = []
            for ep in batch.episodes:
                concat_path = getattr(ep, "concat_path", None)
                if concat_path is None:
                    continue
                idx = path_to_episode_idx.get(str(concat_path))
                if idx is None or idx >= len(episode_locations):
                    continue
                candidate = (episode_locations[idx] or "").strip()
                if candidate:
                    labels.append(candidate)
            if labels:
                # Majority vote across the day's episode labels — first-episode-wins
                # stamped a two-town day with the morning town. Ties keep
                # first-seen order (Counter preserves insertion order).
                counts = Counter(labels)
                day_loc, _n = counts.most_common(1)[0]
                if len(counts) > 1:
                    logger.warning(
                        "  day %s: %d distinct episode labels %s — majority "
                        "vote picked %r; verify it represents the whole day",
                        batch.date,
                        len(counts),
                        dict(counts),
                        day_loc,
                    )
            else:
                # Zero-location day: inheriting the drive-wide label can
                # stamp another region's specific label on every LF/short
                # of this day in a multi-region drive — leave a loud
                # breadcrumb.
                day_loc = drive_wide_location
                logger.warning(
                    "  day %s: no episode-level location labels — inheriting "
                    "drive-wide label %r for every LF/short of this day; "
                    "verify it matches the day's footage (use "
                    "--per-lf-location to correct)",
                    batch.date,
                    drive_wide_location,
                )
            per_day_locations[batch.date] = day_loc
        state.per_day_locations = per_day_locations

        # -- Build or load the real DrivePlan (with real segments +
        #    montage peaks) so run_all_days can assemble LFs segment-
        #    by-segment. Prefer an existing drive_plan.json on disk
        #    (resume), fall back to building from drive_scores in-memory.
        drive_plan_path = stages_dir / "drive_plan.json"
        real_plan: Optional[DrivePlan] = None
        if drive_plan_path.exists():
            try:
                real_plan = DrivePlan.load(drive_plan_path)
                logger.info(
                    "  day-first: loaded existing drive_plan.json (%d LFs)",
                    len(real_plan.regular_lfs),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "  day-first: failed to load drive_plan.json (%s), rebuilding",
                    exc,
                )
                real_plan = None
        if real_plan is None:
            real_plan = _build_drive_plan_from_state(self.config, state)
            if real_plan is not None:
                try:
                    real_plan.save(drive_plan_path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "  day-first: failed to write drive_plan.json: %s",
                        exc,
                    )
        # A failed plan build (real_plan None) leaves lf_plans_by_date
        # empty, so the per-LF resolvable gate iterates nothing and run_all_days
        # would encode on the drive-wide fallback cascade. Hard-fail here — but
        # only when the plan SHOULD have built (footage queued + complete
        # inputs), so a legitimate no-footage / no-scores path is unaffected.
        _assert_drive_plan_built(real_plan, state, day_batches)
        enrichment = _enrich_per_lf_gps(
            state=state,
            real_plan=real_plan,
            day_batches=day_batches,
            episode_files=episode_files,
            per_day_locations=per_day_locations,
            drive_wide_location=drive_wide_location,
            gps_cache_dir=str(self.config.drive.resolved_geocode_cache_dir()),
        )
        gps_points_by_lf = enrichment.gps_points_by_lf
        lf_plans_by_date = enrichment.lf_plans_by_date
        geocoder = enrichment.geocoder
        per_lf_overrides = enrichment.per_lf_overrides
        per_lf_locations_by_date = enrichment.per_lf_locations_by_date
        per_lf_ost_text_by_date = enrichment.per_lf_ost_text_by_date
        destination_by_lf = getattr(enrichment, "destination_by_lf", None) or {}
        # Per-LF A→B endpoints shifted by a head/tail GPS dropout, withheld
        # by build_resolved_location so a wrong origin/destination never ships.
        endpoint_loss_by_lf = getattr(enrichment, "endpoint_loss_by_lf", None) or {}

        recorded, skipped_artifacts, remaining_batches = self._partition_batches_by_resume_state(
            state, day_batches, stages_dir
        )

        # Snapshot the persisted (already-burned) OST text and the set of
        # days this resume SKIPS (in ``per_day_completed``, master/finals still
        # on disk → NOT re-encoded this run). A skipped day's OST pixels are
        # fixed; re-deriving its title/thumbnail text after a geocode-cache
        # roll or code change would ship a label that DIVERGES from the burned
        # master (Phase D reads ``state.per_lf_ost_text`` for both). Below we
        # PREFER the persisted text for those LFs and re-derive only the days
        # actually re-rendered this run.
        _persisted_ost_text = dict(getattr(state, "per_lf_ost_text", None) or {})
        _completed_dates = {getattr(a, "date", None) for a in skipped_artifacts}
        _completed_dates.discard(None)

        # ── Pre-seed audit-visible state ───────────────────────
        # The OST audit gate inside ``run_day_pipeline`` (reached below
        # via ``run_all_days``) writes ``ost_audit_<date>.json`` from
        # ``state["per_lf_ost_text"]`` / ``per_lf_titles`` /
        # ``per_lf_recording_date`` BEFORE the per-LF ``apply_ost`` loop.
        # Those keys are otherwise only populated AFTER ``run_all_days``
        # returns (see end of this method), and the gate has no kwarg
        # fallback for titles/recording-date — so an operator reviewing
        # the audit on a fresh run saw an empty ``youtube_title`` and
        # ``recording_date`` and a verb-less ``ost_text``. Pre-seed them
        # here from the per-date enrichment maps (the canonical OST text
        # already carries the Highlights/Driving verb via
        # ``build_lf_title``) and the partition-derived recording dates.
        # These are best-effort previews; the authoritative final
        # population still runs after ``run_all_days`` (with the
        # on-the-fly fallback + recording-date invariant), so the guarded
        # ``if`` assignments never regress the happy path.
        _preseed_ost_text: dict[int, str] = {}
        for _day_map in per_lf_ost_text_by_date.values():
            for _lf_idx, _canonical in _day_map.items():
                if _canonical:
                    _preseed_ost_text[int(_lf_idx)] = _canonical
        # Refresh the dedup ledger from the channel ONCE per run, so a
        # BC that starts before the previous upload's Phase D sync doesn't
        # dedup against a stale on-disk ledger. Reused at every
        # ``load_published_titles`` site below. Degrades to {} on any
        # credential/API failure.
        from ..longform_metadata import refresh_channel_title_ledger

        _channel_title_ids = refresh_channel_title_ledger()

        if _preseed_ost_text:
            from ..longform_metadata import load_published_titles

            state.per_lf_ost_text = dict(_preseed_ost_text)
            state.per_lf_titles = ensure_unique_titles(
                dict(_preseed_ost_text),
                published_titles=load_published_titles() | set(_channel_title_ids),
                published_title_ids=_channel_title_ids,
            )
        _preseed_recording_date = _build_per_lf_recording_date_map(lf_plans_by_date)
        if _preseed_recording_date:
            state.per_lf_recording_date = _preseed_recording_date

        # ── Pre-encode reconciliation gate ─────────────────────────
        # The burned OST/title text is FINAL here, so reconcile it against
        # the GPS-resolved regions NOW — refusing to start a multi-hour
        # encode on a desynced label instead of catching it post-encode in
        # Phase D, where it would be too late. Building the resolved models here also
        # persists them ost_signature-stamped, so the post-run_all_days
        # ``_populate_resolved_locations`` call fast-paths on the signature
        # match — no extra geocoding on the happy path. Phase D's
        # post-encode check stays as the backstop (resume paths, late
        # title mutations).
        _flat_lf_plans = [
            lf for day_lfs in (lf_plans_by_date or {}).values() for lf in (day_lfs or [])
        ]
        if _preseed_ost_text and _flat_lf_plans:
            _populate_resolved_locations(
                state,
                _flat_lf_plans,
                gps_points_by_lf,
                geocoder,
                destination_by_lf,
                per_lf_ost_text=_preseed_ost_text,
                endpoint_loss_by_lf=endpoint_loss_by_lf,
            )
            _ladder_evidence = _build_ladder_evidence(
                _preseed_ost_text,
                gps_points_by_lf,
                per_lf_overrides,
                lf_plans=_flat_lf_plans,
                resolved_by_lf=state.per_lf_resolved_location or {},
                geocode_fn=_ladder_forward_geocode,
            )
            # Re-render any below-tier-1 (but renderable) LF at its awarded
            # tier BEFORE the gate, so a demoted label reaches the gate no longer
            # tier-1-shaped (and passes). Tier 5 stays as-is → the gate still
            # refuses to encode a genuine no-claim.
            _demoted = _degrade_labels_to_awarded_tier(
                _preseed_ost_text,
                state.per_lf_resolved_location or {},
                _ladder_evidence,
                state=state,
            )
            if _demoted:
                _apply_demoted_labels(_demoted, _preseed_ost_text, per_lf_ost_text_by_date, state)
            _assert_pre_encode_location_invariant(
                _preseed_ost_text,
                state.per_lf_resolved_location or {},
                evidence_by_lf=_ladder_evidence,
            )

        # Sibling gate, one question further out: the invariant above reconciles
        # the label with the GPS→geocoder chain; this reconciles that chain with
        # the independent road-sign evidence. Opt-in, and outside the
        # ``_preseed_ost_text`` guard so a resume with persisted resolved models
        # is still gated. Upstream of ``run_all_days`` — no encode, no upload.
        _assert_sign_verify(state, self.config)

        if remaining_batches:
            # Per-day write-through callback: record the day in
            # ``per_day_completed`` and persist state to disk BEFORE the
            # next day starts. A mid-phase crash (e.g. day N master
            # fails after days 0..N-1 landed) must leave the already-done
            # days recorded so the next resume short-circuits them.
            def _on_day_complete(artifacts, st) -> None:
                # Route Story trail film: a sibling deliverable, attached before
                # the record so it persists (and resumes) with the day.
                if isinstance(artifacts, Deliverables):
                    attach_trail_film(artifacts, st, self.config)
                _record_day_and_save(artifacts, st, state_save_fn)
                _maybe_drop_day_master(artifacts, st, stages_dir)

            # Build per-date GPS (in day-master
            # timeline) and a geocoder-backed OST-text closure so the
            # shorts loop's ``trim_gps_to_clip_timeline`` branch is
            # actually exercised in production. Without this, the clip-timeline
            # plumbing inside ``day_pipeline`` receives no GPS and no
            # consumer, and every short burns the drive-wide ``location``
            # fallback.
            gps_points_by_date = _build_gps_points_by_date(
                remaining_batches,
                episode_files=state.episode_files or [],
                gps_per_episode=state.gps_per_episode,
            )
            short_ost_text_fn = _make_short_ost_text_fn(geocoder, places_fn=_forward_geocode_places)
            # Regions closure for the cross-surface shorts location gate
            # (hard-fails a short whose burned county is in zero of its own clip's
            # sampled GPS regions). Same geocoder, same trimmed-clip points.
            from ..validation.content_gates import _make_short_regions_fn

            short_regions_fn = _make_short_regions_fn(geocoder)

            fresh_artifacts = run_all_days(
                day_batches=remaining_batches,
                drive_scores=state.drive_scores,
                file_boundaries=state.file_boundaries or [],
                output_base=stages_dir,
                location=drive_wide_location,
                config=self.config,
                episode_paths=state.episode_files,
                per_day_locations=per_day_locations,
                lf_plans_by_date=lf_plans_by_date,
                # Thread per-LF OST labels so run_day_pipeline's
                # burn-in step picks the right text per LF (not the
                # drive-wide fallback).
                per_lf_locations_by_date=per_lf_locations_by_date,
                # Thread CANONICAL per-LF OST text so the burn-in
                # uses "{start} to {end} via {road}, {state}" instead of
                # the county-level fallback. run_day_pipeline falls back
                # to per_lf_locations when an entry is missing (preserves
                # per-LF-label behaviour for degraded inputs).
                per_lf_ost_text_by_date=per_lf_ost_text_by_date,
                # Per-date GPS in day-master timeline + closure that
                # reverse-geocodes each trimmed-clip centroid into a per-
                # short OST label.
                gps_points_by_date=gps_points_by_date or None,
                short_ost_text_fn=short_ost_text_fn,
                short_regions_fn=short_regions_fn,
                state=state,
                on_day_complete=_on_day_complete,
                # Persist the front-load OST hash markers before the first
                # master encode so a mid-encode kill can't lose them.
                state_save_fn=state_save_fn,
            )
        else:
            fresh_artifacts = []
        artifacts = skipped_artifacts + fresh_artifacts
        logger.info(
            "  day-first: %d day(s) produced artifacts (%d skipped via per_day_completed, %d fresh)",
            len(artifacts),
            len(skipped_artifacts),
            len(fresh_artifacts),
        )

        # ── Completion gate: all planned days must produce artifacts ──
        # If any day raised an exception that was caught in run_all_days,
        # the artifact list will be shorter than day_batches. Detect this
        # here rather than letting Phase D proceed silently with partial
        # results (e.g. 4 LFs when 9 were expected).
        completed_dates = {a.date for a in artifacts}
        planned_dates = {b.date for b in day_batches}
        failed_dates = planned_dates - completed_dates
        if failed_dates:
            raise RuntimeError(
                f"Phase BC incomplete — {len(failed_dates)} day(s) produced no artifacts: "
                f"{sorted(failed_dates)}. Check per-day errors above, fix the root cause, then resume."
            )

        # ── Per-day bookkeeping: record successful day masters ────────
        # After each day's artifact lands successfully, write an entry
        # to `state.per_day_completed` keyed by date-string with the
        # master's absolute path. Subsequent resumes can short-circuit
        # the ffprobe via the skip path above. Only newly-produced
        # artifacts need recording (skipped days already have entries).
        # Bookkeeping is best-effort; the existing gate remains the
        # authoritative file-level check. Integrity on resume is covered
        # by the `master_record` size:mtime fingerprint, NOT a
        # whole-file hash of the ~60 GB master.
        # Base on live state: on_day_complete may already hold this date.
        per_day_completed = dict(state.per_day_completed or {})
        for art in fresh_artifacts:
            master = getattr(art, "master_path", None)
            if master is None or art.date in per_day_completed:
                continue
            master_path = Path(master)
            if not master_path.exists():
                logger.warning(
                    "  day-first: day %s master %s vanished before bookkeeping",
                    art.date,
                    master_path,
                )
                continue
            entry = {
                "master_path": str(master_path),
            }
            # Merge encode-time integrity metadata for cached-master
            # validation on resume after the raw episodes are purged.
            record = getattr(art, "master_record", None)
            if isinstance(record, dict):
                entry.update(record)
            # Mirror ``_record_day_and_save``'s post-cap bookkeeping
            # for any day this end-of-phase pass records first.
            planned_shorts_count = getattr(art, "planned_shorts_count", None)
            if planned_shorts_count is not None:
                entry["shorts_target_in_force"] = getattr(art, "shorts_target_in_force", None)
                entry["shorts_expected"] = planned_shorts_count
            per_day_completed[art.date] = entry
        # Phase D reads rendered paths from this record (Deliverables), never
        # from the flat state lists. Added beside the existing keys for EVERY
        # day (incl. reconstructed ones); a day with finals but no entry gets
        # the same master_path=None recovery marker ``_record_day_and_save``
        # writes.
        for art in artifacts:
            if art.date not in per_day_completed:
                if not getattr(art, "regular_lf_paths", None):
                    continue
                per_day_completed[art.date] = {"master_path": None}
            per_day_completed[art.date]["deliverables"] = _deliverables_record(art, state)
        state.per_day_completed = per_day_completed

        # -- Fan artifacts back out to the state keys Phase D reads
        regular_lf_paths: list = []
        shorts_per_lf: dict = {}
        per_lf_locations: dict = {}
        skeleton_lfs: list = []
        # Start fallback LF numbering above any real plan index so we
        # don't collide when the rendered artifact count exceeds plans.
        max_real_index = 0
        if real_plan and real_plan.regular_lfs:
            max_real_index = max(int(getattr(lf, "index", 0) or 0) for lf in real_plan.regular_lfs)
        next_lf_index = max_real_index + 1

        # Geocoder was built above (pre-``run_all_days``) so both the
        # OST burn-in path AND the title-builder step below share one
        # session budget. Reuse it here — do NOT re-construct.

        for artifact in artifacts:
            date_key = artifact.date
            # The previous ``day_lf_location`` local was
            # only used as the deleted ``day_fallback`` arg to
            # ``derive_per_lf_locations``. Removed; ``per_day_locations``
            # is still consulted by other parts of the loop via
            # ``per_day_locations.get(date_key, ...)``.

            # Real LF plans for this day (from the partitioning above).
            # If none, we still record the rendered LF paths but use
            # the day-wide label as a fallback.
            day_real_lfs = (lf_plans_by_date or {}).get(date_key, []) or []

            # Attribute the day's rendered shorts to its LFs by SOURCE-TIME
            # WINDOW. Each short was rendered from
            # a [start_sec, end_sec) slice of the day master; each LF is
            # assembled from one or more day-local segments that translate
            # to master-time intervals. A short belongs to LF X if its
            # midpoint lies inside any of LF X's segment intervals; orphans
            # (footage culled from every LF) attach to the nearest LF.
            #
            # Prior behaviour was equal-split by rank-position
            # (``divmod(len(shorts), len(lfs))``). That produced the
            # correct *count* per LF for the consolidated Metricool CSV
            # but the linkage was arbitrary — a short whose source frames
            # lived in LF #7's footage could be attributed to LF #1
            # because rank #1 always landed in bucket 0. Operator-side
            # auditing of "shorts under LF X come from LF X's footage"
            # was impossible. See
            # ``day_pipeline.assign_shorts_to_lfs_by_source_window``.
            from ..day_pipeline import (
                assign_shorts_to_lfs_by_source_window,
                compute_day_file_boundaries,
            )

            day_short_paths = list(getattr(artifact, "short_paths", None) or [])
            day_short_ranges = list(getattr(artifact, "short_ranges", None) or [])
            day_lf_count = max(1, len(artifact.regular_lf_paths))
            # Pair each rendered path to its plan by CARRIED INDEX, never by
            # list position: a mid-day render skip leaves a gap in
            # ``regular_lf_paths`` and any positional lookup would then shift a
            # later LF's file onto an earlier LF's plan/label/bucket.
            plan_by_index = {int(getattr(lf, "index", 0)): lf for lf in day_real_lfs}
            rendered_lf_indices = list(getattr(artifact, "regular_lf_indices", None) or [])
            use_carried = bool(rendered_lf_indices) and len(rendered_lf_indices) == len(
                artifact.regular_lf_paths
            )
            if use_carried:
                # Assign shorts against the ACTUAL rendered LFs' plans (matched
                # by index), not the first N plans — so a skipped LF's source
                # window is never mistaken for its later sibling's.
                day_lf_plans_for_assign = [
                    plan_by_index[ix] for ix in rendered_lf_indices if ix in plan_by_index
                ]
            else:
                day_lf_plans_for_assign = day_real_lfs[:day_lf_count]

            # Build day-local file_boundaries for source-window translation.
            day_batch_for_assign = next(
                (b for b in day_batches if b.date == date_key),
                None,
            )

            day_lf_shorts_buckets: list[list] = [[] for _ in range(day_lf_count)]
            if (
                day_short_ranges
                and len(day_short_ranges) == len(day_short_paths)
                and day_lf_plans_for_assign
                and day_batch_for_assign is not None
            ):
                file_boundaries = compute_day_file_boundaries(day_batch_for_assign)
                short_idx_by_lf = assign_shorts_to_lfs_by_source_window(
                    short_ranges=day_short_ranges,
                    lf_plans=day_lf_plans_for_assign,
                    file_boundaries=file_boundaries,
                )
                # Map LF.index → its position in the day's plan list so we
                # can fill the buckets aligned with the rendered LF order.
                idx_to_pos = {
                    int(getattr(lf, "index", 0)): pos
                    for pos, lf in enumerate(day_lf_plans_for_assign)
                }
                for lf_idx_key, short_indices in short_idx_by_lf.items():
                    pos = idx_to_pos.get(int(lf_idx_key))
                    if pos is None or pos >= day_lf_count:
                        continue
                    # Sort by source-time start so shorts within each LF
                    # publish chronologically (short_01 = earliest in LF),
                    # not by rank: everything is chronological by capture time.
                    valid_indices = [i for i in short_indices if 0 <= i < len(day_short_paths)]
                    valid_indices.sort(
                        key=lambda i: float(getattr(day_short_ranges[i], "start_sec", 0.0))
                    )
                    day_lf_shorts_buckets[pos] = [day_short_paths[i] for i in valid_indices]
            elif day_short_paths and day_lf_count > 1:
                # The legacy
                # equal-split fallback attributed shorts to arbitrary parent
                # LFs — wrong "▶ Full drive:" first-comment link, inherited
                # recording date, and trim ratio. With >1 LF and no
                # source-window data the attribution is ambiguous: hard-fail
                # instead of guessing.
                raise RuntimeError(
                    f"day {date_key}: cannot attribute {len(day_short_paths)} "
                    f"short(s) across {day_lf_count} LFs — short_ranges/"
                    f"lf_plans/day_batch unavailable (short_ranges="
                    f"{len(day_short_ranges)}, short_paths="
                    f"{len(day_short_paths)}). The legacy equal-split guess "
                    "links shorts to the wrong parent LF; re-run the day so "
                    "short_ranges persist in the day artifact, or repair it."
                )
            elif day_short_paths:
                # Exactly one LF: attribution is unambiguous by construction.
                day_lf_shorts_buckets[0] = list(day_short_paths)

            # Map each rendered LF path to its plan BY CARRIED INDEX so a
            # mid-day render skip leaves an index-keyed gap instead of
            # attributing a later LF's file to the previous LF's plan.
            for i, lf_path in enumerate(artifact.regular_lf_paths):
                regular_lf_paths.append(lf_path)
                if use_carried:
                    carried_idx = rendered_lf_indices[i]
                    lf_plan = plan_by_index.get(carried_idx)
                elif i < len(day_real_lfs):
                    carried_idx = None
                    lf_plan = day_real_lfs[i]
                else:
                    carried_idx = None
                    lf_plan = None
                if lf_plan is None:
                    # No plan for this rendered file (a render-skip gap with no
                    # carried plan, or more finals than plans). Synthesize,
                    # keyed off the carried index when we have one.
                    duration = _probe(lf_path) or 0.0
                    lf_plan = RegularLFPlan(
                        index=carried_idx if carried_idx is not None else next_lf_index,
                        segments=[],
                        montage_peaks=[],
                        shorts_target=volume_source.shorts_per_lf().value,
                        # Fall back to "ireland", matching
                        # the canonical --region default resolved in
                        # drive_orchestrator (region resume-preservation) —
                        # "usa" here steered publish slots/cadence for the
                        # other country's footage on region-less state.
                        region=state.region or "ireland",
                        total_duration=duration,
                    )
                skeleton_lfs.append(lf_plan)
                lf_idx_int = int(getattr(lf_plan, "index", next_lf_index))
                shorts_per_lf[lf_idx_int] = (
                    day_lf_shorts_buckets[i] if i < len(day_lf_shorts_buckets) else []
                )
                next_lf_index = max(
                    next_lf_index + 1,
                    int(getattr(lf_plan, "index", next_lf_index)) + 1,
                )

            # Reuse the per-LF labels computed BEFORE ``run_all_days``
            # (single source of truth for OST burn-in, the title
            # builder pass below, and downstream state). Fall back to an
            # on-the-fly derive for days with no pre-computed entry —
            # defensive: ``lf_plans_by_date`` could theoretically be
            # missing a key if a day has no real LF plans.
            day_labels = per_lf_locations_by_date.get(date_key)
            if day_labels is None:
                day_labels = derive_per_lf_locations(
                    day_real_lfs or [],
                    geocoder=geocoder,
                    gps_fallback=None,
                    overrides=per_lf_overrides,
                    # Defensive path: same wiring as the
                    # pre-run_all_days call above.
                    gps_points_by_lf=gps_points_by_lf or None,
                )
                per_lf_locations_by_date[date_key] = day_labels
            per_lf_locations.update(day_labels)

        # ── Flatten canonical OST text / titles ────────────────
        # Flatten the per-date canonical OST text map into a flat
        # ``{lf_index: canonical}`` dict — the single source of truth
        # for OST burn-in, YouTube title, and thumbnail text.
        # Defensive fallback:
        # any LF in ``skeleton_lfs`` that wasn't captured in the
        # pre-run_all_days derivation (e.g. ``lf_plans_by_date`` missed
        # the day) gets an on-the-fly build_lf_title() using its
        # rendered segments + per_lf_location.
        per_lf_ost_text: dict[int, str] = {}
        for _date_key, day_map in per_lf_ost_text_by_date.items():
            for lf_idx, canonical in day_map.items():
                if canonical:
                    per_lf_ost_text[int(lf_idx)] = canonical
        # For LFs on completed/skipped days, prefer the persisted
        # (already-burned) OST text over this run's fresh derivation — the
        # single source of truth Phase D reads for title + thumbnail. This runs
        # BEFORE _populate_resolved_locations so a completed LF's resolved model
        # is signature-matched to the burned text (fast-path, no re-geocode) and
        # BEFORE the defensive fallback (which skips indices already present).
        per_lf_ost_text = _prefer_persisted_ost_for_completed_days(
            per_lf_ost_text, _persisted_ost_text, per_lf_ost_text_by_date, _completed_dates
        )
        # Persist one canonical ResolvedLocation per LF (modal
        # county vote + regions/vote shares for the location invariant + is_us/country_code +
        # the winning-county road) so resume reloads instead of
        # re-deriving and every surface consumes it. Runs BEFORE the
        # defensive title fallback so that fallback's build_lf_title can read
        # the resolved (winning-county) road. Back-compat: a None field is
        # lazily initialised; LFs without a usable trace are skipped.
        _populate_resolved_locations(
            state,
            skeleton_lfs,
            gps_points_by_lf,
            geocoder,
            destination_by_lf,
            # The fresh per-LF OST text is this run's model signature —
            # a resume re-derives any model stamped against a stale title.
            per_lf_ost_text=per_lf_ost_text,
            endpoint_loss_by_lf=endpoint_loss_by_lf,
        )
        resolved_by_lf = state.per_lf_resolved_location or {}
        # Defensive on-the-fly fallback for any skeleton_lf missing a
        # canonical entry (degraded path: real LF plans were missing
        # pre-run_all_days, so defensively compute from rendered state).
        for lf in skeleton_lfs:
            lf_idx = int(getattr(lf, "index", 0))
            if lf_idx in per_lf_ost_text:
                continue
            segments = list(getattr(lf, "segments", None) or [])
            # Never let a skeleton LF silently inherit the drive-wide label.
            _assert_skeleton_lf_has_location(
                lf_idx,
                per_lf_locations,
                state.per_lf_overrides or {},
                drive_wide_location,
            )
            lf_location = per_lf_locations.get(lf_idx, drive_wide_location)
            # Read the winning-county road from the
            # canonical resolved model so the via-road disambiguator follows
            # the vote, not the centroid that produced the (road) suffix.
            _resolved = resolved_by_lf.get(lf_idx)
            _resolved_road = getattr(_resolved, "road", None) if _resolved else None
            # Thread the canonical travel-ordered place sequence so the
            # title anchors on a recognizable settlement when the per-endpoint
            # geocode misses, instead of collapsing to the bare county.
            _resolved_places = (
                getattr(_resolved, "ordered_place_sequence", None) if _resolved else None
            )
            # The once-resolved destination POI anchors the title
            # ahead of the A→B cascade (prefer the persisted model, else the
            # precomputed map for an LF missing a resolved entry).
            _resolved_destination = (
                getattr(_resolved, "destination", None) if _resolved else None
            ) or destination_by_lf.get(lf_idx)
            # Operator override is AUTHORITATIVE (mirrors the primary derivation
            # in gps_enrichment) — burn it verbatim rather than letting the GPS
            # cascade reshape it.
            _override = (state.per_lf_overrides or {}).get(lf_idx)
            if isinstance(_override, str) and _override.strip():
                canonical = _override.strip()
                canonical = validate_burned_ost_text(canonical, lf_location, lf_idx)
                if canonical:
                    per_lf_ost_text[lf_idx] = canonical
                continue
            try:
                canonical = build_lf_title(
                    segments,
                    geocoder=geocoder,
                    per_lf_location=lf_location,
                    gps_points=gps_points_by_lf.get(lf_idx) if gps_points_by_lf else None,
                    resolved_road=_resolved_road,
                    resolved_places=_resolved_places,
                    resolved_destination=_resolved_destination,
                )
            except Exception as exc:  # noqa: BLE001 — best-effort
                logger.warning(
                    "title_builder.build_lf_title failed for LF %d: %s — "
                    "falling back to per-LF location string",
                    lf_idx,
                    exc,
                )
                canonical = lf_location
            canonical = validate_burned_ost_text(canonical, lf_location, lf_idx)
            if canonical:
                per_lf_ost_text[lf_idx] = canonical
        # ``state["per_lf_ost_text"]`` is the UNSUFFIXED canonical
        # text — downstream surfaces (OST burn-in, thumbnail) read this.
        # YouTube titles need cross-LF uniqueness, so we build
        # ``per_lf_titles`` as the dedup-suffixed variant on top. The
        # OST burn itself has already happened (in run_day_pipeline via
        # ``per_lf_ost_text_by_date``), so this flat dict is for
        # downstream surfaces only.
        # Snapshot per-LF recording
        # date from the day-batch partition. Phase D used to derive
        # this via ``episode_dates_list[segments[0].source_file_index]``
        # but ``_partition_lf_plans_by_date`` already remapped that
        # index to DAY-LOCAL — so the lookup always returned the first
        # global episode's date. The day-batch keys are the truth.
        #
        # The day-batch derivation only fires on the FRESH-encode path —
        # it requires ``lf_plans_by_date`` (which itself requires a real
        # ``DrivePlan`` and a non-empty episode_dates list). Both resume
        # fast-paths in this phase (master-on-disk and finals-only)
        # populate ``regular_lf_paths`` from the ``day_<DATE>/regular_lfs``
        # layout WITHOUT going through ``_partition_lf_plans_by_date``,
        # so they would leave ``per_lf_recording_date`` empty and the
        # YouTube description would lack its "📅 Filmed:" line. The day_<DATE>
        # directory name is the on-disk source of truth, so we layer a
        # path-derived fallback on top of the partition-derived map for
        # any LF that the partition path missed.
        per_lf_recording_date = _build_per_lf_recording_date_map(lf_plans_by_date)
        path_derived = _derive_per_lf_recording_date_from_paths(regular_lf_paths, skeleton_lfs)
        _record_recording_date_recovery(state, per_lf_recording_date, path_derived)
        for lf_idx, date_str in path_derived.items():
            per_lf_recording_date.setdefault(lf_idx, date_str)
        state.per_lf_recording_date = per_lf_recording_date

        # ── State-coherence invariant ────────────────────────────────
        # Every LF rendered by this phase must have a recording date.
        # An empty map at this exit point means a future Phase BC code
        # path was added (a third fast-path, an alternate ingest plan,
        # etc.) without populating the field — fail loudly here, NOT
        # silently mid-Phase D where the symptom is "no Filmed line in
        # the published description". Skeleton-LF indices are the
        # authoritative roster (matches ``regular_lf_paths`` 1:1).
        missing_indices = [
            int(getattr(lf, "index", 0))
            for lf in skeleton_lfs
            if int(getattr(lf, "index", 0)) not in per_lf_recording_date
        ]
        if missing_indices:
            raise RuntimeError(
                "phase_bc_day_first: per_lf_recording_date map is missing "
                f"entries for LF indices {missing_indices} (rendered "
                f"{len(skeleton_lfs)} LFs, mapped "
                f"{len(per_lf_recording_date)}). The day_<DATE> directory "
                "structure on disk is the source of truth — "
                "_derive_per_lf_recording_date_from_paths should cover "
                "any LF that the partition path missed."
            )
        from ..longform_metadata import load_published_titles

        state.per_lf_ost_text = dict(per_lf_ost_text)
        # Dedup via (Part N of M) suffix on collisions — TITLE ONLY. Also
        # dedup FINAL titles against already-published channel titles so an
        # upload can't silently duplicate a live video's title.
        # ``_channel_title_ids`` was refreshed once, earlier in this method.
        per_lf_titles = ensure_unique_titles(
            dict(per_lf_ost_text),
            published_titles=load_published_titles() | set(_channel_title_ids),
            published_title_ids=_channel_title_ids,
        )
        state.per_lf_titles = per_lf_titles

        _assert_post_encoding_invariants(
            state=state,
            skeleton_lfs=skeleton_lfs,
            regular_lf_paths=regular_lf_paths,
        )

        # Operator-side per-LF route stats sidecar (additive, never gates).
        # See `gomoveshift/pipeline/route_stats_sidecar.py`.
        from ..route_stats_sidecar import write_route_stats_sidecar

        write_route_stats_sidecar(output_base, state)

        # Per-LF provenance sidecar (additive, never gates).
        from ..lf_provenance_sidecar import (
            write_lf_provenance_sidecar,
            write_shorts_provenance_sidecar,
        )

        write_lf_provenance_sidecar(output_base, state)
        # Same moment, same contract: the shorts' own cut-from-master
        # identity, so the surface with the largest audience per file can say
        # which master it came from. No-ops when the run rendered no shorts.
        write_shorts_provenance_sidecar(output_base, state)

        # Calculate aggregate durations for the synthetic DrivePlan
        total_source_duration = sum(state.file_durations or [])
        total_highlight_duration = sum(lf.total_duration for lf in skeleton_lfs)

        # Sort by LF index so Phase D's slot-assignment loop assigns schedule
        # dates in recording order regardless of artifact processing order.
        # Finals-only days process before fresh-encode days, so without this
        # sort later-trip LFs accumulate first and get the earliest publish slots.
        if len(skeleton_lfs) > 1:
            _paired = sorted(
                zip(skeleton_lfs, regular_lf_paths),
                key=lambda x: int(getattr(x[0], "index", 0)),
            )
            skeleton_lfs, regular_lf_paths = map(list, zip(*_paired))

        drive_plan = DrivePlan(
            source_files=[Path(p) for p in state.episode_files or []],
            processed_files=[Path(p) for p in state.episode_files or []],
            regular_lfs=skeleton_lfs,
            score_threshold=0.0,
            total_source_duration=total_source_duration,
            total_highlight_duration=total_highlight_duration,
        )

        state.day_artifacts = artifacts
        state.drive_plan = drive_plan
        state.regular_lf_paths = regular_lf_paths
        state.shorts_per_lf = shorts_per_lf
        state.per_lf_locations = per_lf_locations

        # Assign each planned short a STABLE global index (plan
        # order — ascending LF index, then list order within each LF), so
        # Phase D consumes a persisted index instead of recomputing
        # global_index_offset from len(posts). ``existing=`` honours any
        # indexes already persisted on a resumed run — an index, once
        # assigned, never changes and is never reused.
        from ..social_generator import assign_global_short_indexes

        state.short_global_index = assign_global_short_indexes(
            shorts_per_lf, existing=state.short_global_index
        )

        # Aggregated BC→D boundary contract. Runs AFTER every
        # per-field _assert_per_lf_*_complete and AFTER every state[...]
        # write above — so a single PhaseBCExitContractError surfaces ALL
        # missing per-LF maps in one shot. Symmetric to the D-entry
        # aggregator at PhaseD.run start.
        assert_phase_bc_per_lf_complete(state, skeleton_lfs)

        logger.info(
            "  day-first: %d regular LF(s) total",
            len(regular_lf_paths),
        )

    def _reconstructed_shortfall_target(
        self, state, num_lfs: int, entry: Optional[dict] = None
    ) -> Optional[int]:
        """Required TOTAL shorts for a reconstructed day at the CURRENT EXPLICIT
        target, times the day's LF count. The only explicit target is
        ``--shorts-target`` (``state.shorts_target``; config carries none), and it
        is a cap on Waypost's decided count, never a raise
        (``volume_source.apply_flag``) -- the same per-LF number a fresh day plans.

        When ``entry`` (the day's ``per_day_completed`` record) carries
        both ``shorts_target_in_force`` and ``shorts_expected`` AND the CURRENT
        per-LF target is unchanged from the one the day actually ran under,
        return the recorded post-per-LF-capacity-cap expectation instead —
        a day legitimately capped below ``per_lf * num_lfs`` by
        ``cap_shorts_to_per_lf_capacity`` must not be misread as short. A
        target that has since been RAISED (or an entry with no keys, e.g. a
        day recorded before these keys existed) falls through to the strict derivation below.

        Deliberately NOT the pure publish-plan derivation otherwise: a
        reconstructed day already satisfied that derived count when it was
        completed under the same config, so re-deriving it can't reveal a new
        shortfall. The case this guards is a resume that RAISED the explicit
        target — that's what the strict fallback re-checks. ``None`` (no
        explicit target set) disables the gate.
        """
        flag = getattr(state, "shorts_target", None)
        if flag is None:
            return None
        per_lf = volume_source.apply_flag(volume_source.shorts_per_lf().value, flag)
        if (
            isinstance(entry, dict)
            and "shorts_target_in_force" in entry
            and "shorts_expected" in entry
            and entry["shorts_target_in_force"] == per_lf
        ):
            return entry["shorts_expected"]
        return per_lf * num_lfs

    def _enforce_reconstructed_shortfall(
        self, state, reconstructed, entry: Optional[dict] = None
    ) -> None:
        """Re-run the shorts-shortfall gate on a resume-reconstructed day.

        The ``per_day_completed`` fast-paths trust a day's on-disk finals without
        re-planning shorts — so a resume that RAISED ``--shorts-target`` would
        quietly ship the old, lower count. Count-match the reconstructed shorts
        against the CURRENT required target here (``--allow-short-count`` still
        escapes), exactly as a fresh day is gated before shipping. ``entry`` is
        this day's ``per_day_completed`` record, threaded through to
        :meth:`_reconstructed_shortfall_target` so a per-LF-capacity-capped day
        can be recognized rather than misread as a shortfall.
        """
        num_lfs = len(getattr(reconstructed, "regular_lf_paths", None) or [])
        if num_lfs == 0:
            return
        required = self._reconstructed_shortfall_target(state, num_lfs, entry)
        produced = len(getattr(reconstructed, "short_paths", None) or [])
        allow = bool(getattr(state, "allow_short_count", None))
        _enforce_shorts_contract(
            produced, required, reconstructed.date, allow, stage="reconstructed"
        )

    def _partition_batches_by_resume_state(self, state, day_batches, stages_dir):
        # ── Resume safety net: skip days already recorded as done ─────
        # `per_day_completed` is the outer
        # bookkeeping layer on top of the per-stage file-existence +
        # duration-probe gate inside `encode_day_master`. When present,
        # it lets us skip a day's master re-probe entirely: if the
        # recorded master_path still exists on disk, we trust it and
        # reconstruct `DayArtifacts` from the day directory instead of
        # calling `run_day_pipeline` (which would ffprobe the master).
        # Legacy state files with no record fall through to the
        # existing gate — backward-compat is preserved.
        recorded = state.per_day_completed or {}
        skipped_artifacts: list = []
        remaining_batches: list = []
        # State first (merge_on_resume has already resolved CLI intent onto it
        # by the time a resume reaches here), config as the non-resume default.
        superlong_required = bool(
            getattr(state, "superlong_enabled", None)
            or getattr(self.config.drive, "superlong_enabled", False)
        )
        for batch in day_batches:
            entry = recorded.get(batch.date) if isinstance(recorded, dict) else None
            master_path_str = (entry or {}).get("master_path") if entry else None
            day_dir = stages_dir / f"day_{batch.date}"

            if superlong_required and not day_superlong_complete(batch.date, day_dir, state):
                # Superlong's only producer lives in run_day_pipeline's tail,
                # which NEITHER fast-path below reaches. A day with LFs+shorts
                # but no sl_NN.mp4 is therefore not complete for THIS run —
                # route it back to the encoder rather than report a green run
                # with zero superlong output.
                master = day_dir / f"day_{batch.date}_master.mp4"
                logger.warning(
                    "  day-first: day %s has finals but an incomplete superlong set — "
                    "re-running it (--superlong). Day master %s.",
                    batch.date,
                    "on disk: reused, no re-encode"
                    if master.is_file()
                    else "ABSENT: it must be re-encoded first — superlong parts can "
                    "only be cut from the day master",
                )
                remaining_batches.append(batch)
                continue

            master_path_obj = Path(master_path_str) if master_path_str else None
            master_passes_integrity = False
            if master_path_obj is not None and master_path_obj.exists():
                expected_duration = float(getattr(batch, "total_duration", 0.0) or 0.0)
                master_passes_integrity = _probe_master_integrity(
                    master_path_obj,
                    expected_duration,
                    batch.date,
                    self.config.drive.ffprobe_timeout_sec,
                )

            if master_passes_integrity:
                # Fast-path #1: master still on disk and (when expected
                # duration is known) passes the integrity probe.
                logger.info(
                    "  day-first: day %s already in per_day_completed with "
                    "master %s — skipping re-probe + re-encode",
                    batch.date,
                    master_path_str,
                )
                reconstructed = _reconstruct_day_artifacts_from_disk(
                    date=batch.date,
                    master_path=master_path_obj,
                    day_dir=day_dir,
                )
                self._enforce_reconstructed_shortfall(state, reconstructed, entry)
                skipped_artifacts.append(reconstructed)
            elif entry and _day_dir_has_finals(day_dir):
                # Fast-path #2: master was dropped post-run but the
                # day's finals are all on disk. Downstream phases (Phase D)
                # read only ``regular_lf_paths`` / ``short_paths`` — no
                # consumer needs the master after ``run_day_pipeline``
                # returns. Reconstruct with master_path=None.
                logger.info(
                    "  day-first: day %s master dropped but finals present — "
                    "skipping re-encode (finals-only reconstruction)",
                    batch.date,
                )
                reconstructed = _reconstruct_day_artifacts_from_disk(
                    date=batch.date,
                    master_path=None,
                    day_dir=day_dir,
                )
                self._enforce_reconstructed_shortfall(state, reconstructed, entry)
                skipped_artifacts.append(reconstructed)
            else:
                remaining_batches.append(batch)

        return recorded, skipped_artifacts, remaining_batches


def _prefer_persisted_ost_for_completed_days(
    fresh_by_lf: dict,
    persisted_by_lf: dict,
    per_lf_ost_text_by_date: dict,
    completed_dates: set,
) -> dict:
    """Preserve the already-burned OST text for LFs on completed days.

    On resume, a day recorded in ``per_day_completed`` whose master/finals are
    still on disk (``completed_dates`` — the dates this run SKIPS) has FIXED OST
    pixels. Re-deriving its title/thumbnail text from a fresh geocode pass (after
    a geocode-cache roll or a code change) can DIVERGE from the burned frames,
    and Phase D reads ``state.per_lf_ost_text`` for both the YouTube title and
    the thumbnail. For every LF on such a day, prefer the persisted
    (confirmed-burned) text; LFs on days actually re-encoded this run keep the
    fresh derivation. A fresh run (no persisted text / no completed days) is a
    verbatim no-op, so the happy path never regresses.
    """
    merged = dict(fresh_by_lf)
    if not persisted_by_lf or not completed_dates:
        return merged
    for date_key, day_map in (per_lf_ost_text_by_date or {}).items():
        if date_key not in completed_dates:
            continue
        for lf_idx in day_map:
            lf_i = int(lf_idx)
            persisted = persisted_by_lf.get(lf_i)
            if isinstance(persisted, str) and persisted.strip():
                merged[lf_i] = persisted
    return merged


def _assert_drive_plan_built(real_plan, state, day_batches) -> None:
    """Hard-fail a silent plan-build failure before the encode.

    ``real_plan`` None → empty ``lf_plans_by_date`` → the per-LF resolvable
    gate iterates nothing and ``run_all_days`` encodes on the drive-wide
    fallback cascade. Refuse — but ONLY when the plan SHOULD have built:
    footage queued (a day batch with episodes) AND every
    ``_build_drive_plan_from_state`` input present (scores + boundaries +
    files). A legitimate no-footage / inputs-missing path is left untouched.
    """
    if real_plan is not None:
        return
    has_footage = any(getattr(b, "episodes", None) for b in (day_batches or []))
    inputs_ready = (
        getattr(state, "drive_scores", None) is not None
        and bool(getattr(state, "file_boundaries", None) or [])
        and bool(getattr(state, "episode_files", None) or [])
    )
    if not (has_footage and inputs_ready):
        return
    n_batches = sum(1 for b in (day_batches or []) if getattr(b, "episodes", None))
    raise ValueError(
        "phase_bc: cannot enter encoding — drive plan build FAILED (no "
        f"DrivePlan) for {n_batches} day batch(es) with footage despite "
        "complete inputs (drive scores + file boundaries + episode files). "
        "Encoding now would resolve ZERO per-LF labels and burn the drive-wide "
        "fallback on every LF/short. Re-run with --fresh to rebuild "
        "drive_plan.json, or supply per-LF overrides "
        "(--per-lf-location '0=<route>'; --list-lfs shows indices)."
    )


def _assert_skeleton_lf_has_location(
    lf_idx: int,
    per_lf_locations: dict,
    overrides: dict,
    drive_wide_location: str,
) -> None:
    """Gate a synthesized skeleton LF against the drive-wide fallback.

    The post-``run_all_days`` fallback OST loop synthesizes a ``RegularLFPlan``
    (empty segments) for a render-skip gap. A skeleton absent from
    ``per_lf_locations`` used to silently inherit the drive-wide label via
    ``per_lf_locations.get(idx, drive_wide_location)`` — raise instead.

    Scope: fires only when SOME labels resolved (``per_lf_locations`` non-empty)
    but THIS skeleton is missing — an LF synthesized beyond the resolved plan.
    A wholly-empty map means the plan never resolved (real_plan None), governed
    upstream by ``_assert_drive_plan_built``; a real run_day_pipeline
    yields zero LFs there, so there is nothing to gate.
    """
    if not (per_lf_locations or {}):
        return
    if lf_idx in per_lf_locations:
        return
    _override = (overrides or {}).get(lf_idx)
    if isinstance(_override, str) and _override.strip():
        return
    raise ValueError(
        "phase_bc: cannot burn OST — skeleton LF index "
        f"{lf_idx} has no resolved per-LF location and no --per-lf-location "
        f"override; refusing the drive-wide fallback label {drive_wide_location!r}. "
        f"Re-run with --list-lfs, then add --per-lf-location '{lf_idx}=<route>'."
    )


def _build_drive_plan_from_state(config: Config, state) -> Optional[DrivePlan]:
    """Build a DrivePlan (with real segments + montage peaks) from the
    in-memory state produced by phase_a. Returns None when required
    inputs are missing."""
    drive_scores = state.drive_scores
    file_boundaries = state.file_boundaries or []
    episode_files = state.episode_files or []
    if drive_scores is None or not file_boundaries or not episode_files:
        logger.info(
            "  day-first: cannot build drive_plan — scores/boundaries/files missing",
        )
        return None
    processed = [Path(p) for p in episode_files]
    episode_dates = state.episode_dates or []
    region = state.region or ""
    gps_per_episode = state.gps_per_episode
    assembler = LongFormAssembler(config.drive)
    # full_drive planning REQUIRES a stop mask (the planner raises without
    # one, and nothing here built it — a full_drive run would
    # die at _assert_drive_plan_built). Build the drive-wide mask from the
    # same GPS-speed/optical knobs the day pipeline uses. best_moments keeps
    # its threshold path unchanged (the day pipeline masks its own shorts).
    stop_mask = None
    if getattr(config.drive, "lf_selection_mode", "best_moments") == "full_drive":
        stop_mask = _build_drive_stop_mask(config, state)
    try:
        if (
            episode_dates
            and len(episode_dates) == len(file_boundaries)
            and getattr(config.drive, "adaptive_threshold_enabled", False)
        ):
            return assembler.build_drive_plan_per_day(
                scores=drive_scores,
                processed_files=processed,
                file_boundaries=file_boundaries,
                episode_dates=episode_dates,
                gps_per_episode=gps_per_episode,
                region=region,
                stop_mask=stop_mask,
            )
        return assembler.build_drive_plan(
            scores=drive_scores,
            processed_files=processed,
            file_boundaries=file_boundaries,
            gps_per_episode=gps_per_episode,
            region=region,
            stop_mask=stop_mask,
        )
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        # Expected transient/IO/parse/validation failures (e.g. a missing
        # score-cache file, a corrupt JSON sidecar, or the assembler's
        # ``ValueError`` on a length-mismatched ``episode_dates`` /
        # ``file_boundaries``) degrade gracefully to ``None`` so the caller
        # falls back to a fresh rebuild. Programming errors
        # (``AttributeError`` / ``TypeError`` / ``AssertionError``) are NOT
        # caught here — swallowing them would silently rebuild LFs from
        # possibly-corrupt segments and mask a real defect.
        logger.warning("  day-first: build_drive_plan failed: %s", exc)
        return None


def _build_drive_stop_mask(config: Config, state) -> Optional[np.ndarray]:
    """Drive-wide per-second STOP mask aligned to ``state.drive_scores`` rows.

    GPS ground speed is authoritative (``_build_drive_gps_points`` projects
    every episode's trackpoints onto the drive timeline); seconds with no
    measurable speed fall back to optical motion — exactly what
    ``run_day_pipeline`` does per day, with the same ``stop_mask_kwargs``.
    ``LongFormAssembler`` slices it per day with the same row selector it
    uses for the scores. ``None`` when ``config.drive.stop_mask_enabled`` is
    off or there are no scores.
    """
    if not getattr(getattr(config, "drive", None), "stop_mask_enabled", True):
        return None
    scores = state.drive_scores
    if scores is None or len(scores) == 0:
        return None
    points = _build_drive_gps_points(state.file_boundaries or [], state.gps_per_episode)
    # build_stop_mask indexes rows as seconds from row 0; rebase the GPS to
    # the first score row so both share one origin.
    base = float(scores[0, COL_TIMESTAMP])
    if base:
        points = [(t - base, lat, lon) for t, lat, lon in points]
    mask = build_stop_mask(scores, points or None, **stop_mask_kwargs(config))
    summary = mask_summary(mask)
    logger.info(
        "  day-first: drive stop mask — %d/%d s masked (%.1f%%) from %d GPS fix(es)",
        summary["masked_seconds"],
        summary["total_seconds"],
        summary["masked_fraction"] * 100,
        len(points),
    )
    return mask


def _build_per_lf_recording_date_map(lf_plans_by_date) -> dict:
    """Translate ``{date: [lf, ...]}`` (the output of
    ``_partition_lf_plans_by_date``) into ``{lf.index: "YYYY-MM-DD"}``.

    phase_d cannot derive a per-LF
    recording_date by indexing into ``state["episode_dates"]`` because
    ``_partition_lf_plans_by_date`` has already remapped every segment's
    ``source_file_index`` from GLOBAL to DAY-LOCAL. The day-batch keys
    of ``lf_plans_by_date`` are the authoritative source of each LF's
    recording day. We snapshot that mapping into state here so phase_d
    can read it directly. ``lf.index`` is NOT remapped, so it is safe
    as the dict key.

    Date values may be ``str`` ("YYYY-MM-DD") or ``datetime.date``;
    both are normalized to the canonical "YYYY-MM-DD" string format
    that phase_d / ``LongFormMetadataGenerator`` expect.
    """
    if not lf_plans_by_date:
        return {}
    out: dict[int, str] = {}
    for date, day_lfs in lf_plans_by_date.items():
        if isinstance(date, str):
            date_str = date.strip()
        elif hasattr(date, "strftime"):
            date_str = date.strftime("%Y-%m-%d")
        else:
            continue
        if not date_str:
            continue
        for lf in day_lfs or []:
            lf_idx = getattr(lf, "index", None)
            if lf_idx is None:
                continue
            out[int(lf_idx)] = date_str
    return out


def _ladder_forward_geocode(town: str):
    """Defensive forward-geocode for the ladder's endpoint-proximity check.

    Wraps ``_nominatim_search`` (the one (name)->(lat, lon) geocoder in the
    codebase) so any failure — import, network, malformed response — degrades
    to ``None`` and the gate skips the check instead of crashing pre-encode.
    """
    try:
        from ..chapters import nominatim_source

        return nominatim_source._nominatim_search(town)
    except Exception:  # noqa: BLE001 — a geocoder failure never blocks an encode
        return None


def _forward_geocode_places(query: str) -> list:
    """Defensive settlement-candidate lookup for trace-verifying label endpoints.

    Wraps ``_nominatim_places`` so any failure degrades to ``[]`` — nothing
    verified, and the location invariant then fails closed exactly as before.
    The failure is logged so it is not mistaken for "no settlement of that name".
    """
    try:
        from ..chapters import nominatim_source

        return nominatim_source._nominatim_places(query)
    except Exception as exc:  # noqa: BLE001 — unverified, never a crash
        logger.warning("TRACE-UNVERIFIED-PLACE: %r — geocoder failed (%s)", query, exc)
        return []


def _verify_trace_places(rl, label, points, places_fn) -> None:
    """Stamp ``rl.trace_verified_places`` for ``label`` from the LF's own trace.

    No-op (leaves ``None``, "not computed") without a geocoder, a label or GPS.
    """
    if places_fn is None or not label or not points:
        return
    from ..validation.content_gates import (
        resolved_location_gate_dict,
        trace_verified_endpoint_towns,
    )

    rl.trace_verified_places = trace_verified_endpoint_towns(
        label, resolved_location_gate_dict(rl), points, places_fn
    )


def _build_ladder_evidence(
    per_lf_ost_text,
    gps_points_by_lf,
    per_lf_overrides,
    lf_plans=None,
    resolved_by_lf=None,
    geocode_fn=None,
) -> dict:
    """Per-LF evidence for the pre-encode specificity ladder.

    Wires, per LF: the raw trace (``gps_points``), the FOOTAGE DURATION from
    the LF plan's segments (sum of segment spans; the gate's fallback
    window is the fixes' own min/max, which cannot see a dead prefix, and a
    single same-clock window can't represent stop-removed non-contiguous
    footage), the winning-county road refs off the resolved model normalized
    through ``_road_ref_members`` (raw OSM ``"I 81;US 11"`` would
    substring-match a hyphenated label 0% and false-fail a correct via-road),
    and a forward geocoder for the endpoint-proximity check. Missing
    inputs omit their key → the gate no-ops that check. An LF with an operator
    ``--per-lf-location`` override is ``authoritative`` (bypassed).

    A below-tier-1 LF that still VOTED its OWN dominant county /
    state supplies a demotion signal FROM THAT MODEL — a voted county is the one
    corroboration covering the window (``corroborations=1`` → tier 3), else a
    voted state alone stands in for the in-state bracket (``has_bracket_route``
    → tier 4) — so ``award_tier`` demotes it and ``render_degraded_tier_label``
    burns the LF's own broad-but-true claim instead of the specificity-ladder gate hard-failing
    a tier-1 label nothing could demote. Region source is THIS LF's
    ``ResolvedLocation`` only (no bracket routing / drive-wide / adjacent-LF
    fallback — a fully GPS-dead LF with no model is still batch-fatal). The
    signal only bites below tier 1: ``award_tier`` returns tier 1 for
    coverage ≥ the tier-1 floor regardless of these keys.
    """
    from ..title_builder import _road_ref_members

    gps_by_lf = gps_points_by_lf or {}
    resolved = resolved_by_lf or {}
    durations: dict[int, float] = {}
    for lf in lf_plans or []:
        total = sum(
            max(
                0.0,
                float(getattr(s, "end_time", 0.0) or 0.0)
                - float(getattr(s, "start_time", 0.0) or 0.0),
            )
            for s in getattr(lf, "segments", None) or []
        )
        if total > 0:
            durations[int(getattr(lf, "index", 0))] = total

    out: dict[int, dict] = {}
    for i in per_lf_ost_text or {}:
        idx = int(i)
        rl = resolved.get(idx)
        ev: dict = {"gps_points": gps_by_lf.get(idx)}
        duration = durations.get(idx)
        if duration:
            ev["duration"] = duration
        raw_refs = getattr(rl, "road_refs", None) or []
        refs = [";".join(_road_ref_members(str(r))) for r in raw_refs if str(r).strip()]
        if refs:
            ev["road_refs"] = refs
        if getattr(rl, "dominant_county", None):
            ev["corroborations"] = 1
        elif getattr(rl, "dominant_state", None):
            ev["has_bracket_route"] = True
        if callable(geocode_fn):
            ev["geocode_fn"] = geocode_fn
        out[idx] = ev
    return out


_TIER_EVIDENCE_SCALAR_KEYS = (
    "corroborations",
    "has_bracket_route",
    "duration",
    "road_refs",
)


def _serializable_tier_evidence(ev: dict) -> dict:
    """JSON-safe projection of the ladder evidence, for the provenance record.

    The live evidence dict carries the LF's raw GPS trace and a ``geocode_fn``
    callable — neither can go through ``pipeline_state.json``. Recording a fixed
    scalar projection (plus the trace's point COUNT) makes an unserializable
    state file impossible by construction instead of gating on one later.
    """
    out: dict = {k: ev[k] for k in _TIER_EVIDENCE_SCALAR_KEYS if ev.get(k) is not None}
    points = ev.get("gps_points")
    out["gps_point_count"] = len(points) if points is not None else 0
    return out


def _degrade_labels_to_awarded_tier(
    per_lf_ost_text: dict, resolved_by_lf: dict, evidence_by_lf: dict, *, state=None
) -> "dict[int, str]":
    """Re-render each below-tier-1 (but renderable) LF's burned OST at its
    AWARDED specificity tier → ``{lf_idx: degraded_label}`` for LFs that
    demoted. Tier from the same evidence the specificity-ladder gate reads
    (``award_tier_for_evidence``); the FORM from the LF's ``ResolvedLocation``
    (road/county/state) via ``render_degraded_tier_label``. Tier 1 keeps its
    full label; tier 5 renders nothing (the gate hard-fails a no-claim); an
    operator ``authoritative`` override is never demoted. The re-rendered label
    is no longer tier-1-shaped, so the gate then PASSES it.

    ``state``, when supplied, additionally RECORDS every evaluated LF's award
    (``per_lf_gps_coverage_pct`` as a 0–100 percent, ``per_lf_awarded_tier``,
    ``per_lf_tier_evidence``) so the provenance sidecar can prove which tier
    each burned label was awarded and on what evidence. Recording only — the
    demotion decision itself is untouched.
    """
    from ..title_builder import normalize_state_abbr, render_degraded_tier_label
    from ..validation.content_gates import award_tier_for_evidence

    demoted: dict[int, str] = {}
    coverage_by_lf: dict[int, float] = {}
    tier_by_lf: dict[int, int] = {}
    evidence_record: dict[int, dict] = {}
    for lf_idx in per_lf_ost_text or {}:
        idx = int(lf_idx)
        ev = (evidence_by_lf or {}).get(idx)
        rl = (resolved_by_lf or {}).get(idx)
        if not ev or rl is None or ev.get("authoritative"):
            continue
        coverage, tier = award_tier_for_evidence(ev)
        if tier is not None:
            tier_by_lf[idx] = int(tier)
            evidence_record[idx] = _serializable_tier_evidence(ev)
            if coverage is not None:
                coverage_by_lf[idx] = round(float(coverage) * 100, 1)
        if tier is None or tier == 1 or tier >= 5:
            continue
        state_full = getattr(rl, "dominant_state", None)
        label = render_degraded_tier_label(
            tier,
            road=getattr(rl, "road", None),
            county=getattr(rl, "dominant_county", None),
            region=normalize_state_abbr(state_full) or state_full,
            state=state_full,
        )
        if label:
            demoted[idx] = label
    if state is not None and tier_by_lf:
        state.per_lf_gps_coverage_pct = {**(state.per_lf_gps_coverage_pct or {}), **coverage_by_lf}
        state.per_lf_awarded_tier = {**(state.per_lf_awarded_tier or {}), **tier_by_lf}
        state.per_lf_tier_evidence = {**(state.per_lf_tier_evidence or {}), **evidence_record}
    return demoted


def _apply_demoted_labels(
    demoted: dict, per_lf_ost_text: dict, per_lf_ost_text_by_date: dict, state
) -> None:
    """Write the demoted labels to every burn surface in lockstep: the flat
    pre-seed map (specificity-ladder gate + ``state.per_lf_ost_text``/``per_lf_titles`` for
    Phase D's YouTube title) AND the per-date map threaded into ``run_all_days``
    (the OST burn-in) — so pixels, audit preview, and title agree.

    This is a pre-encode PREVIEW: the authoritative post-encode dedup below
    (which does see the channel refresh) overwrites ``per_lf_titles``
    before Phase D reads it, so an unrefreshed ledger here is a missed
    early-preview detection at worst, never a shipped duplicate.
    """
    from ..longform_metadata import load_published_titles

    for lf_idx, label in demoted.items():
        per_lf_ost_text[int(lf_idx)] = label
        for day_map in (per_lf_ost_text_by_date or {}).values():
            if int(lf_idx) in day_map:
                day_map[int(lf_idx)] = label
    state.per_lf_ost_text = dict(per_lf_ost_text)
    state.per_lf_titles = ensure_unique_titles(
        dict(per_lf_ost_text), published_titles=load_published_titles()
    )


def _assert_pre_encode_location_invariant(
    per_lf_ost_text: dict, resolved_by_lf: dict, evidence_by_lf: "dict | None" = None
) -> None:
    """Refuse to START an encode whose burned OST/title text fails the
    location invariant (the pre-encode half of the gate) OR the
    specificity ladder.

    The per-LF title/OST text is fixed HERE in Phase BC, but the desync gate
    used to run only in Phase D (post-encode, pre-upload) — a far-road label
    cost the full multi-hour encode before it was caught. Reconcile each LF's
    canonical OST text against its freshly-built ``ResolvedLocation`` BEFORE
    ``run_all_days`` via the same ``check_location_invariant`` Phase D uses,
    and raise the same ``RuntimeError`` contract as
    ``assert_location_invariant_hard_fail``. LFs with no resolved model
    (GPS-less / quorum-degraded) skip, matching the Phase D no-op; Phase D's
    post-encode check remains the backstop for resume paths and late title
    mutations (dedup suffixes). ``evidence_by_lf`` (production call site)
    additionally reconciles each label against its RAW footage evidence via
    :func:`check_specificity_ladder`; ``None`` is a full no-op (no ladder evidence).
    """
    from ..validation.content_gates import (
        check_location_invariant,
        check_specificity_ladder,
        resolved_location_gate_dict,
    )

    failures: list[str] = []
    for lf_idx, ost_text in sorted(per_lf_ost_text.items()):
        rl = resolved_by_lf.get(int(lf_idx))
        if rl is None:
            continue
        gate = check_location_invariant(
            {
                "title": ost_text,
                "location": ost_text,
                "resolved_location": resolved_location_gate_dict(rl),
            }
        )
        if not gate.passed:
            failures.append(f"  LF {lf_idx}: {gate.message}")
        if evidence_by_lf is not None:
            ladder = check_specificity_ladder(
                {"title": ost_text, "location": ost_text}, evidence_by_lf.get(int(lf_idx))
            )
            if not ladder.passed:
                failures.append(f"  LF {lf_idx}: {ladder.message}")
    if failures:
        raise RuntimeError(
            "Pre-encode location invariant HARD FAIL — at least one LF's "
            "burned OST/title text names a location in ZERO sampled GPS "
            "regions (far-road class). Refusing to start the encode; "
            "fix the label (--per-lf-location override) or the GPS "
            "derivation first.\nAffected LFs:\n" + "\n".join(failures)
        )


def _assert_sign_verify(state, config) -> None:
    """Refuse to START an encode whose burned label is CONTRADICTED by the road
    signs in its own footage (opt-in ``config.drive.sign_verify_strict``).

    Sibling of :func:`_assert_pre_encode_location_invariant`, same call site and
    the same ``RuntimeError`` contract, but a different question: that gate asks
    whether the label agrees with the GPS→geocoder chain, this one asks whether
    that chain agrees with *independent* evidence. When the trace and the
    geocoder are both wrong, only a physical sign in the frame can tell — and
    the only useful moment to hear it is before the multi-hour encode.

    Opt-in by design: sign reading is advisory evidence, so the default posture
    stays the ``--label-dry-run`` print + WARN. No-ops when the flag is off,
    when Phase A recorded no hits, and per-LF when there is no resolved model
    (GPS-less / legacy), matching the sibling gate's skip.
    """
    drive_cfg = getattr(config, "drive", None)
    if not bool(getattr(drive_cfg, "sign_verify_strict", False)):
        return
    if not getattr(state, "sign_hits_by_date", None):
        return
    resolved_by_lf = {
        int(k): v
        for k, v in (getattr(state, "per_lf_resolved_location", None) or {}).items()
        if v is not None
    }
    if not resolved_by_lf:
        return

    from ...analysis.sign_evidence import STATUS_CONFLICTS, hits_by_date_from_state
    from ...analysis.sign_verify import verify_all

    # Build a FRESH plan rather than reusing ``real_plan``: by this point
    # ``_partition_lf_plans_by_date`` has remapped every segment's
    # ``source_file_index`` from global to day-local, and hits are bucketed in
    # the global index space. This is the same unpartitioned source
    # ``--label-dry-run`` compares against, so both gates see one bucketing.
    drive_plan = _build_drive_plan_from_state(config, state)
    if drive_plan is None:
        logger.warning(
            "  sign-verify strict: no drive plan could be built — skipping the "
            "sign gate (the pre-encode location invariant still applies)"
        )
        return

    verdicts = (
        verify_all(
            hits_by_date_from_state(state.sign_hits_by_date),
            drive_plan,
            resolved_by_lf,
            min_corroborating_hits=int(
                getattr(drive_cfg, "sign_verify_min_corroborating_hits", 2) or 2
            ),
            # Deliberately NO geocode_fn: a HARD gate must not depend on a live
            # network service — a Nominatim outage would block an encode on an
            # infrastructure blip rather than on real evidence. Proximity
            # refinement stays on the advisory --label-dry-run surface.
            geocode_fn=None,
        )
        or {}
    )

    failures: list[str] = []
    for lf_idx in sorted(verdicts):
        verdict = verdicts[lf_idx]
        if getattr(verdict, "status", "") != STATUS_CONFLICTS:
            continue
        rl = resolved_by_lf.get(int(lf_idx))
        claimed = ", ".join(str(c) for c in (getattr(verdict, "conflicting_counties", None) or []))
        regions = ", ".join(str(r) for r in (getattr(rl, "regions", None) or []))
        sign_text = "; ".join(str(r) for r in (getattr(verdict, "reasons", None) or []))
        failures.append(
            f"  LF {lf_idx}: signs claim {claimed or '<unnamed county>'}; label resolved "
            f"{getattr(rl, 'dominant_county', None)}, {getattr(rl, 'dominant_state', None)}; "
            f"claimed county absent from sampled regions [{regions or '<none>'}]; "
            f"sign text: {sign_text or '<no reason recorded>'}"
        )
    if failures:
        raise RuntimeError(
            "SIGN-VERIFY-CONFLICT strict HARD FAIL — the road signs in at least "
            "one LF's own footage contradict the label about to be burned. "
            "Refusing to start the encode; confirm with --label-dry-run, then "
            "fix the label (--per-lf-location override) or the GPS derivation — "
            "or drop --sign-verify-strict if the signs are the wrong ones.\n"
            "Affected LFs:\n" + "\n".join(failures)
        )


def _populate_resolved_locations(
    state,
    skeleton_lfs,
    gps_points_by_lf,
    geocoder,
    destination_by_lf=None,
    per_lf_ost_text=None,
    endpoint_loss_by_lf=None,
    places_fn=_forward_geocode_places,
) -> None:
    """Populate ``state.per_lf_resolved_location`` (the populate half).

    Wires the canonical :class:`~gomoveshift.models.ResolvedLocation` model and
    its ``build_resolved_location`` builder into Phase BC: every LF with
    a usable GPS trace gets ONE resolved model (modal-county vote, regions /
    vote shares for the location invariant, ``is_us`` / ``country_code``
    resolved once, the winning-county road) persisted into
    ``state.per_lf_resolved_location[lf_index]`` so resume reloads instead of
    re-deriving, and downstream surfaces consume it.

    The ``destination`` field is populated here from ``destination_by_lf``
    (the per-LF destination POI resolved ONCE in ``_enrich_per_lf_gps`` from
    each LF's montage-peak footage). Reusing the precomputed map honours the
    computed-once invariant — no second Overpass query. ``None`` map / missing
    entry leaves ``destination`` at its ``None`` default.

    The persisted model is versioned like the color-analysis cache. Each
    stored model carries ``ost_signature``: the fresh ``per_lf_ost_text`` it was
    derived against. ``per_lf_ost_text`` is recomputed on EVERY run, so reusing a
    rehydrated model unconditionally let the shipped title diverge from the model
    feeding units/playlist/tags + the content gates whenever the trace/geocoder
    changed. A rehydrated entry is now reused ONLY when its signature matches this
    run's fresh OST text (fast path); on a mismatch — or a legacy unversioned
    ``None`` signature — it is dropped and re-derived so title and model can never
    diverge. When this run has no fresh OST text for an LF (defensive/GPS-less
    path) there is nothing to compare, so the entry is preserved.

    Each model also carries ``trace_verified_places`` for its ``ost_signature``
    label: the endpoint towns a settlement of exactly that name within 2 km of
    the LF's own fixes proves (``places_fn`` proposes the candidates; ``None``
    disables). Computed once with the signature; a reused model that predates
    the field (``None``) is verified in place on the fast path.

    Robust-by-construction back-compat: the field defaults ``None`` (an
    older state file has no key), so it is initialised lazily; an LF
    with no usable trace (or no geocoder) yields an empty
    ``ResolvedLocation`` which we DO NOT store, leaving the slot open for a
    later recompute rather than persisting a useless empty.
    """
    destination_by_lf = destination_by_lf or {}
    per_lf_ost_text = per_lf_ost_text or {}
    endpoint_loss_by_lf = endpoint_loss_by_lf or {}
    resolved = state.per_lf_resolved_location
    if resolved is None:
        resolved = {}
        state.per_lf_resolved_location = resolved
    if geocoder is None or not gps_points_by_lf:
        return
    overrides = getattr(state, "per_lf_overrides", None) or {}
    # Collect genuine derivation FAILURES (an exception from
    # build_resolved_location with GPS PRESENT and no operator override) and
    # hard-fail the run after the loop. A warn+continue here left the LF
    # model-less, which silently SKIPS the pre-encode far-road gate
    # exactly when the geocode layer most likely produced a WRONG label.
    derivation_failures: list[str] = []
    for lf in skeleton_lfs or []:
        lf_idx = int(getattr(lf, "index", 0))
        # Version/invalidate the persisted model like the color cache.
        # ``per_lf_ost_text`` is recomputed FRESH on every (resume) run, but a
        # rehydrated model used to be reused unconditionally — so a changed
        # trace/geocoder shipped a title that silently diverged from the model
        # feeding units/playlist/tags + the content gates. Re-derive whenever
        # the stamped ``ost_signature`` no longer matches this run's fresh OST
        # text (or is a legacy ``None``); keep the fast-path skip on a match /
        # when this run has no fresh text to compare against.
        signature = per_lf_ost_text.get(lf_idx)
        existing = resolved.get(lf_idx)
        # A state file persisted by an older builder can
        # carry a quorum-degraded model (dominant_* None but regions/vote/
        # road/sequence kept — the only dominant-less shape the storage
        # branch below ever persisted). Its sub-floor evidence would keep
        # feeding the title fallback + location-invariant gates through the signature
        # fast-path, so drop it unconditionally and re-derive; the fixed
        # builder never re-stores a degraded shell.
        if existing is not None and getattr(existing, "dominant_county", None) is None:
            del resolved[lf_idx]
            existing = None
        if existing is not None:
            if signature is None or getattr(existing, "ost_signature", None) == signature:
                # fast-path: signature matches (or nothing to compare). A model
                # persisted before trace verification existed gets it now.
                if getattr(existing, "trace_verified_places", None) is None:
                    _verify_trace_places(
                        existing, signature, gps_points_by_lf.get(lf_idx), places_fn
                    )
                continue
            del resolved[lf_idx]  # signature drift → drop stale, re-derive below
        points = gps_points_by_lf.get(lf_idx)
        if not points:
            continue  # GPS-less LF — a documented, legitimate no-op
        try:
            # ``build_resolved_location`` now populates rl.road from the
            # WINNING county's vote itself, so no second dominant_location pass
            # is needed here (that doubled the geocode quota on uncached runs).
            # Withhold any A→B endpoint the enrichment pass flagged as
            # shifted by a per-segment head/tail GPS dropout.
            rl = build_resolved_location(
                points, geocoder=geocoder, withhold_endpoints=endpoint_loss_by_lf.get(lf_idx)
            )
        except Exception as exc:  # noqa: BLE001
            # GPS is present, so this is a derivation FAILURE, not a
            # GPS-less no-op. An authoritative --per-lf-location override
            # supplies the label verbatim (model not needed) → degrade quietly;
            # otherwise record the failure and hard-fail below.
            override = overrides.get(lf_idx)
            if isinstance(override, str) and override.strip():
                logger.warning(
                    "build_resolved_location failed for LF %d (%s) — using "
                    "operator --per-lf-location override",
                    lf_idx,
                    exc,
                )
                continue
            derivation_failures.append(f"  LF {lf_idx}: {exc}")
            continue
        # Anchor the once-resolved destination POI onto the canonical
        # model so every surface (and a resume reload) sees it.
        _destination = destination_by_lf.get(lf_idx)
        if _destination:
            rl.destination = _destination
        # Stamp the OST/title text this model was derived against so a later
        # resume can detect divergence and re-derive instead of shipping a stale
        # model beside a freshly-recomputed title.
        rl.ost_signature = signature
        # Skip empty resolutions (no dominant county AND no regions) so resume
        # can recompute later rather than locking in a useless shell.
        if rl.dominant_county or rl.regions:
            _verify_trace_places(rl, signature, points, places_fn)
            resolved[lf_idx] = rl
    # Per-run degrade counter — any genuine derivation failure (>0) hard-
    # fails. Refusing to start the encode is the whole point: a model-less LF
    # silently skips the pre-encode far-road gate precisely when the
    # geocode/GPX layer is most likely to have produced a wrong label.
    if derivation_failures:
        raise RuntimeError(
            "build_resolved_location HARD FAIL — location derivation raised for "
            f"{len(derivation_failures)} LF(s) that HAVE GPS and no operator "
            "override. A warn+continue would leave each model-less and silently "
            "skip the pre-encode far-road gate. Restore/fix the "
            "GPS/geocoder or pass an explicit --per-lf-location override.\n"
            "Affected LFs:\n" + "\n".join(derivation_failures)
        )


# Anchored to the WHOLE component: the recording date is the
# ``day_<YYYY-MM-DD>`` directory that is the IMMEDIATE parent of the
# ``regular_lfs`` dir, never an arbitrary ``day_*`` token elsewhere in
# the path (a relocated/nested path must not skew the date).
_DAY_DIR_DATE_RE = __import__("re").compile(r"^day_(\d{4}-\d{2}-\d{2})$")


def _day_date_from_lf_path(lf_path) -> "str | None":
    """Extract ``YYYY-MM-DD`` from the ``day_<DATE>`` dir anchoring an LF path.

    Canonical layout: ``<stages>/day_<YYYY-MM-DD>/regular_lfs/lf_NN.mp4``.
    The date directory is the parent of the ``regular_lfs`` dir (i.e. the
    LF file's grandparent). Anchoring to that exact component — rather than
    ``re.search`` over the full path string — means a relocated path whose
    ANCESTORS happen to contain a different ``day_*`` token can't skew the
    derived date.

    Falls back to scanning path components for a single ``day_<DATE>`` dir
    when ``regular_lfs`` is absent (e.g. a flattened test fixture); returns
    ``None`` when no anchored match exists.
    """
    from pathlib import Path

    parts = Path(str(lf_path)).parts
    # Preferred anchor: the dir immediately above ``regular_lfs``.
    for i, comp in enumerate(parts):
        if comp == "regular_lfs" and i > 0:
            m = _DAY_DIR_DATE_RE.match(parts[i - 1])
            if m is not None:
                return m.group(1)
    # Fallback: a single unambiguous ``day_<DATE>`` component anywhere.
    matches = [m.group(1) for comp in parts if (m := _DAY_DIR_DATE_RE.match(comp))]
    if len(matches) == 1:
        return matches[0]
    return None


def _derive_per_lf_recording_date_from_paths(
    regular_lf_paths: list,
    skeleton_lfs: list,
) -> dict:
    """Derive ``{lf.index: "YYYY-MM-DD"}`` from on-disk LF paths.

    Day-first layout writes every LF under
    ``<stages>/day_<YYYY-MM-DD>/regular_lfs/lf_NN.mp4`` — see
    ``_reconstruct_day_artifacts_from_disk`` for the canonical glob.
    The ``day_<DATE>`` directory name IS the LF's recording date by
    construction (``run_day_pipeline`` enforces single-day LFs), so
    this helper is the resume-safe source of truth for
    ``state["per_lf_recording_date"]`` when the partition-derived
    path can't run (resume fast-path, missing DrivePlan, missing
    episode_dates).

    ``skeleton_lfs[i]`` corresponds 1:1 to ``regular_lf_paths[i]``;
    the LF's index is taken from the plan, though the ``lf_NN``
    filename carries the same value — ``run_day_pipeline`` names each
    final by its drive-global ``lf_plan.index`` (``lf_{idx:02d}.mp4``),
    not a day-local counter.

    Returns an empty dict when paths can't be parsed — callers layer
    this on top of the partition-derived map and never overwrite it.
    """
    if not regular_lf_paths or not skeleton_lfs:
        return {}
    out: dict[int, str] = {}
    # Key by the skeleton LF's drive-global index (NOT a positional
    # accumulation): the date for skeleton_lfs[i] is read from its OWN
    # path regular_lf_paths[i], anchored to that path's day_<DATE> dir.
    for lf, lf_path in zip(skeleton_lfs, regular_lf_paths):
        lf_idx = getattr(lf, "index", None)
        if lf_idx is None:
            continue
        date_str = _day_date_from_lf_path(lf_path)
        if date_str is None:
            continue
        out[int(lf_idx)] = date_str
    return out


def _record_recording_date_recovery(
    state,
    partition_map: dict,
    path_map: dict,
) -> None:
    """Observe the recording-date path backstop.

    Records LF indices that the path-derivation supplied which the
    partition-derived map missed. Silent when partition map already
    covers the path-derived index set (fresh-encode happy path).
    """
    if not path_map:
        return
    recovered = sorted(int(idx) for idx in path_map if idx not in partition_map)
    if not recovered:
        return
    from ..state_invariants import record_backstop_recovery

    record_backstop_recovery(
        state,
        "per_lf_recording_date",
        recovered,
        "partition-derived map missed entries; recovered from "
        "day_<YYYY-MM-DD>/regular_lfs/lf_NN.mp4 directory layout "
        "(resume fast-path, no DrivePlan / empty episode_dates)",
    )


def _resolve_ingest_plan(state):
    """Return an ingest plan ready for ``group_episodes_by_day``.

    On resume, ``PipelineState.from_dict`` calls ``IngestPlan.from_dict``,
    which is a shallow rehydrate — its ``episodes`` are passed through as
    the raw ``list[dict]`` they were serialized as, not ``Episode``
    instances. ``group_episodes_by_day`` then does ``getattr(ep, "date",
    None)`` / ``getattr(ep, "creation_time", None)`` which both return
    ``None`` for a ``dict``, every episode is silently dropped, and the
    phase processes 0 batches while still returning success.

    This helper detects the dict-children case and synthesizes a fresh
    plan from the persisted parallel arrays (``episode_files``,
    ``episode_dates``, etc.) which the resume path always populates.
    """
    plan = state.ingest_plan
    episodes = getattr(plan, "episodes", None) if plan is not None else None
    if not episodes or any(isinstance(ep, dict) for ep in episodes):
        return _synthesize_ingest_plan_from_state(state)
    return plan


def _synthesize_ingest_plan_from_state(state):
    """Build a minimal ingest-plan object from persisted state fields."""
    episode_files = state.episode_files or []
    episode_dates = state.episode_dates or []
    file_durations = state.file_durations or []
    gpx_files = state.gpx_files or []

    if not episode_files:
        return None

    episodes = []
    for i, ep_file in enumerate(episode_files):
        date_str = episode_dates[i] if i < len(episode_dates) else None
        if hasattr(date_str, "strftime"):
            date_str = date_str.strftime("%Y-%m-%d")
        duration = float(file_durations[i]) if i < len(file_durations) else 0.0
        gpx_path = gpx_files[i] if i < len(gpx_files) else None

        ep = SimpleNamespace(
            index=i + 1,
            concat_path=Path(ep_file),
            date=date_str,
            total_duration=duration,
            gpx_path=gpx_path,
            creation_time=None,
        )
        episodes.append(ep)

    plan = SimpleNamespace(episodes=episodes)
    logger.info(
        "  day-first: synthesized %d episode(s) from resume state",
        len(episodes),
    )
    return plan


def _deliverables_record(artifacts, state) -> dict:
    """JSON-safe ``Deliverables`` record for ``artifacts``, stored beside the
    day's ``per_day_completed`` entry so Phase D (and a resume) rebuild it."""
    stages = getattr(state, "stages_base", None) or (
        Path(getattr(state, "output_base", None) or ".") / "_stages"
    )
    return deliverables_to_state(lift_artifacts(artifacts, Path(stages)))


def _record_day_and_save(artifacts, state, state_save_fn) -> None:
    """Record a just-completed day in ``state['per_day_completed']`` and
    persist state to disk via ``state_save_fn``.

    Called per-day from inside ``run_all_days`` so a mid-phase crash
    preserves already-done days' markers. Called again at end of phase
    for any days missed by the callback — this function is idempotent
    (setting the same key with the same value twice is a no-op).

    Best-effort — the save call catches its own errors and logs; it does
    not propagate because losing bookkeeping is recoverable but killing
    the run is not.

    When ``artifacts.master_path``
    is None AND the day's finals or raw concat exist on disk, a
    *recovery marker* is still written so the next resume can take the
    finals-only reconstruction fast-path. Without this, a crashed day
    re-encodes from scratch on resume even when work had already landed.
    The caller (``run_all_days._maybe_fire_recovery_callback``) already
    gates on the on-disk check before firing this path.
    """
    master = getattr(artifacts, "master_path", None)
    if master is None:
        # Recovery path: the day's master never landed.
        # Write a marker with master_path=None so the next resume picks
        # up via the finals-only fast-path (for days with finals on
        # disk) or falls through to re-encode from raw (for raw-only
        # days). Either way we don't re-do the whole day.
        per_day = state.per_day_completed
        if not isinstance(per_day, dict):
            per_day = {}
        per_day[artifacts.date] = {
            "master_path": None,
        }
        state.per_day_completed = per_day
        logger.warning(
            "  day-first: day %s recovery marker written "
            "(master_path=None; %d LFs, %d shorts on disk)",
            artifacts.date,
            len(getattr(artifacts, "regular_lf_paths", []) or []),
            len(getattr(artifacts, "short_paths", []) or []),
        )
        if state_save_fn is not None:
            try:
                state_save_fn(state)
            except Exception as exc:  # noqa: BLE001 — bookkeeping never blocks
                logger.warning(
                    "  day-first: state save failed after day %s recovery "
                    "marker: %s — next resume may re-encode this day",
                    artifacts.date,
                    exc,
                )
        return
    master_path = Path(master)
    if not master_path.exists():
        logger.warning(
            "  day-first: day %s master %s vanished before bookkeeping",
            artifacts.date,
            master_path,
        )
        return
    per_day = state.per_day_completed
    if not isinstance(per_day, dict):
        per_day = {}
    entry = {
        "master_path": str(master_path),
    }
    # Merge encode-time integrity metadata for cached-master validation
    # on resume after the raw episodes are purged.
    record = getattr(artifacts, "master_record", None)
    if isinstance(record, dict):
        entry.update(record)
    # Persist the post-per-LF-capacity-cap expectation so a later
    # resume's ``_enforce_reconstructed_shortfall`` can recognize a
    # correctly-capped day instead of re-deriving the pre-cap total.
    # Omitted (not just None) when no explicit/derived target was in force —
    # matches ``DayArtifacts.planned_shorts_count``'s own None-ness.
    planned_shorts_count = getattr(artifacts, "planned_shorts_count", None)
    if planned_shorts_count is not None:
        entry["shorts_target_in_force"] = getattr(artifacts, "shorts_target_in_force", None)
        entry["shorts_expected"] = planned_shorts_count
    entry["deliverables"] = _deliverables_record(artifacts, state)
    per_day[artifacts.date] = entry
    state.per_day_completed = per_day
    if state_save_fn is not None:
        try:
            state_save_fn(state)
        except Exception as exc:  # noqa: BLE001 — bookkeeping never blocks
            logger.warning(
                "  day-first: state save failed after day %s: %s — "
                "next resume may re-encode this day",
                artifacts.date,
                exc,
            )


def _day_dir_has_finals(day_dir: Path) -> bool:
    """Return True when a day directory has at least one final LF and short.

    Used by the resume fast-path to decide whether a day with a dropped
    master can still be reconstructed from disk alone. Matches the layout
    glob used by ``_reconstruct_day_artifacts_from_disk``.
    """
    lf_dir = day_dir / "regular_lfs"
    shorts_dir = day_dir / "shorts"
    has_lf = lf_dir.is_dir() and any(
        p.is_file() and p.stat().st_size > 0 for p in lf_dir.glob("lf_[0-9][0-9].mp4")
    )
    has_short = shorts_dir.is_dir() and any(
        p.is_file() and p.stat().st_size > 0 for p in shorts_dir.glob("short_[0-9][0-9].mp4")
    )
    return has_lf and has_short


def day_superlong_complete(date: str, day_dir: Path, state) -> bool:
    """True when this day's superlong parts are all final on disk.

    ``_day_dir_has_finals`` answers "are LFs and shorts here", which is the
    wrong question on a ``--superlong`` run: superlong has exactly ONE
    producer, ``day_pipeline._produce_superlong_parts``, and it runs inside
    ``run_day_pipeline`` — a day the resume skips can never reach it.

    Expected count comes from ``state.superlong_parts[date]["part_count"]``,
    the record that producer writes on success. NO record means superlong
    never ran, which is INCOMPLETE — not complete: the count cannot be
    re-derived here, because planning it needs the day's stop mask, which only
    exists inside ``run_day_pipeline``.
    """
    from ..cleanup import finalized_superlong_count

    record = (getattr(state, "superlong_parts", None) or {}).get(date)
    expected = record.get("part_count") if isinstance(record, dict) else None
    if not expected:
        return False
    return finalized_superlong_count(day_dir) >= int(expected)


def _reconstruct_day_artifacts_from_disk(
    date: str,
    master_path: Optional[Path],
    day_dir: Path,
) -> DayArtifacts:
    """Rebuild a `DayArtifacts` record from the on-disk output of a
    previously-successful day run.

    Used on resume when `per_day_completed` says a day is done. Two cases:

      1. Master still on disk — we pass it as ``master_path`` and glob finals
         from ``day_dir``.
      2. Master has been dropped (post-phase-BC cleanup) — ``master_path``
         is ``None``. Downstream phases don't read the master, so a None value
         propagates harmlessly.

    Layout produced by ``run_day_pipeline``:

        day_<date>/
            <master>.mp4          (may be absent post-drop)
            regular_lfs/lf_NN.mp4
            shorts/short_NN.mp4

    Missing subdirs yield empty lists — safe, because fresh-LF/shorts
    extraction is cheap and the next phase will see no paths rather
    than stale ones.
    """
    lf_dir = day_dir / "regular_lfs"
    regular_lf_paths: list = []
    if lf_dir.is_dir():
        # Match the final OST-applied LFs only (lf_NN.mp4), not the
        # intermediate lf_NN_with_montage.mp4 etc. Require st_size>0 (mirrors
        # _day_dir_has_finals): a 0-byte / partial final left by a crashed or
        # recovery-marked day must NOT enter the upload set on resume, where it
        # would bypass the OST audit gate and the apply_ost re-burn loop.
        regular_lf_paths = sorted(
            p for p in lf_dir.glob("lf_[0-9][0-9].mp4") if p.is_file() and p.stat().st_size > 0
        )

    shorts_dir = day_dir / "shorts"
    short_paths: list = []
    if shorts_dir.is_dir():
        # Final OST-applied shorts only (short_NN.mp4), not short_NN_raw.mp4.
        # Same st_size>0 guard as the LFs above.
        short_paths = sorted(
            p
            for p in shorts_dir.glob("short_[0-9][0-9].mp4")
            if p.is_file() and p.stat().st_size > 0
        )

    return DayArtifacts(
        date=date,
        master_path=master_path,
        regular_lf_paths=regular_lf_paths,
        # Carry each final's drive-global LF index, parsed from its
        # ``lf_NN`` stem — ``run_day_pipeline`` names finals by
        # ``lf_plan.index`` (``lf_{idx:02d}.mp4``), so the number IS the
        # plan index, and the ``lf_[0-9][0-9]`` glob above guarantees the
        # stem parses. Omitting this made the consumer fall back to
        # POSITIONAL pairing, so a resume after a mid-day render skip
        # paired a later LF's file with an earlier LF's plan — uploading
        # it under the wrong title/OST metadata/thumbnail/schedule.
        regular_lf_indices=[int(p.stem.split("_")[1]) for p in regular_lf_paths],
        short_paths=short_paths,
        # Restore the per-short source windows so the resume path keeps
        # the source-window shorts↔LF attribution instead of the legacy
        # equal-split fallback. The documented workflow (--skip-upload audit
        # → --youtube-upload re-run) means the SHIPPED run is a resume; a
        # dropped short_ranges here silently misattributed every real
        # publish's shorts by rank position.
        short_ranges=_reconstruct_short_ranges_from_sidecar(date, day_dir, short_paths),
    )


def _reconstruct_short_ranges_from_sidecar(
    date: str,
    day_dir: Path,
    short_paths: list,
) -> list:
    """Rebuild the per-short ``ShortRange`` list from the day's
    ``shorts_ost_audit_<date>.json`` sidecar, PARALLEL to ``short_paths``.

    ``run_day_pipeline`` writes the sidecar to the day's PARENT (the stages
    dir), one entry per rendered short carrying ``rank`` / ``start_sec`` /
    ``end_sec`` in the day-master timeline — the exact fields
    ``assign_shorts_to_lfs_by_source_window`` reads. Each short file is
    named ``short_<rank>.mp4``, so we key the sidecar by ``rank`` and align
    it to ``short_paths`` by the filename number.

    Returns ``[]`` (→ genuine equal-split last resort) when the sidecar is
    absent/unreadable or can't cover every short path — never a partial,
    misaligned list.
    """
    if not short_paths:
        return []
    sidecar = day_dir.parent / f"shorts_ost_audit_{date}.json"
    if not sidecar.is_file():
        return []
    try:
        payload = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return []
    by_rank: dict[int, ShortRange] = {}
    for entry in payload.get("shorts") or []:
        try:
            rank = int(entry["rank"])
        except (KeyError, TypeError, ValueError):
            continue
        by_rank[rank] = ShortRange(
            start_sec=float(entry.get("start_sec", 0.0) or 0.0),
            end_sec=float(entry.get("end_sec", 0.0) or 0.0),
            integrated_score=0.0,
            rank=rank,
        )
    ranges: list = []
    for path in short_paths:
        try:
            rank = int(Path(path).stem.split("_")[1])
        except (IndexError, ValueError):
            return []
        sr = by_rank.get(rank)
        if sr is None:
            return []
        ranges.append(sr)
    return ranges


def _probe_duration(video_path: Path) -> Optional[float]:
    """Return the duration of a video file in seconds via ffprobe."""
    # Late-bind subprocess from drive_orchestrator for test patch compat
    orch_mod = sys.modules.get("gomoveshift.pipeline.drive_orchestrator")
    subprocess = getattr(orch_mod, "subprocess", None)
    if subprocess is None:
        import subprocess
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            timeout=15,
        )
    except Exception as e:
        logger.debug("_probe_duration: ffprobe failed for %s: %s", video_path, e)
        return None
    if result.returncode != 0:
        return None
    raw = (result.stdout or b"").decode("utf-8", errors="replace").strip()
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
