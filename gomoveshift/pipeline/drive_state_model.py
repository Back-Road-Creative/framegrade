"""Typed pipeline state for the Drive Pipeline.

Replaces the untyped 39-key ``state`` dict with a proper dataclass so that
missing keys, type mismatches, and JSON-round-trip coercion bugs are caught
at attribute access time rather than silently producing ``None``.

Key provenance
--------------
Orchestrator.run():  source_dir, output_base, region, publish_day, location,
                     fresh, skip_upload, skip_csv
Phase A:             ingest_plan, episode_locations, episode_profiles, location,
                     episode_files, gpx_files, gps_per_episode, drive_scores,
                     file_boundaries, file_durations, episode_dates
Phase B:             processed_files,
                     delivered_lfs, delivered_lf_paths, regular_lf_paths,
                     shorts_per_lf, metadata_per_lf, social_posts_per_lf
Phase BC day-first:  per_day_locations, day_artifacts, drive_plan,
                     regular_lf_paths, shorts_per_lf, per_lf_locations
Phase C:             drive_plan, regular_lf_paths,
                     shorts_per_lf, route_profiles
Phase D:             metadata_per_lf, drive_schedule, social_posts_per_lf,
                     upload_manifests, youtube_uploads
Orchestrator misc:   phases_completed, per_day_completed,
                     _per_day_skip_indices
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Optional

# Sentinel for ``PipelineState.get(key, default)`` — distinguishes
# "caller did not supply a default" from "caller supplied ``None`` as
# default". Without this, ``state.get(key, default)`` cannot tell a
# field whose value is ``None`` apart from a field that does not exist:
# both used to return ``default``. The sentinel restores Python-dict
# ``get`` semantics — present-but-None values are returned as ``None``;
# only truly-absent attributes fall through to ``default``.
_MISSING = object()


def _coerce_int_key_dict(raw: Any) -> Optional[dict]:
    """Convert a dict with string keys to int keys (JSON round-trip fix)."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return raw
    result = {}
    for k, v in raw.items():
        try:
            result[int(k)] = v
        except (ValueError, TypeError):
            result[k] = v
    return result


def _coerce_int_key_dict_paths(raw: Any) -> Optional[dict]:
    """Like _coerce_int_key_dict but also wraps values in Path."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return raw
    result = {}
    for k, v in raw.items():
        try:
            int_k = int(k)
        except (ValueError, TypeError):
            int_k = k
        result[int_k] = Path(v) if isinstance(v, str) else v
    return result


# Single source of truth for every PipelineState field that holds an
# int-keyed dict (``{lf_index: value, ...}`` or similar). JSON serializes
# int keys as strings; ``_coerce_int_key_dict`` rehydrates them at load.
# Downstream code MUST NOT do ``d.get(str(lf.index))`` fallbacks — the
# load-time normalization is the contract boundary.
_INT_KEY_DICT_FIELDS = (
    "shorts_per_lf",
    "metadata_per_lf",
    "social_posts_per_lf",
    "upload_manifests",
    "youtube_uploads",
    "route_profiles",
    "per_lf_locations",
    "per_lf_ost_text",
    "per_lf_titles",
    "per_lf_recording_date",
    "per_lf_trim_ratio",
    "per_lf_gpx_path",
    "per_lf_gpx_episode_paths",
    "per_lf_distance_km",
    "per_lf_elevation_gain_m",
    "per_lf_elevation_profile",
    "per_lf_speed",
    "per_lf_bearing",
    "per_lf_regions",
    "lf_thumbnail_drive_urls",
    "per_lf_overrides",
    "short_durations_per_lf",
    "per_lf_resolved_location",
    "per_lf_gps_coverage_pct",
    "per_lf_awarded_tier",
    "per_lf_tier_evidence",
    "per_lf_location_evidence",
    "per_lf_route_record",
    "per_lf_pre_render_title_decisions",
)

# Phase-A-derived per-LF label/text fields that must inherit a resumed
# --per-lf-location override (see preserve_per_lf_overrides_on_resume in
# drive_orchestrator.py) — Phase D reads these directly on a Phase-D-only
# resume (Phase A does not re-run), so a stale entry here survives to burn
# the wrong label. Adding a new per-LF label/text field
# that Phase D reads instead of re-deriving? Add it here too.
PER_LF_OVERRIDE_MIRROR_FIELDS = (
    "per_lf_locations",
    "per_lf_ost_text",
    "per_lf_titles",
)

# Subset of ``_INT_KEY_DICT_FIELDS`` whose keys MUST all be ``int`` after
# load (the canonical-key invariant fires when violated). ``route_profiles``
# is intentionally excluded — it admits hybrid keys (``"all"`` /
# ``"day1"`` / LF ints) by design, and ``_coerce_int_key_dict`` already
# converts the int-stringified subset back to int.
_STRICT_INT_KEY_DICT_FIELDS = tuple(f for f in _INT_KEY_DICT_FIELDS if f != "route_profiles")

# Large Phase-A products that are recoverable from an on-disk artifact and
# therefore NOT written into pipeline_state.json. Serializing them was
# pure I/O waste — tens of MB rewritten on EVERY checkpoint:
#   * ``drive_scores`` — the numpy score matrix, reloaded from
#     ``drive_scores.npz`` on resume (``drive_orchestrator._try_resume``).
#   * ``gps_per_episode`` — the full per-episode trackpoint traces, rebuilt
#     by re-parsing the ``.gpx`` paths in ``gpx_files``
#     (``gps_enrichment.rebuild_gps_per_episode`` on resume).
# ``to_dict`` skips them on the WRITE side only; ``from_dict`` still READS
# them from a legacy checkpoint that predates this exclusion, so an in-flight
# resume of an old run stays safe.
_UNSERIALIZED_FIELDS = frozenset(
    # ``ost_audit_confirmed_this_run``: a per-invocation confirmation
    # signal that must NOT survive a resume — persisting it is exactly the hole
    # (a saved ``True`` blanket-bypassing a changed burn after a kill).
    # ``force_sidecar_reset``: same hazard for the one-shot sidecar
    # reconciliation flag — a saved ``True`` re-wipes both upload sidecars and
    # re-uploads every published LF on every later resume. This exclusion is
    # write-side only (``from_dict`` still reads legacy keys) — it stops the
    # value from propagating past the run that set it, it does NOT make an
    # already-poisoned on-disk checkpoint safe on its very next read.
    # ``drive_orchestrator.reset_force_sidecar_reset_after_resume`` closes
    # that gap by forcing the field back to THIS invocation's CLI intent
    # after the resume merge re-imports whatever is on disk.
    # ``ost_system_review_this_run``: the same per-invocation rule
    # for the ``--ost-system-review`` opt-in.
    {
        "drive_scores",
        "gps_per_episode",
        "ost_audit_confirmed_this_run",
        "ost_system_review_this_run",
        "force_sidecar_reset",
    }
)


def _list_to_tuples(raw: Any) -> Optional[list]:
    """Convert a list of 2-element lists to a list of tuples (JSON fix)."""
    if raw is None:
        return None
    return [tuple(item) if isinstance(item, (list, tuple)) else item for item in raw]


@dataclasses.dataclass
class PipelineState:
    """Typed pipeline state for the Drive Pipeline.

    All fields that are only populated in later phases default to ``None``.
    """

    # ── Orchestrator init ──────────────────────────────────────────────
    source_dir: Optional[Path] = None
    output_base: Optional[Path] = None
    region: str = ""
    publish_day: str = "wed"
    location: str = ""
    fresh: bool = False
    skip_upload: bool = False
    # Split skip-upload into Drive vs YouTube. Defaults False; the
    # legacy ``skip_upload=True`` is treated as setting BOTH at the gate.
    skip_youtube_upload: bool = False
    skip_drive_upload: bool = False
    skip_csv: bool = False
    stages_base: Optional[Path] = None  # ext4-backed intermediate dir (None = output_base/_stages)

    # ── Phase A ────────────────────────────────────────────────────────
    ingest_plan: Any = None
    episode_locations: Optional[list] = None
    episode_profiles: Optional[list] = None
    episode_files: Optional[list] = None
    gpx_files: Optional[list] = None
    gps_per_episode: Optional[list] = None
    drive_scores: Any = None  # numpy array
    file_boundaries: Optional[list] = None  # list[tuple[float, float]]
    file_durations: Optional[list] = None
    episode_dates: Optional[list] = None
    # Median sharpness per source, in ``episode_files`` order (V2 source gate).
    # JSON-safe: an undecodable source is ``None`` here (NaN in the npz), so the
    # strict-JSON state file never carries a NaN token.
    source_sharpness: list = dataclasses.field(default_factory=list)

    # ── Phase B ────────────────────────────────────────────────────────
    processed_files: Optional[dict] = None  # dict[int, Path]
    delivered_lfs: Optional[set] = None
    delivered_lf_paths: Optional[dict] = None  # dict[int, Path]
    regular_lf_paths: Optional[list] = None
    shorts_per_lf: Optional[dict] = None  # dict[int, list]
    metadata_per_lf: Optional[dict] = None  # dict[int, dict]
    social_posts_per_lf: Optional[dict] = None  # dict[int, list]

    # ── Phase BC day-first ─────────────────────────────────────────────
    per_day_locations: Optional[dict] = None
    day_artifacts: Optional[list] = None
    per_lf_locations: Optional[dict] = None
    # Canonical per-LF OST burn-in text.
    # Shape: ``{lf_index: "{start} to {end} via {road}, {state}"}`` for the
    # five-level fallback cascade in ``title_builder.build_lf_title``. Unlike
    # ``per_lf_titles``, this field is captured BEFORE
    # ``title_builder.ensure_unique_titles`` so it never carries the
    # ``(Part N of M)`` dedup suffix. Downstream surfaces (OST burn-in,
    # YouTube title, thumbnail) should all read from this single source.
    per_lf_ost_text: Optional[dict] = None
    # Per-LF YouTube title text (canonical + dedup suffix when needed).
    # Declared here so dataclass-based state resume preserves the mapping
    # through JSON round-trip. Stored before/after the dedup pass (after,
    # so collision suffixes are present when needed for the YouTube title).
    per_lf_titles: Optional[dict] = None
    # Per-LF recording date ``{lf.index: "YYYY-MM-DD"}`` — drives the
    # ``📅 Filmed: <date>`` line in the YouTube description. A past run
    # shipped LFs with no Filmed line because this
    # field used to live as an undeclared ``state[...]`` write that
    # ``from_dict`` silently dropped on JSON round-trip. Now a declared
    # contract member with a construction-time path-derivation migration
    # (``PipelineState._migrate_per_lf_recording_date_from_paths``, run
    # from ``__post_init__``) — the in-phase backstops in phase_d are
    # unreachable by construction.
    per_lf_recording_date: Optional[dict] = None
    # Per-LF trim ratio ``{lf.index: final_duration / source_clip_duration}``
    # — drives the title verb band (``Driving`` vs ``Highlights from``),
    # description hook, and consolidated shorts CSV first-comment lead.
    # Companion to ``per_lf_recording_date`` for the same
    # reason: declared so JSON round-trip preserves the mapping, with a
    # load-time ffprobe-based migration as the resume backstop.
    per_lf_trim_ratio: Optional[dict] = None
    # Per-LF GPX file path captured BEFORE ``_partition_lf_plans_by_date``
    # remaps segment indices to day-local. Phase D's chapter-generation
    # reads ``state["per_lf_gpx_path"][lf.index]`` to pick the correct GPX
    # for each LF on multi-day drives. Declared here so the mapping
    # survives JSON round-trip — without this field, resume runs would
    # silently fall back to a day-local-index lookup against the global
    # ``gpx_files`` list and pick the wrong GPX for day 2+ LFs.
    per_lf_gpx_path: Optional[dict] = None
    # Additive per-episode span map
    # ``{lf.index: {gpx_path_str: [episode_local_ranges]}}``. Lets the
    # post-encoding distance/elevation computes SUM across every episode an LF
    # spans (a multi-episode LF was otherwise understated); single-rep
    # ``per_lf_gpx_path`` above stays authoritative for chapters/regions. Int
    # top-level keys → registered in _INT_KEY_DICT_FIELDS; None → computes fall
    # back to the single-rep path.
    per_lf_gpx_episode_paths: Optional[dict] = None
    # Per-LF GPS-derived measurements written by Phase BC and surfaced
    # in the long-form description (``🛣️ Distance:`` / ``⛰️ Elevation
    # gain:`` / ``🗺️ Through:`` lines). Declared so JSON round-trip
    # preserves the int-keyed shape — pre-declaration, ``from_dict``
    # silently dropped them as unknown keys and Phase D's
    # ``state.get(...)`` fallback never saw them on resume.
    per_lf_distance_km: Optional[dict] = None
    per_lf_elevation_gain_m: Optional[dict] = None
    # per-LF elevation PROFILE ({lf.index: {min_m, max_m, net_m,
    # descent_m}}) — the shape complement to the scalar gain above, surfacing
    # the ``📈 Elevation`` range line. Plain JSON-native dict-of-dict so it
    # round-trips with no _dict_value_dataclass_fields_map entry; int-keyed so
    # registered in _INT_KEY_DICT_FIELDS. None default → a state file written before
    # this field existed loads cleanly (line just omitted until recompute).
    per_lf_elevation_profile: Optional[dict] = None
    # per-LF speed/character ({lf.index: {avg_kmh, max_kmh, std_kmh}}) —
    # surfaces the ``🚗 Pace:`` description line. Same plain JSON-native
    # dict-of-dict + int-key contract as per_lf_elevation_profile above.
    per_lf_speed: Optional[dict] = None
    # per-LF net travel bearing ({lf.index: "NE"}) — surfaces the
    # ``🧭 Heading:`` description line. Dict-of-str, int-keyed (same JSON
    # round-trip contract).
    per_lf_bearing: Optional[dict] = None
    per_lf_regions: Optional[dict] = None
    # Legitimately-omitted GPX-keyed per-LF facts (``{field_name: [lf_index,
    # ...]}``) — an LF whose cleaned trace yields no usable distance
    # (stationary/over-filtered) or is plausibility-rejected is dropped from
    # ``per_lf_distance_km``/``per_lf_elevation_gain_m`` ON PURPOSE. Both the
    # BC-exit and D-entry boundary contracts read this to exclude those indices
    # from the GPX-keyed coverage check (else they false-fail like missing
    # data). STRING keys (field names) — NOT an int-keyed map.
    per_lf_gpx_omitted: Optional[dict] = None
    # Per-LF Drive thumbnail URLs (``{lf.index: "https://drive..."}``)
    # — used by Phase D's per-LF Metricool CSV writer. Same JSON
    # round-trip contract as the per_lf_* maps above.
    lf_thumbnail_drive_urls: Optional[dict] = None
    # Canonical per-LF ``ResolvedLocation`` (label-substrate
    # hardening). ``{lf.index: ResolvedLocation}`` — computed once by the
    # builder on the quality-filtered GPS trace and consumed by every
    # location surface (title/OST, description, chapters, country/units/
    # playlist) so they stop re-deriving independently. ``None`` default:
    # a state file written before this field existed (no key) loads cleanly
    # and routes to the recompute-on-resume path (the resolver is a pure addition until a
    # surface reads it). Registered in ``_INT_KEY_DICT_FIELDS`` (int keys
    # survive JSON round-trip) AND ``_dict_value_dataclass_fields_map``
    # (values rehydrate to ``ResolvedLocation``, not plain dicts). On
    # resume the builder consumes the per-LF GPS trace rebuilt by
    # ``gps_enrichment._build_gps_points_by_lf`` from the persisted
    # ``gps_per_episode`` (no new persisted field).
    per_lf_resolved_location: Optional[dict] = None
    # One LocationEvidence record per LF, as its plain
    # dict form, built in Phase BC from a bounded GPS sample. Phase D hands it
    # to Waypost's select_title as the title facts. ``None`` default: a
    # state file written before this field loads cleanly and the LF keeps its
    # baseline title.
    per_lf_location_evidence: Optional[dict] = None
    # Each LF's retained-route evidence (``final_text_record.route_record``)
    # from Phase BC — place names, seconds, gaps and region claims, no fixes.
    # Phase D builds the final-text review from it and upload re-verifies.
    per_lf_route_record: Optional[dict] = None
    # Waypost's chosen title, approved against the retained route before the
    # LF encode. Phase D uses this exact record and checks its route fingerprint
    # and baseline rather than making a later independent title choice.
    per_lf_pre_render_title_decisions: Optional[dict] = None
    # The specificity-ladder award, RECORDED (never gating) where it is
    # already computed — the pre-encode demotion pass
    # (``phase_bc_day_first._degrade_labels_to_awarded_tier``). Declared so the
    # maps survive JSON round-trip: the provenance sidecar used to read them
    # through ``getattr(state, ..., None)``, which on a dataclass with no
    # ``__getattr__`` and an unknown-key-dropping ``from_dict`` could only ever
    # return the default — a dead read that looked like a working feature.
    # ``{lf.index: coverage_percent}`` (0–100, NOT the 0–1 fraction
    # ``award_tier_for_evidence`` returns), ``{lf.index: tier_int}`` and
    # ``{lf.index: {scalar evidence}}`` — the evidence is a JSON-safe projection
    # (no GPS trace, no ``geocode_fn`` callable). Int keys → registered in
    # ``_INT_KEY_DICT_FIELDS``. ``None`` default: a state file written before
    # this field existed loads cleanly and the sidecar records null-with-reason
    # as before.
    per_lf_gps_coverage_pct: Optional[dict] = None
    per_lf_awarded_tier: Optional[dict] = None
    per_lf_tier_evidence: Optional[dict] = None

    # ── Phase C ────────────────────────────────────────────────────────
    drive_plan: Any = None  # DrivePlan
    route_profiles: Optional[dict] = None

    # Each planned short's STABLE global index, assigned once when the
    # shorts ladder is planned (``social_generator.assign_global_short_indexes``,
    # per LF in plan order) and persisted here so Phase D consumes it instead
    # of recomputing ``global_index_offset`` from ``len(posts)`` — a render
    # failure then leaves a hole (the index is never reused/shifted), never a
    # silent renumber of every later short. Keyed by ``str(short_path)``, NOT
    # an LF index — deliberately excluded from ``_INT_KEY_DICT_FIELDS``,
    # whose int-key coercion would corrupt these path-string keys. ``None``
    # default: a checkpoint written before this field existed loads cleanly and
    # Phase D falls back to the legacy recompute with a logged warning (see phase_d_scheduling.py).
    short_global_index: Optional[dict] = None  # dict[str, int]
    # The set of GLOBAL short indexes flagged ``interactive`` in
    # ``titles.yaml``, derived once via ``social_generator.load_hand_titles``
    # + ``derive_interactive_indexes`` at plan time and persisted here —
    # never re-parsed at render time. Stored as a sorted list (JSON has no
    # set type); consumers treat it as a set. ``None``/absent means "not yet
    # derived for this checkpoint", not "no interactive shorts".
    interactive_indexes: Optional[list] = None  # list[int]

    # ── Phase D ────────────────────────────────────────────────────────
    drive_schedule: Any = None
    upload_manifests: Optional[dict] = None
    youtube_uploads: Optional[dict] = None
    youtube_uploads_sha: Optional[str] = None  # SHA of durable youtube_uploads.json sidecar

    # ── CLI schedule overrides ───────────────────────────────────────────
    start_date: Optional[str] = None
    shorts_per_day: Optional[int] = None
    shorts_target: Optional[int] = None
    shorts_gap: Optional[float] = None
    youtube_upload: bool = False
    # Single-LF publish-date override (``--lf-publish-date YYYY-MM-DD``).
    # When set, Phase D's scheduler skips per-day reseeding and uses this
    # date verbatim as the LF's publish anchor. Read by
    # ``phase_d._resolve_lf_publish_anchor``; mirrors ``start_date`` in
    # round-trip handling.
    lf_publish_date: Optional[str] = None
    # ``--no-youtube-upload`` flag — when True, Phase D's YouTube
    # uploader emits the per-LF Drive + Metricool deliverables but
    # skips the actual ``videos.insert`` API call. ALWAYS_CLI:
    # written concrete every invocation, twin of youtube_upload —
    # omitting the flag clears a saved True.
    no_youtube_upload: Optional[bool] = None
    # ``--keep-intermediates`` / ``--keep-masters`` — both opt out of
    # Phase D's cleanup pass. Declared so resume preserves the operator
    # choice; ``cleanup.py`` and ``phase_bc_day_first`` read these.
    # Tri-state (None = flag absent) so the ``schedule_overrides``
    # CLI-wins allowlist in ``_try_resume`` can distinguish "operator did
    # not pass the flag" (saved state wins) from an explicit ``--keep-*``
    # (CLI wins) — a plain ``False`` default would re-stomp a saved True.
    keep_intermediates: Optional[bool] = None
    keep_masters: Optional[bool] = None
    # ``--allow-short-count`` — operator accepts a shorts shortfall:
    # suppresses the SHORTS-CONTRACT plan/render hard-fails in
    # ``run_day_pipeline`` and lets the master-deletion gate count-match
    # against the actual (reduced) count. Same tri-state sentinel as the
    # ``keep_*`` flags above.
    allow_short_count: Optional[bool] = None
    # ``--no-ledger-reconcile`` — skip the startup YouTube↔ledger
    # reconciliation. Tri-state sentinel; readers treat None as False.
    no_ledger_reconcile: Optional[bool] = None
    # ``--sign-verify`` / ``--sign-verify-strict`` — road-sign GPS
    # verification. The first enables the (opt-in) per-day sign OCR pass plus
    # the advisory per-LF comparison; the second escalates a corroborated
    # state/county conflict to a pre-encode hard fail. Same tri-state sentinel
    # as the ``keep_*`` flags: None = flag absent, so a CLI-absent resume
    # preserves the operator's saved intent instead of silently re-deciding.
    # Both carry a ``config.drive`` mirror (the pipeline read path) via the
    # flag registry.
    sign_verify: Optional[bool] = None
    sign_verify_strict: Optional[bool] = None
    # Per-day road-sign observations recorded by the Phase A sign pass:
    # ``{"YYYY-MM-DD": [SignHit-as-dict, ...]}``. Plain JSON (each hit is a
    # flat dict of JSON-native fields plus a list of claim dicts) so the state
    # serializer round-trips it with no codec; consumers hydrate through
    # ``analysis.sign_evidence.hits_by_date_from_state`` rather than reaching
    # into raw dicts. Persisted so a resume never re-decodes footage for
    # evidence it already gathered.
    sign_hits_by_date: Optional[dict] = None
    # ``--no-audio`` — strip the audio track from the day master (encoded
    # ``-an``), so every LF stream-cut from it ships silent (shorts are
    # always muted). Mirrored to ``config.drive.skip_audio`` (the pipeline
    # read path) by the flag registry. Tri-state sentinel: None = flag
    # absent, so a CLI-absent resume preserves a saved True instead of
    # re-encoding a silent-intent day master WITH audio; readers treat
    # None as False.
    skip_audio: Optional[bool] = None
    # Explicit --lf-mode (cli | resumed): None = never stated, config decides.
    lf_mode: Optional[str] = None
    # Per-LF operator-supplied location overrides — argparse
    # ``--per-lf-location 'INDEX=LABEL'``. Phase A / Phase BC consume
    # this via ``gps_enrichment._derive_per_lf_locations`` when GPS is
    # missing for a given LF. Round-tripped to survive resume.
    per_lf_overrides: Optional[dict] = None
    # Force the Phase D scheduler to skip rebasing publish anchors when
    # the operator re-runs after a partial Phase D. Set by
    # ``drive_orchestrator._build_state`` when ``--force-no-rebase`` is
    # supplied. Read by Phase D's scheduler entry. Tri-state sentinel: None
    # means the flag was absent, so a CLI-absent resume cannot re-stomp a saved
    # True; readers treat None as False.
    force_no_rebase: Optional[bool] = None
    # ``--allow-bad-shorts`` — diagnostic escape hatch that disables the
    # 1080x1920/9:16 dimension validation in Phase D. Read at
    # ``phase_d_gates._assert_shorts_quality_gate`` entry, which coerces
    # via ``bool(getattr(...))`` so None is falsy (gate active). Tri-state
    # sentinel: None = flag absent, so a CLI-absent resume preserves a
    # saved True instead of re-stomping it with a concrete False.
    skip_shorts_quality_gate: Optional[bool] = None
    # ``--ost-audit-confirmed`` — top-level confirmation signal for
    # the OST audit gate (third option after ``ost_audit_*.confirmed``
    # marker file and ``state.ost_audit_passed[date]`` map). Read by
    # ``ost_audit_sidecar.gate_or_raise``. Tri-state (None = flag absent)
    # so the resume CLI-wins allowlist preserves a saved confirmation.
    ost_audit_confirmed: Optional[bool] = None
    # The confirmation supplied on THIS invocation (never persisted — see
    # ``_UNSERIALIZED_FIELDS``). The gate honors a bypass ONLY when the flag was
    # passed on the current run; a persisted ``ost_audit_confirmed`` alone can no
    # longer auto-pass a changed, never-reviewed burn after a mid-encode kill lost
    # the audit hash. A resume re-passes through the hash-bound ``ost_audit_passed``
    # + ``ost_audit_hash`` markers, not this flag.
    ost_audit_confirmed_this_run: Optional[bool] = None
    # ``--ost-system-review`` on THIS invocation (never persisted).
    # Lets ``ost_system_review`` pass the OST gates; see that module.
    ost_system_review_this_run: Optional[bool] = None
    # Phase BC's route-claim record per day, sealed by its writer
    # (``ost_system_review.record_system_reviews``). Persisted so a resume
    # keeps it; route claims re-running rewrites it.
    ost_system_review: Optional[dict] = None
    # ``--gate-cd-confirmed`` — bypasses the Gate C/D content-review
    # prompt when the persisted content hash still matches.
    # Set by ``review_gate`` after a successful gate; round-tripped
    # for resume so the operator doesn't re-confirm identical content.
    gate_cd_confirmed: Optional[bool] = None
    gate_cd_confirmation_hash: Optional[str] = None
    # Per-day OST audit resume markers (``{date: bool}`` / ``{date: hex}``).
    # ``ost_audit_sidecar.gate_or_raise`` populates these when audit
    # passes; on resume, a matching ``ost_audit_hash`` short-circuits
    # the re-prompt. Drift-checked by ``state_invariants._check_ost_audit_hash_drift``.
    ost_audit_passed: Optional[dict] = None
    ost_audit_hash: Optional[dict] = None
    # Per-day SHORTS OST audit resume markers (``{date: bool}`` / ``{date: hex}``),
    # the pre-burn shorts gate's analog of the per-LF markers above.
    # ``ost_audit_sidecar.shorts_gate_or_raise`` populates these on a pass so a
    # resume with a matching ``shorts_ost_audit_hash`` short-circuits the
    # re-prompt; a material change to any short's burned text reopens the gate.
    shorts_ost_audit_passed: Optional[dict] = None
    shorts_ost_audit_hash: Optional[dict] = None
    # The Shorts / superlong route-claim records per day, sealed by their
    # writer (``ost_system_review.record_surface_reviews``); each surface's
    # gate reads only its own. Rewritten by the measuring loop every run.
    shorts_ost_system_review: Optional[dict] = None
    superlong_ost_system_review: Optional[dict] = None
    # Phase D fingerprint for partial-resume safety. ``drive_orchestrator``
    # writes the SHA + inputs at Phase D entry; resume compares the
    # fingerprint of the just-computed inputs against the persisted SHA
    # before re-using cached Phase D artifacts.
    phase_d_fingerprint: Optional[str] = None
    phase_d_fingerprint_inputs: Optional[dict] = None
    # ``{phase_name: code_version}`` stamped when each phase completes, so a
    # resume can drop a checkpointed phase_a/phase_bc whose code-rule version
    # changed (e.g. a planner/audio rule change) and re-run it instead of
    # shipping pre-fix renders. Mirrors the phase_d fingerprint invalidation.
    phase_code_versions: Optional[dict] = None
    # Per-LF measured short durations (``{lf_index: [duration_sec, ...]}``)
    # written by Phase D after shorts encode + validation. Read by
    # ``preflight_plan`` to estimate Phase D wall-clock on resume runs.
    short_durations_per_lf: Optional[dict] = None
    # Declared here so
    # ``PipelineState.to_dict()`` serializes them and ``from_dict()``
    # restores them on resume. Prior to this, setting them via
    # ``state[key] = value`` worked for hasattr/get reads but silently
    # dropped at serialization because ``dataclasses.fields()`` never
    # listed them. The resume merge
    # (``flag_registry.merge_on_resume``) sets them with ``object.__setattr__``,
    # which depends on these being declared fields to round-trip.
    #
    # Tri-state for the argparse-sentinel pattern: None
    # means "CLI did not supply the flag — state wins on resume".
    # True/False means "CLI explicitly said so — CLI wins on resume".
    # Legacy callers that set these via ``state[key] = bool(value)``
    # before any argparse wiring worked (the orchestrator ``run()``
    # method does this) still work because bool is a valid value.
    allow_seo_degraded: Optional[bool] = None
    force_sidecar_reset: Optional[bool] = None
    # ``--superlong`` sentinel — mirrored onto config.drive.superlong_enabled
    # by flag_registry.py.
    superlong_enabled: Optional[bool] = None
    # ``--superlong-day`` sentinel — mirrored onto
    # config.drive.superlong_publish_day by flag_registry.py.
    superlong_publish_day: Optional[str] = None

    # ── Orchestrator bookkeeping ───────────────────────────────────────
    run_start: Optional[str] = None  # ISO-8601 anchor for DoD freshness checks
    phases_completed: Optional[list] = None
    # Per-day bookkeeping for `phase_bc_day_first` resume safety.
    # Keyed by date-string (e.g. "2025-01-15"), each value is a dict
    # with at least:
    #     {"master_path": "<abs path>"}
    # A day that ran under an explicit/derived shorts target also carries
    # {"shorts_target_in_force": <per-LF int>, "shorts_expected": <post-cap
    # int>} — the resume shortfall gate prefers this recorded
    # post-per-LF-capacity-cap expectation over re-deriving the pre-cap
    # total when the current target matches what the day actually ran under.
    # On resume at phase_bc_day_first entry, days present in this map
    # whose master artifact still exists on disk at `master_path` are
    # SKIPPED — they do not re-run encode. Days absent from the map
    # (or whose artifact is missing) fall through to the existing
    # file-existence + duration-probe gate inside day_master_encoder
    # (backward-compat path for legacy state files that pre-date this
    # field). `from_dict` returns None for missing entries, which the
    # skip logic treats as "not recorded → fall through".
    per_day_completed: Optional[dict] = None
    # The insert flags the finished days were rendered with (``inserts.insert_request``); the
    # flags themselves stay per-invocation, a resume is refused if they differ. Bookkeeping.
    insert_request: Optional[dict] = None
    _per_day_skip_indices: Optional[set] = None
    # ``{date: {"part_count": int, "records": [...]}}``; gates
    # ``cleanup.drop_day_master`` and skips already-final parts on resume.
    # A record may carry ``"lower_thirds": [{"label", "time_sec"}, ...]`` —
    # the reviewable burned text, derived from that part's OWN
    # ``record["chapters"]`` via ``chapter_town_events``: the canonical,
    # ``ResolvedLocation``-reconciled chapter pipeline every town-name
    # surface uses, never raw GPX. (This comment used to name a
    # caller-supplied ``place_transitions_by_part`` override; that
    # parameter is gone from ``_produce_superlong_parts`` and the
    # chapter-derived path is the only production shape.)
    superlong_parts: Optional[dict] = None

    # ── Resume-time backstop observability ────
    # ``{field_name: [lf_index, ...]}`` populated by
    # ``state_invariants.record_backstop_recovery`` whenever a resume-time
    # fallback reconstructs missing state from on-disk artefacts. ``[-1]``
    # sentinel = whole-state recovery (``file_boundaries`` /
    # ``shorts_per_lf`` fallback). Operators read this + the
    # ``"backstop fired:"`` WARN log to see which phase boundaries got
    # breached on resume.
    per_lf_state_recovered_from_disk: Optional[dict] = None

    def __post_init__(self) -> None:
        """Apply disk-derivable migrations once at construction.

        These once ran only inside ``from_dict``; moving them here
        guarantees every code path that constructs a ``PipelineState``
        (``from_dict``, ``cls(**kwargs)`` in tests, direct calls) lands
        with the same coerced shape. Each migration is conditional and
        short-circuits when its inputs are missing, so empty
        ``PipelineState()`` construction stays a no-op.
        """
        self._migrate_per_lf_recording_date_from_paths()
        self._migrate_per_lf_trim_ratio_from_paths()
        self._migrate_per_lf_gpx_path_from_state()
        self._migrate_per_lf_gpx_path_legacy_state()

    def _migrate_per_lf_recording_date_from_paths(self) -> None:
        """Populate ``per_lf_recording_date`` from on-disk LF paths if absent.

        Day-first layout writes every LF under
        ``<stages>/day_<YYYY-MM-DD>/regular_lfs/lf_NN.mp4`` and the
        ``day_<DATE>`` directory name is the LF's recording date by
        construction (``run_day_pipeline`` enforces single-day LFs).
        """
        if self.per_lf_recording_date:
            return
        regular_lf_paths = self.regular_lf_paths
        if not regular_lf_paths:
            return
        drive_plan = self.drive_plan
        skeleton_lfs = getattr(drive_plan, "regular_lfs", None) if drive_plan is not None else None
        if not skeleton_lfs:
            return

        out: dict[int, str] = {}
        for lf, lf_path in zip(skeleton_lfs, regular_lf_paths):
            lf_idx = getattr(lf, "index", None)
            if lf_idx is None:
                continue
            match = _DAY_DIR_DATE_RE.search(str(lf_path))
            if match is None:
                continue
            out[int(lf_idx)] = match.group(1)
        if out:
            object.__setattr__(self, "per_lf_recording_date", out)

    def _migrate_per_lf_trim_ratio_from_paths(self) -> None:
        """Populate ``per_lf_trim_ratio`` from ffprobe + ``file_durations``.

        When the field is missing but LFs already exist on disk, ffprobe
        each LF's final duration and combine with the same source-file
        durations the fresh-encode path uses. LFs whose final-duration
        probe returns 0, whose source duration is unresolvable, or whose
        path doesn't exist are OMITTED — downstream code handles
        missing-key by falling back to the legacy "Driving" framing.
        """
        if self.per_lf_trim_ratio:
            return
        regular_lf_paths = self.regular_lf_paths
        if not regular_lf_paths:
            return
        drive_plan = self.drive_plan
        skeleton_lfs = getattr(drive_plan, "regular_lfs", None) if drive_plan is not None else None
        if not skeleton_lfs:
            return
        file_durations = self.file_durations or []

        # Late imports to avoid pulling the phase_bc machinery at module load.
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import _probe_duration
        from gomoveshift.pipeline.post_encoding_invariants import _day_local_source_durations

        # Segment ``source_file_index`` is DAY-LOCAL post-partition while
        # ``file_durations`` is DRIVE-WIDE. Slice each LF's durations to its
        # own recording day (same fix as the fresh-encode path and resume
        # backstop); an empty map (single-day / legacy state without
        # ``episode_dates``) falls back to the drive-wide list.
        per_lf_source_durations = _day_local_source_durations(
            self.per_lf_recording_date or {}, self.episode_dates or [], file_durations
        )

        out: dict[int, float] = {}
        for lf, lf_path in zip(skeleton_lfs, regular_lf_paths):
            lf_idx = getattr(lf, "index", None)
            if lf_idx is None:
                continue
            path_obj = Path(str(lf_path))
            if not path_obj.exists():
                continue
            try:
                final_duration = float(_probe_duration(path_obj) or 0.0)
            except Exception:  # noqa: BLE001
                continue
            if final_duration <= 0:
                continue
            durations = per_lf_source_durations.get(int(lf_idx), file_durations)
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
        if out:
            object.__setattr__(self, "per_lf_trim_ratio", out)

    def _migrate_per_lf_gpx_path_from_state(self) -> None:
        """Populate ``per_lf_gpx_path`` from ``gpx_files`` + ``episode_dates``.

        Backstop for state files pre-dating the field declaration.
        Each LF's path lives under ``day_<YYYY-MM-DD>/...``; ``episode_dates``
        is parallel to ``gpx_files``. Match the LF's day directory against
        ``episode_dates`` to find the matching GPX index.
        """
        if self.per_lf_gpx_path:
            return
        regular_lf_paths = self.regular_lf_paths
        if not regular_lf_paths:
            return
        gpx_files = self.gpx_files or []
        episode_dates = self.episode_dates or []
        if not gpx_files or not episode_dates:
            return
        drive_plan = self.drive_plan
        skeleton_lfs = getattr(drive_plan, "regular_lfs", None) if drive_plan is not None else None
        if not skeleton_lfs:
            return

        out: dict[int, str] = {}
        for lf, lf_path in zip(skeleton_lfs, regular_lf_paths):
            lf_idx = getattr(lf, "index", None)
            if lf_idx is None:
                continue
            match = _DAY_DIR_DATE_RE.search(str(lf_path))
            if match is None:
                continue
            lf_date = match.group(1)
            gpx_idx = None
            for i, ep_date in enumerate(episode_dates):
                if ep_date == lf_date and i < len(gpx_files) and gpx_files[i] is not None:
                    gpx_idx = i
                    break
            if gpx_idx is None:
                continue
            out[int(lf_idx)] = str(gpx_files[gpx_idx])
        if out:
            object.__setattr__(self, "per_lf_gpx_path", out)

    def _migrate_per_lf_gpx_path_legacy_state(self) -> None:
        """Durable resume invariant for ``per_lf_gpx_path``.

        Legacy state files written before ``per_lf_gpx_path`` was emitted
        as a total map (one entry per regular LF, ``None`` for GPS-less
        LFs) resume-fail at ``state_invariants._check_per_lf_completeness``.
        Walks each LF's on-disk ``day_<YYYY-MM-DD>/`` directory, probes
        for a sibling ``.gpx`` file, fills the discovered path or explicit
        ``None`` (covers deleted runs). WARNING-logged so operators see
        why a previously-broken resume now succeeds.
        """
        import logging

        if "phase_bc_day_first" not in (self.phases_completed or []):
            return
        regular_lf_paths = self.regular_lf_paths or []
        if not regular_lf_paths:
            return
        drive_plan = self.drive_plan
        skeleton_lfs = getattr(drive_plan, "regular_lfs", None) if drive_plan is not None else None
        if not skeleton_lfs:
            return

        expected_indices = [
            int(getattr(lf, "index"))
            for lf in skeleton_lfs
            if getattr(lf, "index", None) is not None
        ]
        if not expected_indices:
            return

        existing = self.per_lf_gpx_path or {}
        present = {int(k) for k in existing}
        missing = [i for i in expected_indices if i not in present]
        if not missing:
            return

        lf_idx_to_path: dict[int, Any] = {}
        for lf, lf_path in zip(skeleton_lfs, regular_lf_paths):
            idx = getattr(lf, "index", None)
            if idx is not None:
                lf_idx_to_path[int(idx)] = lf_path

        new_entries: dict[int, Optional[str]] = {}
        for lf_idx in missing:
            lf_path = lf_idx_to_path.get(lf_idx)
            gpx_value: Optional[str] = None
            if lf_path is not None:
                try:
                    # Canonical day-first layout: day_<DATE>/regular_lfs/lf_NN.mp4
                    day_dir = Path(str(lf_path)).resolve().parent.parent
                except Exception:  # noqa: BLE001
                    day_dir = None
                if day_dir is not None and day_dir.exists() and day_dir.is_dir():
                    try:
                        gpx_candidates = sorted(day_dir.glob("*.gpx"))
                    except OSError:
                        gpx_candidates = []
                    if gpx_candidates:
                        gpx_value = str(gpx_candidates[0])
            new_entries[lf_idx] = gpx_value

        merged: dict[int, Optional[str]] = {int(k): v for k, v in existing.items()}
        merged.update(new_entries)
        object.__setattr__(self, "per_lf_gpx_path", merged)

        logging.getLogger(__name__).warning(
            "per_lf_gpx_path migration: missing %d of %d expected "
            "LF entries on resume; filled from filesystem probe. before=%r after=%r",
            len(missing),
            len(expected_indices),
            dict(existing),
            merged,
        )

    # ── Dict-protocol syntactic sugar ─────────────────────────────────
    # The dual-shape contract is gone — every ``state``
    # threaded through the pipeline is a ``PipelineState``. These methods
    # remain as pure syntactic sugar so historical test code can keep
    # using ``state["x"]`` / ``state.get("x")`` (equivalent to attribute
    # access on this dataclass — no type ambiguity, no rehydration trap).
    # Do NOT reintroduce ``state_get`` / ``state_set`` helper functions:
    # the dual-shape contract they papered over no longer exists.

    def __getitem__(self, key: str) -> Any:
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        object.__setattr__(self, key, value)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return hasattr(self, key) and getattr(self, key) is not None

    def get(self, key: str, default: Any = _MISSING) -> Any:
        """Dict-style ``get`` with Python-dict ``None``-vs-missing semantics.

        Returns ``getattr(self, key)`` if the attribute exists, including
        when its value is ``None``. Only falls through to ``default`` when
        the attribute is genuinely absent — mirroring ``dict.get``. If no
        ``default`` is supplied, returns ``None`` for missing keys (also
        matching ``dict.get``).

        The earlier implementation conflated ``None`` and missing by
        returning ``default`` for both, which masked the "explicitly unset"
        signal that ``to_dict`` deliberately preserves for non-None-default
        fields.
        """
        val = getattr(self, key, _MISSING)
        if val is _MISSING:
            return None if default is _MISSING else default
        return val

    def setdefault(self, key: str, default: Any = None) -> Any:
        val = getattr(self, key, None)
        if val is None:
            object.__setattr__(self, key, default)
            return default
        return val

    def update(self, other) -> None:  # type: ignore[override]
        """Merge another PipelineState or dict into this one.

        When merging from a raw dict (e.g. JSON-loaded checkpoint), apply
        the same type coercions as ``from_dict()`` — int-key dicts, Path
        fields, tuple-list fields, and set fields.
        """
        if isinstance(other, PipelineState):
            # Merge only fields ``other`` carries a *non-default* value for, so
            # a freshly-constructed ``other`` (whose unset fields hold their
            # dataclass defaults) cannot clobber a meaningful value already on
            # ``self``. The old ``val is not None`` guard skipped only ``None``,
            # so a default-``False`` bool (e.g. ``skip_upload``) would overwrite
            # a saved ``True``. Latent today (the resume path copies fields via
            # ``object.__setattr__``, not this branch) — fixed defensively.
            for f in dataclasses.fields(other):
                val = getattr(other, f.name)
                if val is None:
                    continue
                default = _field_default(f)
                if val == default:
                    continue
                object.__setattr__(self, f.name, val)
        else:
            field_names = {f.name for f in dataclasses.fields(self)}
            singleton_dc_map = _dataclass_fields_map()
            dict_value_dc_map = _dict_value_dataclass_fields_map()
            for key, val in other.items():
                if key not in field_names:
                    continue
                # Apply the same coercions as from_dict()
                if key in ("source_dir", "output_base", "stages_base"):
                    if val is not None and not isinstance(val, Path):
                        val = Path(val)
                elif key in ("file_boundaries", "gps_per_episode"):
                    val = _list_to_tuples(val)
                elif key in _INT_KEY_DICT_FIELDS:
                    val = _coerce_int_key_dict(val)
                elif key == "processed_files":
                    val = _coerce_int_key_dict_paths(val)
                elif key == "delivered_lf_paths":
                    val = _coerce_int_key_dict_paths(val)
                elif key == "delivered_lfs":
                    if val is not None:
                        val = set(val) if not isinstance(val, set) else val
                elif key == "_per_day_skip_indices":
                    if val is not None:
                        val = set(val) if not isinstance(val, set) else val
                # Compound dataclass-shape coercions — mirror from_dict so
                # both load paths agree on the in-memory shape.
                if key in singleton_dc_map:
                    val = _deserialize_dataclass(val, singleton_dc_map[key])
                elif key in dict_value_dc_map:
                    val = _coerce_dict_value_dataclasses(val, dict_value_dc_map[key])
                object.__setattr__(self, key, val)

    def keys(self) -> list[str]:
        return [f.name for f in dataclasses.fields(self) if getattr(self, f.name) is not None]

    def items(self) -> list[tuple[str, Any]]:
        return [
            (f.name, getattr(self, f.name))
            for f in dataclasses.fields(self)
            if getattr(self, f.name) is not None
        ]

    def __iter__(self):
        return iter(self.keys())

    # ── Serialization ──────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict.

        Delegates to ``_drive_state._serialize_value`` for each field so
        that Path, tuple, set, numpy, and dataclass values are handled
        consistently with the existing save logic.

        None-handling: for fields
        whose dataclass default is non-None (e.g. ``region: str = ""``,
        ``publish_day: str = "wed"``, ``fresh: bool = False``), an
        explicit runtime ``None`` MUST survive serialization — otherwise
        the JSON round-trip silently rehydrates to the class default and
        loses the "explicitly unset" signal. Fields whose default is
        ``None`` are still skipped (they round-trip identically either
        way; emitting them just bloats the on-disk JSON).
        """
        from ._drive_state import _serialize_value

        result = {}
        for f in dataclasses.fields(self):
            if f.name in _UNSERIALIZED_FIELDS:
                # Recoverable from .npz / .gpx on disk — never persisted.
                continue
            val = getattr(self, f.name)
            if val is None:
                if _field_default(f) is not None:
                    # Non-None default: preserve the explicit None so
                    # from_dict can restore it instead of falling back
                    # to the class default.
                    result[f.name] = None
                continue
            result[f.name] = _serialize_value(val)
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineState":
        """Deserialize from a raw dict (including legacy JSON-loaded dicts).

        Handles:
        - Path coercion for source_dir / output_base
        - list -> tuple for file_boundaries / gps_per_episode
        - string -> int key coercion for processed_files, shorts_per_lf,
          delivered_lf_paths, metadata_per_lf, social_posts_per_lf, etc.
        - list -> set for delivered_lfs
        """
        kwargs: dict[str, Any] = {}
        field_names = {f.name for f in dataclasses.fields(cls)}

        for key, val in d.items():
            if key not in field_names:
                continue
            kwargs[key] = val

        # Path fields
        for pf in ("source_dir", "output_base", "stages_base"):
            if pf in kwargs and kwargs[pf] is not None:
                kwargs[pf] = Path(kwargs[pf]) if not isinstance(kwargs[pf], Path) else kwargs[pf]

        # Tuple-list fields (JSON serializes tuples as lists)
        for tf in ("file_boundaries", "gps_per_episode"):
            if tf in kwargs:
                kwargs[tf] = _list_to_tuples(kwargs[tf])

        # Int-key dict fields
        for ikf in _INT_KEY_DICT_FIELDS:
            if ikf in kwargs:
                kwargs[ikf] = _coerce_int_key_dict(kwargs[ikf])

        # Int-key + Path-value dict fields
        if "processed_files" in kwargs:
            kwargs["processed_files"] = _coerce_int_key_dict_paths(kwargs["processed_files"])
        if "delivered_lf_paths" in kwargs:
            kwargs["delivered_lf_paths"] = _coerce_int_key_dict_paths(kwargs["delivered_lf_paths"])

        # Set fields
        if "delivered_lfs" in kwargs and kwargs["delivered_lfs"] is not None:
            kwargs["delivered_lfs"] = set(kwargs["delivered_lfs"])
        if "_per_day_skip_indices" in kwargs and kwargs["_per_day_skip_indices"] is not None:
            kwargs["_per_day_skip_indices"] = set(kwargs["_per_day_skip_indices"])

        # Numpy-marker fields — reconstruct ndarrays from the {"__numpy__": ...}
        # tagged dict written by _serialize_value.
        if "drive_scores" in kwargs:
            kwargs["drive_scores"] = _deserialize_numpy(kwargs["drive_scores"])

        # `per_day_completed` migrated from Optional[list] → Optional[dict]
        # in an earlier version. Legacy state files may
        # carry a list (the former type) or even an empty list literal the
        # old code never populated — in either case, coerce to an empty
        # dict so the new skip logic falls through to the file-existence
        # gate without KeyError. A well-formed dict passes through.
        if "per_day_completed" in kwargs:
            val = kwargs["per_day_completed"]
            if val is None or isinstance(val, dict):
                pass  # well-formed or absent → keep as-is
            else:
                # Legacy list (or any other shape): discard so resume
                # falls through to the existing gate. Backward-compat.
                kwargs["per_day_completed"] = None

        # Dataclass-shaped fields — serialized via .to_dict() or
        # dataclasses.asdict() and must be re-hydrated on load. Without
        # this, downstream code calling e.g. drive_plan.regular_lfs on a
        # resumed state gets an AttributeError because the field is a
        # plain dict.
        for fname, dc_cls in _dataclass_fields_map().items():
            if fname in kwargs:
                kwargs[fname] = _deserialize_dataclass(kwargs[fname], dc_cls)

        # Dict-VALUE dataclass fields — e.g.
        # ``upload_manifests: dict[int, UploadManifest]``. Int-key
        # coercion above handles the keys; this pass walks each value
        # through the same ``from_dict``/constructor path that
        # ``_deserialize_dataclass`` uses for singleton fields. Closes
        # a ``_manifest_files()`` shim failure mode at the
        # contract boundary instead of at each consumer.
        for fname, dc_cls in _dict_value_dataclass_fields_map().items():
            if fname in kwargs:
                kwargs[fname] = _coerce_dict_value_dataclasses(kwargs[fname], dc_cls)

        # ``cls(**kwargs)`` triggers ``__post_init__``, which runs the
        # per-LF disk-derivable migrations (formerly inlined here as
        # module-level ``_migrate_*`` helpers). The migrations are no-ops
        # when the relevant inputs are missing, so direct construction
        # without resume context (tests, fresh runs) stays cheap.
        instance = cls(**kwargs)

        _assert_canonical_key_shape(instance)
        return instance


def _field_default(f: "dataclasses.Field") -> Any:
    """Return the static default value for a dataclass field, or None.

    Used by ``PipelineState.to_dict`` to decide whether a runtime
    ``None`` carries information (non-None default → preserve) or is
    indistinguishable from "field never set" (None default → skip).
    """
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        try:
            return f.default_factory()  # type: ignore[misc]
        except Exception:
            return None
    return None


def _dataclass_fields_map() -> dict[str, type]:
    """Lazy accessor for the ``field_name -> dataclass type`` mapping.

    Imports are deferred to call time to avoid circular-import risk
    between ``drive_state_model`` (imported by pipeline internals) and
    ``gomoveshift.models`` / ``gomoveshift.pipeline.ingest.gopro_parser``.
    """
    from gomoveshift.models import DrivePlan
    from gomoveshift.pipeline.ingest.gopro_parser import IngestPlan
    from gomoveshift.pipeline.scheduler import DriveReleaseSchedule

    return {
        "drive_plan": DrivePlan,
        "ingest_plan": IngestPlan,
        "drive_schedule": DriveReleaseSchedule,
    }


def _dict_value_dataclass_fields_map() -> dict[str, type]:
    """``field_name -> dataclass type`` for ``dict[int, Dataclass]`` fields.

    A past failure (a CSV with empty
    ``Picture Url 1`` on every row) traced to ``state["upload_manifests"]`` coming back
    as ``dict[int, dict]`` instead of ``dict[int, UploadManifest]`` after
    a JSON round-trip — ``getattr(manifest, "files", {})`` then returned
    ``{}`` for every manifest. This map is the contract path that closes
    the bug class: ``PipelineState.from_dict`` walks every value through
    ``_deserialize_dataclass`` so consumers can rely on attribute access.
    """
    from gomoveshift.models import ResolvedLocation
    from gomoveshift.pipeline.uploader import UploadManifest

    return {
        "upload_manifests": UploadManifest,
        # ``per_lf_resolved_location: dict[int, ResolvedLocation]``.
        # ``ResolvedLocation`` is a plain dataclass (no ``from_dict``), so
        # ``_deserialize_dataclass`` rehydrates via ``cls(**value)``.
        "per_lf_resolved_location": ResolvedLocation,
    }


def _coerce_dict_value_dataclasses(raw: Any, cls: type) -> Optional[dict]:
    """Re-hydrate each value of an int-keyed dict to ``cls``.

    Assumes keys are already coerced to ``int`` by
    ``_coerce_int_key_dict``. Tolerates already-hydrated instances
    (in-memory round-trip) and None (field never populated).
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return raw
    return {k: _deserialize_dataclass(v, cls) for k, v in raw.items()}


def _deserialize_dataclass(value: Any, cls: type) -> Any:
    """Re-hydrate a serialized dataclass dict into an instance of ``cls``.

    ``value`` is either already a ``cls`` instance (in-memory round trip),
    ``None`` (field never populated), or a ``dict`` (JSON-loaded). For
    the dict case, prefer a ``cls.from_dict`` classmethod when available
    and fall back to ``cls(**value)`` for plain dataclasses.
    """
    if value is None or isinstance(value, cls):
        return value
    if not isinstance(value, dict):
        return value
    from_dict = getattr(cls, "from_dict", None)
    if callable(from_dict):
        return from_dict(value)
    return cls(**value)


def _deserialize_numpy(v: Any) -> Any:
    """Reconstruct a numpy ndarray from the marker dict, or pass through."""
    if not (isinstance(v, dict) and v.get("__numpy__")):
        return v
    import numpy as np

    arr = np.array(v["data"], dtype=v["dtype"])
    shape = tuple(v.get("shape", ()))
    if shape:
        arr = arr.reshape(shape)
    return arr


# ─── Per-LF disk-derivable migrations ─────────────
#
# Migration LOGIC lives on ``PipelineState`` itself as ``_migrate_*``
# methods invoked from ``__post_init__``. The former module-level
# ``_migrate_*`` helpers were collapsed into methods so every
# construction path (``from_dict``, direct ``cls(**kwargs)``, tests)
# applies the same coercions. The methods are idempotent and short-circuit
# when the relevant inputs are missing, so empty-state construction
# stays cheap. Only the shared regex stays at module scope.

_DAY_DIR_DATE_RE = __import__("re").compile(r"day_(\d{4}-\d{2}-\d{2})")


def _assert_canonical_key_shape(instance: "PipelineState") -> None:
    """Raise if any int-keyed-dict field still carries non-int keys.

    Defense-in-depth backstop for the ``_coerce_int_key_dict`` pass
    inside ``from_dict``. Catches regressions where a new field is
    declared on ``PipelineState`` but the author forgot to add it to
    ``_INT_KEY_DICT_FIELDS`` — the resulting dict-of-str-keys would
    otherwise silently miss every ``d.get(int_key)`` lookup downstream.

    Only checks fields enumerated in ``_INT_KEY_DICT_FIELDS``. Empty /
    None fields are no-ops. The ``per_day_completed`` field is keyed by
    date string and is intentionally excluded.
    """
    for fname in _STRICT_INT_KEY_DICT_FIELDS:
        val = getattr(instance, fname, None)
        if not isinstance(val, dict) or not val:
            continue
        bad = [k for k in val if not isinstance(k, int)]
        if bad:
            raise ValueError(
                f"PipelineState.{fname}: non-int keys after load "
                f"({bad[:3]}{'...' if len(bad) > 3 else ''}) — "
                "JSON int-key coercion missed this field. Add it to "
                "_INT_KEY_DICT_FIELDS in drive_state_model.py."
            )
