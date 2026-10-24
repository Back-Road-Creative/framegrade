"""Phase D: Metadata & Publish.

Orchestration:

  1. Build per-LF metadata.
  1.5. Generate YouTube chapters from GPS data (non-blocking).
  2. Upload shorts to Google Drive (existing behaviour).
  3. Upload LFs to YouTube via ``YouTubeUploader`` (late-imported).
  4. Generate + attach YouTube thumbnails.
  5. Write ONE consolidated Metricool CSV for all shorts, with each row's
     First Comment Text pointing at the parent LF's YouTube URL.
  6. Final validation gate.

Not done here:
  * Super LF metadata / upload / CSV generation.
  * Drive upload of LF videos (they go to YouTube).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ...config import Config
from ...tracking import MetricsCollector, TrackingStore  # post-upload metrics hook
from ..chapters import ChapterGenerator
from ..csv_generator import CSVGenerator  # noqa: F401 — patched in tests
from ..exceptions import YouTubeQuotaError  # noqa: F401 — re-exported for tests
from ..lf_segment_window import (  # noqa: F401 — re-exported; the engine imports them from here
    _build_included_segments_for_lf,
    _pick_representative_episode,
)
from ..longform_metadata import LongFormMetadataGenerator
from ..media_api import deliverables_for_publish, publish_lf_paths
from ..pipeline_validator import PipelineValidator
from ..related_video_sidecar import emit_related_video_worklist
from ..scheduler import PipelineScheduler, ReleaseSchedule  # noqa: F401 — patched in tests
from ..social_generator import (
    SocialGenerator,
    assert_title_prefixes_unique,
    derive_interactive_indexes,
    load_hand_titles,
)
from ..state_invariants import assert_phase_d_per_lf_inputs_present
from ..uploader import PipelineUploader, UploadManifest  # noqa: F401 — patched in tests
from .external_systems_coordinator import (  # noqa: F401 — re-exported for tests + bare-name lookups
    _resolve_drive_subfolder_id,
    _resolve_shorts_paths,
    _resolve_youtube_uploader_class,
    _upload_all_lfs_shorts_to_drive,
    _upload_lf_shorts_to_drive,
    _upload_lf_thumbnails_to_drive,
    _upload_lf_to_youtube,
    _upload_lfs_to_youtube,
)

# Backwards-compat alias for a static-method helper call site.
_LFMeta = LongFormMetadataGenerator

logger = logging.getLogger(__name__)


def _claim_native_only() -> bool:
    """The brand MediaProfile's ``claim_native_resolution_only`` (strict ``True`` if unloadable)."""
    from ...console.settings import current_brand
    from ...publishing.brand_policy import load_media_profile

    try:
        return bool(load_media_profile(current_brand()).claim_native_resolution_only)
    except Exception:  # noqa: BLE001 — fail closed to the strict policy
        logger.warning("phase D: media profile unreadable -- native-only claim policy assumed")
        return True


def log_upload_lfs_banner(
    *,
    youtube_upload_enabled: bool,
    skip_youtube_upload: bool,
    no_youtube_upload: bool,
    publish_set: bool,
) -> None:
    """Greppable ``D-UPLOAD-LFS:`` banner at Phase D upload entry.

    Prefix is the contract for operator tmux-log grep. ENABLED/DISABLED
    derives from the SAME predicate as the actual gate in
    ``_upload_lfs_to_youtube`` (``skip_youtube_upload or not
    youtube_upload_enabled``); ``no_youtube_upload`` only flavors the
    provenance line, so banner and gate can never disagree.
    """
    if not (skip_youtube_upload or not youtube_upload_enabled):
        suffix = " (--publish set)" if publish_set else " (--youtube-upload)"
        logger.info("D-UPLOAD-LFS: ENABLED%s", suffix)
        return
    if no_youtube_upload:
        logger.info(
            "D-UPLOAD-LFS: DISABLED (--no-youtube-upload, this invocation — schedule + "
            "CSV still generated; re-run with --youtube-upload to publish)"
        )
    elif skip_youtube_upload:
        logger.info(
            "D-UPLOAD-LFS: DISABLED (--skip-upload/--skip-youtube-upload, this "
            "invocation — re-run with --youtube-upload to publish)"
        )
    else:
        logger.info(
            "D-UPLOAD-LFS: DISABLED (no --publish / --youtube-upload this invocation — "
            "pass --youtube-upload to enable)"
        )


def _collect_metrics_best_effort(
    *,
    project_base: str,
    output_base: Path,
    youtube_uploads: dict,
) -> None:
    """Best-effort post-upload metrics collection.

    Registers each uploaded LF's YouTube id into the tracking record, then
    pulls metrics via :class:`MetricsCollector`. BEST-EFFORT: uploads are
    already done, so metrics are additive. ANY failure (unavailable
    collector, missing or revoked credential, platform API error, store IO
    error) is logged ``METRICS-COLLECT:`` and swallowed; it NEVER raises.
    ``collect_all`` / ``collect_for_short`` also swallow per-collector
    errors, so one unavailable platform can't abort the others.
    """
    try:
        # ``output_base`` is the PROJECT dir (<output root>/<project>); the
        # store appends <project>/tracking itself, so root it one level up
        # -- the same root ``metrics collect``/``velocity``/``report`` read
        # by default (``TrackingStore()`` -> ``gms_output_dir()``).
        store = TrackingStore(root=Path(output_base).parent)
        if not youtube_uploads or not store.exists(project_base):
            logger.info("METRICS-COLLECT: nothing to collect for %s, skipping", project_base)
            return

        # Register each LF's uploaded YouTube id against the matching short
        # index (store keys shorts by global index; youtube_uploads by LF).
        # A miss (short not registered) raises ValueError — non-fatal here.
        for lf_index, record in youtube_uploads.items():
            video_id = (record or {}).get("video_id", "")
            if not video_id:
                continue
            try:
                store.update_platform_id(project_base, lf_index, "youtube", video_id)
            except ValueError:
                logger.debug("METRICS-COLLECT: no short #%s to tag", lf_index)

        record = store.load(project_base)
        updated = MetricsCollector().collect_all(record)
        store.save(record)
        logger.info(
            "METRICS-COLLECT: collected metrics for %d short(s) in %s", updated, project_base
        )
    except Exception as exc:  # noqa: BLE001 — never break the pipeline
        logger.warning("METRICS-COLLECT: failed (uploads unaffected): %s", exc)


# ─── Phase D output-generator code version ──────────────────────────────
#
# Bumped on every change to any Phase D output generator
# (``_build_gms_description``, ``_location_tags``,
# ``csv_generator.build_consolidated_shorts_csv``, the YouTube metadata
# JSON shape, the consolidated Metricool CSV shape, the per-LF thumbnail
# pipeline, etc.). The orchestrator's resume path includes this in the
# Phase D fingerprint, so bumping invalidates every existing
# ``phase_d`` checkpoint and forces re-run on the next resume.
#
# ``compute_phase_d_fingerprint`` reads it via the function parameter
# rather than the module attribute so tests can monkeypatch the
# constant and assert invalidation without re-importing the module.
PHASE_D_CODE_VERSION = 1


# Lazy-loaded exception types from google API libraries.
# Resolved at usage time so the module can still be imported when the
# google packages are absent (the upload path raises RuntimeError instead).
try:
    from google.auth.exceptions import RefreshError as _RefreshError
except ImportError:
    _RefreshError = type(None)  # never matched

try:
    from googleapiclient.errors import HttpError as _HttpError
except ImportError:
    _HttpError = type(None)  # never matched


def _build_recording_dates_by_short_index(
    *,
    shorts_by_lf: "dict[int, list]",
    per_lf_recording_date: "dict",
) -> "dict[int, str]":
    """Map ``short_index → recording date`` from ``state["per_lf_recording_date"]``.

    The consolidated Metricool
    shorts CSV writer requires a per-short recording-date map (no default).
    Each short inherits its parent LF's date; this helper produces the map
    Phase D passes to ``build_consolidated_shorts_csv``.

    Parent maps are int-keyed by ``PipelineState.from_dict`` —
    ``_assert_canonical_key_shape`` guarantees the contract at load time.
    Raises ``ValueError`` if any parent LF referenced by ``shorts_by_lf``
    is missing from ``per_lf_recording_date`` — defense-in-depth ahead of
    the CSV writer's own contract check. The Phase BC invariant
    ``_assert_per_lf_recording_date_complete`` is the upstream guarantee
    that this raise is unreachable from a valid run.
    """
    result: dict[int, str] = {}
    for lf_idx, posts in shorts_by_lf.items():
        if not posts:
            # No shorts under this LF → nothing to map a date for. The
            # invariant only applies to LFs that actually parent shorts.
            continue
        date_val = per_lf_recording_date.get(lf_idx)
        if hasattr(date_val, "strftime"):
            date_val = date_val.strftime("%Y-%m-%d")
        if not isinstance(date_val, str) or not date_val.strip():
            raise ValueError(
                f"per_lf_recording_date missing entry for LF index {lf_idx} "
                f"(parent of {len(posts)} short(s)); Phase BC invariant "
                "_assert_per_lf_recording_date_complete should have caught this"
            )
        date_str = date_val.strip()
        for post in posts:
            result[int(post.short_index)] = date_str
    return result


def assert_audio_loudness_hard_fail(audio_failures: list[str]) -> None:
    """Hard-fail Phase D when any LF's post-encode integrated loudness is
    LOUDER than the near-silent ceiling (or breaches the true-peak cap).

    This is the long-form audio quality gate under the near-silent
    policy: every published LF must be practically silent (attenuate-only,
    never boosted toward a loudness target). Phase D's pre-existing
    ``logger.warning`` was insufficient, so this helper raises ``RuntimeError``
    so the run aborts before any Drive/YouTube upload — its caller
    ``_run_final_validation_gate`` runs ahead of the upload block in
    ``PhaseD.run`` (do not move it back below the uploads).

    Pulled out of ``PhaseD.run`` for direct test coverage.
    """
    if not audio_failures:
        return
    raise RuntimeError(
        "Audio loudness HARD FAIL — at least one LF is LOUDER than the "
        "near-silent ceiling after attenuation (near-silent policy). "
        "A published video must be practically silent. Do NOT publish.\n"
        "Affected LFs:\n" + "\n".join(audio_failures)
    )


def assert_location_invariant_hard_fail(location_failures: list[str]) -> None:
    """Hard-fail Phase D when any LF's title county is in ZERO sampled GPS
    regions (the far-road class).

    The ``location_invariant`` content gate is WARN-only on the drive path
    (``phase_d_gates`` demotes every non-audio gate to ``logger.warning``),
    which left the cross-surface check toothless. This escalates the
    HARD case to ``RuntimeError`` so the run aborts before any Drive/YouTube
    upload (its caller ``_run_final_validation_gate`` runs ahead of the
    upload block in ``PhaseD.run``), mirroring ``assert_audio_loudness_hard_fail``.
    """
    if not location_failures:
        return
    raise RuntimeError(
        "Location invariant HARD FAIL — at least one LF's title county is "
        "in ZERO sampled GPS regions (far-road class). The displayed "
        "location desynced from the GPS trace. Do NOT publish.\n"
        "Affected LFs:\n" + "\n".join(location_failures)
    )


def assert_file_integrity_hard_fail(integrity_failures: list[str]) -> None:
    """Hard-fail Phase D when any LF's FINAL.mp4 fails the deterministic
    ``file_integrity`` gate — missing, under the 1 MiB floor, or
    ffprobe-unreadable, i.e. genuinely corrupt.

    The ``file_integrity`` content gate is WARN-only on the drive path
    (``phase_d_gates`` demotes every non-escalated gate to ``logger.warning``),
    which let a corrupt or truncated long-form sail through to Drive/YouTube
    upload. This escalates it to ``RuntimeError`` so the run aborts before any
    upload (its caller ``_run_final_validation_gate`` runs ahead of the
    upload block in ``PhaseD.run``), mirroring ``assert_audio_loudness_hard_fail``.
    Zero false-positive risk: the gate fails only on genuinely-broken files.
    """
    if not integrity_failures:
        return
    raise RuntimeError(
        "File integrity HARD FAIL — at least one LF's FINAL.mp4 is missing, "
        "under the 1 MiB floor, or ffprobe-unreadable (genuinely corrupt). "
        "Do NOT publish.\n"
        "Affected LFs:\n" + "\n".join(integrity_failures)
    )


def assert_content_safety_hard_fail(content_failures: list[str]) -> None:
    """Hard-fail Phase D when any LF's title/description/captions violate the
    curated ``content_safety`` blocklist.

    The ``content_safety`` content gate is WARN-only on the drive path
    (``phase_d_gates`` demotes every non-escalated gate to ``logger.warning``),
    which left a blocklist-violating title free to ship. The gate's entire
    purpose is to block, so this escalates it to ``RuntimeError`` ahead of any
    Drive/YouTube upload, mirroring ``assert_audio_loudness_hard_fail``.

    Caveat: the blocklist could in theory false-match a legitimate place name.
    """
    if not content_failures:
        return
    raise RuntimeError(
        "Content safety HARD FAIL — at least one LF's title, description, or "
        "captions violate the curated blocklist. Do NOT publish.\n"
        "Affected LFs:\n" + "\n".join(content_failures)
    )


def assert_metadata_quality_hard_fail(metadata_failures: list[str]) -> None:
    """Hard-fail Phase D when any LF fails the ``metadata_quality`` gate —
    title/location keyword mismatch, business-name-in-title, fragmented
    location tags, duplicate chapter blocks, operational data in the
    description.

    Previously WARN-only on the drive path, which left the desc↔title
    location-agreement check toothless: ``lf_location`` (the ``📍 Location:``
    line source) and ``lf_title_override`` are INDEPENDENT inputs in
    ``_build_metadata_per_lf``, and this gate's keyword check is the one
    place they are compared — demoting it to a warning let a title whose
    location diverged from its own description ship. Escalates to
    ``RuntimeError`` ahead of any Drive/YouTube upload, mirroring
    ``assert_audio_loudness_hard_fail``. Metadata regenerates without a
    re-encode, so an abort here costs seconds, not hours.
    """
    if not metadata_failures:
        return
    raise RuntimeError(
        "Metadata quality HARD FAIL — at least one LF's metadata failed the "
        "quality gate (title/location mismatch, business name in title, "
        "fragmented location tags, or malformed description). Fix the "
        "metadata inputs and resume; do NOT publish.\n"
        "Affected LFs:\n" + "\n".join(metadata_failures)
    )


def _reconcile_shorts_per_lf_against_disk(state) -> None:
    """Guard ``state["shorts_per_lf"]`` against the day-first cache on
    disk at phase_d entry — **compare-only, never rewrite**.

    ``state["shorts_per_lf"]`` is the source-window attribution authority
    (``assign_shorts_to_lfs_by_source_window``). On-disk files carry no
    LF attribution, so an equal-split guess over the filename-sorted
    shorts is almost never the true per-LF membership; rewriting the dict
    to it corrupts every short's LF location label, parent link, and slot.
    This helper NEVER overwrites the dict, and stays **additive**:
    state-only (ghost) entries survive.

    The genuine split-brain it still catches
    (a stale ``shorts_per_lf`` vs the day-first cache): shorts on disk the dict
    doesn't account for (empty/stale legacy state, reset, render/serialise
    crash). Equal-splitting those burns the wrong location into each
    short, so it **hard-fails** (``RuntimeError``) upstream of any upload.

    Disk layout: ``<stages_base>/day_<YYYY-MM-DD>/shorts/short_NN.mp4``.
    """
    stages_base = state.stages_base
    if not stages_base:
        return
    stages_dir = Path(stages_base)
    if not stages_dir.is_dir():
        return

    # Scope to days THIS run assembles day-first (the Deliverables' dates). A
    # run whose LFs live elsewhere owns no day-first shorts here.
    dayfirst_dates: set[str] = {d.date for d in deliverables_for_publish(state) if d.date}

    if not dayfirst_dates:
        return

    # Flat set of on-disk day-first shorts for those days.
    disk_paths: set[str] = set()
    for date in dayfirst_dates:
        shorts_dir = stages_dir / f"day_{date}" / "shorts"
        if not shorts_dir.is_dir():
            continue
        for p in shorts_dir.glob("short_[0-9][0-9].mp4"):
            if p.is_file():
                disk_paths.add(str(p))

    if not disk_paths:
        return  # No on-disk shorts — nothing to reconcile; leave state.

    state_shorts = state.shorts_per_lf or {}
    state_paths = {str(p) for paths in state_shorts.values() if paths for p in paths}

    missing = disk_paths - state_paths
    if not missing:
        # Every on-disk short is already attributed — leave the dict as-is.
        return

    # Genuine split-brain: disk has shorts the dict never attributed to
    # any LF, and their true source-window LF is unrecoverable here. An
    # equal-split guess would ship mis-located shorts — abort upstream.
    state_total = sum(len(v) for v in state_shorts.values() if v)
    raise RuntimeError(
        "phase_d: shorts_per_lf SPLIT-BRAIN — state dict attributes "
        f"{state_total} short(s) but {len(disk_paths)} exist on disk; "
        f"{len(missing)} short(s) are unattributed to any LF:\n    "
        + "\n    ".join(sorted(missing))
        + "\nRefusing to guess an equal-split (would burn the wrong "
        "location into each short). Rebuild shorts_per_lf from the "
        "day-first plan or re-run phase_bc."
    )


# Publish day abbreviation -> full name
PUBLISH_DAYS = {
    "mon": "Monday",
    "tue": "Tuesday",
    "wed": "Wednesday",
    "thu": "Thursday",
    "fri": "Friday",
    "sat": "Saturday",
    "sun": "Sunday",
}


# _resolve_drive_subfolder_id + _resolve_youtube_uploader_class moved to
# external_systems_coordinator. See the re-export at
# module top alongside _upload_lf_to_youtube.


def _build_included_segments_by_episode_for_lf(
    lf_plan,
) -> dict[int, list[tuple[float, float]]]:
    """Like :func:`_build_included_segments_for_lf`, but keyed per source
    EPISODE and NOT collapsed to the representative — segments from every
    episode are kept (distance/elevation must sum over all of them).

    Groups ``lf_plan.segments`` by ``int(source_file_index)``; ranges use
    episode-local ``start_time``/``end_time`` (same rebasing caveat the
    single-rep helper documents). MUST be called while ``source_file_index`` is
    still GLOBAL (before ``_partition_lf_plans_by_date``).
    """
    by_episode: dict[int, list[tuple[float, float]]] = {}
    for seg in list(getattr(lf_plan, "segments", []) or []):
        idx = int(getattr(seg, "source_file_index", 0))
        start = float(getattr(seg, "start_time", 0.0))
        end = float(getattr(seg, "end_time", start))
        if end > start:
            by_episode.setdefault(idx, []).append((start, end))
    return by_episode


# Validation gates live in phase_d_gates; scheduling helpers in phase_d_scheduling.
# Both re-exported so existing call sites + test patch targets keep resolving.
from .phase_d_gates import (  # noqa: E402,F401
    _assert_shorts_quality_gate,
    _lf_day_records,
    _lf_picture_inputs,
    _run_final_validation_gate,
    _run_sidecar_integrity_gate,
)

# _upload_lfs_to_youtube lives in external_systems_coordinator. See the
# re-export at the top of this module.
# _upload_lf_thumbnails_to_drive lives in external_systems_coordinator. See the
# re-export below (alongside _upload_all_lfs_shorts_to_drive).
#
# _build_metadata_per_lf lives in phase_d_metadata. The re-export
# preserves the public import path and test monkey-patch surface.
# _generate_chapters_all_lfs lives in phase_d_metadata. The
# function late-binds back to this module (``from . import phase_d as
# _pd``) so monkey-patches of ``phase_d._generate_chapters_for_lf`` and
# ``phase_d._LFMeta._embed_chapters`` still flow.
# _generate_lf_thumbnails lives in phase_d_metadata. It references no
# phase_d-resident symbols, so no late-binding shim is needed.
from .phase_d_metadata import (  # noqa: E402,F401
    _attach_lf_thumbnails,
    _attach_superlong_thumbnails,
    _build_metadata_per_lf,
    _generate_chapters_all_lfs,
    _generate_lf_thumbnails,
    _generate_superlong_thumbnails,
)
from .phase_d_scheduling import (  # noqa: E402,F401
    _build_and_persist_drive_schedule,
    _build_social_posts_per_lf,
    _maybe_rebase_drive_schedule,
    _reresolve_passed_lf_slots,
    assert_all_places_verified,
)


class DrivePlanReconstructionError(RuntimeError):
    """Raised when a JSON-resumed drive_plan dict cannot be parsed.

    A resumed drive_plan that fails to reconstruct is NOT the same as a
    genuinely empty plan: the plan on disk exists but this run
    couldn't read it, so the caller must fail loudly rather than treat it
    as "nothing to publish" — treating it as empty wipes
    ``drive_schedule``/``upload_manifests``/etc, destroying the run-wide
    scheduling lock and letting a later run reschedule already-published
    LFs onto different slots.
    """


def _reconstruct_drive_plan(drive_plan_input):
    """Coerce a JSON-resumed drive_plan dict back into a ``DrivePlan``.

    Pass-through for non-dict inputs (None, an already-typed ``DrivePlan``).
    Raises ``DrivePlanReconstructionError`` if the dict shape is
    unparseable — a failed READ must be distinguishable from a
    legitimately empty plan (bare ``None``/``regular_lfs=[]``), which the
    caller still treats as "nothing to publish this run".
    """
    if not isinstance(drive_plan_input, dict):
        return drive_plan_input
    from ...models import DrivePlan, RegularLFPlan

    try:
        return DrivePlan(
            source_files=[Path(p) for p in drive_plan_input.get("source_files", [])],
            processed_files=[Path(p) for p in drive_plan_input.get("processed_files", [])],
            regular_lfs=[
                RegularLFPlan.from_dict(lf) for lf in drive_plan_input.get("regular_lfs", [])
            ],
            score_threshold=drive_plan_input.get("score_threshold", 0.0),
            total_source_duration=drive_plan_input.get("total_source_duration", 0.0),
            total_highlight_duration=drive_plan_input.get("total_highlight_duration", 0.0),
            best_moments_filtered=drive_plan_input.get("best_moments_filtered", False),
        )
    except Exception as e:
        logger.error("Failed to reconstruct DrivePlan from dict: %s", e)
        raise DrivePlanReconstructionError(
            f"Failed to reconstruct DrivePlan from resumed state: {e}"
        ) from e


def _emit_consolidated_shorts_csv(
    *,
    output_base: Path,
    project_base: str,
    drive_plan,
    social_posts_per_lf: dict,
    upload_manifests: dict,
    drive_schedule,
    youtube_uploads: dict,
    per_lf_trim_ratio: dict,
    per_lf_recording_date: dict,
    skip_drive_upload: bool = False,
    config: Optional[Config] = None,
    region: Optional[str] = None,
) -> Optional[Path]:
    """Write the ONE consolidated Metricool shorts CSV and return its path.

    All per-short maps are built here from the inputs:
      * ``drive_links_by_short_index`` — position-based zip of social
        posts against manifest files (a parse-from-filename
        merge would collide across LFs, since day-first manifest
        filenames are day-LOCAL).
      * ``parent_ids`` — short_index → parent LF YouTube video id.
      * ``parent_trims`` — short_index → parent LF trim_ratio
        so the first-comment lead picks the right band.
      * ``recording_dates_by_short_idx`` — short_index → ISO date,
        backed by Phase BC's ``_assert_per_lf_recording_date_complete``.
    """
    from ..csv_generator import build_consolidated_shorts_csv, platforms_for_publishing_routes

    drive_links: dict[int, str] = _build_drive_links_by_short_index(
        social_posts_per_lf=social_posts_per_lf,
        upload_manifests=upload_manifests,
    )

    # Media gate: the Metricool CSV is only importable when each
    # short has a Drive media URL. When the shorts were not Drive-uploaded
    # (``--skip-upload`` audit pass) there are no links, so writing the canonical
    # CSV would ship a dead deliverable (every ``Picture Url 1`` blank — what
    # would make the shorts silently vanish from Metricool). DEFER instead:
    # skip the write and log a greppable line. The CSV is produced on the real
    # Drive-upload run. A NON-skip run with no links is a genuine upload bug —
    # there we fall through and let ``build_consolidated_shorts_csv`` raise
    # ``METRICOOL-CSV-NO-MEDIA`` rather than masking it as "deferred".
    has_media = any(v for v in drive_links.values())
    if not has_media and skip_drive_upload:
        logger.warning(
            "  D-csv: METRICOOL-CSV-DEFERRED — shorts not Drive-uploaded "
            "(skip_drive_upload); no importable Metricool CSV written. Run the "
            "Drive-upload pass to produce it."
        )
        return None

    slots_by_idx: dict[int, object] = {
        s.short_index: s for s in (drive_schedule.shorts_slots if drive_schedule else [])
    }

    parent_ids: dict[int, str] = {}
    parent_trims: dict[int, float] = {}
    parent_publish_at: dict[int, str] = {}
    shorts_by_lf: dict[int, list] = {}
    for lf in drive_plan.regular_lfs:
        posts = social_posts_per_lf.get(lf.index, [])
        shorts_by_lf[lf.index] = posts
        _rec = youtube_uploads.get(lf.index) or {}
        vid = _rec.get("video_id", "")
        _pub_at = _rec.get("publish_at")
        _ratio = per_lf_trim_ratio.get(lf.index)
        for p in posts:
            parent_ids[p.short_index] = vid
            if _pub_at:
                parent_publish_at[p.short_index] = _pub_at
            if isinstance(_ratio, (int, float)):
                parent_trims[p.short_index] = float(_ratio)

    recording_dates_by_short_idx = _build_recording_dates_by_short_index(
        shorts_by_lf=shorts_by_lf,
        per_lf_recording_date=per_lf_recording_date,
    )

    csv_path = output_base / f"{project_base}_shorts_metricool.csv"
    # Wire the config knob through so it is live, not decorative: the shorts
    # CSV publishes to TikTok unless config.drive.publish_tiktok is set False.
    publish_tiktok = config.drive.publish_tiktok if config is not None else True
    # Suppress the CSV toggle for any platform on the "first_party" route --
    # Phase D's publish-plan sidecar already posts that platform directly
    # (see _emit_publish_plan_sidecar), so a live Metricool toggle here would
    # double-post it.
    platforms = platforms_for_publishing_routes(config.drive if config is not None else None)
    build_consolidated_shorts_csv(
        output_path=csv_path,
        shorts_by_lf=shorts_by_lf,
        slots_by_short_index=slots_by_idx,
        drive_links_by_short_index=drive_links,
        parent_lf_video_id_by_short_index=parent_ids,
        parent_recording_date_by_short_index=recording_dates_by_short_idx,
        parent_trim_ratio_by_short_index=parent_trims,
        parent_publish_at_by_short_index=parent_publish_at,
        platforms=platforms,
        publish_tiktok=publish_tiktok,
        require_media=not skip_drive_upload,
    )
    # Transactional booking: shorts entries book ONLY here, after the
    # media gate passed (real Drive link per short, or the builder raised).
    from ..content_ledger import ContentEntry, record_confirmed_upload

    entries = [
        ContentEntry(
            date=slot.date,
            time=slot.time,
            content_type="shorts",
            project_name=project_base,
            source="pipeline",
            region=region,
        )
        for idx, slot in slots_by_idx.items()
        if drive_links.get(idx)
    ]
    if entries:
        ledger_path = Path(config.seo.storage_path) / ".content_ledger.json" if config else None
        if not record_confirmed_upload(entries, ledger_path):
            # A live upload is never silently unbooked (see the LF sibling
            # in external_systems_coordinator._book_lf_upload_in_ledger).
            logger.error(
                "  LEDGER-UNBOOKED: %d confirmed shorts slot(s) for %s were "
                "NOT booked into the ledger — risk of a later double-book.",
                len(entries),
                project_base,
            )
    return csv_path


def _build_local_media_paths_by_short_index(
    *,
    social_posts_per_lf: dict,
    shorts_paths_by_lf: Optional[dict],
) -> "dict[int, str]":
    """Position-zip social posts against LOCAL rendered short paths
    (``state.shorts_per_lf``) -- mirrors ``_build_drive_links_by_short_index``
    but for on-disk media, so first-party publishing needs no Drive manifest.
    """
    local_paths: dict[int, str] = {}
    if not shorts_paths_by_lf:
        return local_paths
    for lf_idx, paths in shorts_paths_by_lf.items():
        paths = list(paths or [])
        if not paths:
            continue
        posts = social_posts_per_lf.get(lf_idx) or []
        if not posts or len(posts) != len(paths):
            logger.error(
                "  D-plan-merge: LF %s length mismatch (%d posts vs %d local "
                "short paths) -- skipping LF (mirrors D-csv-merge)",
                lf_idx,
                len(posts),
                len(paths),
            )
            continue
        for post, path in zip(posts, paths):
            short_index = getattr(post, "short_index", None)
            if short_index is None:
                continue
            local_paths[short_index] = str(path)
    return local_paths


def _persist_shorts_to_output(
    *,
    output_base: Path,
    shorts_paths_by_lf: Optional[dict],
) -> "dict[int, list[str]]":
    """Copy each rendered short into the run's OUTPUT dir and return the same
    ``{lf_index: [path, ...]}`` shape pointing at the copies.

    Shorts render to ``<stages_base>/day_<date>/shorts/short_NN.mp4``, and
    ``stages_base`` defaults under the cache home (``paths.py``). A cache
    cleaner may delete any per-drive stage cache that has sat idle for some
    days, while a shorts ladder can schedule out for ~66 days. A publish plan
    recording cache paths would therefore lose its media before its last
    delivery fires.

    Copying is the robust-by-construction fix: the plan cannot reference
    ephemeral storage because the media sits beside the plan and its sidecar
    on the drive the operator keeps. Rendering still happens on the fast
    local cache; only the finished shorts (~37 MB each) are copied, never the
    multi-GB day masters.

    Unconditional, NOT gated on a ``first_party`` route: a conditional
    artifact would make the run's output differ by config, and the Gate C/D
    hash covers that output. Never raises -- a missing rendered short is
    skipped, because the CSV path does not need these copies and a shorts
    shortfall already hard-fails at its own upstream gate.
    """
    persisted: dict[int, list[str]] = {}
    if not shorts_paths_by_lf:
        return persisted

    for lf_index, paths in shorts_paths_by_lf.items():
        kept: list[str] = []
        for source in list(paths or []):
            src = Path(source)
            if not src.is_file():
                logger.warning("  D-media: rendered short missing, not copied: %s", src)
                continue
            dest_dir = output_base / "shorts" / f"lf_{int(lf_index):02d}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            try:
                # Size match is the resume check: a re-run must not recopy
                # gigabytes it already wrote, and a truncated earlier copy
                # must not be trusted.
                if not dest.exists() or dest.stat().st_size != src.stat().st_size:
                    shutil.copyfile(src, dest)
                    try:
                        shutil.copystat(src, dest)
                    except OSError:
                        # drvfs (a /mnt/<letter> mount) refuses the mode/utime
                        # update with EPERM after the bytes landed fine; the
                        # copy itself succeeded, so carry on.
                        pass
            except OSError as exc:
                # Keep the ladder complete on the rendered file rather than
                # dropping the short: a plan with a hole fails NO-MEDIA.
                logger.warning(
                    "  D-media: could not copy %s -> %s: %s -- recording the stage-cache path",
                    src,
                    dest,
                    exc,
                )
                kept.append(str(src))
                continue
            kept.append(str(dest))
        persisted[int(lf_index)] = kept
    return persisted


def _json_datetime_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _serialize_publish_plan(plan) -> str:
    """Deterministic JSON: sorted keys + explicit ``datetime`` encoder (never
    ``default=str``) so ``review_gate.compute_gate_cd_hash`` (which hashes
    this file's content) is byte-stable across runs with identical inputs.
    Drops each record's wall-clock ``generated_at`` -- the same exclusion
    ``compute_gate_cd_hash`` already applies to its other inputs.
    """
    payload = asdict(plan)
    for record in payload.get("shorts", []):
        record.pop("generated_at", None)
    return json.dumps(payload, sort_keys=True, indent=2, default=_json_datetime_default) + "\n"


#: Synthetic parent identity used ONLY by the preview; never a real YouTube id.
_PREVIEW_PARENT_ID_PREFIX = "PREVIEW-NO-LF-ID-"
#: Last-resort parent publish time when neither the upload record nor the drive
#: schedule has one (epoch: earlier than every slot, so no ordering gate trips).
_PREVIEW_PARENT_PUBLISH_AT = "1970-01-01T00:00:00+00:00"


_PREVIEW_MEDIA_SHA256 = "PREVIEW-NOT-HASHED"


def _build_first_party_plan(
    *,
    project_base,
    route_selected,
    drive_cfg,
    drive_plan,
    social_posts_per_lf,
    drive_schedule,
    youtube_uploads,
    per_lf_recording_date,
    shorts_paths_by_lf,
    synthetic_parents: bool,
):
    """Assemble ``build_publish_plan`` inputs and build the plan in memory.

    Shared by the real sidecar and the preview so both run the SAME gates over
    the SAME inputs. ``synthetic_parents`` (preview only) fills a missing LF
    video id / publish time with a labelled placeholder instead of letting the
    builder raise PUBLISH-PLAN-NO-PARENT-ID. Returns ``(plan,
    recording_dates_by_short_idx, placeholders_by_short_idx)``; the last is
    empty for the real path. Writes nothing.
    """
    from ...publishing.media_fingerprint import fingerprint_media
    from ...publishing.models import MediaFingerprint
    from ...publishing.plan_builder import build_publish_plan

    slots_by_idx: dict[int, object] = {
        s.short_index: s for s in (drive_schedule.shorts_slots if drive_schedule else [])
    }
    lf_slots = list(getattr(drive_schedule, "lf_slots", None) or [])
    parent_ids: dict[int, str] = {}
    parent_publish_at: dict[int, str] = {}
    placeholders: dict[int, list] = {}
    shorts_by_lf: dict[int, list] = {}
    for lf_pos, lf in enumerate(drive_plan.regular_lfs):
        posts = social_posts_per_lf.get(lf.index, [])
        shorts_by_lf[lf.index] = posts
        _rec = youtube_uploads.get(lf.index) or {}
        vid = _rec.get("video_id", "")
        _pub_at = _rec.get("publish_at")
        synthetic_id = synthetic_parents and not vid
        synthetic_at = synthetic_parents and not _pub_at
        if synthetic_id:
            vid = f"{_PREVIEW_PARENT_ID_PREFIX}{lf.index}"
        if synthetic_at:
            slot_at = (
                getattr(lf_slots[lf_pos], "datetime_est", "") if lf_pos < len(lf_slots) else ""
            )
            _pub_at = slot_at or _PREVIEW_PARENT_PUBLISH_AT
        for p in posts:
            parent_ids[p.short_index] = vid
            if _pub_at:
                parent_publish_at[p.short_index] = _pub_at
            if synthetic_parents:
                marks = []
                if synthetic_id:
                    marks.append("parent_lf_video_id")
                if synthetic_at:
                    marks.append("parent_lf_publish_at")
                if synthetic_id:
                    marks.append("payloads.youtube.description")
                placeholders[p.short_index] = marks

    recording_dates_by_short_idx = _build_recording_dates_by_short_index(
        shorts_by_lf=shorts_by_lf,
        per_lf_recording_date=per_lf_recording_date,
    )

    local_paths = _build_local_media_paths_by_short_index(
        social_posts_per_lf=social_posts_per_lf,
        shorts_paths_by_lf=shorts_paths_by_lf,
    )
    if synthetic_parents:
        # Preview: no ffprobe/hashing on an audit pass (the deferral contract);
        # the media identity is a labelled placeholder, flagged per short.
        media_by_short_index = {
            short_idx: MediaFingerprint(
                path=str(path),
                size_bytes=0,
                duration_seconds=0.0,
                width=0,
                height=0,
                codec="",
                sha256=_PREVIEW_MEDIA_SHA256,
            )
            for short_idx, path in local_paths.items()
        }
        for marks in placeholders.values():
            marks.append("media")
    else:
        media_by_short_index = {
            short_idx: fingerprint_media(Path(path)) for short_idx, path in local_paths.items()
        }

    plan = build_publish_plan(
        project_name=project_base,
        shorts_by_lf=shorts_by_lf,
        slots_by_short_index=slots_by_idx,
        media_by_short_index=media_by_short_index,
        parent_lf_video_id_by_short_index=parent_ids,
        parent_recording_date_by_short_index=recording_dates_by_short_idx,
        parent_publish_at_by_short_index=parent_publish_at,
        route_selected=route_selected,
        tiktok_privacy_level=getattr(drive_cfg, "publishing_tiktok_privacy_level", "SELF_ONLY"),
    )
    return plan, recording_dates_by_short_idx, placeholders


def _emit_publish_preview_sidecar(
    *,
    output_base: Path,
    project_base: str,
    route_selected,
    drive_cfg,
    drive_plan,
    social_posts_per_lf: dict,
    drive_schedule,
    youtube_uploads: dict,
    per_lf_recording_date: dict,
    shorts_paths_by_lf: Optional[dict],
) -> None:
    """Write ``<project>_publish_preview.json`` so a run that cannot write the
    real plan (audit pass, ``--skip-upload``) still shows the final Short
    payloads for review.

    NOT a plan: a separate file name that no consumer globs
    (``*_publish_plan.json``), a top-level ``"kind": "publish_preview"`` /
    ``"not_for_publish": true`` marker that ``load_publish_plan`` refuses, and a
    per-short ``placeholders`` list naming each synthetic field. Returns
    ``None`` on purpose -- there is no path for a caller to pass on as the plan.
    Built through the same ``build_publish_plan``; any error is logged as a
    loud warning and nothing is written, so it can never fail the audit run.
    """
    try:
        plan, _dates, placeholders = _build_first_party_plan(
            project_base=project_base,
            route_selected=route_selected,
            drive_cfg=drive_cfg,
            drive_plan=drive_plan,
            social_posts_per_lf=social_posts_per_lf,
            drive_schedule=drive_schedule,
            youtube_uploads=youtube_uploads,
            per_lf_recording_date=per_lf_recording_date,
            shorts_paths_by_lf=shorts_paths_by_lf,
            synthetic_parents=True,
        )
        payload = asdict(plan)
        for record in payload.get("shorts", []):
            record.pop("generated_at", None)
            record["placeholders"] = placeholders.get(record["short_index"], [])
        payload = {"kind": "publish_preview", "not_for_publish": True, **payload}
        preview_path = output_base / f"{project_base}_publish_preview.json"
        preview_path.write_text(
            json.dumps(payload, sort_keys=True, indent=2, default=_json_datetime_default) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001 -- preview must never fail the run
        logger.warning(
            "  D-publish-preview: PREVIEW NOT WRITTEN -- %s: %s (no preview of the Shorts payload)",
            type(exc).__name__,
            exc,
        )
        return
    logger.warning(
        "  D-publish-preview: wrote %s -- PREVIEW ONLY, not a publish plan "
        "(placeholder parent fields are listed per short; never enqueue it)",
        preview_path.name,
    )


def _emit_publish_plan_sidecar(
    *,
    output_base: Path,
    project_base: str,
    drive_plan,
    social_posts_per_lf: dict,
    drive_schedule,
    youtube_uploads: dict,
    per_lf_recording_date: dict,
    shorts_paths_by_lf: Optional[dict],
    publish_pass: bool,
    config: Optional[Config] = None,
    region: Optional[str] = None,
    stages_base: Optional[Path] = None,
) -> Optional[Path]:
    """Write the first-party publish-plan JSON sidecar. This function is
    ADDITIVE ONLY -- it never calls or alters ``_emit_consolidated_shorts_csv``.

    Note the ROUTE still reaches the CSV by a separate path: a platform on
    ``first_party`` has its Metricool toggle forced FALSE by
    ``csv_generator.platforms_for_publishing_routes``, so the platform this
    sidecar publishes is not also scheduled through Metricool. Flipping a
    route therefore DOES change CSV content, just not via this function.

    Entered iff a ``DriveConfig.publishing_route_*`` is ``"first_party"``,
    checked FIRST so an all-metricool run pays no fingerprinting/ffprobe/IO.
    Deliberately skips the CSV's Drive-media gate: a first-party
    route uploads the LOCAL rendered file, so this must still write on a
    ``skip_drive_upload``/no-Drive-links run where the CSV itself defers.
    Propagates any fingerprint/build error rather than writing a partial
    sidecar -- the JSON is fully built in memory before the one write.

    ``region`` lets the plan carry a per-platform publish hour where the
    data path moved a platform off the slot; ``None`` writes the plain
    shape (every platform on the slot).

    ``stages_base`` is where the encoder wrote each day's privacy attestation;
    the plan's directory gets the set of its days' attestations (see
    ``_sync_privacy_attestations``), because dispatch reads them from there.

    ``publish_pass`` is the run's publish intent, ``bool(state.youtube_upload)``
    -- the predicate the verified-place and hand-title gates enforce on. Only a
    publish pass writes. An audit pass merely WARNS on an unverified place, yet
    an audit re-run over an uploaded drive still carries every LF's video id, so
    writing here would hand ``enqueue-plan --apply`` shorts no gate refused. The
    audit pass leaves the publish pass's plan on disk as it is. Keyword-only and
    required, so no caller can leave the pass unstated.
    """
    drive_cfg = getattr(config, "drive", None) if config is not None else None
    route_selected = tuple(
        platform
        for platform, field in (
            ("youtube", "publishing_route_youtube"),
            ("instagram", "publishing_route_instagram"),
            ("facebook", "publishing_route_facebook"),
            ("tiktok", "publishing_route_tiktok"),
        )
        if drive_cfg is not None and getattr(drive_cfg, field, "off") == "first_party"
    )
    if not route_selected:
        return None
    if not publish_pass:
        logger.warning(
            "  D-publish-plan: DEFERRED -- audit pass (youtube_upload off); only a "
            "publish pass writes the plan, where the place and title gates enforce"
        )
        _emit_publish_preview_sidecar(
            output_base=output_base,
            project_base=project_base,
            route_selected=route_selected,
            drive_cfg=drive_cfg,
            drive_plan=drive_plan,
            social_posts_per_lf=social_posts_per_lf,
            drive_schedule=drive_schedule,
            youtube_uploads=youtube_uploads,
            per_lf_recording_date=per_lf_recording_date,
            shorts_paths_by_lf=shorts_paths_by_lf,
        )
        return None
    # A publish pass that uploaded nothing (--skip-youtube-upload) records no
    # LF video id, and every short's parent link needs one (plan_builder raises
    # PUBLISH-PLAN-NO-PARENT-ID), which would fail the whole dry pass on this
    # last step. Defer loudly; the upload run writes the sidecar.
    if not any((rec or {}).get("video_id") for rec in (youtube_uploads or {}).values()):
        logger.warning(
            "  D-publish-plan: DEFERRED -- no YouTube upload recorded for any LF "
            "(skip-upload run?); the sidecar is written on the upload run"
        )
        _emit_publish_preview_sidecar(
            output_base=output_base,
            project_base=project_base,
            route_selected=route_selected,
            drive_cfg=drive_cfg,
            drive_plan=drive_plan,
            social_posts_per_lf=social_posts_per_lf,
            drive_schedule=drive_schedule,
            youtube_uploads=youtube_uploads,
            per_lf_recording_date=per_lf_recording_date,
            shorts_paths_by_lf=shorts_paths_by_lf,
        )
        return None

    plan, recording_dates_by_short_idx, _ = _build_first_party_plan(
        project_base=project_base,
        route_selected=route_selected,
        drive_cfg=drive_cfg,
        drive_plan=drive_plan,
        social_posts_per_lf=social_posts_per_lf,
        drive_schedule=drive_schedule,
        youtube_uploads=youtube_uploads,
        per_lf_recording_date=per_lf_recording_date,
        shorts_paths_by_lf=shorts_paths_by_lf,
        synthetic_parents=False,
    )

    plan_path = output_base / f"{project_base}_publish_plan.json"
    plan_path.write_text(_serialize_publish_plan(plan), encoding="utf-8")
    _sync_privacy_attestations(
        output_base=output_base,
        stages_base=stages_base,
        dates=sorted({d for d in recording_dates_by_short_idx.values() if d}),
    )
    return plan_path


#: A DAY's attestation name -- the only kind Phase D may delete (enqueue-asset's
#: ``day_asset_privacy_attestation.json`` matches the dispatch glob too).
_DAY_ATTESTATION_RE = re.compile(r"^day_\d{4}-\d{2}-\d{2}_privacy_attestation\.json$")


def _sync_privacy_attestations(
    *, output_base: Path, stages_base: Optional[Path], dates: list
) -> None:
    """Place each plan day's privacy attestation beside the publish plan.

    The encoder writes ``day_<d>_privacy_attestation.json`` into the stage dir
    (a cache cleaner may remove it once idle); dispatch globs the PLAN's
    directory. A day with no stage attestation gets an explicit ``inconclusive``
    record -- the loader merges what the glob finds, so a silently absent day
    would let its attested siblings merge to a pass. Attestations for any other
    day are deleted: a stale ``verified`` must never stand in; a non-day one
    (enqueue-asset's bundle) is left alone -- deleting it would let the asset's
    rows merge to the drive's verified days. Content-only
    copy (drvfs refuses ``copystat``/``utime``). Never raises for a missing
    attestation; the dispatch hold is the gate.
    """
    from ..day_master_encoder import (
        PRIVACY_ATTESTATION_NAME,
        PRIVACY_ATTESTATION_SCHEMA,
        PRIVACY_INCONCLUSIVE,
        REQUIRED_PRIVACY_CLASSES,
    )

    keep = set()
    for date in dates:
        name = PRIVACY_ATTESTATION_NAME.format(date=date)
        keep.add(name)
        dest = output_base / name
        src = Path(stages_base) / f"day_{date}" / name if stages_base else None
        if src is not None and src.is_file():
            dest.write_bytes(src.read_bytes())
            continue
        where = src if src is not None else f"<no stages_base>/day_{date}/{name}"
        logger.warning(
            "  D-publish-plan: no privacy attestation at %s; writing an inconclusive "
            "record beside the plan (deliveries for this run will HOLD)",
            where,
        )
        reason = f"no privacy attestation was written for this day's master at {where}"
        record = {
            "schema": PRIVACY_ATTESTATION_SCHEMA,
            "date": date,
            "required_classes": list(REQUIRED_PRIVACY_CLASSES),
            "classes": {
                c: {"state": PRIVACY_INCONCLUSIVE, "reason": reason}
                for c in REQUIRED_PRIVACY_CLASSES
            },
        }
        dest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for stale in output_base.glob("day_*_privacy_attestation.json"):
        if stale.name not in keep and _DAY_ATTESTATION_RE.match(stale.name):
            stale.unlink()


# _resolve_shorts_paths + _upload_lf_shorts_to_drive live in
# external_systems_coordinator. See the re-export at
# module top alongside _upload_all_lfs_shorts_to_drive.


# _upload_lf_to_youtube lives in external_systems_coordinator. See the
# re-export at module top.


def _generate_chapters_for_lf(
    config: Config,
    gpx_path: Path,
    video_duration: float,
    location: str,
    stages_dir: Optional[Path] = None,
    included_segments: Optional[list[tuple[float, float]]] = None,
    resolved_location=None,
    day_journey=None,
) -> tuple[list[dict], str]:
    """Generate YouTube chapters for a single LF from its GPS data.

    Args:
        included_segments: Optional source-timeline ``(start, end)``
            ranges (see :func:`_build_included_segments_for_lf`). When
            set, the chapter generator trims the GPX to these ranges
            before segmentation so chapters describe only the footage
            that survived editing.

    Returns:
        Tuple of (chapters_list, formatted_chapter_text).

    Raises:
        Any exception from ChapterGenerator (caller must handle).
    """
    chapter_gen = ChapterGenerator(config)
    chapters = chapter_gen.generate(
        gpx_path=gpx_path,
        video_duration=video_duration,
        start_location=location,
        end_location=location,
        stages_dir=stages_dir,
        included_segments=included_segments,
        resolved_location=resolved_location,
        day_journey=day_journey,
    )
    chapter_text = chapter_gen.format_chapters(chapters)
    return chapters, chapter_text


class PhaseD:
    """Phase D: Metadata & Publish."""

    name = "phase_d"

    def __init__(
        self,
        config: Config,
        save_callback: Optional[callable] = None,
    ) -> None:
        self.config = config
        self._save_callback = save_callback

    def _save_state(self, state) -> None:
        """Persist state via the orchestrator-provided callback, if any."""
        if self._save_callback is not None:
            self._save_callback(state)

    def run(self, state, drive_plan=None, pre_upload_gate=None) -> None:
        logger.info("Phase D: Metadata & Publish")
        if drive_plan is None:
            drive_plan = state.drive_plan
        # Resume-path reconstruction (JSON-deserialised DrivePlan).
        drive_plan = _reconstruct_drive_plan(drive_plan)

        regular_lf_paths = publish_lf_paths(state)
        # Reconcile shorts_per_lf against the
        # day-first cache on disk BEFORE any consumer reads it. Catches
        # split-brain where disk has files but state dict is empty/stale
        # (legacy state file, manual reset, crash between rendering and
        # state serialisation).
        _reconcile_shorts_per_lf_against_disk(state)
        shorts_per_lf = state.shorts_per_lf or {}
        region = state.region
        publish_day = state.publish_day
        location = state.location
        output_base = Path(state.output_base)
        skip_upload = state.skip_upload
        # skip-upload is split into Drive vs YouTube. Legacy
        # ``skip_upload=True`` (set by --skip-upload alias OR persisted from
        # an older run) implies BOTH. The fine-grained flags let an
        # operator regen Metricool CSVs (Drive uploads needed) without
        # re-publishing already-shipped YouTube videos.
        skip_drive_upload = bool(state.skip_drive_upload or skip_upload)
        skip_youtube_upload = bool(state.skip_youtube_upload or skip_upload)
        skip_csv = state.skip_csv
        force_sidecar_reset = bool(state.force_sidecar_reset)

        # ── Sidecar integrity gate ─────────────────────────────
        _run_sidecar_integrity_gate(
            state=state,
            drive_plan=drive_plan,
            output_base=output_base,
            force_sidecar_reset=force_sidecar_reset,
        )

        if not drive_plan or not drive_plan.regular_lfs:
            logger.warning("Phase D: No qualifying segments — drive_plan is empty, skipping")
            logger.warning(
                "⚠ phase_d: drive_plan empty or no regular_lfs; "
                "emitting no-op deliverables "
                "(0 metadata / 0 schedule / 0 posts / 0 uploads)"
            )
            state.metadata_per_lf = {}
            state.drive_schedule = None
            state.social_posts_per_lf = {}
            state.upload_manifests = {}
            # Preserve state.youtube_uploads — a failed run that hits the
            # empty-plan path must NOT destroy prior upload history, or
            # re-runs happily upload duplicates.
            return

        # titles.yaml is an optional hand-title override on every run (a hand
        # title still wins and is still place-checked); it no longer has to
        # cover every short.
        titles_file = output_base / "social" / "titles.yaml"

        # ── Verified-place gate — same slot, UPSTREAM of chapters + schedule ──
        # A published short may name ONLY the place its own clip's GPS verified:
        # every planned short needs a sidecar label, non-empty clip-GPS regions
        # and place_source == "clip_gps". Publish run: raises naming every
        # offender. Audit pass: one UNVERIFIED-PLACES warning, never raises.
        # Universe = EVERY short in shorts_per_lf (generate_all posts each of
        # them), NOT the short_global_index subset: a pre-upgrade checkpoint has no
        # index, and a short missing from it is still posted.
        assert_all_places_verified(
            {lf.index: list(shorts_per_lf.get(lf.index, [])) for lf in drive_plan.regular_lfs},
            enforce=bool(state.youtube_upload),
        )

        source_dir = Path(state.source_dir)
        project_base = source_dir.stem.replace(" ", "_").lower()

        stages_dir = Path(state.stages_base or (output_base / "_stages"))
        stages_dir.mkdir(parents=True, exist_ok=True)

        publish_day_full = PUBLISH_DAYS.get(publish_day[:3].lower(), "Wednesday")

        plan_count = len(drive_plan.regular_lfs)
        assembled_count = len(regular_lf_paths)
        if plan_count != assembled_count:
            raise RuntimeError(
                f"LF count mismatch: plan has {plan_count} LF(s) but "
                f"{assembled_count} path(s) were assembled in Phase C"
            )

        # Aggregated D-entry boundary contract — symmetric to the
        # BC-exit aggregator. Catches the resume case where state on disk
        # was written by an older revision and bypassed BC's exit gate.
        # Runs AFTER the LF count mismatch check (different failure class)
        # but BEFORE any defensive ``state.get("per_lf_*") or {}`` read.
        state_for_contract = dict(state)
        state_for_contract["drive_plan"] = drive_plan
        assert_phase_d_per_lf_inputs_present(state_for_contract)

        # ── 1. Metadata per LF ─────────────────────────────────────────────
        logger.info("  D1: Generating metadata per LF...")
        meta_gen = LongFormMetadataGenerator()
        metadata_per_lf = _build_metadata_per_lf(
            meta_gen=meta_gen,
            drive_plan=drive_plan,
            regular_lf_paths=regular_lf_paths,
            state=state,
            location=location,
            publish_day_full=publish_day_full,
            project_base=project_base,
            output_base=output_base,
        )
        # per_lf_locations is guaranteed present by the D-entry
        # contract above — read directly rather than defending against absence.
        per_lf_locations_override = state.per_lf_locations

        # ── Title-text quality gate — UPSTREAM of every upload ─────────
        # Title distinctness + seo_degraded are computable from the metadata
        # just built, so enforce them here — before the irreversible YouTube
        # upload — instead of only at end-of-Phase-D. A violation now blocks
        # the bad-title video from ever reaching YouTube. The end-of-Phase-D gate
        # below covers the rules that need upload-/CSV-derived state.
        from ..quality_gates import assert_phase_d_title_quality

        assert_phase_d_title_quality(
            metadata_per_lf,
            allow_seo_degraded=bool(state.allow_seo_degraded),
        )

        # ── D1.5. Generate YouTube chapters from GPS ──────────────────────
        _generate_chapters_all_lfs(
            config=self.config,
            drive_plan=drive_plan,
            state=state,
            metadata_per_lf=metadata_per_lf,
            location=location,
            project_base=project_base,
            output_base=output_base,
            stages_dir=stages_dir,
        )

        # ── Drive schedule (still needed for shorts slot mapping) ──────────
        # Rebase the start_date when a resumed run's schedule has
        # drifted into the past. Three-guard contract (computed_at >12h
        # old, earliest publish_at in past, no YouTube uploads yet) plus
        # two operator backstops (--lf-publish-date, --force-no-rebase).
        # See ``_maybe_rebase_drive_schedule`` docstring.
        rebase_info = _maybe_rebase_drive_schedule(state, output_base)
        drive_schedule = _build_and_persist_drive_schedule(
            scheduler_cls=PipelineScheduler,
            config=self.config,
            drive_plan=drive_plan,
            state=state,
            region=region,
            stages_dir=stages_dir,
            # A rebase deliberately discards the old base date, so the
            # build must NOT read it back off the persisted schedule.
            rebased=rebase_info["action"] == "rebase",
            project=project_base,
        )
        if rebase_info["action"] in {"rebase", "explicit"}:
            new_dates = [s.date for s in drive_schedule.lf_slots]
            logger.info(
                "Phase D rebase: NEW lf_slots dates = %s (action=%s)",
                new_dates,
                rebase_info["action"],
            )

        # ── Social posts per LF ────────────────────────────────────────────
        logger.info("  D3: Generating social posts per LF...")
        # Derive+persist state.interactive_indexes once from the
        # hand-authored titles.yaml (never re-parsed at render time). A
        # legacy checkpoint or a run with no titles.yaml leaves it None —
        # "not yet derived", not "no interactive shorts".
        if titles_file.exists() and state.interactive_indexes is None:
            state.interactive_indexes = sorted(
                derive_interactive_indexes(load_hand_titles(titles_file))
            )
        social_posts_per_lf = _build_social_posts_per_lf(
            social_gen_cls=SocialGenerator,
            drive_plan=drive_plan,
            shorts_per_lf=shorts_per_lf,
            per_lf_locations_override=per_lf_locations_override,
            project_base=project_base,
            output_base=output_base,
            per_lf_resolved=state.per_lf_resolved_location,
            titles_file=titles_file,
            require_verified_places=bool(state.youtube_upload),
            # Consume the stable per-short index assigned at plan
            # time (Phase BC) instead of recomputing it from len(posts).
            # None on a legacy pre-upgrade checkpoint — the builder falls back
            # to the old recompute per LF with a logged warning.
            short_global_index=state.short_global_index,
        )
        # Prefix check on the REAL titles (a titles.yaml override is the only
        # thing that can collide -- composed stamps are unique by construction),
        # before the validation gate / Gate C/D / any enqueue.
        assert_title_prefixes_unique(
            {
                (lf_index, p.short_index): p.youtube_title
                for lf_index, posts in social_posts_per_lf.items()
                for p in posts
                if isinstance(p.youtube_title, str)
            },
            where="phase D",
        )

        # ── Content validation gate (audio loudness + location invariant) ─────
        # Runs UPSTREAM of every publish action so a far-road or off-LUFS LF
        # aborts the run before it can be uploaded. The FINAL.mp4 files exist
        # from Phase BC and metadata_per_lf is built above, so the gate has
        # all it needs here; it reads nothing produced by the upload steps.
        # (The separate end-of-Phase-D quality gate stays below, after
        # state writes, because it validates upload-derived state.)
        _run_final_validation_gate(
            validator_cls=PipelineValidator,
            config=self.config,
            drive_plan=drive_plan,
            regular_lf_paths=regular_lf_paths,
            metadata_per_lf=metadata_per_lf,
            picture_inputs=_lf_picture_inputs(state),
            # The day's upscale record + the brand's claim policy, so an
            # upscaled day cannot pass the 4K gate claiming native pixels.
            day_records=_lf_day_records(state),
            claim_native_resolution_only=_claim_native_only(),
        )

        # ── Gate C/D human review — UPSTREAM of every upload ──────────
        # The orchestrator injects the Gate C/D review here (via
        # ``_phase_d_metadata_and_publish``) so a rejection blocks publication
        # ENTIRELY: the upload blocks and the Metricool CSV write below are
        # unreachable once ``gate_cd`` raises ``ReviewFailedError``.
        # Previously the gate ran AFTER phase_d completed — after the
        # irreversible YouTube upload + CSV write — so a rejection could no
        # longer stop publication. ``pre_upload_gate`` is
        # ``None`` for direct unit-test invocations (no orchestrator), which
        # keeps their behaviour identical to before this move.
        if pre_upload_gate is not None:
            pre_upload_gate(state)

        # ── 1.9. LF thumbnails — UPSTREAM of every upload ──────────────────
        # Generation hard-fails, so it runs before the irreversible YouTube
        # upload: a broken generator costs a re-run, not a public long-form
        # with no thumbnail. Attach needs a video ID; it stays at the attach step below.
        _generate_lf_thumbnails(
            drive_plan=drive_plan,
            regular_lf_paths=regular_lf_paths,
            metadata_per_lf=metadata_per_lf,
            state=state,
            output_base=output_base,
        )
        # Each superlong part gets its OWN thumbnail — same
        # upstream-of-upload placement as the LF thumbnails above.
        _generate_superlong_thumbnails(state=state, output_base=output_base)

        # ── 2. Upload shorts to Drive ──────────────────────────────────────
        # Consume day-first cache paths from
        # ``state["shorts_per_lf"]`` instead of looking under
        # ``output_base/shorts/lf_NN/``. Day-first writes shorts to
        # ``<cache>/_stages/day_<DATE>/shorts/short_NN.mp4`` keyed by DAY,
        # not by LF, so the legacy directory is always empty and every
        # ``upload_shorts`` call would otherwise raise
        # ``FileNotFoundError: No .mp4 files found`` and silently leave
        # the Metricool CSV with empty ``Picture Url 1``.
        #
        # If ``state["shorts_per_lf"]`` is empty while regular LFs exist, the
        # upload helper refuses the run rather than fall back to the legacy
        # directory lookup, which would ship shorts with no quality gating.
        # Hard-failing when the upload yields zero links is owned elsewhere.
        # Preserve prior upload_manifests across resume. Initialise
        # from state so a regen / resume invocation does NOT blank Drive
        # links produced by a previous phase_d run. The
        # skip-predicate is strict: a manifest must be
        # COMPLETE (one file per expected short) before we trust it. A
        # partial manifest — e.g. produced by a run that crashed mid-LF —
        # MUST trigger a re-upload, never a silent skip.
        _assert_shorts_quality_gate(state, validator_cls=PipelineValidator)

        upload_manifests, uploader = _upload_all_lfs_shorts_to_drive(
            drive_plan=drive_plan,
            state=state,
            project_base=project_base,
            output_base=output_base,
            uploader_factory=PipelineUploader,
            skip_drive_upload=skip_drive_upload,
            save_state=self._save_state,
        )

        # ── 3. Upload LFs to YouTube ───────────────────────────────────────
        youtube_uploads: dict[int, dict] = state.youtube_uploads or {}
        youtube_upload_enabled = state.youtube_upload
        # Greppable upload banner: same raw inputs as the
        # _upload_lfs_to_youtube gate below — banner derives, never re-decides.
        log_upload_lfs_banner(
            youtube_upload_enabled=bool(youtube_upload_enabled),
            skip_youtube_upload=bool(skip_youtube_upload),
            no_youtube_upload=bool(getattr(state, "no_youtube_upload", False)),
            publish_set=bool(state.publish_day),
        )

        def _reschedule_passed_lfs(now):
            # Everything below this call (shorts CSV, publish plan, state) must
            # see the rebuilt schedule, so rebind the local.
            nonlocal drive_schedule
            fresh = _reresolve_passed_lf_slots(
                current=drive_schedule,
                scheduler_cls=PipelineScheduler,
                config=self.config,
                drive_plan=drive_plan,
                state=state,
                region=region,
                stages_dir=stages_dir,
                output_base=output_base,
                now=now,
                project=project_base,
            )
            if fresh is not None:
                drive_schedule = fresh
            return fresh

        yt, youtube_uploads = _upload_lfs_to_youtube(
            drive_plan=drive_plan,
            regular_lf_paths=regular_lf_paths,
            metadata_per_lf=metadata_per_lf,
            drive_schedule=drive_schedule,
            youtube_uploads=youtube_uploads,
            state=state,
            output_base=output_base,
            config=self.config,
            save_state=self._save_state,
            skip_youtube_upload=skip_youtube_upload,
            youtube_upload_enabled=youtube_upload_enabled,
            project_base=project_base,
            reschedule=_reschedule_passed_lfs,
        )

        # ── 5. Consolidated shorts CSV ─────────────────────────────────────
        # Promote csv_path to the PhaseD.run scope so the end-of-phase
        # quality gate can inspect it regardless of which branch
        # builds it. ``None`` = skip_csv mode → Rules 5+6 are vacuous.
        consolidated_csv_path: Optional[Path] = None
        if not skip_csv:
            logger.info("  D-csv: Writing consolidated shorts CSV...")
            consolidated_csv_path = _emit_consolidated_shorts_csv(
                output_base=output_base,
                project_base=project_base,
                drive_plan=drive_plan,
                social_posts_per_lf=social_posts_per_lf,
                upload_manifests=upload_manifests,
                drive_schedule=drive_schedule,
                youtube_uploads=youtube_uploads,
                # D-entry contract guarantees both maps cover every
                # regular LF — read directly rather than ``.get() or {}``.
                per_lf_trim_ratio=state.per_lf_trim_ratio,
                per_lf_recording_date=state.per_lf_recording_date,
                skip_drive_upload=skip_drive_upload,
                config=self.config,
                region=region,
            )
        else:
            logger.info("  D-csv: SKIPPED")

        # ── 5a. First-party publish-plan sidecar (additive) ────────────────
        # Durable media BEFORE the plan records any path: shorts render into
        # the stage cache, which a cache cleaner may remove once idle, while
        # the ladder schedules out to ~66. Record the output-dir copies so a
        # delivery cannot outlive its own media. See _persist_shorts_to_output.
        durable_shorts = _persist_shorts_to_output(
            output_base=output_base,
            shorts_paths_by_lf=state.shorts_per_lf,
        )

        _emit_publish_plan_sidecar(
            output_base=output_base,
            project_base=project_base,
            drive_plan=drive_plan,
            social_posts_per_lf=social_posts_per_lf,
            drive_schedule=drive_schedule,
            youtube_uploads=youtube_uploads,
            per_lf_recording_date=state.per_lf_recording_date,
            shorts_paths_by_lf=durable_shorts or state.shorts_per_lf,
            # The gates' own predicate: an audit pass only warns on a place, so
            # it must not leave a plan that enqueue-plan would spool.
            publish_pass=bool(state.youtube_upload),
            config=self.config,
            region=region,
            stages_base=state.stages_base,
        )

        # ── Related-video operator worklist (doc-only sidecar) ────────────
        # YouTube exposes NO API for a Short's "Related video" field (the
        # only clickable Short→LF bridge; Studio desktop only), so — like
        # the end-screen sidecar — one worklist line per short goes into
        # operator_checklist.txt for a Studio batch pass. Never fails the
        # batch by construction (see the sidecar module docstring).
        emit_related_video_worklist(
            output_base=output_base,
            social_posts_per_lf=social_posts_per_lf,
            youtube_uploads=youtube_uploads,
        )

        # ── 5b. Attach the already-generated thumbnails to the uploads ─────
        _attach_lf_thumbnails(
            drive_plan=drive_plan,
            output_base=output_base,
            yt=yt,
            youtube_uploads=youtube_uploads,
            state=state,
        )
        _attach_superlong_thumbnails(state=state, output_base=output_base, yt=yt)

        # ── 5c. Upload LF thumbnail JPGs to Drive ──────────────────────────
        uploader = _upload_lf_thumbnails_to_drive(
            drive_plan=drive_plan,
            state=state,
            output_base=output_base,
            project_base=project_base,
            uploader=uploader,
            uploader_factory=PipelineUploader,
            skip_drive_upload=skip_drive_upload,
        )

        # ── 5d. No per-LF Metricool CSV ────────────────────────────────────
        # The consolidated shorts CSV (emitted above) is the only Metricool
        # deliverable. The LF thumbnail Drive URLs above are still uploaded
        # for any consumer (e.g. a manual Metricool LF import) but are not
        # serialized to a per-LF CSV.

        # ── 6. Content validation gate — runs UPSTREAM of uploads ───────────
        # ``_run_final_validation_gate`` runs before the upload blocks
        # (above), so the audio-loudness + location-invariant HARD FAILs abort
        # before any Drive/YouTube publish. See the call site above.

        # ── State writes ───────────────────────────────────────────────────
        state.metadata_per_lf = metadata_per_lf
        state.drive_schedule = drive_schedule
        state.social_posts_per_lf = social_posts_per_lf
        state.upload_manifests = upload_manifests
        state.youtube_uploads = youtube_uploads

        # ── Post-upload metrics collection (best-effort) ──────────────
        # AFTER uploads + state writes (yt ids available), BEFORE the
        # end-of-phase gate (which may raise) so a late-gate abort never costs the
        # additive metrics pass. Failures are logged + swallowed.
        _collect_metrics_best_effort(
            project_base=project_base,
            output_base=output_base,
            youtube_uploads=youtube_uploads,
        )

        # ── End-of-Phase-D quality gate ────────────────────────────
        # Accumulates every semantic violation in the upload-/CSV-derived state
        # Phase D is about to commit (phantom video IDs, publish_day drift,
        # schedule miscount / wrong-day slots, incomplete Drive/YouTube uploads)
        # and raises ``PhaseQualityError`` once. Runs AFTER state writes, BEFORE
        # PhaseD.run returns. The title-text rules moved UPSTREAM of the upload
        # into ``assert_phase_d_title_quality``, not here.
        from ..quality_gates import assert_phase_d_quality

        assert_phase_d_quality(state, drive_plan, consolidated_csv_path)


def _partition_shorts_slots(
    schedule,
    regular_lfs: list,
) -> dict[int, list]:
    """Partition shorts slots from *schedule* into per-LF buckets.

    Each LF's bucket contains the ``ScheduledSlot`` objects whose
    ``datetime_est`` falls between that LF's slot and the next LF's slot.
    This aligns shorts uploads with their parent LF's weekly window.

    Args:
        schedule: A ``DriveReleaseSchedule`` with ``.lf_slots`` and
            ``.shorts_slots`` attributes.
        regular_lfs: The list of ``RegularLFPlan`` objects (used for
            LF index mapping).

    Returns:
        Dict mapping each LF index to its list of ``ScheduledSlot`` objects.
    """
    lf_slots = schedule.lf_slots
    shorts_slots = schedule.shorts_slots

    if not lf_slots or not shorts_slots:
        return {lf.index: [] for lf in regular_lfs}

    # Compare in UTC, not ISO strings. ISO-string lexicographic compare
    # breaks across DST transitions because two instants with the same
    # wall-clock representation can carry different offsets
    # (``-04:00`` vs ``-05:00``) — string order no longer matches UTC
    # instant order. ``datetime.fromisoformat`` + ``.astimezone(UTC)``
    # normalizes both sides before compare.
    def _to_utc(iso_dt: str) -> datetime:
        return datetime.fromisoformat(iso_dt).astimezone(timezone.utc)

    result: dict[int, list] = {}
    for i, lf in enumerate(regular_lfs):
        lf_dt = _to_utc(lf_slots[i].datetime_est) if i < len(lf_slots) else None
        next_dt = _to_utc(lf_slots[i + 1].datetime_est) if i + 1 < len(lf_slots) else None

        window: list = []
        for slot in shorts_slots:
            slot_dt = _to_utc(slot.datetime_est)
            if lf_dt is not None and slot_dt <= lf_dt:
                continue
            if next_dt is not None and slot_dt >= next_dt:
                continue
            window.append(slot)
        result[lf.index] = window

    return result


def _build_drive_links_by_short_index(
    *,
    social_posts_per_lf: dict[int, list],
    upload_manifests: dict[int, object],
) -> dict[int, str]:
    """Position-zip per-LF social posts against per-LF upload manifests.

    Parsing short_index from manifest filenames would collide under
    day-local naming (``short_01.mp4`` per LF) and last-write-wins. This
    helper instead zips
    ``social_posts_per_lf[lf]`` (each post carries the canonical global
    ``short_index``) against the manifest's ``files`` dict in insertion
    order. Manifests are built in :func:`PhaseD.run` from
    ``zip(path_strs, links)`` using the same ordered list as the per-LF
    shorts state, so the two iterables are position-aligned.

    Length-mismatch handling: when ``len(posts) != len(manifest.files)``
    the entire LF is skipped with an ERROR log line (the CSV audit
    will catch the resulting empty ``Picture Url 1`` cells).

    Args:
        social_posts_per_lf: ``state["social_posts_per_lf"]`` —
            ``{lf_index: [SocialPost, ...]}`` where each post has a
            global ``short_index`` attribute.
        upload_manifests: ``state["upload_manifests"]`` /
            local ``upload_manifests`` dict — ``{lf_index: UploadManifest}``
            where ``UploadManifest.files`` is an ordered dict of
            ``{filename: drive_url}``.

    Returns:
        ``{global_short_index: drive_url}`` ready for
        :func:`build_consolidated_shorts_csv`.
    """
    drive_links: dict[int, str] = {}
    for lf_idx, manifest in upload_manifests.items():
        # State contract: ``upload_manifests`` values are
        # ``UploadManifest`` instances even on the resume path —
        # ``PipelineState.from_dict`` rehydrates them at load time.
        files = manifest.files or {}
        if not files:
            # Empty manifest — no-op for this LF (already-warned upstream).
            continue
        posts = social_posts_per_lf.get(lf_idx)
        if not posts:
            logger.error(
                "  D-csv-merge: LF %s has %d manifest entries but no "
                "social_posts_per_lf entry — skipping LF",
                lf_idx,
                len(files),
            )
            continue
        if len(posts) != len(files):
            logger.error(
                "  D-csv-merge: LF %s length mismatch — %d posts vs "
                "%d manifest files; skipping LF to avoid corrupt "
                "links. The CSV audit will surface empty cells.",
                lf_idx,
                len(posts),
                len(files),
            )
            continue
        for post, link in zip(posts, files.values()):
            short_index = getattr(post, "short_index", None)
            if short_index is None:
                logger.error(
                    "  D-csv-merge: LF %s post is missing short_index "
                    "attribute — skipping single entry",
                    lf_idx,
                )
                continue
            drive_links[short_index] = link
    return drive_links


def _parse_short_index(filename: str) -> Optional[int]:
    """Parse the short index from a manifest filename.

    Looks for a ``short_(\\d+)`` pattern first (e.g. ``lf_02_short_07.mp4`` → 7).
    Falls back to the last numeric token if no ``short_`` prefix is found
    (e.g. ``03_short.mp4`` → 3).
    """
    m = re.search(r"short_(\d+)", filename)
    if m:
        return int(m.group(1))
    # Fallback: last numeric token in the stem.
    nums = re.findall(r"\d+", filename.rsplit(".", 1)[0])
    if nums:
        return int(nums[-1])
    return None
