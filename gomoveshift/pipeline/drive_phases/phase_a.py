"""Phase A: Ingest & Analyze — scan GoPro files, score episodes, derive OST."""

import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from ...analysis.gps_parser import parse_gpx_file
from ...analysis.shorts_models import effective_offset
from ...analysis.unified_scorer import (
    COL_MOTION_ENERGY,
    COL_TIMESTAMP,
    NUM_COLUMNS,
    UnifiedSegmentScorer,
)
from ...config import Config, resolve_lf_mode
from ...video_types import VideoType, classify_video, suggest_action
from .._drive_helpers import _extract_episode_ost_meta, differentiate_duplicate_ost
from ..gopro_gps import (
    assert_gpx_video_correspondence,
    assert_slice_gpx_correspondence,
    extract_location_coords,
)
from ..ingest import (
    _get_route_analyzer,
    _get_shared_geocoder,
    create_ingest_plan,
    derive_drive_ost_text,
    execute_ingest,
)
from ..ost_overrides import apply_overrides, load_overrides
from ..ost_validator import validate_ost
from ..quality import (
    SHARPEN_DETAIL_MAX_DIM,
    SOURCE_SHARPNESS_FLOOR,
    SOURCE_SHARPNESS_SAMPLES,
    sharpness_score,
)
from ..route_profile import build_route_profile

logger = logging.getLogger(__name__)


# Code-rule version for Phase A (ingest + scoring). Bumped whenever the Phase A
# rule changes such that a checkpointed phase_a's cached outputs
# (``drive_scores.npz``, ingest plan) can no longer be trusted on resume. The
# orchestrator's resume path re-runs phase_a when the persisted version differs
# from this constant — mirrors ``PHASE_D_CODE_VERSION``.
PHASE_A_CODE_VERSION = 2


def _score_inputs_key(episode_files: list) -> list:
    """Validity key for the cached ``drive_scores.npz``.

    The expensive Phase A scoring pass is a pure function of the episode
    source files it scores. The key is the ordered list of each episode
    concat path with its size + mtime, so an added / removed / replaced /
    re-ingested episode invalidates the cache and forces a fresh score.
    """
    key = []
    for path in episode_files:
        try:
            st = Path(path).stat()
            key.append([str(path), st.st_size, int(st.st_mtime)])
        except OSError:
            # A missing/unstattable input can't be matched against a cache
            # entry — record a sentinel so it never spuriously equals one.
            key.append([str(path), -1, -1])
    return key


def _synthetic_score_rows(duration: float, sample_interval: float = 1.0) -> np.ndarray:
    """A score chunk spanning ``duration`` at ``sample_interval``, GPU-free.

    Used only for a ``labels_only`` ``full_drive`` Phase A run (see
    ``PhaseA.run``): the label/list printers need real per-second
    ``COL_TIMESTAMP`` values (the assembler slices LF segments and the
    drive-wide stop mask by timestamp), and never read composite quality
    columns, so those stay zero rather than paying for a GPU frame-decode
    pass whose output nobody consumes. ``COL_MOTION_ENERGY`` is pinned to
    1.0 (not 0.0): ``build_stop_mask`` falls back to optical motion when GPS
    speed is unmeasurable, and a zero-valued motion column would mask every
    GPS-less second as a stop, collapsing/losing LFs the real scoring pass
    would keep.
    """
    n = max(1, round(max(duration, 0.0) / sample_interval))
    rows = np.zeros((n, NUM_COLUMNS), dtype=np.float64)
    rows[:, COL_TIMESTAMP] = np.arange(n, dtype=np.float64) * sample_interval
    rows[:, COL_MOTION_ENERGY] = 1.0
    return rows


def _assert_no_scored_gaps(valid_episodes: list, file_scores: list) -> None:
    """Hard-fail Phase A when any episode with real duration scored ZERO rows.

    A per-episode scoring failure returns a ``(0, N)`` array (no frames
    extracted / decode failure). ``merge_file_scores`` concatenates it as
    nothing — the episode's minutes silently vanish from the drive-wide
    timeline, so every downstream planner sees a GAP instead of an error. Raise
    instead of merging that gap (greppable tag: ``PHASE-A``).
    """
    for i, ep in enumerate(valid_episodes):
        scores_i = file_scores[i] if i < len(file_scores) else None
        n_rows = 0 if scores_i is None else int(getattr(scores_i, "shape", (0,))[0])
        if n_rows == 0 and float(getattr(ep, "total_duration", 0.0) or 0.0) > 0:
            raise RuntimeError(
                f"PHASE-A: episode {getattr(ep, 'index', i)} "
                f"({float(getattr(ep, 'total_duration', 0.0)):.0f}s) produced ZERO "
                "score samples — scoring failed (no frames extracted?). Refusing to "
                "merge a scored-timeline gap; investigate the source file instead of "
                "silently concatenating nothing."
            )


def _sample_source_sharpness(path, duration: float, n: int = SOURCE_SHARPNESS_SAMPLES) -> list:
    """Sharpness of ``n`` evenly spaced frames of one source (undecodable frames dropped).

    The scorer keeps no frames, so each sample is one reduced-size ffmpeg
    extraction. Every value comes from ``quality.sharpness_score``.
    """
    values = []
    for i in range(n):
        ts = max(duration, 0.0) * (i + 0.5) / n
        cmd = [
            "ffmpeg", "-loglevel", "error", "-ss", f"{ts:.3f}", "-i", str(path),
            "-frames:v", "1", "-vf", f"scale={int(SHARPEN_DETAIL_MAX_DIM)}:-2",
            "-f", "image2pipe", "-c:v", "png", "-",
        ]  # fmt: skip
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode != 0 or not r.stdout:
            continue
        gray = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            values.append(sharpness_score(gray))
    return values


def _json_safe_sharpness(values) -> list:
    """Per-source medians as floats; an unmeasured (NaN) source becomes ``None``."""
    return [None if np.isnan(v) else float(v) for v in values]


def _assert_sources_sharp(valid_episodes: list, per_source: list) -> list:
    """Median sharpness per source; hard-fail on a soft one before anything merges.

    A source with no decodable sample is recorded as NaN (the scored-gap gate
    owns undecodable footage); it is never silently passed off as sharp.
    """
    medians = []
    for ep, samples in zip(valid_episodes, per_source):
        median = float(np.median(samples)) if samples else float("nan")
        medians.append(median)
        if samples and median < SOURCE_SHARPNESS_FLOOR:
            raise RuntimeError(
                f"PHASE-A: source {Path(str(ep.concat_path)).name} is soft: median "
                f"sharpness {median:.1f} < floor {SOURCE_SHARPNESS_FLOOR:.1f} "
                f"(samples: {', '.join(f'{v:.1f}' for v in samples)}). Refusing to "
                "merge a source that cannot be graded sharp; re-shoot or replace it."
            )
        if not samples:
            logger.warning("  Source %s: no frame decoded for sharpness", ep.concat_path)
    return medians


class PhaseA:
    """Phase A: Ingest & Analyze."""

    name = "phase_a"

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self, state, labels_only: bool = False) -> None:
        """Run Phase A.

        ``labels_only`` is set by the orchestrator for ``--label-dry-run`` /
        ``--list-lfs`` invocations. In ``full_drive`` mode it skips the
        GPU-bound scoring pass entirely (see ``_synthetic_score_rows``) and
        never persists a ``drive_scores.npz`` cache, so a --label-dry-run run
        never waits on / uses the shared GPU lock and never poisons a later
        LIVE run's score-cache resume gate with fabricated scores. Ignored outside
        ``full_drive``: ``best_moments`` LF boundaries are themselves derived
        from real composite scores, so labels there are only meaningful with
        a real scoring pass.
        """
        logger.info("Phase A: Ingest & Analyze")
        source_dir = state.source_dir
        output_base = state.output_base
        # Heavy Phase A working files — the concatenated per-episode .mp4s
        # (hundreds of GB) and the drive_scores cache — live on the
        # orchestrator-resolved ext4 intermediate dir (state.stages_base),
        # NEVER the output dir. A v9fs/drvfs output mount (e.g. /mnt/<letter>) is slow
        # and unstable for a multi-hour batch; writing the ingest working
        # set there would sink hundreds of GB onto the slow mount and also
        # miss the orchestrator's cache reload, which reads from
        # stages_base first (output_base/_stages is the legacy fallback).
        # Mirrors Phase BC, which already routes day masters to stages_base.
        stages_dir = Path(state.stages_base) if state.stages_base else Path(output_base) / "_stages"
        stages_dir.mkdir(parents=True, exist_ok=True)

        # A1 + A2: Ingest
        logger.info("  A1-A2: Scanning and ingesting GoPro files...")
        plan = create_ingest_plan(
            source_dir,
            mode=self.config.drive.ingest_mode,
            extensions=self.config.drive.video_extensions,
        )
        plan = execute_ingest(plan, stages_dir, derive_locations=False)
        state.ingest_plan = plan

        # Score each episode file
        logger.info("  A3: Scoring episode files...")
        scorer = UnifiedSegmentScorer(self.config.drive.scoring_weights)
        # No weather pass. The ``weather`` scoring weight was removed (its
        # mass moved to the per-second ``brightness`` signal), and the
        # classifier behind it reports the same class for every input — so
        # the column it fills is a constant that ranks nothing. Measuring it
        # cost a full extra ffmpeg brightness decode of every episode.
        # Omitting the detector leaves the scorer on its unmeasured
        # fallback, which yields that identical constant column, so cached
        # drive_scores.npz files stay shape- and value-compatible.
        # The detector class itself lives on for the single-video social
        # stage; restore this pass only alongside a change that gives
        # weather real weight again.
        episode_files = []
        gpx_files = []
        gps_per_episode = []
        file_durations = []
        file_boundaries = []
        episode_dates = []
        offset = 0.0

        valid_episodes = []
        for ep in plan.episodes:
            if not ep.concat_path:
                logger.warning("  Episode %d: no concat file, skipping", ep.index)
                continue

            episode_files.append(ep.concat_path)
            gpx_files.append(ep.gpx_path)
            episode_dates.append(ep.date)

            gps_data = None
            gps_offset = 0.0
            if ep.gpx_path:
                try:
                    gps_data = parse_gpx_file(ep.gpx_path)
                    gps_offset = effective_offset(gps_data, gps_data.start_time or 0.0)
                except Exception as e:
                    logger.warning("  Episode %d: GPS parse failed: %s", ep.index, e)

            # GPX↔video correspondence gate. GoPro GPMF is
            # frame-aligned by construction (extracted from the same MP4)
            # so it is skipped at zero cost; the gate fires only on the
            # external-provenance path (user-supplied generic sidecar
            # or a resume-state pairing) where a swapped GPX could stamp a
            # wrong-drive location. Upstream of every irreversible artifact
            # (encode, OST burn, upload), so a failure costs nothing irreversible.
            gpx_source = getattr(ep, "gpx_source", "gpmf")
            if gps_data is not None and gpx_source == "external":
                embedded = extract_location_coords(ep.concat_path)
                assert_gpx_video_correspondence(
                    gps_data,
                    ep.creation_time,
                    ep.total_duration,
                    embedded_coord=embedded,
                )
            elif gps_data is not None and gpx_source == "gpmf_slice":
                # A GPMF sub-slice can't be a swapped track, but a gross
                # anchoring/slicing error can hand it a mismatched window.
                # Slice-tuned gate: start-time + one-sided over-span,
                # no embedded-coord (a stream-copy slice inherits the
                # parent's start-location tag, wrong for a later slice).
                assert_slice_gpx_correspondence(
                    gps_data,
                    ep.creation_time,
                    ep.total_duration,
                )

            gps_per_episode.append((gps_data, gps_offset))

            duration = ep.total_duration
            file_durations.append(duration)
            file_boundaries.append((offset, offset + duration))
            offset += duration
            valid_episodes.append(ep)

        # Auto-derive per-episode + drive-level OST from GPS
        try:
            geocoder = _get_shared_geocoder()
            route_analyzer = _get_route_analyzer(geocoder)
        except Exception as e:
            logger.warning("  Route analyzer init failed: %s -- OST will be empty", e)
            geocoder = None
            route_analyzer = None

        episode_locations = []
        episode_profiles = []
        episode_meta: list[dict] = []
        for i, gpx_path in enumerate(gpx_files):
            # Build structured profile from GPS
            profile = build_route_profile(
                index=i,
                gpx_files=gpx_files,
                episode_indices=[i],
                geocoder=geocoder,
                route_analyzer=route_analyzer,
            )
            episode_profiles.append(profile.to_dict())

            # Derive display string from profile title
            ep_ost = profile.title()
            if ep_ost:
                ok, reason = validate_ost(ep_ost, "lf")
                if not ok:
                    logger.warning(
                        "  Episode %d OST failed validation (%s): %r -- dropping",
                        i,
                        reason,
                        ep_ost,
                    )
                    ep_ost = ""
            episode_locations.append(ep_ost)
            episode_meta.append(_extract_episode_ost_meta(gpx_path))
            if ep_ost:
                logger.info("  Episode %d OST: %s", i, ep_ost)

        differentiated = differentiate_duplicate_ost(episode_locations, episode_meta)
        for i, (before, after) in enumerate(zip(episode_locations, differentiated)):
            if before and before != after:
                logger.info("  Episode %d OST differentiated: %r -> %r", i, before, after)
        episode_locations = differentiated
        state.episode_locations = episode_locations
        state.episode_profiles = episode_profiles

        valid_gpx = [p for p in gpx_files if p]
        drive_ost = ""
        if valid_gpx:
            drive_ost = derive_drive_ost_text(
                valid_gpx,
                route_analyzer=route_analyzer,
                geocoder=geocoder,
                episode_locations=episode_locations,
                episode_profiles=episode_profiles,
            )
            if drive_ost:
                ok, reason = validate_ost(drive_ost, "lf")
                if not ok:
                    logger.warning(
                        "  Drive-level OST failed validation (%s): %r -- dropping",
                        reason,
                        drive_ost,
                    )
                    drive_ost = ""

        overrides = load_overrides(output_base)
        if overrides:
            episode_locations, drive_ost = apply_overrides(
                episode_locations,
                drive_ost,
                overrides,
            )
            state.episode_locations = episode_locations

        if drive_ost:
            logger.info("  Drive-level OST: %s", drive_ost)
            state.location = drive_ost

        # Road-sign OCR evidence pass (advisory, opt-in via --sign-verify).
        # Reads signs off the RAW, PRE-BLUR episodes — blur covers faces /
        # plates / occupants, never signs — and records evidence ONLY: it
        # writes no location on any surface.
        #
        # Placed BEFORE the cached-score early return below so a resume that
        # reuses drive_scores.npz still back-fills the evidence. It is keyed
        # independently of that cache (``state.sign_hits_by_date is None``), and
        # PHASE_A_CODE_VERSION is deliberately NOT bumped — adding this pass
        # must not invalidate a 3-4h scoring checkpoint. A populated field means
        # the footage was already read, so a resume never re-decodes it.
        if getattr(self.config.drive, "sign_verify", False) and state.sign_hits_by_date is None:
            try:
                from ...analysis.sign_evidence import hits_by_date_to_dicts
                from ...analysis.sign_pass import collect_sign_hits

                sign_hits = collect_sign_hits(
                    self.config,
                    plan,
                    episode_files,
                    gps_per_episode,
                )
                state.sign_hits_by_date = hits_by_date_to_dicts(sign_hits)
                logger.info(
                    "  A6: sign pass recorded %d hit(s) across %d day(s)",
                    sum(len(v) for v in (sign_hits or {}).values()),
                    len(sign_hits or {}),
                )
            except Exception as e:  # noqa: BLE001 — advisory evidence never fails a run
                logger.warning("  A6: sign pass failed (%s) -- continuing without sign evidence", e)

        # Score-cache resume gate — reload the cached drive_scores.npz instead of
        # re-running the 3-4h scoring pass when nothing the scores depend on
        # has changed. The scoring pass is a pure function of the episode
        # source files, so the validity key is each concat path + size +
        # mtime, persisted in the .npz metadata sidecar (``score_inputs``)
        # when saved. A mismatch (added/removed/replaced/re-ingested episode),
        # a missing key (pre-resume-gate cache), or any load error safely falls through
        # to a fresh score. ``--fresh`` does NOT delete drive_scores.npz
        # (_purge_stale_stage_artifacts only wipes drive_plan/pipeline_state),
        # so it is honoured here explicitly: a fresh run always re-scores.
        scores_path = stages_dir / "drive_scores.npz"
        current_key = _score_inputs_key(episode_files)
        if not state.fresh and scores_path.exists():
            try:
                cached_scores, cached_meta = scorer.load_scores(scores_path)
                cached_key = cached_meta.get("score_inputs")
                with np.load(scores_path) as npz:
                    cached_sharpness = (
                        npz["source_sharpness"] if "source_sharpness" in npz else None
                    )
            except Exception as e:  # corrupt npz / unreadable sidecar
                logger.warning("  resume: cached drive_scores.npz unreadable (%s) -- re-scoring", e)
            else:
                if cached_sharpness is None or len(cached_sharpness) != len(episode_files):
                    logger.info(
                        "  resume: cached drive_scores.npz has no usable source_sharpness -- re-scoring"
                    )
                elif cached_key == current_key and len(cached_scores) > 0:
                    logger.info("  resume: reusing cached drive_scores.npz, skipping re-score")
                    state.episode_files = episode_files
                    state.gpx_files = gpx_files
                    state.gps_per_episode = gps_per_episode
                    state.drive_scores = cached_scores
                    state.source_sharpness = _json_safe_sharpness(cached_sharpness)
                    state.file_boundaries = file_boundaries
                    state.file_durations = file_durations
                    state.episode_dates = episode_dates
                    return
                else:
                    logger.info(
                        "  resume: cached drive_scores.npz inputs changed -- re-scoring "
                        "(cached %d episodes, current %d)",
                        len(cached_key) if isinstance(cached_key, list) else -1,
                        len(current_key),
                    )

        # --label-dry-run / --list-lfs in full_drive mode never needs the
        # real (GPU frame-decode) scoring pass -- full_drive's LF-per-day (or K balanced parts)
        # boundaries come from GPS-speed stop mask + calendar day, and the
        # label printers (lf_listing.py) only read segment start/end times
        # and raw GPX points, never composite quality columns. Skip straight
        # to a synthetic score array (motion pinned to 1.0, see
        # ``_synthetic_score_rows``) so the caller never takes the shared GPU
        # lock. GPS-less spans are shown here unmasked-by-motion; a LIVE run
        # (real scoring) may still mask a genuine stop there once optical
        # motion is measured. Not applied to best_moments: there, LF
        # boundaries themselves come from real composite-score thresholds, so
        # a synthetic array would print meaningless labels.
        labels_only_skip_scoring = labels_only and (
            resolve_lf_mode(self.config.drive) == "full_drive"
        )
        if labels_only_skip_scoring:
            logger.info(
                "  A3: --label-dry-run/--list-lfs (full_drive) -- skipping GPU "
                "scoring pass; labels resolve from GPX/episode metadata only"
            )
            file_scores = [_synthetic_score_rows(ep.total_duration) for ep in valid_episodes]
        else:
            # Score episodes in parallel
            max_scoring_workers = min(3, len(valid_episodes))

            def _score_episode(args):
                idx, ep, gps_entry = args
                gps_data, gps_off = gps_entry
                scores = scorer.score_file(
                    ep.concat_path,
                    gps_data=gps_data,
                    gps_offset=gps_off,
                )
                logger.info(
                    "  Episode %d: %d score samples (%.0f min)",
                    ep.index,
                    len(scores),
                    ep.total_duration / 60,
                )
                return idx, scores

            score_args = [(i, ep, gps_per_episode[i]) for i, ep in enumerate(valid_episodes)]

            file_scores = [None] * len(valid_episodes)
            if max_scoring_workers > 1:
                logger.info(
                    "  Scoring %d episodes in parallel (%d workers)...",
                    len(valid_episodes),
                    max_scoring_workers,
                )
                with ThreadPoolExecutor(max_workers=max_scoring_workers) as pool:
                    for idx, scores in pool.map(_score_episode, score_args):
                        file_scores[idx] = scores
            else:
                for args in score_args:
                    idx, scores = _score_episode(args)
                    file_scores[idx] = scores

        # Refuse to merge a scored-timeline gap from a failed episode score.
        _assert_no_scored_gaps(valid_episodes, file_scores)

        # Source picture gate -- measured per source, refused before the merge.
        # labels_only dry runs never score, so they never gate either.
        source_sharpness: list = []
        if not labels_only_skip_scoring:
            source_sharpness = _assert_sources_sharp(
                valid_episodes,
                [
                    _sample_source_sharpness(ep.concat_path, ep.total_duration)
                    for ep in valid_episodes
                ],
            )

        # Log suggested action for dead-zone episodes
        for i, ep in enumerate(valid_episodes):
            ep_width, ep_height = 3840, 2160
            if ep.concat_path and Path(ep.concat_path).exists():
                try:
                    import subprocess as _sp

                    _probe = _sp.run(
                        [
                            "ffprobe",
                            "-v",
                            "quiet",
                            "-select_streams",
                            "v:0",
                            "-show_entries",
                            "stream=width,height",
                            "-of",
                            "csv=p=0",
                            str(ep.concat_path),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if _probe.returncode == 0 and "," in _probe.stdout.strip():
                        parts = _probe.stdout.strip().split(",")
                        ep_width, ep_height = int(parts[0]), int(parts[1])
                except Exception:
                    pass

            duration = ep.total_duration
            vtype = classify_video(duration, ep_width, ep_height)
            if vtype == VideoType.UNCLASSIFIED:
                action = suggest_action(duration, ep_width, ep_height)
                logger.warning(
                    "  Episode %d: UNCLASSIFIED (%.0fs) -- suggested action: %s",
                    ep.index,
                    duration,
                    action,
                )

        # Merge into drive-wide time series
        logger.info("  A4: Merging %d file scores into drive-wide timeline...", len(file_scores))
        drive_scores = scorer.merge_file_scores(file_scores, file_durations)

        # Save scores to .npz (with the resume-cache validity key).
        # Skipped for the synthetic labels_only_skip_scoring array -- persisting
        # it would let a later LIVE run's score-cache resume gate reuse fabricated
        # zero-valued scores instead of ever really scoring the episodes.
        if not labels_only_skip_scoring:
            scorer.save_scores(
                drive_scores,
                scores_path,
                {
                    "num_files": len(file_scores),
                    "total_duration": sum(file_durations),
                    "file_durations": file_durations,
                    "score_inputs": current_key,
                    "source_sharpness": _json_safe_sharpness(source_sharpness),
                },
                extra={"source_sharpness": np.asarray(source_sharpness, dtype=np.float64)},
            )

        state.episode_files = episode_files
        state.gpx_files = gpx_files
        state.gps_per_episode = gps_per_episode
        state.drive_scores = drive_scores
        state.source_sharpness = _json_safe_sharpness(source_sharpness)
        state.file_boundaries = file_boundaries
        state.file_durations = file_durations
        state.episode_dates = episode_dates
