"""
Pipeline Cleanup — safe removal of intermediate files with dry-run by default.

Cleans up large intermediate artifacts (~20GB per project) while enforcing
safety gates to prevent data loss.

Usage:
    from gomoveshift.pipeline.cleanup import PipelineCleanup

    cleanup = PipelineCleanup()
    projects = cleanup.discover_projects()
    plan = cleanup.plan_cleanup(projects, min_age_days=7)
    # Inspect plan.files and plan.skipped before executing
    cleanup.execute_plan(plan)
"""

import functools
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Intermediate files in _stages/ that are safe to delete once _FINAL.mp4 exists
STAGE_INTERMEDIATES = {
    "_no_stops.mp4",
    "_color.mp4",
    "_analysis.json",
    "_stops.json",
}

# Terminal completion markers across the two pipelines that share the
# ``_stages/pipeline_state.json`` file:
#
#   * Main video pipeline (``gomoveshift pipeline``) writes
#     ``stages_completed`` with ``csv_generation`` as the terminal marker.
#   * Drive orchestrator (``gomoveshift drive``) writes ``phases_completed``
#     with ``phase_d`` as the terminal marker.
#
# Cleanup must accept either. These are legitimately distinct logical
# buckets — a rename would collide with real usage — so the reader is a
# migration shim that recognizes both schemas.
_MAIN_COMPLETION_KEY = "stages_completed"
_MAIN_COMPLETION_MARKER = "csv_generation"
_DRIVE_COMPLETION_KEY = "phases_completed"
_DRIVE_COMPLETION_MARKER = "phase_d"


@dataclass
class CleanupPlan:
    """Plan describing files to delete and items skipped with reasons."""

    files: List[Dict] = field(default_factory=list)  # [{"path": str, "size": int, "reason": str}]
    skipped: List[Dict] = field(default_factory=list)  # [{"path": str, "reason": str}]
    # Copied-source dirs (``<stage_cache_base>/sources/<project>``) being reclaimed.
    source_dirs: List[Dict] = field(default_factory=list)  # [{"path": str, "size": int}]


class PipelineCleanup:
    """
    Safe cleanup of local pipeline artifacts.

    Safety rules:
    - Dry-run by default (--confirm required)
    - Age-based retention (default 7 days)
    - Pipeline must be complete (main-pipeline csv_generation in
      stages_completed, OR drive-pipeline phase_d in phases_completed)
    - _stages/ intermediates only deleted if _FINAL.mp4 exists, is non-zero,
      AND passes ffprobe (a truncated-but-nonzero final must not destroy the
      only re-encode path)
    - WSL output directory's deliverable only deleted if a *distinct*
      ``d_drive_base`` is explicitly configured and holds a verified D:
      drive copy (size match). The ``cleanup`` CLI command's
      ``--archive-base`` flag is the only way to supply it and defaults to
      unset, so by default the deliverable is never deleted; comparing
      ``output_base`` against itself is not verification.
    - WSL staging input only deleted if pipeline completed
    - Drive runs (``phase_d`` done, no ``_FINAL.mp4``; found under
      ``stage_cache_base``): an ``lf_NN.mp4`` render is deleted only when its
      LF has a YouTube ``video_id``; the episode concat only when every LF is
      uploaded and ``source_dir`` still holds the footage to rebuild it.
      Symlinks are never listed.
    - Copied sources (``<stage_cache_base>/sources/<project>``, the ext4 copy
      preflight tells operators to make of a cross-filesystem source) are
      reclaimed with a published drive. Symlinks inside are unlinked, never
      followed; a symlinked ``sources/<project>`` is skipped outright.
    - NEVER touches: Google Drive files, D: drive output, source videos on D:
    """

    def __init__(
        self,
        output_base: Optional[Path] = None,
        staging_dir: Optional[Path] = None,
        d_drive_base: Optional[Path] = None,
        stage_cache_base: Optional[Path] = None,
    ):
        from ..paths import gms_output_dir, gms_staging_dir

        self.output_base = output_base or gms_output_dir()
        self.staging_dir = staging_dir or gms_staging_dir()
        self.d_drive_base = d_drive_base
        # Drive runs' ext4 stage cache (``paths.gms_intermediate_base``). Unset:
        # only ``output_base`` is scanned. The ``cleanup`` CLI always sets it.
        self.stage_cache_base = stage_cache_base

    def discover_projects(self) -> List[Path]:
        """Find project directories in the output base and the stage cache."""
        projects: List[Path] = []
        seen: set = set()
        for base in (self.output_base, self.stage_cache_base):
            if base is None or not base.exists():
                continue
            for entry in sorted(base.iterdir()):
                if entry.is_dir() and (entry / "_stages").exists():
                    key = entry.resolve()
                    if key not in seen:
                        seen.add(key)
                        projects.append(entry)
        return projects

    def _load_pipeline_state(self, project_dir: Path) -> Optional[Dict]:
        """Load pipeline_state.json for a project."""
        state_file = project_dir / "_stages" / "pipeline_state.json"
        if not state_file.exists():
            return None
        try:
            with open(state_file) as f:
                return json.load(f)
        except Exception as exc:
            logger.warning(
                "Failed to read pipeline_state.json at %s (%s: %s); treating project as incomplete and skipping cleanup",
                state_file,
                type(exc).__name__,
                exc,
            )
            return None

    def _is_pipeline_complete(self, state: Optional[Dict]) -> bool:
        """Check if pipeline completed its terminal stage/phase.

        Accepts both schemas written to ``_stages/pipeline_state.json``:

        * main pipeline: ``csv_generation`` in ``stages_completed``
        * drive pipeline: ``phase_d`` in ``phases_completed``

        A project is considered complete if either marker is present. The
        reader is a migration shim rather than a rename because the two keys
        are distinct logical buckets.
        """
        if not state:
            return False
        main_done = _MAIN_COMPLETION_MARKER in state.get(_MAIN_COMPLETION_KEY, [])
        drive_done = _DRIVE_COMPLETION_MARKER in state.get(_DRIVE_COMPLETION_KEY, [])
        return main_done or drive_done

    def _project_age_days(self, project_dir: Path) -> float:
        """Get age of project in days based on _stages/ modification time."""
        stages_dir = project_dir / "_stages"
        if stages_dir.exists():
            mtime = stages_dir.stat().st_mtime
        else:
            mtime = project_dir.stat().st_mtime
        return (time.time() - mtime) / 86400

    def _d_drive_copy_verified(self, project_dir: Path) -> bool:
        """Check if D: drive has matching deliverables (file exists + size match).

        Returns False whenever no *distinct* D: drive location is configured
        — ``d_drive_base`` unset, or resolving to the same directory as the
        WSL output — because comparing a file against itself is not
        verification and must never be treated as a pass.
        """
        if self.d_drive_base is None:
            return False

        project_name = project_dir.name
        d_dir = self.d_drive_base / project_name

        if d_dir.resolve() == project_dir.resolve():
            return False

        if not d_dir.exists():
            return False

        final_src = project_dir / f"{project_name}_FINAL.mp4"
        final_dst = d_dir / f"{project_name}_FINAL.mp4"

        if not final_src.exists() or not final_dst.exists():
            return False

        return final_src.stat().st_size == final_dst.stat().st_size

    def plan_cleanup(
        self,
        projects: List[Path],
        min_age_days: int = 7,
        *,
        ffprobe_fn: Optional[object] = None,
    ) -> CleanupPlan:
        """
        Build a cleanup plan listing files to delete with reasons.

        Returns a CleanupPlan with files (to delete) and skipped (with reasons).

        ``ffprobe_fn`` is injectable for tests; default is
        ``ffmpeg_runner.validate_video`` (lazy-imported). It must accept a
        ``Path`` and return an object with a ``valid: bool`` attribute (and
        ``reason: str``). The ``_FINAL.mp4`` is ffprobe-validated before any
        ``_stages/`` intermediate is queued for deletion — a truncated-but-
        nonzero final must not destroy the only re-encode path.
        """
        if ffprobe_fn is None:
            from .ffmpeg_runner import validate_video as _vv

            ffprobe_fn = _vv

        plan = CleanupPlan()

        for project_dir in projects:
            project_name = project_dir.name
            age_days = self._project_age_days(project_dir)
            state = self._load_pipeline_state(project_dir)
            pipeline_complete = self._is_pipeline_complete(state)

            # Age check
            if age_days < min_age_days:
                plan.skipped.append(
                    {
                        "path": str(project_dir),
                        "reason": f"Too recent ({age_days:.1f} days < {min_age_days} day minimum)",
                    }
                )
                continue

            # Pipeline completion check
            if not pipeline_complete:
                logger.warning(
                    "⚠ cleanup: project %s skipped — pipeline not complete "
                    "(neither %s in %s nor %s in %s); preferred path "
                    "(reclaim intermediates for completed run) unavailable; "
                    "falling back to no-op (appended to plan.skipped)",
                    project_dir,
                    _MAIN_COMPLETION_MARKER,
                    _MAIN_COMPLETION_KEY,
                    _DRIVE_COMPLETION_MARKER,
                    _DRIVE_COMPLETION_KEY,
                )
                plan.skipped.append(
                    {
                        "path": str(project_dir),
                        "reason": (
                            f"Pipeline not complete (neither "
                            f"{_MAIN_COMPLETION_MARKER} in {_MAIN_COMPLETION_KEY} "
                            f"nor {_DRIVE_COMPLETION_MARKER} in {_DRIVE_COMPLETION_KEY})"
                        ),
                    }
                )
                continue

            final_video = project_dir / f"{project_name}_FINAL.mp4"
            drive_done = _DRIVE_COMPLETION_MARKER in (state or {}).get(_DRIVE_COMPLETION_KEY, [])
            if drive_done and not final_video.exists():
                # Drive layout: renders + concat live in _stages, no _FINAL.
                self._plan_drive_stage_cache(project_dir, state or {}, plan)
                continue

            # 1. _stages/ intermediates — gate: _FINAL.mp4 exists, non-zero,
            #    AND passes ffprobe. A truncated-but-nonzero final must not
            #    trigger deletion of the only re-encode path (mirrors the
            #    drop_day_master shorts-integrity guard).
            stages_dir = project_dir / "_stages"

            final_ok = final_video.exists() and final_video.stat().st_size > 0
            probe_reason = "_FINAL.mp4 missing or empty"
            if final_ok:
                probe = ffprobe_fn(final_video)  # type: ignore[misc]
                if not bool(getattr(probe, "valid", False)):
                    final_ok = False
                    probe_reason = (
                        f"_FINAL.mp4 failed ffprobe ({getattr(probe, 'reason', 'invalid')})"
                    )

            if final_ok:
                if stages_dir.exists():
                    for f in sorted(stages_dir.iterdir()):
                        if f.is_file() and f.suffix in {".mp4", ".json"}:
                            # Keep pipeline_state.json
                            if f.name == "pipeline_state.json":
                                continue
                            plan.files.append(
                                {
                                    "path": str(f),
                                    "size": f.stat().st_size,
                                    "reason": "Stage intermediate (_FINAL.mp4 verified)",
                                }
                            )
            else:
                plan.skipped.append(
                    {
                        "path": str(stages_dir),
                        "reason": f"{probe_reason} — refusing to delete intermediates",
                    }
                )

            # 2. WSL output directory — gate: D: drive copy verified
            d_verified = self._d_drive_copy_verified(project_dir)
            if not d_verified:
                plan.skipped.append(
                    {
                        "path": str(project_dir),
                        "reason": "D: drive copy not verified — refusing to delete WSL output",
                    }
                )
            # Note: we only clean intermediates above, not the whole output dir.
            # Full output dir cleanup is gated on D: drive verification
            # and could remove _FINAL.mp4 + metadata, so we add those separately.
            if d_verified:
                # Large deliverables that are now on D: drive
                for name in [
                    f"{project_name}_FINAL.mp4",
                ]:
                    src = project_dir / name
                    if src.exists():
                        plan.files.append(
                            {
                                "path": str(src),
                                "size": src.stat().st_size,
                                "reason": "Deliverable verified on D: drive (size match)",
                            }
                        )

            # 3. WSL staging files — gate: pipeline completed
            if pipeline_complete and self.staging_dir.exists():
                for f in sorted(self.staging_dir.glob("*.mp4")):
                    # Clean staging files that match this project:
                    # - Source video copy (same name as input)
                    # - FINAL copy (project_name_FINAL.mp4)
                    source_video = state.get("source_video", "") if state else ""
                    is_source = f.name == Path(source_video).name
                    is_final = f.name == f"{project_name}_FINAL.mp4"
                    if is_source or is_final:
                        plan.files.append(
                            {
                                "path": str(f),
                                "size": f.stat().st_size,
                                "reason": "Staging copy (pipeline complete)",
                            }
                        )

                # Also clean color analysis artifacts
                for f in sorted(self.staging_dir.glob("*_color_analysis.json")):
                    plan.files.append(
                        {
                            "path": str(f),
                            "size": f.stat().st_size,
                            "reason": "Staging color analysis artifact",
                        }
                    )

        return plan

    def _plan_drive_stage_cache(self, project_dir: Path, state: Dict, plan: CleanupPlan) -> None:
        """Queue a published drive's rebuildable renders (see class docstring)."""
        stages_dir = project_dir / "_stages"
        uploads = state.get("youtube_uploads") or {}
        video_ids = {str(k): (v or {}).get("video_id") for k, v in uploads.items()}
        all_uploaded = bool(video_ids) and all(video_ids.values())
        if all_uploaded:
            self._plan_copied_sources(project_dir.name, plan)

        for lf in sorted(stages_dir.glob("day_*/regular_lfs/lf_*.mp4")):
            m = _LF_FINAL_RE.match(lf.name)
            if not m or lf.is_symlink() or not lf.is_file():
                continue
            vid = video_ids.get(str(int(m.group(1))))
            if vid:
                plan.files.append(
                    {
                        "path": str(lf),
                        "size": lf.stat().st_size,
                        "reason": f"LF render uploaded to YouTube ({vid})",
                    }
                )
            else:
                plan.skipped.append(
                    {"path": str(lf), "reason": "LF not uploaded (no YouTube video_id)"}
                )

        episodes = [
            p
            for p in sorted(stages_dir.glob("episode_*.mp4"))
            if p.is_file() and not p.is_symlink()
        ]
        if not episodes:
            return
        if not all_uploaded:
            plan.skipped.append(
                {
                    "path": str(stages_dir),
                    "reason": "Episode concat kept — not every LF uploaded (no YouTube video_id)",
                }
            )
            return
        source_dir = Path(state.get("source_dir") or "")
        if not state.get("source_dir") or not source_dir.is_dir() or not any(source_dir.iterdir()):
            plan.skipped.append(
                {
                    "path": str(stages_dir),
                    "reason": f"Episode concat kept — source_dir {source_dir} missing or empty",
                }
            )
            return
        for ep in episodes:
            plan.files.append(
                {
                    "path": str(ep),
                    "size": ep.stat().st_size,
                    "reason": f"Episode concat (all LFs uploaded; rebuildable from source {source_dir})",
                }
            )

    def _plan_copied_sources(self, project_name: str, plan: CleanupPlan) -> None:
        """Queue ``<stage_cache_base>/sources/<project>``: every entry, symlinks included.

        Walks without following links, so a symlink (to a file or directory) is
        queued as a link to unlink and its target is never read or touched.
        """
        if self.stage_cache_base is None:
            return
        src = self.stage_cache_base / "sources" / project_name
        if src.is_symlink():
            plan.skipped.append(
                {"path": str(src), "reason": "sources dir is a symlink — never followed"}
            )
            return
        if not src.is_dir():
            return
        total = 0
        for root, dirnames, filenames in os.walk(src, followlinks=False):
            entries = [(n, True) for n in dirnames if (Path(root) / n).is_symlink()]
            entries += [(n, (Path(root) / n).is_symlink()) for n in filenames]
            for name, is_link in sorted(entries):
                path = Path(root) / name
                size = 0 if is_link else path.stat().st_size
                total += size
                plan.files.append(
                    {
                        "path": str(path),
                        "size": size,
                        "reason": (
                            "Copied source symlink (link only; target untouched)"
                            if is_link
                            else "Copied source (published drive; original stays on its drive)"
                        ),
                    }
                )
            dirnames[:] = [n for n in dirnames if not (Path(root) / n).is_symlink()]
        plan.source_dirs.append({"path": str(src), "size": total})

    def execute_plan(self, plan: CleanupPlan) -> int:
        """
        Execute a cleanup plan by deleting the listed files.

        Returns the number of files deleted.
        """
        deleted = 0
        for f in plan.files:
            path = Path(f["path"])
            try:
                if path.is_symlink() or path.exists():
                    size_mb = f["size"] / (1024 * 1024)
                    logger.info(f"  Deleting: {path} ({size_mb:.1f} MB)")
                    path.unlink()
                    deleted += 1
            except Exception as e:
                logger.warning(f"  Failed to delete {path}: {e}")

        for d in plan.source_dirs:
            # Bottom-up rmdir of the now-empty copied-sources tree; never
            # recursive-delete, so anything left over (failed unlink) survives.
            root = Path(d["path"])
            if root.is_symlink() or not root.is_dir():
                continue
            for cur, _dirs, _files in os.walk(root, topdown=False):
                try:
                    os.rmdir(cur)
                except OSError:
                    pass

        return deleted


# ---------------------------------------------------------------------------
# Per-day intermediate purge — inline cleanup during run_day_pipeline.
# ---------------------------------------------------------------------------

# Final-file names emitted by ``run_day_pipeline`` that gate each intermediate.
# Each final is the *non*-suffixed OST-applied artifact; work files live
# alongside with a known suffix.
_LF_FINAL_RE = re.compile(r"^lf_(\d{2})\.mp4$")
_SHORT_FINAL_RE = re.compile(r"^short_(\d{2})\.mp4$")
_SUPERLONG_FINAL_RE = re.compile(r"^sl_(\d{2})\.mp4$")


def _finalized_indices(directory: Path, pattern: re.Pattern) -> set[str]:
    """Return the set of 2-digit indices whose final artifact exists + is non-empty."""
    if not directory.is_dir():
        return set()
    indices: set[str] = set()
    for entry in directory.iterdir():
        match = pattern.match(entry.name)
        if not match:
            continue
        try:
            if entry.is_file() and entry.stat().st_size > 0:
                indices.add(match.group(1))
        except OSError:
            continue
    return indices


def finalized_superlong_count(day_dir: Path) -> int:
    """Number of non-empty ``superlong/sl_NN.mp4`` finals in ``day_dir``.

    The single definition of "a superlong part is on disk", shared by
    ``drop_day_master``'s retention gate and the resume-completeness predicate
    in ``phase_bc_day_first`` — the two places that must agree on whether a
    day's superlong work is finished.
    """
    return len(_finalized_indices(Path(day_dir) / "superlong", _SUPERLONG_FINAL_RE))


def _purge_lf_work(lf_dir: Path, idx: str, summary: Dict[str, int]) -> None:
    """Delete one finalized LF's work files, adding to ``summary``.

    ``idx`` is the 2-digit index; the caller has verified ``lf_{idx}.mp4``.
    """
    for name in (f"lf_{idx}_segments.mp4", f"lf_{idx}_with_montage.mp4"):
        path = lf_dir / name
        try:
            size = path.stat().st_size if path.is_file() else 0
        except OSError:
            continue
        if size == 0 and not path.is_file():
            continue
        try:
            path.unlink()
        except OSError as exc:
            logger.warning("  purge: failed to delete %s: %s", path, exc)
            continue
        summary["files_deleted"] += 1
        summary["bytes_freed"] += size
        logger.debug("  purge: removed %s (%d bytes)", path, size)

    work = lf_dir / f".lf_{idx}_work"
    if work.is_dir():
        try:
            work_size = sum(f.stat().st_size for f in work.rglob("*") if f.is_file())
        except OSError:
            work_size = 0
        try:
            shutil.rmtree(work)
        except OSError as exc:
            logger.warning("  purge: failed to remove %s: %s", work, exc)
            return
        summary["dirs_deleted"] += 1
        summary["bytes_freed"] += work_size
        logger.debug("  purge: removed %s (%d bytes)", work, work_size)


def purge_lf_intermediates(day_dir: Path, lf_index: int) -> Dict[str, int]:
    """Remove ONE long-form's work files once its final exists.

    Called by ``run_day_pipeline`` right after an LF's burn is fingerprinted,
    so a multi-part day holds one LF's working set at a time instead of every
    part's until day end (a 133-minute two-part day needed ~233 GB the old
    way, ~124 GB this way). Same targets and same gate as the LF half of
    ``purge_day_intermediates``: ``regular_lfs/lf_XX.mp4`` must exist with
    non-zero size, else nothing is touched.

    Returns ``{"files_deleted", "dirs_deleted", "bytes_freed"}``.
    """
    summary = {"files_deleted": 0, "dirs_deleted": 0, "bytes_freed": 0}
    lf_dir = Path(day_dir) / "regular_lfs"
    idx = f"{int(lf_index):02d}"
    if idx in _finalized_indices(lf_dir, _LF_FINAL_RE):
        _purge_lf_work(lf_dir, idx, summary)
        if summary["bytes_freed"]:
            logger.info(
                "  purge: LF %s work files released (%.2f GB)",
                idx,
                summary["bytes_freed"] / 1_073_741_824,
            )
    return summary


def purge_day_intermediates(day_dir: Path) -> Dict[str, int]:
    """Remove LF + short work files for a single day once finals exist.

    Called at the end of ``run_day_pipeline`` to bound the intermediate cache
    at ~one day's working set. The gate is per-LF / per-short: each final
    ``lf_XX.mp4`` or ``short_XX.mp4`` must exist with non-zero size before its
    siblings are deleted. Missing or empty finals leave everything in place
    (resume-safe).

    Targets (gated on ``lf_XX.mp4``):
      - ``regular_lfs/lf_XX_segments.mp4``
      - ``regular_lfs/lf_XX_with_montage.mp4``
      - ``regular_lfs/.lf_XX_work/`` (directory)

    Targets (gated on ``short_XX.mp4``):
      - ``shorts/short_XX_raw.mp4``

    Targets (gated on ``superlong/sl_XX.mp4``):
      - ``superlong/sl_XX_segments.mp4``
      - ``superlong/.sl_XX_work/`` (directory)

    Never touches ``day_*_master.mp4``, ``day_*_color_cache.json``, or the
    final deliverables themselves. Missing ``day_dir`` is a silent no-op.

    Returns a dict: ``{"files_deleted", "dirs_deleted", "bytes_freed"}``.
    """
    summary = {"files_deleted": 0, "dirs_deleted": 0, "bytes_freed": 0}
    day_dir = Path(day_dir)
    if not day_dir.is_dir():
        return summary

    lf_dir = day_dir / "regular_lfs"
    shorts_dir = day_dir / "shorts"
    sl_dir = day_dir / "superlong"

    lf_ready = _finalized_indices(lf_dir, _LF_FINAL_RE)
    short_ready = _finalized_indices(shorts_dir, _SHORT_FINAL_RE)
    sl_ready = _finalized_indices(sl_dir, _SUPERLONG_FINAL_RE)

    for idx in lf_ready:
        _purge_lf_work(lf_dir, idx, summary)

    for idx in short_ready:
        path = shorts_dir / f"short_{idx}_raw.mp4"
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
            path.unlink()
        except OSError as exc:
            logger.warning("  purge: failed to delete %s: %s", path, exc)
            continue
        summary["files_deleted"] += 1
        summary["bytes_freed"] += size
        logger.debug("  purge: removed %s (%d bytes)", path, size)

    for idx in sl_ready:
        path = sl_dir / f"sl_{idx}_segments.mp4"
        if path.is_file():
            try:
                size = path.stat().st_size
                path.unlink()
            except OSError as exc:
                logger.warning("  purge: failed to delete %s: %s", path, exc)
            else:
                summary["files_deleted"] += 1
                summary["bytes_freed"] += size
                logger.debug("  purge: removed %s (%d bytes)", path, size)

        work = sl_dir / f".sl_{idx}_work"
        if work.is_dir():
            try:
                work_size = sum(f.stat().st_size for f in work.rglob("*") if f.is_file())
            except OSError:
                work_size = 0
            try:
                shutil.rmtree(work)
            except OSError as exc:
                logger.warning("  purge: failed to remove %s: %s", work, exc)
            else:
                summary["dirs_deleted"] += 1
                summary["bytes_freed"] += work_size
                logger.debug("  purge: removed %s (%d bytes)", work, work_size)

    if summary["files_deleted"] or summary["dirs_deleted"]:
        logger.info(
            "  purge: day %s — removed %d file(s) + %d dir(s), freed %.2f GB",
            day_dir.name,
            summary["files_deleted"],
            summary["dirs_deleted"],
            summary["bytes_freed"] / (1024**3),
        )

    return summary


def verify_shorts_integrity(
    day_dir: Path,
    *,
    expected_count: Optional[int] = None,
    ffprobe_fn: Optional[object] = None,
) -> Dict[str, object]:
    """Verify a day's shorts are publishable before letting the day master be dropped.

    The day master is the only economical source for re-derived shorts (a
    fresh encode is ~6h per day). Shorts once came out at 16:9 instead of
    9:16 — unpublishable on Reels/Shorts/TikTok — after the master had already
    been deleted. This check is the gate that blocks ``drop_day_master`` until every short on disk
    is verified as a real, decodable file.

    Three predicates, each must pass for ``ok=True``:

    1. **Existence** — every expected short file (``shorts/short_NN.mp4``)
       exists with non-zero size on disk.
    2. **ffprobe integrity** — each short's container parses, has at least
       one video stream, and reports ``duration > 0``.
    3. **Count match** (only when ``expected_count`` is provided) — the number
       of present shorts equals ``expected_count``. Useful when the caller
       knows from state how many shorts the day was supposed to emit.

    On any failure, the result names the offending shorts so the operator can
    investigate without grepping logs.

    ``ffprobe_fn`` is injectable for tests. Default is
    ``ffmpeg_runner.validate_video`` (lazy-imported so this module stays
    cheap to import). The callable must accept a ``Path`` and return an
    object with attributes ``valid: bool``, ``duration: Optional[float]``,
    and ``reason: str`` (matching ``VideoValidation``).

    Returns a dict with keys:
      - ``ok`` (bool) — True iff all predicates passed.
      - ``reason`` (str) — short human description; empty when ok.
      - ``missing`` (list[str]) — short filenames expected but missing/empty.
      - ``bad`` (list[dict]) — entries ``{"name": str, "reason": str}`` for
        shorts that failed ffprobe.
      - ``count_actual`` (int) — number of non-empty short files found.
      - ``count_expected`` (Optional[int]) — passthrough of input.
    """
    summary: Dict[str, object] = {
        "ok": False,
        "reason": "",
        "missing": [],
        "bad": [],
        "count_actual": 0,
        "count_expected": expected_count,
    }
    day_dir = Path(day_dir)
    shorts_dir = day_dir / "shorts"
    if not shorts_dir.is_dir():
        summary["reason"] = f"shorts dir missing: {shorts_dir}"
        return summary

    # Discover present shorts via the same regex used elsewhere.
    present: list[Path] = []
    for entry in sorted(shorts_dir.iterdir()):
        match = _SHORT_FINAL_RE.match(entry.name)
        if not match:
            continue
        try:
            if entry.is_file() and entry.stat().st_size > 0:
                present.append(entry)
        except OSError:
            continue
    summary["count_actual"] = len(present)

    # Existence: if expected_count given, identify gaps by index.
    missing: list[str] = []
    if expected_count is not None and expected_count > 0:
        present_indices = {
            int(_SHORT_FINAL_RE.match(p.name).group(1))  # type: ignore[union-attr]
            for p in present
        }
        for i in range(1, expected_count + 1):
            if i not in present_indices:
                missing.append(f"short_{i:02d}.mp4")
    summary["missing"] = missing

    # ffprobe each present short. Shorts carry no audio stream by design
    # (`-an`, see the audio hard rules), so the default probe must not
    # require one — pinning require_audio=False here rather than in
    # validate_video's own default keeps every other caller unaffected.
    if ffprobe_fn is None:
        from .ffmpeg_runner import validate_video as _vv

        ffprobe_fn = functools.partial(_vv, require_audio=False)

    bad: list[Dict[str, str]] = []
    for short in present:
        try:
            result = ffprobe_fn(short)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001 — any probe error is "bad"
            bad.append({"name": short.name, "reason": f"probe raised: {exc}"})
            continue
        valid = bool(getattr(result, "valid", False))
        duration = getattr(result, "duration", None)
        if not valid:
            bad.append(
                {
                    "name": short.name,
                    "reason": getattr(result, "reason", "ffprobe invalid"),
                }
            )
            continue
        if duration is None or duration <= 0:
            bad.append({"name": short.name, "reason": f"duration={duration}"})
    summary["bad"] = bad

    # Final verdict.
    if missing:
        summary["reason"] = (
            f"missing {len(missing)} short(s) "
            f"(expected {expected_count}, found {summary['count_actual']})"
        )
        return summary
    if bad:
        summary["reason"] = f"{len(bad)} short(s) failed ffprobe"
        return summary
    if expected_count is not None and summary["count_actual"] != expected_count:
        # Caught by the missing-loop above, but defence in depth: also catches
        # the case where MORE shorts exist than expected (treat as fail).
        summary["reason"] = (
            f"count mismatch: expected {expected_count}, found {summary['count_actual']}"
        )
        return summary
    if expected_count is None and summary["count_actual"] == 0:
        summary["reason"] = "no shorts on disk"
        return summary

    summary["ok"] = True
    return summary


def drop_day_master(
    day_dir: Path,
    *,
    date: str,
    expected_shorts_count: Optional[int] = None,
    ffprobe_fn: Optional[object] = None,
    expected_superlong_parts: Optional[int] = None,
) -> Dict[str, object]:
    """Delete ``day_{date}_master.mp4`` once the day's finals are on disk.

    ``expected_superlong_parts`` (opt-in): when set, the master ADDITIONALLY
    survives until every planned ``superlong/sl_NN.mp4`` final is non-empty.

    Called after ``phase_bc_day_first`` records the day in ``per_day_completed``.
    The master is a working artifact: encode_day_master → extract LFs/shorts
    → apply OST. By the time this function runs, every downstream consumer
    is done with it. Resume still works because
    ``_reconstruct_day_artifacts_from_disk`` can rebuild ``DayArtifacts``
    from the final ``lf_XX.mp4`` and ``short_XX.mp4`` files alone.

    Gate: at least one ``regular_lfs/lf_XX.mp4`` AND one ``shorts/short_XX.mp4``
    must exist with non-zero size. A day that produced zero LFs or zero shorts
    is treated as incomplete — master stays. Zero-byte finals (mid-write
    stubs from a crashed writer) do not count.

    Shorts-integrity gate: in addition to
    the existence checks above, the day's shorts must pass
    :func:`verify_shorts_integrity` (existence + ffprobe + optional count
    match against ``expected_shorts_count``). On failure the master is
    PRESERVED and an ERROR is logged with the master path so the operator
    can recover without a 6h per-day re-encode. The check is silent
    belt-and-suspenders when ``expected_shorts_count is None``: we still
    ffprobe whatever shorts are present.

    Never touches ``day_*_color_cache.json`` — the color cache is small and
    survives to short-circuit colour re-analysis on resume.

    Returns ``{"dropped": bool, "bytes_freed": int, "integrity": dict}``.
    """
    summary: Dict[str, object] = {
        "dropped": False,
        "bytes_freed": 0,
        "integrity": {},
    }
    day_dir = Path(day_dir)
    if not day_dir.is_dir():
        return summary

    master = day_dir / f"day_{date}_master.mp4"
    if not master.is_file():
        return summary

    lf_dir = day_dir / "regular_lfs"
    shorts_dir = day_dir / "shorts"
    has_lf = bool(_finalized_indices(lf_dir, _LF_FINAL_RE))
    has_short = bool(_finalized_indices(shorts_dir, _SHORT_FINAL_RE))
    if not (has_lf and has_short):
        logger.debug(
            "  master-drop: day %s has_lf=%s has_short=%s — keeping master",
            date,
            has_lf,
            has_short,
        )
        return summary

    if expected_superlong_parts:
        n_ready = finalized_superlong_count(day_dir)
        if n_ready < expected_superlong_parts:
            logger.debug(
                "  master-drop: day %s superlong %d/%d", date, n_ready, expected_superlong_parts
            )
            return summary

    # Shorts-integrity gate. Master stays on failure; ERROR identifies
    # the master path so the operator can re-derive shorts without a 6h
    # re-encode of the day.
    integrity = verify_shorts_integrity(
        day_dir,
        expected_count=expected_shorts_count,
        ffprobe_fn=ffprobe_fn,
    )
    summary["integrity"] = integrity
    if not integrity["ok"]:
        logger.error(
            "  master-drop: day %s shorts integrity FAILED (%s) — preserving "
            "master at %s for recovery. missing=%s bad=%s",
            date,
            integrity["reason"],
            master,
            integrity["missing"],
            [b["name"] for b in integrity["bad"]],  # type: ignore[index]
        )
        return summary

    try:
        size = master.stat().st_size
        master.unlink()
    except OSError as exc:
        logger.warning("  master-drop: failed to delete %s: %s", master, exc)
        return summary

    summary["dropped"] = True
    summary["bytes_freed"] = size
    logger.info(
        "  master-drop: day %s — removed %s (%.2f GB); resume will reconstruct from finals",
        date,
        master.name,
        size / (1024**3),
    )
    return summary
